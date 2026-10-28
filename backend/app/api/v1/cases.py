import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.user import User, UserRole
from app.models.case import Case, CaseAssignment, CompositeSession, Utterance, GeneratedImage
from app.models.audit import AuditAction, AuditOutcome
from app.audit.ledger import AuditLedger
from app.schemas.case import (
    CaseCreate,
    CaseUpdate,
    CaseResponse,
    CaseAssignRequest,
    DescribeRequest,
    DescribeResponse,
    GenerateInitialRequest,
    GenerateInitialResponse,
    GenerateCandidatesRequest,
    GenerateCandidatesResponse,
    CandidateItemResponse,
    RefineFeedbackRequest,
    RefineFeedbackResponse,
    TimelineResponse,
    TimelineItemResponse,
    RevertRequest,
    RevertResponse,
    UndoRedoResponse,
    CompareResponse,
    CandidateGridRequest,
    CandidateGridResponse,
    GridSelectRequest,
    GridSelectResponse,
    ExportRequest,
)
from app.schemas.face_spec import FaceSpec, FaceSpecPatch
from app.agents.parser import parser_agent, AdversarialCelebrityError
from app.services.image_gen import (
    generate_initial,
    generate_candidates,
    GenerationSafetyBlockError,
)
from app.services.storage import storage_service
from app.services.websocket import ws_manager
from app.services.refinement_pipeline import refinement_pipeline
from app.services.grid_service import grid_service
from app.services.export_service import export_service
from app.api.deps import (
    get_current_user,
    require_roles,
    get_case_for_user,
    require_face_composite_access,
)

router = APIRouter(prefix="/cases", tags=["Case Management & RBAC"])



@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    case_in: CaseCreate,
    request: Request,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR, UserRole.INVESTIGATOR)),
    db: AsyncSession = Depends(get_db),
):
    """
    Creates a new case.
    RBAC: Admin, Supervisor, or Investigator.
    Creator is automatically assigned to the case.
    """
    stmt = select(Case).where(Case.case_number == case_in.case_number)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Case number '{case_in.case_number}' already exists."
        )

    case = Case(
        case_number=case_in.case_number,
        title=case_in.title,
        status=case_in.status,
        created_by_id=current_user.id,
        notes=case_in.notes,
    )
    db.add(case)
    await db.flush()

    # Automatically assign creator
    assignment = CaseAssignment(
        case_id=case.id,
        user_id=current_user.id,
        assigned_by=current_user.id,
    )
    db.add(assignment)

    # Assign additional users if requested
    for uid in case_in.assigned_user_ids:
        if uid != current_user.id:
            db.add(CaseAssignment(
                case_id=case.id,
                user_id=uid,
                assigned_by=current_user.id,
            ))

    await db.flush()

    # Audit log
    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.CREATE_CASE,
        resource=f"/api/v1/cases/{case.id}",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case.id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={"case_number": case.case_number, "title": case.title}
    )
    await db.commit()

    reloaded_case = await db.execute(
        select(Case).where(Case.id == case.id).options(selectinload(Case.assignments))
    )
    return CaseResponse.model_validate(reloaded_case.scalar_one().to_dict())


@router.get("", response_model=List[CaseResponse])
async def list_cases(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Lists cases with need-to-know filtering:
    - Supervisor, Admin, and Auditor see all cases.
    - Investigator and ForensicArtist only see cases they are assigned to.
    """
    user_role = current_user.role
    if isinstance(user_role, str):
        user_role = UserRole(user_role)

    stmt = select(Case).options(selectinload(Case.assignments))

    if user_role not in [UserRole.ADMIN, UserRole.SUPERVISOR, UserRole.AUDITOR]:
        stmt = stmt.join(Case.assignments).where(CaseAssignment.user_id == current_user.id)

    result = await db.execute(stmt)
    cases = result.scalars().all()
    return [CaseResponse.model_validate(c.to_dict()) for c in cases]


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieves case metadata.
    Enforces need-to-know access (assigned users or supervisory/admin/audit role).
    """
    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.VIEW_CASE,
        resource=f"/api/v1/cases/{case.id}",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case.id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={"case_number": case.case_number}
    )
    await db.commit()
    return CaseResponse.model_validate(case.to_dict())


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    update_data: CaseUpdate,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR, UserRole.INVESTIGATOR)),
    db: AsyncSession = Depends(get_db),
):
    """
    Updates case details.
    RBAC: Admin, Supervisor, or assigned Investigator.
    """
    if update_data.title is not None:
        case.title = update_data.title
    if update_data.status is not None:
        case.status = update_data.status
    if update_data.notes is not None:
        case.notes = update_data.notes

    await db.flush()

    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.UPDATE_CASE,
        resource=f"/api/v1/cases/{case.id}",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case.id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details=update_data.model_dump(exclude_unset=True)
    )
    await db.commit()

    return CaseResponse.model_validate(case.to_dict())


@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_case(
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR)),
    db: AsyncSession = Depends(get_db),
):
    """
    Deletes a case.
    RBAC: Admin and Supervisor only.
    """
    case_number = case.case_number
    case_id = case.id

    await db.delete(case)
    await db.flush()

    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.DELETE_CASE,
        resource=f"/api/v1/cases/{case_id}",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case_id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={"case_number": case_number}
    )
    await db.commit()


@router.post("/{case_id}/assign", response_model=CaseResponse)
async def assign_users_to_case(
    assign_req: CaseAssignRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR)),
    db: AsyncSession = Depends(get_db),
):
    """
    Assigns users to a case for need-to-know access control.
    RBAC: Supervisor or Admin only.
    """
    existing_assigned_ids = set([a.user_id for a in case.assignments])
    new_assignments = []

    for uid in assign_req.user_ids:
        if uid not in existing_assigned_ids:
            u_stmt = select(User).where(User.id == uid)
            u_res = await db.execute(u_stmt)
            if not u_res.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User with ID {uid} does not exist"
                )
            assignment = CaseAssignment(
                case_id=case.id,
                user_id=uid,
                assigned_by=current_user.id,
            )
            db.add(assignment)
            new_assignments.append(uid)

    await db.flush()

    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.ASSIGN_USER,
        resource=f"/api/v1/cases/{case.id}/assign",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case.id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={"newly_assigned_user_ids": new_assignments}
    )
    await db.commit()
    await db.refresh(case, ["assignments"])
    return CaseResponse.model_validate(case.to_dict())


@router.get("/{case_id}/composite/face-image")
async def get_case_face_image(
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    Demonstrates strict Auditor isolation:
    Auditors are strictly prohibited from viewing face composites.
    Only authorized investigators/artists assigned to the case (or supervisor/admin) may view face composites.
    """
    return {
        "case_id": case.id,
        "message": "Authorized access to face composite granted for assigned user/role.",
        "user_role": current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    }


@router.post("/{case_id}/describe", response_model=DescribeResponse, status_code=status.HTTP_200_OK)
async def describe_case_face(
    case_id: str,
    describe_in: DescribeRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR, UserRole.INVESTIGATOR, UserRole.FORENSIC_ARTIST)),
    db: AsyncSession = Depends(get_db),
):
    """
    Phase 2a: Natural Language Face Description Endpoint.
    1. Intercepts and rejects adversarial celebrity comparisons, logging denial to audit ledger.
    2. Stores witness utterance.
    3. Calls NL parser to generate FaceSpecPatch from witness statement.
    4. Applies patch to current FaceSpec (handling hedges, negations, corrections).
    5. Writes an immutable FaceSpec CompositeSession version row.
    6. Logs version update to cryptographic audit chain.
    """
    # 1. Adversarial Check & Handling
    try:
        # Pre-check celebrity reference
        from app.agents.parser import detect_celebrity_reference
        celeb = detect_celebrity_reference(describe_in.utterance)
        if celeb:
            raise AdversarialCelebrityError(
                f"Adversarial celebrity reference detected: '{celeb}'. Forensic composites strictly prohibit celebrity comparisons."
            )
    except AdversarialCelebrityError as ace:
        request.state.audited = True
        await AuditLedger.append_entry(
            session=db,
            action=AuditAction.EDIT,
            resource=f"/api/v1/cases/{case.id}/describe",
            outcome=AuditOutcome.DENIED,
            user_id=current_user.id,
            role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
            case_id=case.id,
            ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            details={
                "rejection_reason": "Adversarial celebrity reference prohibited",
                "utterance": describe_in.utterance,
                "error": str(ace)
            }
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ace)
        )

    # 2. Fetch current version from CompositeSession
    stmt = (
        select(CompositeSession)
        .where(CompositeSession.case_id == case.id)
        .order_by(CompositeSession.version.desc())
    )
    result = await db.execute(stmt)
    latest_session = result.scalars().first()

    if latest_session:
        current_spec = FaceSpec.model_validate_json(latest_session.face_spec_json)
        next_version = latest_session.version + 1
    else:
        current_spec = FaceSpec()
        next_version = 1

    # 3. Store Utterance record
    utterance_record = Utterance(
        case_id=case.id,
        speaker=describe_in.speaker,
        text=describe_in.utterance
    )
    db.add(utterance_record)
    await db.flush()

    # 4. Parse utterance with Gemini agent
    try:
        patch = await parser_agent.parse_utterance(
            utterance=describe_in.utterance,
            utterance_id=utterance_record.id,
            current_spec=current_spec
        )
    except AdversarialCelebrityError as ace:
        request.state.audited = True
        await AuditLedger.append_entry(
            session=db,
            action=AuditAction.EDIT,
            resource=f"/api/v1/cases/{case.id}/describe",
            outcome=AuditOutcome.DENIED,
            user_id=current_user.id,
            role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
            case_id=case.id,
            ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            details={
                "rejection_reason": "Adversarial celebrity reference prohibited",
                "utterance": describe_in.utterance,
                "error": str(ace)
            }
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ace)
        )

    utterance_record.parsed_patch_json = patch.model_dump_json()
    await db.flush()

    # 5. Apply patch to FaceSpec
    new_spec = current_spec.apply_patch(patch, utterance_id=utterance_record.id)
    diff_data = current_spec.diff(new_spec)

    # 6. Write immutable FaceSpec version row (CompositeSession)
    new_session = CompositeSession(
        case_id=case.id,
        version=next_version,
        face_spec_json=new_spec.model_dump_json(),
        locked_regions_json=json.dumps(new_spec.get_locked_regions()),
        utterance_id=utterance_record.id
    )
    db.add(new_session)
    await db.flush()

    # 7. Log version row creation to cryptographically chained AuditLedger
    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.CREATE_VERSION,
        resource=f"/api/v1/cases/{case.id}/describe",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case.id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={
            "version": next_version,
            "utterance_id": utterance_record.id,
            "diff": diff_data
        }
    )
    await db.commit()

    return DescribeResponse(
        case_id=case.id,
        utterance_id=utterance_record.id,
        version=next_version,
        patch=patch,
        face_spec=new_spec,
        diff=diff_data
    )


@router.post("/{case_id}/generate/initial", response_model=GenerateInitialResponse)
async def generate_initial_composite(
    case_id: str,
    gen_in: GenerateInitialRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR, UserRole.INVESTIGATOR, UserRole.FORENSIC_ARTIST)),
    db: AsyncSession = Depends(get_db),
):
    """
    Generates the initial facial composite image deterministically from the current FaceSpec version.
    - Deterministic prompt synthesis based on confirmed attributes.
    - Real-time progress streamed over WebSocket.
    - Envelope encryption with per-case DEK into storage.
    - Version linking row in database.
    - Audit ledger recording with latency and estimated cost.
    """
    stmt = (
        select(CompositeSession)
        .where(CompositeSession.case_id == case.id)
        .order_by(CompositeSession.version.desc())
    )
    res = await db.execute(stmt)
    latest_session = res.scalars().first()

    if latest_session:
        face_spec = FaceSpec.model_validate_json(latest_session.face_spec_json)
        current_version = latest_session.version
    else:
        face_spec = FaceSpec()
        current_version = 1

    try:
        gen_result = await generate_initial(
            face_spec=face_spec,
            case_id=case.id,
            version=current_version,
            utterance_id=gen_in.utterance_id or (latest_session.utterance_id if latest_session else None),
            triggering_call="generate_initial",
            db=db,
            confidence_threshold=gen_in.confidence_threshold,
        )
    except GenerationSafetyBlockError as sbe:
        request.state.audited = True
        await AuditLedger.append_entry(
            session=db,
            action=AuditAction.CREATE_VERSION,
            resource=f"/api/v1/cases/{case.id}/generate/initial",
            outcome=AuditOutcome.DENIED,
            user_id=current_user.id,
            role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
            case_id=case.id,
            ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            details={"error": str(sbe), "safety_refusal": True}
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(sbe)
        )

    # Audit log successful generation
    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.CREATE_VERSION,
        resource=f"/api/v1/cases/{case.id}/generate/initial",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case.id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={
            "image_id": gen_result.image_id,
            "version": gen_result.version,
            "prompt_hash": gen_result.prompt_hash,
            "model_id": gen_result.model_id,
            "latency_ms": gen_result.latency_ms,
            "estimated_cost": gen_result.estimated_cost
        }
    )
    await db.commit()

    return GenerateInitialResponse(
        case_id=case.id,
        image_id=gen_result.image_id,
        version=gen_result.version,
        storage_path=gen_result.storage_path,
        prompt=gen_result.prompt,
        prompt_hash=gen_result.prompt_hash,
        model_id=gen_result.model_id,
        latency_ms=gen_result.latency_ms,
        estimated_cost=gen_result.estimated_cost,
        status="success"
    )


@router.post("/{case_id}/generate/candidates", response_model=GenerateCandidatesResponse)
async def generate_recognition_grid(
    case_id: str,
    cand_in: GenerateCandidatesRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR, UserRole.INVESTIGATOR, UserRole.FORENSIC_ARTIST)),
    db: AsyncSession = Depends(get_db),
):
    """
    Generates N candidate composite variations (default N=9 for recognition grid),
    varying ONLY the requested feature while holding all other features strictly constant.
    - Concurrency bounded by Semaphore.
    - Gracefully tolerates partial failures.
    - Stores each successful image encrypted in object store.
    - Real-time progress streamed over WebSocket.
    """
    stmt = (
        select(CompositeSession)
        .where(CompositeSession.case_id == case.id)
        .order_by(CompositeSession.version.desc())
    )
    res = await db.execute(stmt)
    latest_session = res.scalars().first()

    if latest_session:
        face_spec = FaceSpec.model_validate_json(latest_session.face_spec_json)
        current_version = latest_session.version
    else:
        face_spec = FaceSpec()
        current_version = 1

    grid_result = await generate_candidates(
        face_spec=face_spec,
        n=cand_in.n,
        vary=cand_in.vary,
        case_id=case.id,
        version=current_version,
        utterance_id=cand_in.utterance_id or (latest_session.utterance_id if latest_session else None),
        triggering_call="generate_candidates",
        db=db,
        confidence_threshold=cand_in.confidence_threshold,
    )

    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.EDIT,
        resource=f"/api/v1/cases/{case.id}/generate/candidates",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case.id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={
            "feature": cand_in.vary,
            "total_requested": grid_result.total_requested,
            "total_successful": grid_result.total_successful,
            "total_failed": grid_result.total_failed,
            "total_latency_ms": grid_result.total_latency_ms,
            "total_estimated_cost": grid_result.total_estimated_cost
        }
    )
    await db.commit()

    candidate_responses = [
        CandidateItemResponse(
            candidate_index=c.candidate_index,
            variation_feature=c.variation_feature,
            variation_value=c.variation_value,
            image_id=c.image_id,
            storage_path=c.storage_path,
            latency_ms=c.latency_ms,
            status=c.status,
            error=c.error
        )
        for c in grid_result.candidates
    ]

    return GenerateCandidatesResponse(
        case_id=case.id,
        feature=grid_result.feature,
        total_requested=grid_result.total_requested,
        total_successful=grid_result.total_successful,
        total_failed=grid_result.total_failed,
        candidates=candidate_responses,
        total_latency_ms=grid_result.total_latency_ms,
        total_estimated_cost=grid_result.total_estimated_cost
    )


@router.get("/{case_id}/images/{image_id}")
async def get_rendered_image(
    case_id: str,
    image_id: str,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieves and decrypts a rendered composite image or candidate variation.
    Strictly isolated: Auditor role is prohibited from viewing face images.
    """
    stmt = select(GeneratedImage).where(
        GeneratedImage.id == image_id,
        GeneratedImage.case_id == case.id
    )
    res = await db.execute(stmt)
    img_record = res.scalar_one_or_none()

    if not img_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID {image_id} not found for case {case_id}"
        )

    try:
        decrypted_bytes = await storage_service.retrieve_image(
            case_id=case.id,
            relative_storage_path=img_record.image_path,
            db=db
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Encrypted image artifact not found on storage."
        )

    return Response(content=decrypted_bytes, media_type="image/png")


@router.websocket("/{case_id}/ws/generation")
async def generation_progress_ws(
    websocket: WebSocket,
    case_id: str,
):
    """
    WebSocket endpoint for real-time progress streaming of image generation.
    Streams events: queued -> generating -> done/failed.
    """
    await ws_manager.connect(case_id, websocket)
    try:
        while True:
            # Keep-alive listener / client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')
    except WebSocketDisconnect:
        ws_manager.disconnect(case_id, websocket)
    except Exception:
        ws_manager.disconnect(case_id, websocket)


@router.post("/{case_id}/refine", response_model=RefineFeedbackResponse)
async def refine_composite(
    case_id: str,
    feedback_in: RefineFeedbackRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    Submits conversational feedback from the witness to refine the facial composite.
    RefinementAgent translates feedback into strict tool calls (edit_region, lock_feature, etc.).
    Locality enforcement guarantees that only requested regions take new pixels.
    """
    role_str = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    try:
        pipeline_res = await refinement_pipeline.execute_refinement(
            case_id=case.id,
            witness_message=feedback_in.message,
            db=db,
            utterance_id=feedback_in.utterance_id,
            user_id=current_user.id,
            role=role_str
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )

    return RefineFeedbackResponse(
        case_id=case.id,
        current_version=pipeline_res["current_version"],
        tool_calls_executed=pipeline_res["tool_calls_executed"],
        results=pipeline_res["results"]
    )


@router.get("/{case_id}/timeline", response_model=TimelineResponse)
async def get_version_timeline(
    case_id: str,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    Lists the complete version history timeline of the facial composite.
    Includes locality scores, locked regions, triggering calls, and timestamps.
    """
    stmt = (
        select(CompositeSession)
        .where(CompositeSession.case_id == case.id)
        .order_by(CompositeSession.version.asc())
    )
    res = await db.execute(stmt)
    sessions = res.scalars().all()

    timeline_items = []
    for s in sessions:
        try:
            locked = json.loads(s.locked_regions_json)
        except Exception:
            locked = []

        try:
            fspec = json.loads(s.face_spec_json)
        except Exception:
            fspec = None

        timeline_items.append(
            TimelineItemResponse(
                id=s.id,
                version=s.version,
                created_at=s.created_at,
                triggering_call=s.triggering_call,
                locality_score=s.locality_score,
                locality_unverified=s.locality_unverified,
                locked_regions=locked,
                image_url=s.rendered_image_url,
                image_path=s.image_path,
                utterance_id=s.utterance_id,
                face_spec=fspec
            )
        )

    return TimelineResponse(
        case_id=case.id,
        total_versions=len(timeline_items),
        timeline=timeline_items
    )


@router.post("/{case_id}/revert", response_model=RevertResponse)
async def revert_version(
    case_id: str,
    revert_in: RevertRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    Reverts a specific region (or the whole composite) to any earlier version.
    Composites the earlier version's pixels and synchronizes FaceSpec.
    """
    role_str = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)

    # Fetch latest session
    latest_stmt = (
        select(CompositeSession)
        .where(CompositeSession.case_id == case.id)
        .order_by(CompositeSession.version.desc())
        .limit(1)
    )
    latest_session = (await db.execute(latest_stmt)).scalar_one_or_none()
    if not latest_session:
        raise HTTPException(status_code=404, detail="No existing composite session to revert from.")

    try:
        locked_regions = json.loads(latest_session.locked_regions_json)
    except Exception:
        locked_regions = []

    try:
        revert_res = await refinement_pipeline._handle_revert(
            case_id=case.id,
            to_version=revert_in.to_version,
            region=revert_in.region,
            latest_session=latest_session,
            locked_regions=locked_regions,
            utterance_id=None,
            user_id=current_user.id,
            role=role_str,
            db=db
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if revert_res.get("status") == "blocked":
        raise HTTPException(status_code=400, detail=revert_res.get("message"))

    return RevertResponse(
        case_id=case.id,
        version=revert_res["version"],
        reverted_to_version=revert_res["reverted_to_version"],
        region=revert_res["region"],
        locality_score=revert_res["locality_score"],
        image_url=revert_res.get("image_url"),
        diff=revert_res.get("diff", {})
    )


@router.post("/{case_id}/undo", response_model=UndoRedoResponse)
async def undo_last_edit(
    case_id: str,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    Undoes the most recent edit by reverting to version V - 1.
    """
    role_str = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)

    latest_stmt = (
        select(CompositeSession)
        .where(CompositeSession.case_id == case.id)
        .order_by(CompositeSession.version.desc())
        .limit(1)
    )
    latest_session = (await db.execute(latest_stmt)).scalar_one_or_none()
    if not latest_session or latest_session.version <= 1:
        raise HTTPException(status_code=400, detail="Cannot undo: Already at version 1.")

    target_version = latest_session.version - 1
    try:
        locked_regions = json.loads(latest_session.locked_regions_json)
    except Exception:
        locked_regions = []

    revert_res = await refinement_pipeline._handle_revert(
        case_id=case.id,
        to_version=target_version,
        region=None,  # Full composite revert
        latest_session=latest_session,
        locked_regions=locked_regions,
        utterance_id=None,
        user_id=current_user.id,
        role=role_str,
        db=db
    )

    return UndoRedoResponse(
        case_id=case.id,
        version=revert_res["version"],
        action="undo",
        image_url=revert_res.get("image_url"),
        message=f"Undid changes by restoring version {target_version} as version {revert_res['version']}."
    )


@router.post("/{case_id}/redo", response_model=UndoRedoResponse)
async def redo_edit(
    case_id: str,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    Redoes an edit if the current version was produced by an undo/revert operation.
    """
    role_str = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)

    latest_stmt = (
        select(CompositeSession)
        .where(CompositeSession.case_id == case.id)
        .order_by(CompositeSession.version.desc())
        .limit(1)
    )
    latest_session = (await db.execute(latest_stmt)).scalar_one_or_none()
    if not latest_session:
        raise HTTPException(status_code=400, detail="No composite session found.")

    # Check if the latest session was an undo or revert to an earlier version
    trigger = latest_session.triggering_call or ""
    if "revert:full:to_v" in trigger:
        reverted_to = int(trigger.split("to_v")[-1])
        target_version = reverted_to + 1
        # Check if target version exists
        target_check = (await db.execute(
            select(CompositeSession).where(
                CompositeSession.case_id == case.id,
                CompositeSession.version == target_version
            )
        )).scalar_one_or_none()

        if target_check:
            try:
                locked_regions = json.loads(latest_session.locked_regions_json)
            except Exception:
                locked_regions = []

            revert_res = await refinement_pipeline._handle_revert(
                case_id=case.id,
                to_version=target_version,
                region=None,
                latest_session=latest_session,
                locked_regions=locked_regions,
                utterance_id=None,
                user_id=current_user.id,
                role=role_str,
                db=db
            )
            return UndoRedoResponse(
                case_id=case.id,
                version=revert_res["version"],
                action="redo",
                image_url=revert_res.get("image_url"),
                message=f"Redid changes by advancing to state from version {target_version}."
            )

    raise HTTPException(status_code=400, detail="No redo history available.")


@router.get("/{case_id}/compare", response_model=CompareResponse)
async def compare_versions_api(
    case_id: str,
    a: int,
    b: int,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    Compares two historical versions side-by-side.
    Returns metadata, image URLs, and FaceSpec morphological diff.
    """
    role_str = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    try:
        cmp_res = await refinement_pipeline._handle_compare_versions(
            case_id=case.id,
            a=a,
            b=b,
            user_id=current_user.id,
            role=role_str,
            db=db
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CompareResponse(
        case_id=case.id,
        version_a=cmp_res["version_a"],
        version_b=cmp_res["version_b"],
        diff=cmp_res["diff"]
    )


@router.post("/{case_id}/grid", response_model=CandidateGridResponse)
async def generate_candidate_grid(
    case_id: str,
    grid_in: CandidateGridRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    POST /cases/{id}/grid {feature or "whole_face"} returns a 3x3 set of candidates
    varying only that feature (using generate_candidates + locality.composite so the rest of the face stays identical).
    Supports:
    - standard: standard catalog variations.
    - none_of_these: regenerates a new grid with greater spread.
    - blend_toward: generates a new grid around the picked one #k with smaller variation.
    """
    grid_res = await grid_service.generate_recognition_grid(
        case=case,
        feature=grid_in.feature,
        action=grid_in.action or "standard",
        blend_candidate_index=grid_in.blend_candidate_index,
        spread=grid_in.spread,
        confidence_threshold=grid_in.confidence_threshold,
        db=db,
    )

    request.state.audited = True
    role_str = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.EDIT,
        resource=f"/api/v1/cases/{case.id}/grid",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=role_str,
        case_id=case.id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={
            "feature": grid_in.feature,
            "action": grid_in.action,
            "spread": grid_in.spread,
            "blend_candidate_index": grid_in.blend_candidate_index,
            "total_candidates": grid_res.total_candidates,
            "latency_ms": grid_res.total_latency_ms,
            "estimated_cost": grid_res.total_estimated_cost
        }
    )
    await db.commit()

    return grid_res


@router.post("/{case_id}/grid/select", response_model=GridSelectResponse)
@router.post("/{case_id}/grid/pick", response_model=GridSelectResponse)
async def select_grid_candidate(
    case_id: str,
    select_in: GridSelectRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(require_face_composite_access),
    db: AsyncSession = Depends(get_db),
):
    """
    When the witness picks one candidate from the grid, applies it as a new version (logged).
    """
    request.state.audited = True
    role_str = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    try:
        select_res = await grid_service.select_candidate(
            case=case,
            candidate_index=select_in.candidate_index,
            candidate_id=select_in.candidate_id,
            user_id=current_user.id,
            role=role_str,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return select_res


@router.post("/{case_id}/export")
async def export_case_composite_post(
    case_id: str,
    export_in: ExportRequest,
    request: Request,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Exports facial composite artifact as court-admissible PNG or single-page PDF.
    - Two output modes: photoreal and pencil-sketch.
    - Upscale/enhance to high resolution.
    - Burn in watermark: 'AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY' + case ID + timestamp.
    - Embed metadata (PNG tEXt/EXIF): case ID, version ID, FaceSpec hash, user ID, export time, model IDs.
    - Permission checked: Auditor cannot export (HTTP 403 Forbidden).
    - Written to audit chain with SHA-256 file hash.
    """
    request.state.audited = True
    file_bytes, mime_type, file_hash = await export_service.export_composite(
        case=case,
        current_user=current_user,
        db=db,
        format=export_in.format,
        mode=export_in.mode,
        upscale=export_in.upscale,
        version=export_in.version,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
    )

    filename = f"forensic_composite_{case.case_number}_v{export_in.version or 'latest'}.{export_in.format}"
    return Response(
        content=file_bytes,
        media_type=mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-File-Hash": file_hash,
            "X-Watermark": "AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY"
        }
    )


@router.get("/{case_id}/export")
async def export_case_composite_get(
    case_id: str,
    request: Request,
    format: str = "png",
    mode: str = "photoreal",
    upscale: bool = True,
    version: Optional[int] = None,
    case: Case = Depends(get_case_for_user),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    GET export endpoint for direct browser downloads.
    """
    request.state.audited = True
    file_bytes, mime_type, file_hash = await export_service.export_composite(
        case=case,
        current_user=current_user,
        db=db,
        format=format,
        mode=mode,
        upscale=upscale,
        version=version,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
    )

    filename = f"forensic_composite_{case.case_number}_v{version or 'latest'}.{format}"
    return Response(
        content=file_bytes,
        media_type=mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-File-Hash": file_hash,
            "X-Watermark": "AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY"
        }
    )


