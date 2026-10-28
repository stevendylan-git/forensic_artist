import csv
import io
import math
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User, UserRole
from app.models.audit import AuditLogEntry, AuditAction, AuditOutcome
from app.audit.ledger import AuditLedger
from app.schemas.audit import (
    AuditEntryResponse,
    AuditChainVerificationResult,
    AuditPaginatedResponse,
)
from app.api.deps import get_current_user, require_roles

router = APIRouter(prefix="/audit", tags=["Cryptographic Audit Ledger"])


@router.get("/verify", response_model=AuditChainVerificationResult)
async def verify_audit_chain(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verifies cryptographic hash integrity across the entire append-only audit ledger.
    Walks from genesis (seq=1) to the head and returns the first broken index (if any).
    """
    is_valid, count, broken_seq, expected, actual = await AuditLedger.verify_chain(db)
    message = (
        f"Audit ledger cryptographic chain is 100% intact and valid ({count} entries verified)."
        if is_valid
        else f"Cryptographic integrity violation detected at sequence #{broken_seq}."
    )

    # Log the verification event
    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.VERIFY_AUDIT,
        resource="/api/v1/audit/verify",
        outcome=AuditOutcome.SUCCESS if is_valid else AuditOutcome.ERROR,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={"is_valid": is_valid, "total_entries": count, "broken_at_seq": broken_seq}
    )
    await db.commit()

    return AuditChainVerificationResult(
        is_valid=is_valid,
        total_entries=count,
        broken_at_seq=broken_seq,
        expected_hash=expected,
        actual_hash=actual,
        message=message
    )


@router.get("/entries", response_model=AuditPaginatedResponse)
async def get_audit_entries(
    case_id: Optional[str] = Query(None, description="Filter by case UUID"),
    user_id: Optional[str] = Query(None, description="Filter by user UUID"),
    action: Optional[AuditAction] = Query(None, description="Filter by audit action"),
    start_date: Optional[datetime] = Query(None, description="Start date (UTC)"),
    end_date: Optional[datetime] = Query(None, description="End date (UTC)"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR, UserRole.AUDITOR)),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves filtered, paginated audit entries.
    RBAC: Auditor, Supervisor, or Admin.
    """
    stmt = select(AuditLogEntry)
    count_stmt = select(func.count(AuditLogEntry.id))

    if case_id:
        stmt = stmt.where(AuditLogEntry.case_id == case_id)
        count_stmt = count_stmt.where(AuditLogEntry.case_id == case_id)
    if user_id:
        stmt = stmt.where(AuditLogEntry.user_id == user_id)
        count_stmt = count_stmt.where(AuditLogEntry.user_id == user_id)
    if action:
        stmt = stmt.where(AuditLogEntry.action == action)
        count_stmt = count_stmt.where(AuditLogEntry.action == action)
    if start_date:
        stmt = stmt.where(AuditLogEntry.timestamp_utc >= start_date)
        count_stmt = count_stmt.where(AuditLogEntry.timestamp_utc >= start_date)
    if end_date:
        stmt = stmt.where(AuditLogEntry.timestamp_utc <= end_date)
        count_stmt = count_stmt.where(AuditLogEntry.timestamp_utc <= end_date)

    total_count = (await db.execute(count_stmt)).scalar() or 0
    offset = (page - 1) * limit

    stmt = stmt.order_by(desc(AuditLogEntry.seq)).offset(offset).limit(limit)
    entries = (await db.execute(stmt)).scalars().all()

    pages = math.ceil(total_count / limit) if total_count > 0 else 1

    return AuditPaginatedResponse(
        items=[AuditEntryResponse.model_validate(e.to_dict()) for e in entries],
        total=total_count,
        page=page,
        limit=limit,
        pages=pages
    )


@router.get("/export/csv")
async def export_audit_csv(
    request: Request,
    case_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[AuditAction] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.SUPERVISOR, UserRole.AUDITOR)),
    db: AsyncSession = Depends(get_db)
):
    """
    Exports filtered audit ledger entries to CSV.
    The export action itself is appended to the audit ledger.
    """
    stmt = select(AuditLogEntry)
    if case_id:
        stmt = stmt.where(AuditLogEntry.case_id == case_id)
    if user_id:
        stmt = stmt.where(AuditLogEntry.user_id == user_id)
    if action:
        stmt = stmt.where(AuditLogEntry.action == action)
    if start_date:
        stmt = stmt.where(AuditLogEntry.timestamp_utc >= start_date)
    if end_date:
        stmt = stmt.where(AuditLogEntry.timestamp_utc <= end_date)

    stmt = stmt.order_by(AuditLogEntry.seq.asc())
    entries = (await db.execute(stmt)).scalars().all()

    # Log the export action itself
    request.state.audited = True
    await AuditLedger.append_entry(
        session=db,
        action=AuditAction.EXPORT,
        resource="/api/v1/audit/export/csv",
        outcome=AuditOutcome.SUCCESS,
        user_id=current_user.id,
        role=current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role),
        case_id=case_id,
        ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
        details={
            "exported_entries_count": len(entries),
            "filters": {
                "case_id": case_id,
                "user_id": user_id,
                "action": action.value if action else None,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
            }
        }
    )
    await db.commit()

    # Generate CSV string
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "seq",
        "timestamp_utc",
        "user_id",
        "role",
        "ip",
        "user_agent",
        "case_id",
        "action",
        "resource",
        "outcome",
        "details",
        "prev_hash",
        "entry_hash"
    ])

    for e in entries:
        writer.writerow([
            e.seq,
            e.timestamp_utc.isoformat() if e.timestamp_utc else "",
            e.user_id or "",
            e.role or "",
            e.ip or "",
            e.user_agent or "",
            e.case_id or "",
            e.action.value if isinstance(e.action, AuditAction) else str(e.action),
            e.resource,
            e.outcome.value if isinstance(e.outcome, AuditOutcome) else str(e.outcome),
            e.details,
            e.prev_hash,
            e.entry_hash
        ])

    csv_data = output.getvalue()
    output.close()

    filename = f"audit_ledger_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
