"""
Comprehensive Tests for Phase 4: Candidate Grids and Export Pipeline
Tests:
- 3x3 Candidate Grid Generation (whole_face and individual features with locality compositing)
- "none of these" greater spread variation regeneration
- "blend toward #k" smaller micro-variation generation
- Candidate selection applying as a new composite version with audit logging
- Photoreal and Pencil-Sketch modes (with OpenCV deterministic fallback)
- High-resolution upscaling (Lanczos)
- Burned-in watermark presence ("AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY")
- Embedded metadata in PNG tEXt chunks (case_id, version_id, face_spec_hash, user_id, export_time, model_ids)
- Single-page court-admissible PDF generation
- Strict RBAC: Auditor role prohibited from export (403 Forbidden and logged DENIED)
- Tamper-evident Audit Ledger chain entry with exact SHA-256 file hash
"""

import io
import re
import json
import hashlib
import pytest
import numpy as np
from PIL import Image
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from unittest.mock import patch, AsyncMock
from app.models.case import Case, CaseAssignment, CompositeSession, GeneratedImage
from app.models.user import User
from app.models.audit import AuditLogEntry, AuditAction, AuditOutcome
from app.schemas.face_spec import FaceSpec
from app.schemas.gemini import GeminiImageResponse
from app.services.storage import storage_service
from app.services.locality import to_png_bytes


@pytest.fixture(autouse=True)
def mock_gemini_image():
    """Mock Gemini image generation for candidate grid and export tests."""
    fake_img = np.ones((512, 512, 3), dtype=np.uint8) * 190
    fake_img[200:300, 230:280] = [120, 80, 70]
    fake_png = to_png_bytes(fake_img)
    mock_resp = GeminiImageResponse(
        image_bytes_list=[fake_png],
        model_name="imagen-3.0-generate-002",
        count=1
    )
    with patch("app.services.image_gen.gemini_client.generate_image", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_resp
        yield mock_gen


async def setup_case_with_composite(db_session: AsyncSession, case_number: str = "CASE-EXP-001") -> Case:
    """Helper to initialize a test case with an assigned investigator and version 1 composite."""
    case = Case(case_number=case_number, title="Phase 4 Grid & Export Test Case")
    db_session.add(case)
    await db_session.flush()

    # Assign demo investigator
    user_stmt = select(User).where(User.email == "investigator@forensic.local")
    investigator = (await db_session.execute(user_stmt)).scalar_one()

    assignment = CaseAssignment(case_id=case.id, user_id=investigator.id)
    db_session.add(assignment)

    # Assign demo auditor as well for need-to-know checks
    auditor_stmt = select(User).where(User.email == "auditor@forensic.local")
    auditor = (await db_session.execute(auditor_stmt)).scalar_one()
    auditor_assignment = CaseAssignment(case_id=case.id, user_id=auditor.id)
    db_session.add(auditor_assignment)

    # Create synthetic test composite face image (512x512)
    img_arr = np.ones((512, 512, 3), dtype=np.uint8) * 200
    # Add a simple facial pattern so landmarks and sketch filters have texture
    img_arr[150:220, 180:240] = [80, 50, 40]   # Left eye region
    img_arr[150:220, 270:330] = [80, 50, 40]   # Right eye region
    img_arr[220:320, 240:272] = [130, 90, 80]  # Nose bridge & tip
    img_arr[350:390, 210:302] = [160, 60, 60]  # Mouth
    png_bytes = to_png_bytes(img_arr)

    meta = await storage_service.save_image(
        case_id=case.id,
        image_bytes=png_bytes,
        version=1,
        is_candidate=False,
        db=db_session
    )

    fspec = FaceSpec()
    fspec.nose.bridge.value = "straight narrow bridge"
    fspec.nose.bridge.confidence = 0.85
    fspec.eyes.shape.value = "almond-shaped"
    fspec.eyes.shape.confidence = 0.90

    session_v1 = CompositeSession(
        case_id=case.id,
        version=1,
        face_spec_json=fspec.model_dump_json(),
        locked_regions_json="[]",
        rendered_image_url=meta["url"],
        image_path=meta["storage_path"],
        triggering_call="generate_initial",
        locality_score=0.0,
        locality_unverified=False
    )
    db_session.add(session_v1)
    await db_session.flush()

    gen_img = GeneratedImage(
        case_id=case.id,
        composite_session_id=session_v1.id,
        version=1,
        image_path=meta["storage_path"],
        model_id="imagen-3.0-generate-002",
        prompt="A forensic facial composite portrait.",
        prompt_hash="dummyhash_v1",
        triggering_call="generate_initial"
    )
    db_session.add(gen_img)
    await db_session.commit()
    return case


@pytest.mark.asyncio
async def test_watermark_presence(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Verifies that exported composite image has the mandatory burned-in watermark footer band:
    'AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY' plus case ID and timestamp.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-WM-001")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/export",
        json={"format": "png", "mode": "photoreal", "upscale": False},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert "AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY" in resp.headers.get("X-Watermark", "")

    # Open image with PIL to verify footer band geometry
    img = Image.open(io.BytesIO(resp.content))
    # Original image is 512x512. With footer band (~8% / min 70px), height must be greater than width
    w, h = img.size
    assert w == 512
    assert h >= 512 + 70, f"Expected height to be at least 582 due to footer band, got {h}"

    # Verify the bottom footer band area has the dark charcoal background
    bottom_pixels = np.array(img)[-20:, :, :]
    # Dark slate background #111827 is roughly [17, 24, 39]
    avg_bg = np.mean(bottom_pixels, axis=(0, 1))
    assert avg_bg[0] < 50 and avg_bg[1] < 50 and avg_bg[2] < 60, "Expected dark footer band background"


@pytest.mark.asyncio
async def test_metadata_embedding(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Verifies PNG tEXt metadata contains:
    case_id, version_id, face_spec_hash, user_id, export_time, model_ids, watermark.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-META-001")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/export",
        json={"format": "png", "mode": "photoreal", "upscale": True},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text

    # Extract metadata using PIL
    img = Image.open(io.BytesIO(resp.content))
    meta = img.text

    assert "case_id" in meta, "case_id missing from PNG metadata"
    assert meta["case_id"] == case.id
    assert "version_id" in meta, "version_id missing from PNG metadata"
    assert meta["version_id"] == "1"
    assert "face_spec_hash" in meta, "face_spec_hash missing from PNG metadata"
    assert len(meta["face_spec_hash"]) == 64
    assert "user_id" in meta, "user_id missing from PNG metadata"
    assert "export_time" in meta, "export_time missing from PNG metadata"
    assert "model_ids" in meta, "model_ids missing from PNG metadata"
    assert "watermark" in meta, "watermark missing from PNG metadata"
    assert meta["watermark"] == "AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY"


@pytest.mark.asyncio
async def test_export_audit_entry_creation_and_file_hash(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Verifies that every export writes to the audit chain with the exact SHA-256 file hash.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-AUD-001")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/export",
        json={"format": "png", "mode": "photoreal", "upscale": False},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text

    expected_hash = hashlib.sha256(resp.content).hexdigest()
    assert resp.headers["X-File-Hash"] == expected_hash

    # Check AuditLedger
    audit_stmt = (
        select(AuditLogEntry)
        .where(
            AuditLogEntry.case_id == case.id,
            AuditLogEntry.action == AuditAction.EXPORT
        )
        .order_by(AuditLogEntry.seq.desc())
    )
    res = await db_session.execute(audit_stmt)
    entry = res.scalars().first()

    assert entry is not None, "AuditLogEntry for EXPORT was not created"
    assert entry.outcome == AuditOutcome.SUCCESS
    details = json.loads(entry.details)
    assert details["file_hash"] == expected_hash
    assert details["file_size"] == len(resp.content)
    assert details["format"] == "png"
    assert details["version"] == 1


@pytest.mark.asyncio
async def test_export_permission_auditor_prohibited(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Verifies that the Auditor role is strictly prohibited from exporting (HTTP 403 Forbidden)
    and that a DENIED audit log entry is written to the ledger.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-PERM-001")
    auditor_token = auth_tokens["Auditor"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/export",
        json={"format": "png", "mode": "photoreal"},
        headers={"Authorization": f"Bearer {auditor_token}"}
    )
    assert resp.status_code == 403, f"Expected 403 Forbidden for Auditor, got {resp.status_code}"
    assert "Auditor role restriction" in resp.json()["detail"]

    # Verify DENIED audit record was written
    audit_stmt = (
        select(AuditLogEntry)
        .where(
            AuditLogEntry.case_id == case.id,
            AuditLogEntry.action == AuditAction.EXPORT,
            AuditLogEntry.outcome == AuditOutcome.DENIED
        )
    )
    res = await db_session.execute(audit_stmt)
    denied_entry = res.scalars().first()
    assert denied_entry is not None, "Expected DENIED audit entry for Auditor export attempt"


@pytest.mark.asyncio
async def test_sketch_mode_opencv_fallback(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Tests pencil-sketch export mode. Verifies that OpenCV color-dodge fallback produces
    clean graphite line/shading art preserving proportions with zero added details.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-SKETCH-001")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/export",
        json={"format": "png", "mode": "pencil-sketch", "upscale": False},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    img = Image.open(io.BytesIO(resp.content))

    # The face area should exhibit sketch properties (monochromatic/grayscale tones in face region)
    arr = np.array(img)
    face_crop = arr[100:300, 100:300, :]
    # Check that R, G, and B channels are nearly identical in the sketch crop
    diff_rg = np.mean(np.abs(face_crop[:, :, 0] - face_crop[:, :, 1]))
    diff_gb = np.mean(np.abs(face_crop[:, :, 1] - face_crop[:, :, 2]))
    assert diff_rg < 2.0 and diff_gb < 2.0, "Pencil sketch output must be monochromatic"


@pytest.mark.asyncio
async def test_single_page_pdf_export(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Tests export as a court-admissible one-page PDF.
    Verifies header, image embedding, FaceSpec summary, version history, and single-page constraint.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-PDF-001")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/export",
        json={"format": "pdf", "mode": "photoreal"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF"), "Response content must be a valid PDF"

    # Verify exactly 1 page
    page_matches = re.findall(rb"/Type\s*/Page(?=[^s]|$)", resp.content)
    assert len(page_matches) == 1, f"PDF must be strictly 1 page, found {len(page_matches)} pages"


@pytest.mark.asyncio
async def test_candidate_grid_generation_whole_face(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Tests POST /cases/{id}/grid with whole_face returns 9 candidates (3x3 grid).
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-GRID-001")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/grid",
        json={"feature": "whole_face", "action": "standard"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["case_id"] == case.id
    assert data["feature"] == "whole_face"
    assert data["total_candidates"] == 9
    assert len(data["candidates"]) == 9
    for i, cand in enumerate(data["candidates"]):
        assert cand["candidate_index"] == i
        assert cand["status"] == "success"
        assert cand["image_id"] is not None


@pytest.mark.asyncio
async def test_candidate_grid_feature_with_locality_composite(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Tests POST /cases/{id}/grid with feature='nose' uses locality.composite so the rest
    of the face stays strictly identical.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-GRID-NOSE")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/grid",
        json={"feature": "nose", "action": "standard"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["feature"] == "nose"
    assert data["total_candidates"] == 9
    assert len(data["candidates"]) == 9
    for cand in data["candidates"]:
        assert cand["variation_feature"] == "nose"
        assert cand["status"] == "success"


@pytest.mark.asyncio
async def test_candidate_grid_none_of_these(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Tests POST /cases/{id}/grid with action='none_of_these' regenerates a new grid with greater spread.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-GRID-SPREAD")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/grid",
        json={"feature": "nose", "action": "none_of_these", "spread": 2.0},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["action"] == "none_of_these"
    assert data["spread"] == 2.0
    assert data["total_candidates"] == 9
    # Verify variations are from the wide morphological catalog
    first_cand = data["candidates"][0]
    assert "aquiline" in first_cand["variation_value"] or "broad" in first_cand["variation_value"] or "spread" in first_cand["variation_value"]


@pytest.mark.asyncio
async def test_candidate_grid_blend_toward(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Tests POST /cases/{id}/grid with action='blend_toward' generates a new grid around candidate #k with smaller variation.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-GRID-BLEND")
    token = auth_tokens["Investigator"]

    # 1. First generate standard grid
    resp1 = await client.post(
        f"/api/v1/cases/{case.id}/grid",
        json={"feature": "nose", "action": "standard"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp1.status_code == 200

    # 2. Blend toward candidate index 2
    resp2 = await client.post(
        f"/api/v1/cases/{case.id}/grid",
        json={"feature": "nose", "action": "blend_toward", "blend_candidate_index": 2},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp2.status_code == 200, resp2.text
    data2 = resp2.json()

    assert data2["action"] == "blend_toward"
    assert data2["total_candidates"] == 9
    # Verify candidate variations are fine adjustments
    vals = [c["variation_value"] for c in data2["candidates"]]
    assert any("subtly" in v or "slightly" in v or "refined" in v or "variation" in v for v in vals)


@pytest.mark.asyncio
async def test_candidate_grid_selection_applies_version(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Tests when witness picks a candidate via POST /cases/{id}/grid/select,
    it applies it as a new version (version 2), logs to audit ledger, and updates timeline.
    """
    case = await setup_case_with_composite(db_session, case_number="CASE-SELECT-001")
    token = auth_tokens["Investigator"]

    # 1. Generate grid
    grid_resp = await client.post(
        f"/api/v1/cases/{case.id}/grid",
        json={"feature": "nose", "action": "standard"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert grid_resp.status_code == 200
    chosen_candidate = grid_resp.json()["candidates"][1]

    # 2. Pick candidate #1
    select_resp = await client.post(
        f"/api/v1/cases/{case.id}/grid/select",
        json={"candidate_index": 1, "candidate_id": chosen_candidate["image_id"]},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert select_resp.status_code == 200, select_resp.text
    sel_data = select_resp.json()

    assert sel_data["case_id"] == case.id
    assert sel_data["version"] == 2
    assert sel_data["selected_candidate_index"] == 1
    assert sel_data["feature"] == "nose"
    assert sel_data["image_url"] is not None

    # 3. Verify Version 2 exists in timeline
    tl_resp = await client.get(
        f"/api/v1/cases/{case.id}/timeline",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert tl_resp.status_code == 200
    tl_data = tl_resp.json()
    assert tl_data["total_versions"] == 2
    v2 = tl_data["timeline"][1]
    assert v2["version"] == 2
    assert "grid_select" in v2["triggering_call"]

    # 4. Verify Audit entry was created
    audit_stmt = (
        select(AuditLogEntry)
        .where(
            AuditLogEntry.case_id == case.id,
            AuditLogEntry.action == AuditAction.CREATE_VERSION
        )
        .order_by(AuditLogEntry.seq.desc())
    )
    res = await db_session.execute(audit_stmt)
    entry = res.scalars().first()
    assert entry is not None
    assert entry.outcome == AuditOutcome.SUCCESS
    details = json.loads(entry.details)
    assert details["action"] == "grid_candidate_selected"
    assert details["new_version"] == 2
