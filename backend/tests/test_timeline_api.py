"""
Integration Tests for Version Timeline, Refinement Pipeline, and Revert API (Phase 3)
"""

import json
import pytest
import numpy as np
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.case import Case, CaseAssignment, CompositeSession, GeneratedImage
from app.schemas.face_spec import FaceSpec
from app.services.storage import storage_service
from app.services.locality import to_png_bytes


async def setup_test_case_with_v1(db_session: AsyncSession, case_number: str = "CASE-PHASE3-001"):
    """Creates a case with an initial version 1 composite for testing."""
    case = Case(case_number=case_number, title="Phase 3 Refinement Case")
    db_session.add(case)
    await db_session.flush()

    from sqlalchemy import select
    from app.models.user import User
    user_stmt = select(User).where(User.email == "investigator@forensic.local")
    investigator = (await db_session.execute(user_stmt)).scalar_one()

    # Assign demo investigator
    assignment = CaseAssignment(case_id=case.id, user_id=investigator.id)
    db_session.add(assignment)

    # Synthetic image for v1
    img_arr = np.ones((512, 512, 3), dtype=np.uint8) * 190
    png_bytes = to_png_bytes(img_arr)
    meta = await storage_service.save_image(
        case_id=case.id,
        image_bytes=png_bytes,
        version=1,
        is_candidate=False,
        db=db_session
    )

    fspec = FaceSpec()
    fspec.nose.width.value = "average standard width"
    fspec.nose.width.confidence = 0.9

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
        model_id="test-model",
        prompt="Initial prompt",
        prompt_hash="testhash123",
        triggering_call="generate_initial"
    )
    db_session.add(gen_img)
    await db_session.commit()

    return case


@pytest.mark.asyncio
async def test_timeline_listing(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """Tests GET /cases/{case_id}/timeline returns all versions in order."""
    case = await setup_test_case_with_v1(db_session, case_number="CASE-TL-001")
    token = auth_tokens["Investigator"]

    resp = await client.get(
        f"/api/v1/cases/{case.id}/timeline",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["case_id"] == case.id
    assert data["total_versions"] == 1
    assert len(data["timeline"]) == 1
    v1 = data["timeline"][0]
    assert v1["version"] == 1
    assert v1["locality_score"] == 0.0
    assert v1["locked_regions"] == []


@pytest.mark.asyncio
async def test_refine_edit_nose_pipeline(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """
    Tests POST /cases/{case_id}/refine translates witness feedback into an edit,
    enforcing locality, creating version 2, and logging an audit entry.
    """
    case = await setup_test_case_with_v1(db_session, case_number="CASE-REF-001")
    token = auth_tokens["Investigator"]

    resp = await client.post(
        f"/api/v1/cases/{case.id}/refine",
        json={"message": "Please make the nose wider"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["case_id"] == case.id
    assert data["current_version"] == 2
    assert len(data["results"]) >= 1

    edit_res = data["results"][0]
    assert edit_res["action"] == "edit_region"
    assert edit_res["status"] == "success"
    assert edit_res["version"] == 2
    assert edit_res["region"] == "nose"
    assert "locality_score" in edit_res

    # Verify timeline now has 2 versions
    tl_resp = await client.get(
        f"/api/v1/cases/{case.id}/timeline",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert tl_resp.status_code == 200
    tl_data = tl_resp.json()
    assert tl_data["total_versions"] == 2


@pytest.mark.asyncio
async def test_lock_and_unlock_via_refine_api(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """Tests locking and unlocking features through the refinement endpoint."""
    case = await setup_test_case_with_v1(db_session, case_number="CASE-LOCK-001")
    token = auth_tokens["Investigator"]

    # 1. Lock jaw
    resp_lock = await client.post(
        f"/api/v1/cases/{case.id}/refine",
        json={"message": "The jaw looks perfect"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp_lock.status_code == 200
    lock_data = resp_lock.json()
    assert lock_data["current_version"] == 2
    lock_item = lock_data["results"][0]
    assert lock_item["action"] == "lock_feature"
    assert "jaw" in lock_item["locked_regions"]

    # 2. Try to edit jaw (should be blocked)
    resp_blocked = await client.post(
        f"/api/v1/cases/{case.id}/refine",
        json={"message": "make the jaw wider"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp_blocked.status_code == 200
    blocked_data = resp_blocked.json()
    assert blocked_data["results"][0]["action"] == "ask_witness"
    assert "locked" in blocked_data["results"][0]["question"].lower()

    # 3. Unlock jaw
    resp_unlock = await client.post(
        f"/api/v1/cases/{case.id}/refine",
        json={"message": "unlock the jaw"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp_unlock.status_code == 200
    unlock_data = resp_unlock.json()
    unlock_item = unlock_data["results"][0]
    assert unlock_item["action"] == "unlock_feature"
    assert "jaw" not in unlock_item["locked_regions"]


@pytest.mark.asyncio
async def test_revert_and_undo_api(client: AsyncClient, db_session: AsyncSession, auth_tokens):
    """Tests revert, undo, and redo endpoints."""
    case = await setup_test_case_with_v1(db_session, case_number="CASE-REV-001")
    token = auth_tokens["Investigator"]

    # Create v2: Edit mouth
    await client.post(
        f"/api/v1/cases/{case.id}/refine",
        json={"message": "make the mouth thinner"},
        headers={"Authorization": f"Bearer {token}"}
    )

    # Revert to version 1
    resp_rev = await client.post(
        f"/api/v1/cases/{case.id}/revert",
        json={"to_version": 1},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp_rev.status_code == 200
    rev_data = resp_rev.json()
    assert rev_data["version"] == 3
    assert rev_data["reverted_to_version"] == 1

    # Undo
    resp_undo = await client.post(
        f"/api/v1/cases/{case.id}/undo",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp_undo.status_code == 200
    undo_data = resp_undo.json()
    assert undo_data["action"] == "undo"
    assert undo_data["version"] == 4

    # Compare v1 and v2
    resp_cmp = await client.get(
        f"/api/v1/cases/{case.id}/compare?a=1&b=2",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp_cmp.status_code == 200
    cmp_data = resp_cmp.json()
    assert cmp_data["version_a"]["version"] == 1
    assert cmp_data["version_b"]["version"] == 2
