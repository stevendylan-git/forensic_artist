import pytest
from httpx import AsyncClient
from sqlalchemy import select
from app.models.user import User


@pytest.mark.asyncio
async def test_case_crud_and_assignment(client: AsyncClient, auth_tokens):
    """Verifies Case creation, update, and deletion by authorized roles."""
    inv_token = auth_tokens["investigator@forensic.local"]
    headers = {"Authorization": f"Bearer {inv_token}"}

    # 1. Investigator creates case
    create_resp = await client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "case_number": "CASE-2026-901",
            "title": "Downtown Jewelry Robbery",
            "notes": "Witness saw suspect flee south."
        }
    )
    assert create_resp.status_code == 201
    case_data = create_resp.json()
    case_id = case_data["id"]
    assert case_data["case_number"] == "CASE-2026-901"
    assert len(case_data["assigned_user_ids"]) >= 1

    # 2. Investigator updates case
    patch_resp = await client.patch(
        f"/api/v1/cases/{case_id}",
        headers=headers,
        json={"title": "Downtown Jewelry Robbery - Updated"}
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["title"] == "Downtown Jewelry Robbery - Updated"

    # 3. Investigator tries to delete case -> 403 (Only Admin/Supervisor can delete)
    del_resp = await client.delete(f"/api/v1/cases/{case_id}", headers=headers)
    assert del_resp.status_code == 403

    # 4. Supervisor deletes case -> 204
    sup_token = auth_tokens["supervisor@forensic.local"]
    sup_del_resp = await client.delete(
        f"/api/v1/cases/{case_id}",
        headers={"Authorization": f"Bearer {sup_token}"}
    )
    assert sup_del_resp.status_code == 204


@pytest.mark.asyncio
async def test_case_need_to_know_enforcement(client: AsyncClient, auth_tokens, db_session):
    """
    Verifies Case-level Need-to-know:
    - User can only open cases they are assigned to.
    - Supervisor/Admin can assign users and view any case.
    """
    inv_token = auth_tokens["investigator@forensic.local"]
    art_token = auth_tokens["artist@forensic.local"]
    sup_token = auth_tokens["supervisor@forensic.local"]

    # 1. Investigator creates Case X
    create_resp = await client.post(
        "/api/v1/cases",
        headers={"Authorization": f"Bearer {inv_token}"},
        json={
            "case_number": "CASE-NEED-TO-KNOW-01",
            "title": "Restricted Homicide Investigation"
        }
    )
    case_id = create_resp.json()["id"]

    # 2. Forensic Artist is NOT assigned yet -> 403 Forbidden
    art_resp = await client.get(
        f"/api/v1/cases/{case_id}",
        headers={"Authorization": f"Bearer {art_token}"}
    )
    assert art_resp.status_code == 403
    assert "Need-to-know" in art_resp.json()["detail"]

    # 3. Supervisor can view without explicit assignment
    sup_resp = await client.get(
        f"/api/v1/cases/{case_id}",
        headers={"Authorization": f"Bearer {sup_token}"}
    )
    assert sup_resp.status_code == 200

    # 4. Get artist user ID
    stmt = select(User).where(User.email == "artist@forensic.local")
    artist_user = (await db_session.execute(stmt)).scalar_one()

    # 5. Supervisor assigns Artist to Case X
    assign_resp = await client.post(
        f"/api/v1/cases/{case_id}/assign",
        headers={"Authorization": f"Bearer {sup_token}"},
        json={"user_ids": [artist_user.id]}
    )
    assert assign_resp.status_code == 200
    assert artist_user.id in assign_resp.json()["assigned_user_ids"]

    # 6. Now Forensic Artist CAN access Case X -> 200 OK
    art_resp_after = await client.get(
        f"/api/v1/cases/{case_id}",
        headers={"Authorization": f"Bearer {art_token}"}
    )
    assert art_resp_after.status_code == 200
    assert art_resp_after.json()["id"] == case_id


@pytest.mark.asyncio
async def test_auditor_role_isolation(client: AsyncClient, auth_tokens):
    """
    Verifies Auditor role isolation:
    - Auditor has read-only access to case metadata for audit compliance.
    - Auditor has STRICTLY NO ACCESS to face images or composites.
    """
    inv_token = auth_tokens["investigator@forensic.local"]
    aud_token = auth_tokens["auditor@forensic.local"]

    # 1. Investigator creates case
    create_resp = await client.post(
        "/api/v1/cases",
        headers={"Authorization": f"Bearer {inv_token}"},
        json={
            "case_number": "CASE-AUDIT-ISO-01",
            "title": "Evidentiary Audit Case"
        }
    )
    case_id = create_resp.json()["id"]

    # 2. Auditor reads case metadata -> Allowed for audit tracking
    meta_resp = await client.get(
        f"/api/v1/cases/{case_id}",
        headers={"Authorization": f"Bearer {aud_token}"}
    )
    assert meta_resp.status_code == 200
    assert meta_resp.json()["case_number"] == "CASE-AUDIT-ISO-01"

    # 3. Auditor attempts to view face image -> 403 Forbidden!
    face_resp = await client.get(
        f"/api/v1/cases/{case_id}/composite/face-image",
        headers={"Authorization": f"Bearer {aud_token}"}
    )
    assert face_resp.status_code == 403
    assert "Auditor role restriction" in face_resp.json()["detail"]

    # 4. Assigned Investigator accesses face composite -> 200 OK
    inv_face_resp = await client.get(
        f"/api/v1/cases/{case_id}/composite/face-image",
        headers={"Authorization": f"Bearer {inv_token}"}
    )
    assert inv_face_resp.status_code == 200
