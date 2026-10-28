import pytest
from httpx import AsyncClient
from sqlalchemy import text


@pytest.mark.asyncio
async def test_audit_verify_api_endpoint(client: AsyncClient, auth_tokens):
    """Verifies GET /api/v1/audit/verify returns valid chain status."""
    token = auth_tokens["auditor@forensic.local"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = await client.get("/api/v1/audit/verify", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is True
    assert data["total_entries"] >= 0
    assert data["broken_at_seq"] is None


@pytest.mark.asyncio
async def test_audit_entries_rbac_and_pagination(client: AsyncClient, auth_tokens):
    """
    Verifies /api/v1/audit/entries:
    - Investigator lacks permission -> 403 Forbidden
    - Auditor has permission -> 200 OK with paginated list
    """
    inv_token = auth_tokens["investigator@forensic.local"]
    aud_token = auth_tokens["auditor@forensic.local"]

    # 1. Investigator blocked from viewing audit logs
    inv_resp = await client.get(
        "/api/v1/audit/entries",
        headers={"Authorization": f"Bearer {inv_token}"}
    )
    assert inv_resp.status_code == 403

    # 2. Auditor allowed
    aud_resp = await client.get(
        "/api/v1/audit/entries?limit=10&page=1",
        headers={"Authorization": f"Bearer {aud_token}"}
    )
    assert aud_resp.status_code == 200
    data = aud_resp.json()
    assert "items" in data
    assert "total" in data
    assert data["page"] == 1
    assert len(data["items"]) >= 1

    entry = data["items"][0]
    assert "seq" in entry
    assert "prev_hash" in entry
    assert "entry_hash" in entry


@pytest.mark.asyncio
async def test_audit_export_csv_logged(client: AsyncClient, auth_tokens, db_session):
    """
    Verifies CSV export and verifies that the export event itself
    is recorded in the audit ledger.
    """
    token = auth_tokens["supervisor@forensic.local"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Export CSV
    export_resp = await client.get("/api/v1/audit/export/csv", headers=headers)
    assert export_resp.status_code == 200
    assert "text/csv" in export_resp.headers["content-type"]
    csv_text = export_resp.text
    assert "seq,timestamp_utc,user_id,role" in csv_text

    # 2. Verify that the export itself was logged
    entries_resp = await client.get(
        "/api/v1/audit/entries?action=EXPORT",
        headers=headers
    )
    assert entries_resp.status_code == 200
    export_items = entries_resp.json()["items"]
    assert len(export_items) >= 1
    assert export_items[0]["action"] == "EXPORT"


@pytest.mark.asyncio
async def test_audit_api_tamper_detection(client: AsyncClient, auth_tokens, db_session):
    """
    Directly modifies a database row and asserts that GET /api/v1/audit/verify
    returns is_valid=False and pinpoints the tampered row.
    """
    token = auth_tokens["admin@forensic.local"]
    headers = {"Authorization": f"Bearer {token}"}

    # Generate some entries
    await client.post(
        "/api/v1/cases",
        headers=headers,
        json={"case_number": "CASE-TAMPER-TEST-01", "title": "Tamper Test Case"}
    )

    # Verify chain is initially valid
    verify1 = await client.get("/api/v1/audit/verify", headers=headers)
    assert verify1.json()["is_valid"] is True

    # Drop trigger and corrupt row seq = 1 directly in SQLite
    await db_session.execute(text("DROP TRIGGER IF EXISTS trg_audit_log_prevent_update"))
    await db_session.execute(
        text("UPDATE audit_log SET details = '{\"tampered_directly_in_db\": true}' WHERE seq = 1")
    )
    await db_session.commit()
    db_session.expire_all()

    # Verify endpoint immediately detects tampering at sequence #1
    verify2 = await client.get("/api/v1/audit/verify", headers=headers)
    assert verify2.status_code == 200
    res = verify2.json()
    assert res["is_valid"] is False
    assert res["broken_at_seq"] == 1
    assert "Cryptographic integrity violation" in res["message"]
