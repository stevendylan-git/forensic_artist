import pytest
from sqlalchemy import text
from app.audit.ledger import AuditLedger, GENESIS_HASH
from app.models.audit import AuditAction, AuditOutcome


@pytest.mark.asyncio
async def test_audit_ledger_hash_chain(db_session):
    """Rule 5: Audit log is append-only and cryptographically hash-chained."""
    # 1. Append entry 1
    entry1 = await AuditLedger.append_entry(
        session=db_session,
        action=AuditAction.VIEW_CASE,
        resource="/api/v1/cases/case-123",
        outcome=AuditOutcome.SUCCESS,
        user_id="user-investigator-01",
        role="Investigator",
        case_id="case-123",
        details={"case_number": "CASE-2026-001"}
    )
    await db_session.commit()

    assert entry1.seq == 1
    assert entry1.prev_hash == GENESIS_HASH
    assert len(entry1.entry_hash) == 64

    # 2. Append entry 2
    entry2 = await AuditLedger.append_entry(
        session=db_session,
        action=AuditAction.CREATE_VERSION,
        resource="/api/v1/composite/generate",
        outcome=AuditOutcome.SUCCESS,
        user_id="user-artist-01",
        role="ForensicArtist",
        case_id="case-123",
        details={"version": 1}
    )
    await db_session.commit()

    assert entry2.seq == 2
    # Forward chain verification
    assert entry2.prev_hash == entry1.entry_hash

    # 3. Verify entire chain integrity
    is_valid, count, broken_seq, expected, actual = await AuditLedger.verify_chain(db_session)
    assert is_valid is True
    assert count == 2
    assert broken_seq is None


@pytest.mark.asyncio
async def test_audit_ledger_append_only_trigger(db_session):
    """Verifies that SQLite triggers prohibit direct UPDATE operations on audit_log."""
    entry = await AuditLedger.append_entry(
        session=db_session,
        action=AuditAction.LOGIN,
        resource="/api/v1/auth/login",
        outcome=AuditOutcome.SUCCESS,
        user_id="user-01",
        role="Investigator",
        details={"email": "investigator@forensic.local"}
    )
    await db_session.commit()

    # Attempting normal UPDATE triggers database-level abort
    with pytest.raises(Exception) as exc_info:
        await db_session.execute(
            text(f"UPDATE audit_log SET details = 'tampered' WHERE id = {entry.id}")
        )
        await db_session.commit()

    assert "append-only" in str(exc_info.value).lower()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_audit_ledger_tamper_detection(db_session):
    """
    Simulates direct database compromise by dropping trigger, tampering a row,
    and verifying the cryptographic chain detects the exact tampered sequence.
    """
    entry1 = await AuditLedger.append_entry(
        session=db_session,
        action=AuditAction.LOGIN,
        resource="/api/v1/auth/login",
        outcome=AuditOutcome.SUCCESS,
        user_id="admin-01",
        role="Admin",
        details={"email": "admin@forensic.local"}
    )
    entry2 = await AuditLedger.append_entry(
        session=db_session,
        action=AuditAction.VIEW_CASE,
        resource="/api/v1/cases/case-999",
        outcome=AuditOutcome.SUCCESS,
        user_id="admin-01",
        role="Admin",
        case_id="case-999",
        details={"case_number": "TOP-SECRET-001"}
    )
    entry3 = await AuditLedger.append_entry(
        session=db_session,
        action=AuditAction.EXPORT,
        resource="/api/v1/audit/export/csv",
        outcome=AuditOutcome.SUCCESS,
        user_id="admin-01",
        role="Admin",
        details={"exported": 2}
    )
    await db_session.commit()

    # Temporarily drop trigger to simulate direct out-of-band DB modification by malicious actor
    await db_session.execute(text("DROP TRIGGER IF EXISTS trg_audit_log_prevent_update"))
    await db_session.execute(
        text(f"UPDATE audit_log SET details = '{{\"tampered\": true}}' WHERE seq = 2")
    )
    await db_session.commit()
    db_session.expire_all()

    is_valid, count, broken_seq, expected, actual = await AuditLedger.verify_chain(db_session)
    assert is_valid is False
    assert broken_seq == 2
    assert expected != actual
