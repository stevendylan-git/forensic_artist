import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.audit import AuditLogEntry, AuditAction, AuditOutcome

GENESIS_HASH = "0" * 64


def serialize_canonical_json(data: Any) -> str:
    """Produces sorted, compact canonical JSON without whitespace."""
    if data is None:
        return "{}"
    if isinstance(data, str):
        try:
            # If already valid JSON string, re-normalize
            parsed = json.loads(data)
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return json.dumps({"raw": data}, sort_keys=True, separators=(",", ":"), default=str)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def compute_sha256(data: str) -> str:
    """Computes SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_canonical_entry_dict(
    seq: int,
    timestamp_utc: datetime,
    user_id: Optional[str],
    role: Optional[str],
    ip: Optional[str],
    user_agent: Optional[str],
    case_id: Optional[str],
    action: str,
    resource: str,
    outcome: str,
    details_str: str,
) -> Dict[str, Any]:
    """Builds dictionary with standardized fields for deterministic hashing."""
    # Ensure UTC ISO string with Z
    ts = timestamp_utc if timestamp_utc.tzinfo else timestamp_utc.replace(tzinfo=timezone.utc)
    ts_iso = ts.isoformat()

    return {
        "action": action,
        "case_id": case_id,
        "details": details_str,
        "ip": ip,
        "outcome": outcome,
        "resource": resource,
        "role": role,
        "seq": seq,
        "timestamp_utc": ts_iso,
        "user_agent": user_agent,
        "user_id": user_id,
    }


def compute_entry_hash(prev_hash: str, canonical_dict: Dict[str, Any]) -> str:
    """
    Computes entry_hash = SHA-256(prev_hash + canonical JSON of the entry).
    """
    canonical_json = serialize_canonical_json(canonical_dict)
    combined = prev_hash + canonical_json
    return compute_sha256(combined)


class AuditLedger:
    """
    Append-only, cryptographically hash-chained forensic audit trail service.
    Rule 5: No endpoint may skip audit logging. Log denied attempts too.
    """

    @staticmethod
    async def get_latest_entry(session: AsyncSession) -> Optional[AuditLogEntry]:
        """Retrieves the most recent audit record by sequence number."""
        stmt = select(AuditLogEntry).order_by(desc(AuditLogEntry.seq)).limit(1)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def append_entry(
        session: AsyncSession,
        action: AuditAction | str,
        resource: str,
        outcome: AuditOutcome | str = AuditOutcome.SUCCESS,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        case_id: Optional[str] = None,
        ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Any = None,
    ) -> AuditLogEntry:
        """
        Appends an immutable entry to the audit log with forward cryptographic chaining.
        entry_hash = SHA-256(prev_hash + canonical JSON of the entry)
        """
        action_str = action.value if isinstance(action, AuditAction) else str(action)
        outcome_str = outcome.value if isinstance(outcome, AuditOutcome) else str(outcome)
        details_str = serialize_canonical_json(details if details is not None else {})

        latest = await AuditLedger.get_latest_entry(session)
        if latest:
            seq = latest.seq + 1
            prev_hash = latest.entry_hash
        else:
            seq = 1
            prev_hash = GENESIS_HASH

        now_utc = datetime.now(timezone.utc)

        canonical_dict = build_canonical_entry_dict(
            seq=seq,
            timestamp_utc=now_utc,
            user_id=user_id,
            role=role,
            ip=ip,
            user_agent=user_agent,
            case_id=case_id,
            action=action_str,
            resource=resource,
            outcome=outcome_str,
            details_str=details_str,
        )

        entry_hash = compute_entry_hash(prev_hash, canonical_dict)

        entry = AuditLogEntry(
            seq=seq,
            timestamp_utc=now_utc,
            user_id=user_id,
            role=role,
            ip=ip,
            user_agent=user_agent,
            case_id=case_id,
            action=AuditAction(action_str) if action_str in AuditAction.__members__.values() else AuditAction.API_REQUEST,
            resource=resource,
            outcome=AuditOutcome(outcome_str) if outcome_str in AuditOutcome.__members__.values() else AuditOutcome.SUCCESS,
            details=details_str,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )

        session.add(entry)
        await session.flush()
        return entry

    @staticmethod
    async def verify_chain(session: AsyncSession) -> Tuple[bool, int, Optional[int], Optional[str], Optional[str]]:
        """
        Walks the entire ledger sequentially and verifies cryptographic hash integrity.
        Returns: (is_valid, count, broken_at_seq, expected_hash, actual_hash)
        """
        stmt = select(AuditLogEntry).order_by(AuditLogEntry.seq.asc()).execution_options(populate_existing=True)
        result = await session.execute(stmt)
        entries = result.scalars().all()

        current_prev_hash = GENESIS_HASH
        expected_seq = 1

        for entry in entries:
            # 1. Verify sequence order
            if entry.seq != expected_seq:
                return (
                    False,
                    len(entries),
                    entry.seq,
                    f"Sequential seq={expected_seq}",
                    f"Broken seq={entry.seq}",
                )

            # 2. Verify prev_hash matches prior entry's entry_hash
            if entry.prev_hash != current_prev_hash:
                return (
                    False,
                    len(entries),
                    entry.seq,
                    current_prev_hash,
                    entry.prev_hash,
                )

            # 3. Reconstruct canonical dictionary and verify entry_hash
            action_str = entry.action.value if isinstance(entry.action, AuditAction) else str(entry.action)
            outcome_str = entry.outcome.value if isinstance(entry.outcome, AuditOutcome) else str(entry.outcome)
            canonical_dict = build_canonical_entry_dict(
                seq=entry.seq,
                timestamp_utc=entry.timestamp_utc,
                user_id=entry.user_id,
                role=entry.role,
                ip=entry.ip,
                user_agent=entry.user_agent,
                case_id=entry.case_id,
                action=action_str,
                resource=entry.resource,
                outcome=outcome_str,
                details_str=entry.details,
            )

            expected_entry_hash = compute_entry_hash(entry.prev_hash, canonical_dict)
            if entry.entry_hash != expected_entry_hash:
                return (
                    False,
                    len(entries),
                    entry.seq,
                    expected_entry_hash,
                    entry.entry_hash,
                )

            current_prev_hash = entry.entry_hash
            expected_seq += 1

        return True, len(entries), None, None, None
