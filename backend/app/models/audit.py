import enum
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Enum as SAEnum, event, DDL
)
from app.database import Base


class AuditAction(str, enum.Enum):
    LOGIN = "LOGIN"
    LOGIN_FAILED = "LOGIN_FAILED"
    VIEW_CASE = "VIEW_CASE"
    CREATE_CASE = "CREATE_CASE"
    UPDATE_CASE = "UPDATE_CASE"
    DELETE_CASE = "DELETE_CASE"
    ASSIGN_USER = "ASSIGN_USER"
    CREATE_VERSION = "CREATE_VERSION"
    EDIT = "EDIT"
    LOCK_FEATURE = "LOCK_FEATURE"
    UNLOCK_FEATURE = "UNLOCK_FEATURE"
    REVERT = "REVERT"
    COMPARE = "COMPARE"
    EXPORT = "EXPORT"
    MATCH_QUERY = "MATCH_QUERY"
    ACCESS_DENIED = "ACCESS_DENIED"
    VERIFY_AUDIT = "VERIFY_AUDIT"
    MFA_SETUP = "MFA_SETUP"
    MFA_VERIFY = "MFA_VERIFY"
    API_REQUEST = "API_REQUEST"


class AuditOutcome(str, enum.Enum):
    SUCCESS = "SUCCESS"
    DENIED = "DENIED"
    ERROR = "ERROR"


class AuditLogEntry(Base):
    """
    Append-only, cryptographically hash-chained forensic audit trail.
    Rule 5: No endpoint may skip audit logging. Log denied attempts too.
    """
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    seq = Column(Integer, nullable=False, unique=True, index=True)
    timestamp_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    user_id = Column(String(64), nullable=True, index=True)
    role = Column(String(32), nullable=True, index=True)
    ip = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    case_id = Column(String(64), nullable=True, index=True)
    action = Column(SAEnum(AuditAction, native_enum=False), nullable=False, index=True)
    resource = Column(String(255), nullable=False)
    outcome = Column(SAEnum(AuditOutcome, native_enum=False), nullable=False, default=AuditOutcome.SUCCESS, index=True)
    details = Column(Text, nullable=False, default="{}")
    prev_hash = Column(String(64), nullable=False)
    entry_hash = Column(String(64), nullable=False, unique=True, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "seq": self.seq,
            "timestamp_utc": self.timestamp_utc.isoformat() if self.timestamp_utc else None,
            "user_id": self.user_id,
            "role": self.role,
            "ip": self.ip,
            "user_agent": self.user_agent,
            "case_id": self.case_id,
            "action": self.action.value if isinstance(self.action, AuditAction) else str(self.action),
            "resource": self.resource,
            "outcome": self.outcome.value if isinstance(self.outcome, AuditOutcome) else str(self.outcome),
            "details": self.details,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


# SQLite and Postgres triggers to enforce append-only immutability
_sqlite_trigger_update = DDL(
    """
    CREATE TRIGGER IF NOT EXISTS trg_audit_log_prevent_update
    BEFORE UPDATE ON audit_log
    BEGIN
        SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE operations are strictly prohibited');
    END;
    """
)
_sqlite_trigger_delete = DDL(
    """
    CREATE TRIGGER IF NOT EXISTS trg_audit_log_prevent_delete
    BEFORE DELETE ON audit_log
    BEGIN
        SELECT RAISE(ABORT, 'audit_log is append-only: DELETE operations are strictly prohibited');
    END;
    """
)

event.listen(
    AuditLogEntry.__table__,
    "after_create",
    _sqlite_trigger_update.execute_if(dialect="sqlite")
)
event.listen(
    AuditLogEntry.__table__,
    "after_create",
    _sqlite_trigger_delete.execute_if(dialect="sqlite")
)
