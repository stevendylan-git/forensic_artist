from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict
from app.models.audit import AuditAction, AuditOutcome


class AuditEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    seq: int
    timestamp_utc: datetime
    user_id: Optional[str] = None
    role: Optional[str] = None
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    case_id: Optional[str] = None
    action: str
    resource: str
    outcome: str
    details: str
    prev_hash: str
    entry_hash: str


class AuditChainVerificationResult(BaseModel):
    is_valid: bool
    total_entries: int
    broken_at_seq: Optional[int] = None
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    message: str


class AuditPaginatedResponse(BaseModel):
    items: List[AuditEntryResponse]
    total: int
    page: int
    limit: int
    pages: int
