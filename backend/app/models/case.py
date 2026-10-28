import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, DateTime, Text, Integer, Float, Boolean, ForeignKey, UniqueConstraint, DDL, event
)
from sqlalchemy.orm import relationship
from app.database import Base


class CaseAssignment(Base):
    """
    Case-level need-to-know assignment.
    A user can only open cases they are assigned to (Supervisors and Admins can assign and view).
    """
    __tablename__ = "case_assignments"
    __table_args__ = (
        UniqueConstraint("case_id", "user_id", name="uq_case_user_assignment"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    assigned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    assigned_by = Column(String(36), nullable=True)

    case = relationship("Case", back_populates="assignments")
    user = relationship("User", back_populates="assigned_cases")

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "user_id": self.user_id,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "assigned_by": self.assigned_by,
        }


class CaseEncryptionKey(Base):
    """
    Stores the wrapped/encrypted per-case Data Encryption Key (DEK).
    Envelope encryption: DEK is encrypted with Master KEK using AES-256-GCM.
    """
    __tablename__ = "case_encryption_keys"

    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), primary_key=True)
    encrypted_dek = Column(Text, nullable=False)  # Base64 encoded: nonce (12B) + tag (16B) + ciphertext
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    case = relationship("Case", back_populates="encryption_key")

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Case(Base):
    __tablename__ = "cases"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_number = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=False, default="Untitled Case")
    status = Column(String(32), default="OPEN", nullable=False, index=True)
    created_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    assignments = relationship("CaseAssignment", back_populates="case", cascade="all, delete-orphan", lazy="selectin")
    composites = relationship("CompositeSession", back_populates="case", cascade="all, delete-orphan", lazy="selectin")
    utterances = relationship("Utterance", back_populates="case", cascade="all, delete-orphan", lazy="selectin")
    encryption_key = relationship("CaseEncryptionKey", back_populates="case", uselist=False, cascade="all, delete-orphan", lazy="selectin")
    generated_images = relationship("GeneratedImage", back_populates="case", cascade="all, delete-orphan", lazy="selectin")

    def to_dict(self):
        return {
            "id": self.id,
            "case_number": self.case_number,
            "title": self.title,
            "status": self.status,
            "created_by_id": self.created_by_id,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "assigned_user_ids": [a.user_id for a in self.assignments] if self.assignments else [],
        }


class Utterance(Base):
    """
    Forensic witness or investigator utterance.
    Append-only evidentiary statement record.
    """
    __tablename__ = "case_utterances"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    speaker = Column(String(64), default="witness", nullable=False)
    text = Column(Text, nullable=False)
    parsed_patch_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    case = relationship("Case", back_populates="utterances")
    composites = relationship("CompositeSession", back_populates="utterance", lazy="selectin")
    generated_images = relationship("GeneratedImage", back_populates="utterance", lazy="selectin")

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "speaker": self.speaker,
            "text": self.text,
            "parsed_patch_json": self.parsed_patch_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CompositeSession(Base):
    """
    Forensic Composite Session containing iteration history and FaceSpecs.
    Rule 3: FaceSpec is the source of truth.
    Immutable version rows.
    """
    __tablename__ = "composite_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(Integer, default=1, nullable=False, index=True)
    face_spec_json = Column(Text, nullable=False)  # Serialized FaceSpec Pydantic model
    locked_regions_json = Column(Text, default="[]", nullable=False)
    rendered_image_url = Column(String(255), nullable=True)
    image_path = Column(String(512), nullable=True)
    model_id = Column(String(128), nullable=True)
    prompt_hash = Column(String(64), nullable=True, index=True)
    triggering_call = Column(String(128), nullable=True)
    latency_ms = Column(Float, nullable=True)
    estimated_cost = Column(Float, nullable=True)
    utterance_id = Column(String(36), ForeignKey("case_utterances.id", ondelete="SET NULL"), nullable=True, index=True)
    locality_score = Column(Float, nullable=True)
    locality_unverified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    case = relationship("Case", back_populates="composites")
    utterance = relationship("Utterance", back_populates="composites")
    generated_images = relationship("GeneratedImage", back_populates="composite_session", cascade="all, delete-orphan", lazy="selectin")

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "version": self.version,
            "face_spec_json": self.face_spec_json,
            "locked_regions_json": self.locked_regions_json,
            "rendered_image_url": self.rendered_image_url,
            "image_path": self.image_path,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "triggering_call": self.triggering_call,
            "latency_ms": self.latency_ms,
            "estimated_cost": self.estimated_cost,
            "utterance_id": self.utterance_id,
            "locality_score": self.locality_score,
            "locality_unverified": self.locality_unverified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class GeneratedImage(Base):
    """
    Immutable version record of an AI-rendered facial composite image or candidate variation.
    Links: image path, FaceSpec version, triggering utterance/tool call, model id, prompt hash.
    """
    __tablename__ = "generated_images"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    composite_session_id = Column(String(36), ForeignKey("composite_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    version = Column(Integer, nullable=False, index=True)
    image_path = Column(String(512), nullable=False)
    model_id = Column(String(128), nullable=False)
    prompt = Column(Text, nullable=False)
    prompt_hash = Column(String(64), nullable=False, index=True)
    triggering_call = Column(String(128), nullable=False, default="generate_initial")
    utterance_id = Column(String(36), ForeignKey("case_utterances.id", ondelete="SET NULL"), nullable=True, index=True)
    is_candidate = Column(Boolean, default=False, nullable=False)
    variation_feature = Column(String(128), nullable=True)
    variation_value = Column(String(255), nullable=True)
    latency_ms = Column(Float, nullable=False, default=0.0)
    estimated_cost = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    case = relationship("Case", back_populates="generated_images")
    composite_session = relationship("CompositeSession", back_populates="generated_images")
    utterance = relationship("Utterance", back_populates="generated_images")

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "composite_session_id": self.composite_session_id,
            "version": self.version,
            "image_path": self.image_path,
            "model_id": self.model_id,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "triggering_call": self.triggering_call,
            "utterance_id": self.utterance_id,
            "is_candidate": self.is_candidate,
            "variation_feature": self.variation_feature,
            "variation_value": self.variation_value,
            "latency_ms": self.latency_ms,
            "estimated_cost": self.estimated_cost,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Immutability Triggers for CompositeSession: Prohibit UPDATE
_trg_composite_sessions_prevent_update = DDL(
    """
    CREATE TRIGGER IF NOT EXISTS trg_composite_sessions_prevent_update
    BEFORE UPDATE ON composite_sessions
    BEGIN
        SELECT RAISE(ABORT, 'composite_sessions is immutable: UPDATE operations are strictly prohibited');
    END;
    """
)

# Immutability Triggers for GeneratedImage: Prohibit UPDATE and DELETE
_trg_generated_images_prevent_update = DDL(
    """
    CREATE TRIGGER IF NOT EXISTS trg_generated_images_prevent_update
    BEFORE UPDATE ON generated_images
    BEGIN
        SELECT RAISE(ABORT, 'generated_images is immutable: UPDATE operations are strictly prohibited');
    END;
    """
)

event.listen(
    CompositeSession.__table__,
    "after_create",
    _trg_composite_sessions_prevent_update.execute_if(dialect="sqlite")
)

event.listen(
    GeneratedImage.__table__,
    "after_create",
    _trg_generated_images_prevent_update.execute_if(dialect="sqlite")
)
