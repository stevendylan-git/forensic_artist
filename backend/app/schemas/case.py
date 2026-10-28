from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.face_spec import FaceSpec, FaceSpecPatch


class CaseCreate(BaseModel):
    case_number: str = Field(..., min_length=3, max_length=64, description="Unique case number e.g. CASE-2026-001")
    title: str = Field(..., min_length=1, max_length=255)
    status: str = Field(default="OPEN", max_length=32)
    notes: Optional[str] = None
    assigned_user_ids: List[str] = Field(default_factory=list)


class CaseUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class CaseAssignRequest(BaseModel):
    user_ids: List[str] = Field(..., min_length=1, description="List of user UUIDs to assign to this case")


class CaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    case_number: str
    title: str
    status: str
    created_by_id: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    assigned_user_ids: List[str] = Field(default_factory=list)


class DescribeRequest(BaseModel):
    utterance: str = Field(..., min_length=1, description="Witness verbal statement describing suspect facial features")
    speaker: str = Field(default="witness", description="Speaker role: witness, investigator, victim")


class DescribeResponse(BaseModel):
    case_id: str
    utterance_id: str
    version: int
    patch: FaceSpecPatch
    face_spec: FaceSpec
    diff: Dict[str, Any]


class GenerateInitialRequest(BaseModel):
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence required to render attribute")
    utterance_id: Optional[str] = None


class GenerateInitialResponse(BaseModel):
    case_id: str
    image_id: str
    version: int
    storage_path: Optional[str] = None
    prompt: str
    prompt_hash: str
    model_id: str
    latency_ms: float
    estimated_cost: float
    status: str = "success"


class GenerateCandidatesRequest(BaseModel):
    vary: str = Field(default="nose", description="Feature path to vary, e.g. nose, eyes.shape, hair.style, jawline, chin")
    n: int = Field(default=9, ge=1, le=16, description="Number of candidate variations to generate")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    utterance_id: Optional[str] = None


class CandidateItemResponse(BaseModel):
    candidate_index: int
    variation_feature: str
    variation_value: str
    image_id: Optional[str] = None
    storage_path: Optional[str] = None
    latency_ms: float = 0.0
    status: str  # "success", "safety_refusal", "failed"
    error: Optional[str] = None


class GenerateCandidatesResponse(BaseModel):
    case_id: str
    feature: str
    total_requested: int
    total_successful: int
    total_failed: int
    candidates: List[CandidateItemResponse]
    total_latency_ms: float
    total_estimated_cost: float


class RefineFeedbackRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Witness conversational refinement feedback")
    utterance_id: Optional[str] = None


class RefineFeedbackResponse(BaseModel):
    case_id: str
    current_version: int
    tool_calls_executed: int
    results: List[Dict[str, Any]]


class TimelineItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version: int
    created_at: Optional[datetime] = None
    triggering_call: Optional[str] = None
    locality_score: Optional[float] = None
    locality_unverified: bool = False
    locked_regions: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    utterance_id: Optional[str] = None
    face_spec: Optional[Dict[str, Any]] = None


class TimelineResponse(BaseModel):
    case_id: str
    total_versions: int
    timeline: List[TimelineItemResponse]


class RevertRequest(BaseModel):
    to_version: int = Field(..., ge=1, description="Target version number to revert to")
    region: Optional[str] = Field(default=None, description="Optional region to revert (None for whole composite)")


class RevertResponse(BaseModel):
    case_id: str
    version: int
    reverted_to_version: int
    region: str
    locality_score: float
    image_url: Optional[str] = None
    diff: Dict[str, Any] = Field(default_factory=dict)


class UndoRedoResponse(BaseModel):
    case_id: str
    version: int
    action: str  # "undo" or "redo"
    image_url: Optional[str] = None
    message: str


class CompareResponse(BaseModel):
    case_id: str
    version_a: Dict[str, Any]
    version_b: Dict[str, Any]
    diff: Dict[str, Any]


class CandidateGridRequest(BaseModel):
    feature: str = Field(default="whole_face", description='Feature to vary, e.g. "nose", "eyes", "mouth", or "whole_face"')
    action: Optional[str] = Field(default="standard", description='Grid action: "standard", "none_of_these", or "blend_toward"')
    blend_candidate_index: Optional[int] = Field(default=None, description="Index of candidate #k (0-8) to blend toward")
    spread: float = Field(default=1.0, ge=0.1, le=5.0, description="Spread factor for variation diversity")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CandidateGridResponse(BaseModel):
    case_id: str
    feature: str
    action: str
    spread: float
    total_candidates: int
    candidates: List[CandidateItemResponse]
    total_latency_ms: float
    total_estimated_cost: float


class GridSelectRequest(BaseModel):
    candidate_index: Optional[int] = Field(default=None, description="0-indexed candidate variation number from the 3x3 grid")
    candidate_id: Optional[str] = Field(default=None, description="Image ID of selected candidate")


class GridSelectResponse(BaseModel):
    case_id: str
    version: int
    selected_candidate_index: Optional[int] = None
    feature: str
    variation_value: str
    image_url: Optional[str] = None
    message: str


class ExportRequest(BaseModel):
    format: str = Field(default="png", description='Output format: "png" or "pdf"')
    mode: str = Field(default="photoreal", description='Output mode: "photoreal" or "pencil-sketch"')
    upscale: bool = Field(default=True, description="Upscale/enhance image to high resolution")
    version: Optional[int] = Field(default=None, description="Optional version to export; defaults to latest version")

