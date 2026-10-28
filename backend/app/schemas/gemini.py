from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class GeminiTextRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt text")
    system_instruction: Optional[str] = Field(default=None, description="Forensic system role instruction")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    response_schema: Optional[Any] = Field(default=None, description="Optional Pydantic class or JSON schema")


class GeminiTextResponse(BaseModel):
    content: str
    model_name: str
    usage_metadata: Optional[Dict[str, Any]] = None


class GeminiImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Image rendering prompt")
    number_of_images: int = Field(default=1, ge=1, le=4)
    aspect_ratio: str = Field(default="1:1")
    person_generation: str = Field(default="ALLOW_ADULT")


class GeminiImageResponse(BaseModel):
    image_bytes_list: List[bytes] = Field(default_factory=list)
    model_name: str
    count: int
