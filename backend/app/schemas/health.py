from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Overall system health")
    environment: str = Field(description="Environment mode")
    database_connected: bool = Field(description="Database connectivity status")
    gemini_configured: bool = Field(description="Whether GEMINI_API_KEY is present")
    gemini_text_model: str = Field(description="Active Gemini text model")
    gemini_image_model: str = Field(description="Active Gemini image model")
    version: str = Field(default="1.0.0")
    error: Optional[str] = Field(default=None, description="Detailed diagnostic error if unhealthy")
