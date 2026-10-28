import logging
import time
from typing import Optional, Any, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.config import settings
from app.schemas.gemini import (
    GeminiTextRequest,
    GeminiTextResponse,
    GeminiImageRequest,
    GeminiImageResponse
)

logger = logging.getLogger("gemini_client")


def sanitize_log_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Redacts any API keys or secrets before logging."""
    sanitized = {}
    for k, v in d.items():
        if any(secret_term in k.lower() for secret_term in ["key", "secret", "token", "auth", "password"]):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_log_dict(v)
        else:
            sanitized[k] = v
    return sanitized


class GeminiClient:
    """
    Typed Wrapper for Google GenAI SDK.
    Rules:
    - Rule 1: API keys never logged or exposed.
    - Rule 2: Model IDs sourced from backend/app/config.py.
    - Retries with exponential backoff and timeouts.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.text_model = text_model or settings.GEMINI_TEXT_MODEL
        self.image_model = image_model or settings.GEMINI_IMAGE_MODEL
        self.timeout = timeout or settings.GEMINI_REQUEST_TIMEOUT_SECONDS
        self.max_retries = max_retries or settings.GEMINI_MAX_RETRIES

        # Lazy initialize SDK client
        self._client: Optional[genai.Client] = None

    def get_client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not configured in backend environment.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _log_request(self, call_type: str, model: str, payload: Dict[str, Any]):
        safe_meta = sanitize_log_dict({
            "call_type": call_type,
            "model": model,
            "timeout_seconds": self.timeout,
            "prompt_length": len(payload.get("prompt", "")),
            "payload_preview": payload.get("prompt", "")[:120]
        })
        logger.info(f"Gemini API Request: {safe_meta}")

    def _log_response(self, call_type: str, model: str, duration_sec: float, details: Dict[str, Any]):
        safe_meta = sanitize_log_dict({
            "call_type": call_type,
            "model": model,
            "duration_seconds": round(duration_sec, 3),
            **details
        })
        logger.info(f"Gemini API Response: {safe_meta}")

    @retry(
        retry=retry_if_exception_type((APIError, TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=2.0, max=10.0),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def generate_text(self, request: GeminiTextRequest) -> GeminiTextResponse:
        """
        Executes text generation / structured output using the configured GEMINI_TEXT_MODEL.
        """
        start_time = time.monotonic()
        client = self.get_client()

        config_params: Dict[str, Any] = {
            "temperature": request.temperature,
        }
        if request.system_instruction:
            config_params["system_instruction"] = request.system_instruction
        if request.response_schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = request.response_schema

        config = types.GenerateContentConfig(**config_params)

        self._log_request("generate_content", self.text_model, {"prompt": request.prompt})

        try:
            # Synchronous SDK call wrapped for execution
            response = client.models.generate_content(
                model=self.text_model,
                contents=request.prompt,
                config=config
            )
            duration = time.monotonic() - start_time
            response_text = response.text or ""

            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "candidates_token_count": getattr(response.usage_metadata, "candidates_token_count", 0)
                }

            self._log_response(
                "generate_content",
                self.text_model,
                duration,
                {"response_length": len(response_text), "usage": usage}
            )

            return GeminiTextResponse(
                content=response_text,
                model_name=self.text_model,
                usage_metadata=usage
            )
        except Exception as e:
            logger.error(f"Gemini text generation failed for model {self.text_model}: {str(e)}")
            raise

    @retry(
        retry=retry_if_exception_type((APIError, TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2.0, max=15.0),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def generate_image(self, request: GeminiImageRequest) -> GeminiImageResponse:
        """
        Executes image generation using the configured GEMINI_IMAGE_MODEL (e.g. Imagen 3).
        """
        start_time = time.monotonic()
        client = self.get_client()

        config = types.GenerateImagesConfig(
            number_of_images=request.number_of_images,
            aspect_ratio=request.aspect_ratio,
            person_generation=request.person_generation
        )

        self._log_request("generate_images", self.image_model, {"prompt": request.prompt})

        try:
            response = client.models.generate_images(
                model=self.image_model,
                prompt=request.prompt,
                config=config
            )
            duration = time.monotonic() - start_time

            image_bytes_list = []
            if hasattr(response, "generated_images") and response.generated_images:
                for gen_img in response.generated_images:
                    if hasattr(gen_img, "image") and hasattr(gen_img.image, "image_bytes"):
                        image_bytes_list.append(gen_img.image.image_bytes)

            self._log_response(
                "generate_images",
                self.image_model,
                duration,
                {"images_generated": len(image_bytes_list)}
            )

            return GeminiImageResponse(
                image_bytes_list=image_bytes_list,
                model_name=self.image_model,
                count=len(image_bytes_list)
            )
        except Exception as e:
            logger.error(f"Gemini image generation failed for model {self.image_model}: {str(e)}")
            raise


# Singleton service instance
gemini_client = GeminiClient()
