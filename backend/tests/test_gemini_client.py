import pytest
from unittest.mock import MagicMock
from app.services.gemini_client import GeminiClient, sanitize_log_dict
from app.schemas.gemini import GeminiTextRequest, GeminiImageRequest


def test_sanitize_log_dict_redacts_keys():
    """Rule 1: Secrets are strictly redacted from structured logging."""
    raw = {
        "gemini_api_key": "secret_abc123",
        "nested": {
            "auth_token": "token_xyz",
            "model": "gemini-2.5-flash"
        },
        "prompt": "What is the suspect's eye color?"
    }
    sanitized = sanitize_log_dict(raw)
    assert sanitized["gemini_api_key"] == "[REDACTED]"
    assert sanitized["nested"]["auth_token"] == "[REDACTED]"
    assert sanitized["nested"]["model"] == "gemini-2.5-flash"
    assert sanitized["prompt"] == "What is the suspect's eye color?"


@pytest.mark.asyncio
async def test_mocked_gemini_text_generation():
    """Verifies typed text generation with mocked Google GenAI client."""
    client = GeminiClient(api_key="mock_test_key")

    mock_sdk_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"nose": "straight", "confidence": 0.8}'
    mock_response.usage_metadata.prompt_token_count = 15
    mock_response.usage_metadata.candidates_token_count = 10

    mock_sdk_client.models.generate_content.return_value = mock_response
    client._client = mock_sdk_client

    request = GeminiTextRequest(
        prompt="Describe suspect's nose shape in forensic detail",
        temperature=0.1
    )

    response = await client.generate_text(request)

    assert response.content == '{"nose": "straight", "confidence": 0.8}'
    assert response.model_name == client.text_model
    assert response.usage_metadata["prompt_token_count"] == 15
    mock_sdk_client.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_mocked_gemini_image_generation():
    """Verifies typed image generation with mocked Google GenAI client."""
    client = GeminiClient(api_key="mock_test_key")

    mock_sdk_client = MagicMock()
    mock_gen_image = MagicMock()
    mock_gen_image.image.image_bytes = b"fake_png_header_and_sketch_bytes"

    mock_response = MagicMock()
    mock_response.generated_images = [mock_gen_image]

    mock_sdk_client.models.generate_images.return_value = mock_response
    client._client = mock_sdk_client

    request = GeminiImageRequest(
        prompt="Forensic police sketch of male suspect, oval face, dark brown hair",
        number_of_images=1
    )

    response = await client.generate_image(request)

    assert response.count == 1
    assert response.image_bytes_list[0] == b"fake_png_header_and_sketch_bytes"
    assert response.model_name == client.image_model
    mock_sdk_client.models.generate_images.assert_called_once()
