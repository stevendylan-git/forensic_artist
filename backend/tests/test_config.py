from app.config import settings


def test_config_models_loaded():
    """Rule 2: Model IDs come from backend/app/config.py."""
    assert settings.GEMINI_TEXT_MODEL == "gemini-2.5-flash"
    assert settings.GEMINI_IMAGE_MODEL == "imagen-3.0-generate-002"
    assert settings.WATERMARK_TEXT == "AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY"


def test_config_secrets_never_empty_in_test():
    """Rule 1: Secrets live in backend config."""
    assert settings.GEMINI_API_KEY != ""
