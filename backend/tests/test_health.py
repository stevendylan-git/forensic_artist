import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health check endpoint reporting DB connectivity and models."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "gemini_text_model" in data
    assert "gemini_image_model" in data
    assert data["gemini_text_model"] == "gemini-2.5-flash"
    assert data["gemini_image_model"] == "imagen-3.0-generate-002"
    # Ensure raw API key is NEVER exposed in health output
    assert "test_key_for_unit_tests" not in response.text


@pytest.mark.asyncio
async def test_versioned_health_endpoint(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["ok", "degraded"]
