import os
import hashlib
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from google.genai.errors import APIError
from sqlalchemy import select

from app.config import settings
from app.schemas.face_spec import FaceSpec, FeatureAttribute
from app.schemas.gemini import GeminiImageResponse
from app.models.case import Case, GeneratedImage, CompositeSession
from app.models.user import User, UserRole
from app.services.image_gen import (
    build_forensic_prompt,
    apply_feature_variation,
    generate_initial,
    generate_candidates,
    GenerationSafetyBlockError,
    FEATURE_VARIATIONS,
)
from app.services.storage import (
    storage_service,
    get_or_create_case_dek,
    encrypt_payload,
    decrypt_payload,
)


# --- 1. Deterministic Prompt Builder Tests ---

def test_prompt_builder_includes_confident_attributes():
    """Only attributes with confidence >= threshold and is_stated=True are included."""
    spec = FaceSpec()
    spec.sex_presentation.value = "male"
    spec.sex_presentation.confidence = 0.85
    spec.age_range.value = "30-35"
    spec.age_range.confidence = 0.90
    spec.skin_tone.value = "olive"
    spec.skin_tone.confidence = 0.70
    spec.eyes.color.value = "dark green"
    spec.eyes.color.confidence = 0.80
    spec.hair.style.value = "wavy short crop"
    spec.hair.style.confidence = 0.75

    prompt = build_forensic_prompt(spec, confidence_threshold=0.5)

    assert "male presentation" in prompt
    assert "estimated age 30-35" in prompt
    assert "olive skin tone" in prompt
    assert "dark green iris color" in prompt
    assert "wavy short crop style" in prompt


def test_prompt_builder_describes_low_confidence_or_unknown_attributes_neutrally():
    """Attributes with confidence < threshold or unstated are described neutrally as ambiguous/unspecified."""
    spec = FaceSpec()
    # High confidence attribute
    spec.face_shape.value = "oval"
    spec.face_shape.confidence = 0.90
    # Low confidence attribute (below 0.5 threshold)
    spec.nose.bridge.value = "crooked"
    spec.nose.bridge.confidence = 0.30

    prompt = build_forensic_prompt(spec, confidence_threshold=0.5)

    assert "oval face shape" in prompt
    assert "crooked" not in prompt  # Filtered out due to confidence < 0.5
    assert "ambiguous / unspecified neutral nose" in prompt
    assert "clean-shaven, no facial hair" in prompt
    assert "no glasses, no accessories, no jewelry" in prompt
    assert "no distinguishing marks" in prompt


def test_prompt_builder_mandatory_forensic_directives():
    """Validates passport-style composition, light-gray background, 1:1, and synthetic person disclaimer."""
    spec = FaceSpec()
    prompt = build_forensic_prompt(spec)

    assert "neutral frontal passport-style portrait" in prompt
    assert "Plain light-gray background" in prompt
    assert "even diffused studio lighting" in prompt
    assert "1:1 aspect ratio" in prompt
    assert "head-and-shoulders" in prompt.lower()
    assert "No text, no labels, no watermarks" in prompt
    assert "This is a synthetic, non-existent person." in prompt


def test_prompt_builder_with_accessories_specified():
    """When accessories are stated with high confidence, they are included."""
    spec = FaceSpec()
    spec.glasses_accessories.value = "black rectangular wireframe glasses"
    spec.glasses_accessories.confidence = 0.95

    prompt = build_forensic_prompt(spec, confidence_threshold=0.5)
    assert "accessories: black rectangular wireframe glasses" in prompt


def test_apply_feature_variation_holds_other_features_constant():
    """Variation replaces only the target feature, leaving others untouched."""
    spec = FaceSpec()
    spec.sex_presentation.value = "female"
    spec.sex_presentation.confidence = 0.9
    spec.hair.color.value = "blonde"
    spec.hair.color.confidence = 0.8
    spec.nose.bridge.value = "straight"
    spec.nose.bridge.confidence = 0.8

    varied = apply_feature_variation(spec, "nose.bridge", "aquiline hooked bridge")

    assert varied.nose.bridge.value == "aquiline hooked bridge"
    assert varied.nose.bridge.confidence == 1.0
    # Other features strictly identical
    assert varied.sex_presentation.value == "female"
    assert varied.hair.color.value == "blonde"


# --- 2. AES-256-GCM Envelope Encryption & Storage Tests ---

@pytest.mark.asyncio
async def test_envelope_encryption_roundtrip(db_session):
    """Verifies AES-256-GCM encryption, decryption, and ciphertext tampering detection."""
    case_id = "test-case-enc-001"
    raw_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRTESTIMAGEBYTESFORFORENSIC"

    # 1. Generate/unwrap per-case DEK
    dek = await get_or_create_case_dek(case_id, db_session)
    assert len(dek) == 32

    # 2. Encrypt
    encrypted_blob = encrypt_payload(dek, case_id, raw_image_data)
    assert encrypted_blob != raw_image_data
    assert len(encrypted_blob) == len(raw_image_data) + 12 + 16  # Nonce (12B) + Tag (16B) + Ciphertext

    # 3. Decrypt
    decrypted = decrypt_payload(dek, case_id, encrypted_blob)
    assert decrypted == raw_image_data

    # 4. Storage service store and retrieve
    storage_path = await storage_service.store_image(case_id, "img_123", raw_image_data, db_session)
    retrieved = await storage_service.retrieve_image(case_id, storage_path, db_session)
    assert retrieved == raw_image_data

    # 5. Tampering check: alter 1 byte in ciphertext -> decryption must fail
    tampered_blob = bytearray(encrypted_blob)
    tampered_blob[-1] ^= 0xFF
    with pytest.raises(Exception):
        decrypt_payload(dek, case_id, bytes(tampered_blob))


# --- 3. Mocked Gemini Image Generation Tests ---

@pytest.mark.asyncio
async def test_generate_initial_success_and_db_linking(db_session):
    """Tests generate_initial creates an image, envelope encrypts it, links version row, and logs cost/latency."""
    # Create test case
    case = Case(case_number="CASE-GEN-001", title="Initial Gen Test")
    db_session.add(case)
    await db_session.flush()

    spec = FaceSpec()
    spec.sex_presentation.value = "male"
    spec.sex_presentation.confidence = 0.9
    spec.jawline.value = "chiseled square"
    spec.jawline.confidence = 0.85

    fake_png = b"\x89PNG\r\n\x1a\nfake_rendered_portrait_bytes"
    mock_resp = GeminiImageResponse(
        image_bytes_list=[fake_png],
        model_name="imagen-3.0-generate-002",
        count=1
    )

    with patch("app.services.image_gen.gemini_client.generate_image", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_resp

        result = await generate_initial(
            face_spec=spec,
            case_id=case.id,
            version=1,
            triggering_call="test_initial_call",
            db=db_session
        )

        assert result.image_bytes == fake_png
        assert result.status == "success"
        assert result.storage_path is not None
        assert result.latency_ms > 0
        assert result.estimated_cost == settings.IMAGEN_COST_PER_IMAGE
        assert result.model_id == settings.GEMINI_IMAGE_MODEL

        # Check DB GeneratedImage linking row
        stmt = select(GeneratedImage).where(GeneratedImage.id == result.image_id)
        record = (await db_session.execute(stmt)).scalar_one_or_none()
        assert record is not None
        assert record.case_id == case.id
        assert record.version == 1
        assert record.image_path == result.storage_path
        assert record.prompt_hash == result.prompt_hash
        assert record.is_candidate is False


@pytest.mark.asyncio
async def test_generate_candidates_recognition_grid(db_session):
    """Tests generate_candidates generates 9 variations concurrently and stores records."""
    case = Case(case_number="CASE-GRID-001", title="Recognition Grid Test")
    db_session.add(case)
    await db_session.flush()

    spec = FaceSpec()
    spec.sex_presentation.value = "female"
    spec.sex_presentation.confidence = 0.9

    fake_png = b"\x89PNG\r\n\x1a\ncandidate_grid_image_bytes"
    mock_resp = GeminiImageResponse(
        image_bytes_list=[fake_png],
        model_name="imagen-3.0-generate-002",
        count=1
    )

    with patch("app.services.image_gen.gemini_client.generate_image", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_resp

        grid_result = await generate_candidates(
            face_spec=spec,
            n=9,
            vary="nose",
            case_id=case.id,
            version=1,
            db=db_session
        )

        assert grid_result.total_requested == 9
        assert grid_result.total_successful == 9
        assert grid_result.total_failed == 0
        assert len(grid_result.candidates) == 9
        assert mock_gen.call_count == 9

        # Verify DB records
        stmt = select(GeneratedImage).where(
            GeneratedImage.case_id == case.id,
            GeneratedImage.is_candidate.is_(True)
        )
        records = (await db_session.execute(stmt)).scalars().all()
        assert len(records) == 9
        assert all(r.variation_feature == "nose" for r in records)


@pytest.mark.asyncio
async def test_generate_candidates_partial_failure_tolerance(db_session):
    """Verifies that if 1 or 2 calls fail, the grid does not abort and returns successful candidates."""
    case = Case(case_number="CASE-PARTIAL-001", title="Partial Failure Grid Test")
    db_session.add(case)
    await db_session.flush()

    spec = FaceSpec()
    fake_png = b"\x89PNG\r\n\x1a\nsuccessful_candidate_bytes"

    call_count = 0

    async def mock_generate_with_failure(req):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            # Simulate an API error on the 3rd candidate
            raise APIError(503, {"error": {"message": "503 Service Unavailable temporarily"}})
        return GeminiImageResponse(
            image_bytes_list=[fake_png],
            model_name="imagen-3.0-generate-002",
            count=1
        )

    with patch("app.services.image_gen.gemini_client.generate_image", side_effect=mock_generate_with_failure):
        grid_result = await generate_candidates(
            face_spec=spec,
            n=9,
            vary="chin",
            case_id=case.id,
            version=1,
            db=db_session
        )

        assert grid_result.total_requested == 9
        assert grid_result.total_successful == 8
        assert grid_result.total_failed == 1
        assert any(c.status == "failed" for c in grid_result.candidates)
        assert sum(1 for c in grid_result.candidates if c.status == "success") == 8


@pytest.mark.asyncio
async def test_safety_refusal_surfaces_clear_error_and_does_not_crash(db_session):
    """Safety block from Google GenAI surfaces GenerationSafetyBlockError without crashing."""
    spec = FaceSpec()

    with patch("app.services.image_gen.gemini_client.generate_image", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = APIError(400, {"error": {"message": "Content generation was blocked due to safety policies."}})

        with pytest.raises(GenerationSafetyBlockError) as exc_info:
            await generate_initial(
                face_spec=spec,
                case_id="safety-test-case",
                db=db_session
            )

        assert "safety policy" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_websocket_progress_streaming():
    """Verifies progress events: queued -> generating -> done are received."""
    events = []

    async def mock_callback(payload):
        events.append(payload)

    spec = FaceSpec()
    mock_resp = GeminiImageResponse(
        image_bytes_list=[b"mock_bytes"],
        model_name="imagen-3.0-generate-002",
        count=1
    )

    with patch("app.services.image_gen.gemini_client.generate_image", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_resp

        await generate_initial(
            face_spec=spec,
            case_id="ws-case-test",
            progress_callback=mock_callback
        )

    statuses = [e["status"] for e in events]
    assert "queued" in statuses
    assert "generating" in statuses
    assert "done" in statuses


# --- 4. API Endpoint Integration Tests ---

@pytest.mark.asyncio
async def test_generate_endpoints_via_api(client, auth_tokens, db_session):
    """Tests POST /cases/{id}/generate/initial, candidates, and GET /cases/{id}/images/{image_id}."""
    # 1. Create a case as Investigator
    inv_token = auth_tokens[UserRole.INVESTIGATOR.value]
    headers = {"Authorization": f"Bearer {inv_token}"}

    create_resp = await client.post(
        "/api/v1/cases",
        headers=headers,
        json={"case_number": "CASE-API-GEN-01", "title": "API Image Gen Test"}
    )
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRTESTIMAGEFORAPI"
    mock_resp = GeminiImageResponse(
        image_bytes_list=[fake_png],
        model_name="imagen-3.0-generate-002",
        count=1
    )

    with patch("app.services.image_gen.gemini_client.generate_image", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_resp

        # 2. Call Initial Generation endpoint
        init_resp = await client.post(
            f"/api/v1/cases/{case_id}/generate/initial",
            headers=headers,
            json={"confidence_threshold": 0.5}
        )
        assert init_resp.status_code == 200
        init_data = init_resp.json()
        assert init_data["status"] == "success"
        image_id = init_data["image_id"]

        # 3. Retrieve decrypted image
        img_resp = await client.get(
            f"/api/v1/cases/{case_id}/images/{image_id}",
            headers=headers
        )
        assert img_resp.status_code == 200
        assert img_resp.content == fake_png

        # 4. Auditor isolation test: Auditor cannot view face composite image
        auditor_token = auth_tokens[UserRole.AUDITOR.value]
        auditor_headers = {"Authorization": f"Bearer {auditor_token}"}
        audit_img_resp = await client.get(
            f"/api/v1/cases/{case_id}/images/{image_id}",
            headers=auditor_headers
        )
        assert audit_img_resp.status_code == 403
        assert "Auditor role restriction" in audit_img_resp.json()["detail"]


# --- 5. Optional Live Gemini Imagen Test ---

@pytest.mark.skipif(
    not (os.getenv("RUN_LIVE_GEMINI_TESTS") == "true" and os.getenv("GEMINI_API_KEY") and not os.getenv("GEMINI_API_KEY", "").startswith("test_")),
    reason="Live Gemini image test gated by RUN_LIVE_GEMINI_TESTS=true and valid non-test GEMINI_API_KEY"
)
@pytest.mark.asyncio
async def test_live_gemini_image_generation():
    """Live integration test against Google GenAI Imagen model."""
    spec = FaceSpec()
    spec.sex_presentation.value = "male"
    spec.sex_presentation.confidence = 0.95
    spec.age_range.value = "40-45"
    spec.age_range.confidence = 0.90
    spec.hair.style.value = "short cropped dark brown hair"
    spec.hair.style.confidence = 0.85

    result = await generate_initial(
        face_spec=spec,
        case_id="live-test-case",
        confidence_threshold=0.5
    )
    assert len(result.image_bytes) > 1000
    assert result.image_bytes[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1")
    assert result.latency_ms > 0
