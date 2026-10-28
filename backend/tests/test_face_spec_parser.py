import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from sqlalchemy import select

from app.schemas.face_spec import (
    FaceSpec,
    FaceSpecPatch,
    FeatureAttribute,
    EyesSpec,
    EyesPatch,
    NoseSpec,
    NosePatch,
    HairSpec,
    HairPatch,
    DistinguishingMarksSpec,
    DistinguishingMarksPatch,
)
from app.agents.parser import (
    FaceSpecParserAgent,
    AdversarialCelebrityError,
    detect_celebrity_reference,
    load_system_prompt,
    PROMPT_FILE,
)
from app.models.case import Case, CompositeSession, Utterance
from app.models.audit import AuditLogEntry, AuditAction, AuditOutcome
from app.schemas.gemini import GeminiTextResponse


# ==============================================================================
# 1. FaceSpec Model & Diff Tests
# ==============================================================================

def test_face_spec_default_unknown_semantics():
    """Rule 3: Missing features evaluate to None with 'unknown' semantics."""
    spec = FaceSpec()
    
    # Flat attributes
    assert spec.age_range.value is None
    assert spec.age_range.confidence == 0.0
    assert spec.age_range.source_utterance_ids == []
    assert not spec.age_range.is_stated

    assert spec.sex_presentation.value is None
    assert spec.skin_tone.value is None
    assert spec.face_shape.value is None
    assert spec.forehead.value is None
    assert spec.eyebrows.value is None
    assert spec.mouth_lips.value is None
    assert spec.jawline.value is None
    assert spec.chin.value is None
    assert spec.cheekbones.value is None
    assert spec.ears.value is None
    assert spec.facial_hair.value is None
    assert spec.glasses_accessories.value is None
    assert spec.expression.value is None
    assert spec.notes.value is None

    # Nested attributes
    assert spec.eyes.shape.value is None
    assert spec.eyes.size.value is None
    assert spec.eyes.spacing.value is None
    assert spec.eyes.color.value is None

    assert spec.nose.width.value is None
    assert spec.nose.length.value is None
    assert spec.nose.bridge.value is None
    assert spec.nose.tip.value is None

    assert spec.hair.length.value is None
    assert spec.hair.style.value is None
    assert spec.hair.color.value is None
    assert spec.hair.hairline.value is None

    assert spec.distinguishing_marks.type.value is None
    assert spec.distinguishing_marks.location.value is None


def test_face_spec_diff_computation():
    """Verifies FaceSpec.diff(other) accurately pinpoints modified attributes."""
    spec_a = FaceSpec()
    spec_a.face_shape = FeatureAttribute(value="oval", confidence=0.8, source_utterance_ids=["u1"])
    spec_a.eyes.color = FeatureAttribute(value="brown", confidence=0.7, source_utterance_ids=["u1"])

    spec_b = spec_a.model_copy(deep=True)
    # Modify eyes.color and add nose.width
    spec_b.eyes.color = FeatureAttribute(value="dark hazel", confidence=0.9, source_utterance_ids=["u2"])
    spec_b.nose.width = FeatureAttribute(value="wide", confidence=0.85, source_utterance_ids=["u2"])

    diff = spec_a.diff(spec_b)

    assert "eyes.color" in diff
    assert diff["eyes.color"]["before"]["value"] == "brown"
    assert diff["eyes.color"]["after"]["value"] == "dark hazel"
    assert diff["eyes.color"]["after"]["confidence"] == 0.9

    assert "nose.width" in diff
    assert diff["nose.width"]["before"]["value"] is None
    assert diff["nose.width"]["after"]["value"] == "wide"

    # Unchanged fields must not be in diff
    assert "face_shape" not in diff
    assert "jawline" not in diff


def test_face_spec_locked_regions_respected():
    """Locked regions must be protected against patch overwrites."""
    spec = FaceSpec()
    spec.nose.width = FeatureAttribute(
        value="narrow",
        confidence=0.9,
        source_utterance_ids=["u1"],
        locked=True
    )

    patch = FaceSpecPatch(
        nose=NosePatch(
            width=FeatureAttribute(value="wide", confidence=0.95, source_utterance_ids=["u2"])
        )
    )

    updated_spec = spec.apply_patch(patch, utterance_id="u2")
    # Must remain narrow because nose.width was locked
    assert updated_spec.nose.width.value == "narrow"
    assert updated_spec.nose.width.source_utterance_ids == ["u1"]


# ==============================================================================
# 2. Parser Prompt and Celebrity Detector Tests
# ==============================================================================

def test_parser_system_prompt_content():
    """Verifies that agents/prompts/parser.md matches the forensic extraction specification."""
    prompt = load_system_prompt()
    assert "extract facial descriptions from a crime witness" in prompt
    assert "(a) record ONLY what the witness explicitly stated" in prompt
    assert "(b) never infer ethnicity, skin tone, age or gender unless stated" in prompt
    assert "(c) convert hedging language into confidence between 0 and 1" in prompt
    assert "(d) if the witness corrects an earlier statement" in prompt
    assert "(e) preserve the witness's own wording in a 'verbatim' field" in prompt
    assert "(f) output must validate against the schema" in prompt


def test_celebrity_reference_detection():
    """Detects attempts to compare suspects to celebrities or public figures."""
    assert detect_celebrity_reference("Make him look like Brad Pitt") is not None
    assert detect_celebrity_reference("He looked just like Tom Cruise") is not None
    assert detect_celebrity_reference("Look like Taylor Swift with blonde hair") is not None
    assert detect_celebrity_reference("Suspect had an oval face and dark brown eyes") is None


# ==============================================================================
# 3. Fixtures for NL Parser: Hedged, Correction, Negation, Adversarial
# ==============================================================================

@pytest.mark.asyncio
async def test_parser_hedged_description():
    """
    Fixture 1: Hedged description
    'I think he had bushy eyebrows, maybe light brown eyes.'
    Hedges mapped to lower confidence (0.6, 0.4), verbatim preserved.
    """
    agent = FaceSpecParserAgent()

    mock_gemini_json = json.dumps({
        "eyebrows": {
            "value": "bushy",
            "confidence": 0.6,
            "verbatim": "I think he had bushy eyebrows",
            "source_utterance_ids": ["utt-hedge-1"]
        },
        "eyes": {
            "color": {
                "value": "light brown",
                "confidence": 0.4,
                "verbatim": "maybe light brown eyes",
                "source_utterance_ids": ["utt-hedge-1"]
            }
        }
    })

    with patch("app.agents.parser.gemini_client.generate_text", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = GeminiTextResponse(content=mock_gemini_json, model_name="gemini-2.5-flash")

        patch_res = await agent.parse_utterance(
            utterance="I think he had bushy eyebrows, maybe light brown eyes.",
            utterance_id="utt-hedge-1",
            current_spec=FaceSpec()
        )

        assert patch_res.eyebrows is not None
        assert patch_res.eyebrows.value == "bushy"
        assert patch_res.eyebrows.confidence == 0.6
        assert patch_res.eyebrows.verbatim == "I think he had bushy eyebrows"
        assert "utt-hedge-1" in patch_res.eyebrows.source_utterance_ids

        assert patch_res.eyes is not None
        assert patch_res.eyes.color.value == "light brown"
        assert patch_res.eyes.color.confidence == 0.4
        assert patch_res.eyes.color.verbatim == "maybe light brown eyes"

        # Unstated attributes remain null
        assert patch_res.nose is None
        assert patch_res.jawline is None
        assert patch_res.skin_tone is None


@pytest.mark.asyncio
async def test_parser_correction_overwrites_field_and_source():
    """
    Fixture 2: Correction
    Earlier statement: nose.width = 'wide' (utt-001).
    Correction: 'No, his nose was narrower than that, definitely narrow.'
    Overwrites nose.width and assigns new utterance ID.
    """
    agent = FaceSpecParserAgent()
    current_spec = FaceSpec()
    current_spec.nose.width = FeatureAttribute(
        value="wide",
        confidence=0.8,
        verbatim="wide nose",
        source_utterance_ids=["utt-001"]
    )

    mock_gemini_json = json.dumps({
        "nose": {
            "width": {
                "value": "narrow",
                "confidence": 0.9,
                "verbatim": "No, his nose was narrower than that, definitely narrow",
                "source_utterance_ids": ["utt-002"]
            }
        }
    })

    with patch("app.agents.parser.gemini_client.generate_text", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = GeminiTextResponse(content=mock_gemini_json, model_name="gemini-2.5-flash")

        patch_res = await agent.parse_utterance(
            utterance="No, his nose was narrower than that, definitely narrow.",
            utterance_id="utt-002",
            current_spec=current_spec
        )

        updated_spec = current_spec.apply_patch(patch_res, utterance_id="utt-002")

        assert updated_spec.nose.width.value == "narrow"
        assert updated_spec.nose.width.confidence == 0.9
        assert updated_spec.nose.width.source_utterance_ids == ["utt-002"]

        diff = current_spec.diff(updated_spec)
        assert "nose.width" in diff
        assert diff["nose.width"]["before"]["value"] == "wide"
        assert diff["nose.width"]["after"]["value"] == "narrow"


@pytest.mark.asyncio
async def test_parser_negation_statement():
    """
    Fixture 3: Negation
    'He definitely did not have any facial hair, completely clean shaven.'
    Captures negation as explicit attribute value with high certainty.
    """
    agent = FaceSpecParserAgent()

    mock_gemini_json = json.dumps({
        "facial_hair": {
            "value": "none (clean shaven)",
            "confidence": 0.95,
            "verbatim": "He definitely did not have any facial hair, completely clean shaven",
            "source_utterance_ids": ["utt-neg-1"]
        }
    })

    with patch("app.agents.parser.gemini_client.generate_text", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = GeminiTextResponse(content=mock_gemini_json, model_name="gemini-2.5-flash")

        patch_res = await agent.parse_utterance(
            utterance="He definitely did not have any facial hair, completely clean shaven.",
            utterance_id="utt-neg-1",
            current_spec=FaceSpec()
        )

        assert patch_res.facial_hair is not None
        assert "clean shaven" in patch_res.facial_hair.value.lower()
        assert patch_res.facial_hair.confidence >= 0.9


@pytest.mark.asyncio
async def test_parser_adversarial_celebrity_rejection():
    """
    Fixture 4: Adversarial description trying to name a celebrity
    'Make him look like Brad Pitt with a sharp chin'
    Must raise AdversarialCelebrityError and never call model.
    """
    agent = FaceSpecParserAgent()

    with pytest.raises(AdversarialCelebrityError) as exc_info:
        await agent.parse_utterance(
            utterance="Make him look like Brad Pitt with a sharp chin",
            utterance_id="utt-adv-1",
            current_spec=FaceSpec()
        )

    assert "celebrity" in str(exc_info.value).lower()
    assert "brad pitt" in str(exc_info.value).lower()


# ==============================================================================
# 4. Full API Endpoint Tests (POST /cases/{id}/describe)
# ==============================================================================

@pytest.mark.asyncio
async def test_describe_endpoint_success_and_audit(client: AsyncClient, auth_tokens, db_session):
    """
    Tests POST /api/v1/cases/{id}/describe:
    - Stores utterance
    - Applies patch
    - Writes immutable FaceSpec CompositeSession version row
    - Logs CREATE_VERSION action with diff to the cryptographic audit chain
    """
    inv_token = auth_tokens["investigator@forensic.local"]
    headers = {"Authorization": f"Bearer {inv_token}"}

    # 1. Create a case
    create_resp = await client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "case_number": "CASE-DESCRIBE-01",
            "title": "Eastside Bank Robbery Witness Interview"
        }
    )
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    # 2. Describe suspect's eyebrows and eyes (Hedged description)
    mock_gemini_json_1 = json.dumps({
        "eyebrows": {
            "value": "thick and arched",
            "confidence": 0.6,
            "verbatim": "I think his eyebrows were thick and arched",
            "source_utterance_ids": []
        },
        "eyes": {
            "color": {
                "value": "dark brown",
                "confidence": 0.9,
                "verbatim": "definitely dark brown eyes",
                "source_utterance_ids": []
            }
        }
    })

    with patch("app.agents.parser.gemini_client.generate_text", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = GeminiTextResponse(content=mock_gemini_json_1, model_name="gemini-2.5-flash")

        desc_resp_1 = await client.post(
            f"/api/v1/cases/{case_id}/describe",
            headers=headers,
            json={
                "utterance": "I think his eyebrows were thick and arched, definitely dark brown eyes.",
                "speaker": "witness_1"
            }
        )

        assert desc_resp_1.status_code == 200
        data_1 = desc_resp_1.json()
        assert data_1["version"] == 1
        assert data_1["face_spec"]["eyebrows"]["value"] == "thick and arched"
        assert data_1["face_spec"]["eyes"]["color"]["value"] == "dark brown"
        assert "eyebrows" in data_1["diff"]
        assert "eyes.color" in data_1["diff"]

    # 3. Second utterance: Correction to eyebrows
    mock_gemini_json_2 = json.dumps({
        "eyebrows": {
            "value": "straight and bushy",
            "confidence": 0.85,
            "verbatim": "No, his eyebrows were actually straight and bushy",
            "source_utterance_ids": []
        }
    })

    with patch("app.agents.parser.gemini_client.generate_text", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = GeminiTextResponse(content=mock_gemini_json_2, model_name="gemini-2.5-flash")

        desc_resp_2 = await client.post(
            f"/api/v1/cases/{case_id}/describe",
            headers=headers,
            json={
                "utterance": "No, his eyebrows were actually straight and bushy, not arched.",
                "speaker": "witness_1"
            }
        )

        assert desc_resp_2.status_code == 200
        data_2 = desc_resp_2.json()
        assert data_2["version"] == 2
        # Eyebrows updated to corrected value
        assert data_2["face_spec"]["eyebrows"]["value"] == "straight and bushy"
        # Eyes color persisted from version 1
        assert data_2["face_spec"]["eyes"]["color"]["value"] == "dark brown"
        assert "eyebrows" in data_2["diff"]
        assert data_2["diff"]["eyebrows"]["before"]["value"] == "thick and arched"
        assert data_2["diff"]["eyebrows"]["after"]["value"] == "straight and bushy"

    # 4. Verify immutable version rows in DB
    v_stmt = select(CompositeSession).where(CompositeSession.case_id == case_id).order_by(CompositeSession.version.asc())
    sessions = (await db_session.execute(v_stmt)).scalars().all()
    assert len(sessions) == 2
    assert sessions[0].version == 1
    assert sessions[1].version == 2

    # 5. Verify utterances stored in DB
    u_stmt = select(Utterance).where(Utterance.case_id == case_id)
    utterances = (await db_session.execute(u_stmt)).scalars().all()
    assert len(utterances) == 2

    # 6. Verify cryptographic audit chain entries for CREATE_VERSION
    a_stmt = select(AuditLogEntry).where(
        AuditLogEntry.case_id == case_id,
        AuditLogEntry.action == AuditAction.CREATE_VERSION
    )
    audit_entries = (await db_session.execute(a_stmt)).scalars().all()
    assert len(audit_entries) == 2
    for entry in audit_entries:
        assert entry.outcome == AuditOutcome.SUCCESS
        assert "version" in entry.details


@pytest.mark.asyncio
async def test_describe_endpoint_adversarial_celebrity_rejected_and_logged(
    client: AsyncClient,
    auth_tokens,
    db_session
):
    """
    Tests adversarial celebrity rejection on POST /api/v1/cases/{id}/describe:
    - Attempting to name a celebrity (e.g. 'Make him look like Brad Pitt') must return HTTP 400
    - Must be recorded in the audit ledger with outcome DENIED
    - No version row may be created
    """
    inv_token = auth_tokens["investigator@forensic.local"]
    headers = {"Authorization": f"Bearer {inv_token}"}

    # 1. Create a case
    create_resp = await client.post(
        "/api/v1/cases",
        headers=headers,
        json={
            "case_number": "CASE-ADVERSARIAL-01",
            "title": "Adversarial Test Case"
        }
    )
    case_id = create_resp.json()["id"]

    # 2. Submit adversarial celebrity description
    adv_resp = await client.post(
        f"/api/v1/cases/{case_id}/describe",
        headers=headers,
        json={
            "utterance": "He looks just like Brad Pitt, make him look like Brad Pitt.",
            "speaker": "adversarial_witness"
        }
    )

    # 3. Must be rejected with 400 Bad Request
    assert adv_resp.status_code == 400
    assert "celebrity" in adv_resp.json()["detail"].lower()

    # 4. Verify no CompositeSession version row was created
    cs_stmt = select(CompositeSession).where(CompositeSession.case_id == case_id)
    sessions = (await db_session.execute(cs_stmt)).scalars().all()
    assert len(sessions) == 0

    # 5. Verify audit log entry was appended with outcome=DENIED
    a_stmt = select(AuditLogEntry).where(
        AuditLogEntry.case_id == case_id,
        AuditLogEntry.outcome == AuditOutcome.DENIED
    )
    denied_entries = (await db_session.execute(a_stmt)).scalars().all()
    assert len(denied_entries) >= 1
    assert "celebrity" in denied_entries[0].details.lower()
