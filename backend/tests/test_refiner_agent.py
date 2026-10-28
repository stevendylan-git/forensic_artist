"""
Tests for Refinement Agent and Leading-Question Detector (Phase 3)
Requirements:
(4) leading-question detector test on the agent's own ask_witness outputs.
Strict tool declarations, Rule 8 celebrity blocklist, and locked region handling.
"""

import pytest
from app.agents.refiner import (
    refinement_agent,
    LeadingQuestionDetector,
    EditRegionCall,
    AdjustGlobalCall,
    LockFeatureCall,
    UnlockFeatureCall,
    RevertCall,
    CompareVersionsCall,
    AskWitnessCall,
    RegionEnum,
    MagnitudeEnum,
    REFINEMENT_FUNCTION_DECLARATIONS
)


def test_leading_question_detector_rules():
    """
    Test 4: Leading-question detector tests.
    Validates that:
    - Closed and suggestive questions about facial appearance are detected as LEADING.
    - Neutral cognitive interview questions are accepted as NON-LEADING.
    """
    # 1. Closed appearance questions (MUST be flagged as leading)
    leading_questions = [
        "Was his nose wide?",
        "Did he have blue eyes?",
        "Was the hair curly?",
        "Could the mouth have been larger?",
        "Was his chin pointed?",
        "Did she have a hooked nose?",
        "Would you say his forehead was high?",
        "Was the suspect's skin pale?",
        "Is his jaw square?",
        "Did he have a big nose?",
        "Was he tall and heavy?"
    ]

    for q in leading_questions:
        is_lead, reason = LeadingQuestionDetector.is_leading(q)
        assert is_lead is True, f"Question '{q}' should be flagged as LEADING, but passed!"
        assert reason is not None

    # 2. Open-ended elicitation questions (MUST NOT be flagged as leading)
    open_questions = [
        "What do you remember about the nose?",
        "Can you describe the suspect's eyes?",
        "Tell me about the hair style.",
        "What do you recall about the mouth and lips?",
        "What shape was the chin, if you recall?",
        "Can you describe anything about the facial hair?",
        "What do you recall about the suspect's face overall?",
        "Take your time and tell me what you remember about the eyebrows."
    ]

    for q in open_questions:
        is_lead, reason = LeadingQuestionDetector.is_leading(q)
        assert is_lead is False, f"Question '{q}' should be accepted as NON-LEADING, but was flagged ({reason})!"


def test_leading_question_sanitization():
    """
    Tests that a leading question is automatically sanitized into an approved open-ended elicitation.
    """
    leading_q = "Was his nose wide and crooked?"
    sanitized = LeadingQuestionDetector.sanitize_to_open_ended(leading_q, target_region="nose")
    assert "what do you recall" in sanitized.lower()
    assert "nose" in sanitized.lower()

    # Verify sanitized output now passes the detector
    is_lead, _ = LeadingQuestionDetector.is_leading(sanitized)
    assert is_lead is False, "Sanitized question must pass anti-leading detector!"


@pytest.mark.asyncio
async def test_refiner_agent_tool_schema_coverage():
    """
    Validates that all required tools are declared with strict parameter schemas:
    - edit_region
    - adjust_global
    - lock_feature
    - unlock_feature
    - revert
    - compare_versions
    - ask_witness
    """
    declared_names = {decl.name for decl in REFINEMENT_FUNCTION_DECLARATIONS}
    expected_names = {
        "edit_region",
        "adjust_global",
        "lock_feature",
        "unlock_feature",
        "revert",
        "compare_versions",
        "ask_witness"
    }
    assert expected_names.issubset(declared_names), f"Missing tools: {expected_names - declared_names}"


@pytest.mark.asyncio
async def test_refiner_agent_parses_edit_region():
    """Tests parsing witness request to edit region with direction and magnitude."""
    tools = await refinement_agent.process_feedback(
        message="Please make the nose slightly narrower",
        current_version=1,
        locked_regions=[]
    )
    assert len(tools) == 1
    call = tools[0]
    assert isinstance(call, EditRegionCall)
    assert call.region == RegionEnum.NOSE
    assert call.direction_or_value == "narrower"
    assert call.magnitude == MagnitudeEnum.SLIGHT


@pytest.mark.asyncio
async def test_refiner_agent_respects_locked_features():
    """
    If witness tries to edit a locked region, the agent must NOT output edit_region;
    it should call ask_witness asking if they want to unlock it.
    """
    tools = await refinement_agent.process_feedback(
        message="Make the eyes darker",
        current_version=2,
        locked_regions=["eyes"]
    )
    assert len(tools) == 1
    call = tools[0]
    assert isinstance(call, AskWitnessCall)
    assert "eyes" in call.question.lower()
    assert "locked" in call.question.lower()


@pytest.mark.asyncio
async def test_refiner_agent_parses_lock_feature():
    """Tests witness stating a feature is right/good translates to lock_feature."""
    tools = await refinement_agent.process_feedback(
        message="The jaw looks perfect now",
        current_version=3,
        locked_regions=[]
    )
    assert len(tools) == 1
    call = tools[0]
    assert isinstance(call, LockFeatureCall)
    assert call.region == RegionEnum.JAW


@pytest.mark.asyncio
async def test_refiner_agent_parses_revert_and_compare():
    """Tests revert and compare version commands."""
    tools_revert = await refinement_agent.process_feedback(
        message="Can we go back to version 1?",
        current_version=3,
        available_versions=[1, 2, 3]
    )
    assert len(tools_revert) == 1
    assert isinstance(tools_revert[0], RevertCall)
    assert tools_revert[0].to_version == 1

    tools_compare = await refinement_agent.process_feedback(
        message="Compare version 1 and 3",
        current_version=3,
        available_versions=[1, 2, 3]
    )
    assert len(tools_compare) == 1
    assert isinstance(tools_compare[0], CompareVersionsCall)
    assert tools_compare[0].a == 1
    assert tools_compare[0].b == 3


@pytest.mark.asyncio
async def test_refiner_agent_rejects_named_celebrities():
    """Rule 8: Attempting to reference a real individual raises ValueError."""
    with pytest.raises(ValueError, match="Rule 8"):
        await refinement_agent.process_feedback(
            message="Make him look like Elon Musk",
            current_version=1
        )
