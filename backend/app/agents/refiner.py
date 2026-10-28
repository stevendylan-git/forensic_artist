"""
Forensic Refinement Agent (Phase 3)
Translates witness conversational feedback into strict tool calls.
Rules:
- Output MUST be tool calls, not prose edits.
- Only change what the witness asked to change.
- Never ask leading or yes/no appearance questions.
- If instruction is ambiguous, call ask_witness with an open-ended question.
- If witness says a feature is right/good/perfect, call lock_feature.
- Respect locked regions.
- Never reference named real people (Rule 8).
"""

import os
import re
import json
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field, ConfigDict

from google.genai import types
from app.config import settings
from app.services.gemini_client import gemini_client
from app.agents.interview_agent import check_for_named_real_people

logger = logging.getLogger("refiner_agent")

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "refiner.md")


def load_refiner_system_prompt() -> str:
    """Loads the system prompt from agents/prompts/refiner.md."""
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return (
        "You are a refinement assistant for a witness building a facial composite. "
        "Translate the witness's feedback into tool calls that change ONLY what they asked to change. "
        "Never suggest features, never ask leading or yes/no questions about appearance ('was it wide?'). "
        "If the instruction is ambiguous, call ask_witness with an open question such as 'What do you remember about the nose?'. "
        "If the witness says a feature is right, call lock_feature. Respect locked regions. Never reference real people."
    )


class RegionEnum(str, Enum):
    HAIR = "hair"
    FOREHEAD = "forehead"
    EYEBROWS = "eyebrows"
    EYES = "eyes"
    NOSE = "nose"
    MOUTH = "mouth"
    JAW = "jaw"
    CHIN = "chin"
    CHEEKS = "cheeks"
    EARS = "ears"
    FACIAL_HAIR = "facial_hair"
    SKIN = "skin"
    WHOLE_FACE = "whole_face"


class MagnitudeEnum(str, Enum):
    SLIGHT = "slight"
    MODERATE = "moderate"
    STRONG = "strong"


# Typed tool call models
class EditRegionCall(BaseModel):
    tool: str = "edit_region"
    region: RegionEnum
    attribute: str
    direction_or_value: str
    magnitude: MagnitudeEnum = MagnitudeEnum.MODERATE


class AdjustGlobalCall(BaseModel):
    tool: str = "adjust_global"
    age_delta_years: Optional[int] = None
    weight: Optional[str] = None
    expression: Optional[str] = None


class LockFeatureCall(BaseModel):
    tool: str = "lock_feature"
    region: RegionEnum


class UnlockFeatureCall(BaseModel):
    tool: str = "unlock_feature"
    region: RegionEnum


class RevertCall(BaseModel):
    tool: str = "revert"
    region: Optional[str] = None  # None or "all" / "whole_face" indicates full composite revert
    to_version: int


class CompareVersionsCall(BaseModel):
    tool: str = "compare_versions"
    a: int
    b: int


class AskWitnessCall(BaseModel):
    tool: str = "ask_witness"
    question: str


RefinementToolCall = Union[
    EditRegionCall,
    AdjustGlobalCall,
    LockFeatureCall,
    UnlockFeatureCall,
    RevertCall,
    CompareVersionsCall,
    AskWitnessCall,
]


# Leading question detector patterns
LEADING_PATTERNS = [
    # Yes/No appearance questions
    r"\b(was|were|is|are|did|does|could|would)\s+(he|she|they|the suspect|his|her|their|it|the\s+\w+)\s+(have|had)?\s*(a\s+)?(wide|narrow|big|small|large|thin|crooked|pointed|hooked|flat|curly|straight|dark|light|blue|green|brown|hazel|bushy|pale|tan|round|square|sharp|long|short)",
    r"\b(was|is)\s+(the|his|her|their)?\s*(nose|eye|eyes|mouth|hair|chin|jaw|forehead|face|skin)\s+(wide|narrow|big|small|large|thin|crooked|pointed|hooked|flat|curly|straight|dark|light|blue|green|brown|hazel|bushy|pale|tan|round|square|sharp|long|short)\b",
    r"\b(would you say|do you think|was it more like)\b",
    r"\b(was he|was she|did he|did she)\s+(tall|short|fat|thin|heavy|young|old)\b",
]

LEADING_REGEX = re.compile("|".join(LEADING_PATTERNS), re.IGNORECASE)


class LeadingQuestionDetector:
    """
    Forensic cognitive interview integrity guard.
    Rule 7: Never ask leading or suggestive appearance questions.
    """

    @classmethod
    def is_leading(cls, question: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether a proposed question violates cognitive interviewing protocols.
        Returns (is_leading, reason).
        """
        q = question.strip()

        # Check regex for suggestive/leading structures
        if LEADING_REGEX.search(q):
            return True, "Question suggests specific morphological attributes or forms a closed appearance query."

        # Check for closed Yes/No questions about appearance
        words = q.lower().split()
        if words and words[0] in {"was", "were", "did", "does", "is", "are", "could", "would"}:
            appearance_keywords = {"nose", "eyes", "hair", "mouth", "lips", "chin", "jaw", "face", "skin", "ears"}
            if any(k in words for k in appearance_keywords):
                return True, "Closed yes/no question about facial appearance detected."

        return False, None

    @classmethod
    def sanitize_to_open_ended(cls, question: str, target_region: Optional[str] = None) -> str:
        """
        Transforms a leading/closed question into an approved forensic open elicitation.
        """
        if target_region:
            return f"What do you recall about the suspect's {target_region}?"
        for reg in RegionEnum:
            if reg.value in question.lower():
                return f"What do you remember about the suspect's {reg.value}?"
        return "Can you tell me more about what you recall seeing?"


# Strict Function Declarations for Gemini Function Calling
REFINEMENT_FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="edit_region",
        description="Edits a specific facial region while leaving all other regions untouched.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "region": types.Schema(
                    type="STRING",
                    enum=[e.value for e in RegionEnum],
                    description="The target facial region to modify."
                ),
                "attribute": types.Schema(
                    type="STRING",
                    description="The specific attribute of the region (e.g., width, length, bridge, tip, shape, color, style)."
                ),
                "direction_or_value": types.Schema(
                    type="STRING",
                    description="The desired modification or direction (e.g., narrower, wider, hooked, darker, shorter)."
                ),
                "magnitude": types.Schema(
                    type="STRING",
                    enum=[e.value for e in MagnitudeEnum],
                    description="Magnitude of the change."
                )
            },
            required=["region", "attribute", "direction_or_value", "magnitude"]
        )
    ),
    types.FunctionDeclaration(
        name="adjust_global",
        description="Adjusts holistic or global attributes across the whole face.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "age_delta_years": types.Schema(
                    type="INTEGER",
                    description="Age adjustment delta in years (e.g. +5 or -5)."
                ),
                "weight": types.Schema(
                    type="STRING",
                    description="Weight appearance adjustment (e.g. thinner, fuller, gaunt)."
                ),
                "expression": types.Schema(
                    type="STRING",
                    description="Facial expression adjustment (e.g. neutral, frowning, tense)."
                )
            }
        )
    ),
    types.FunctionDeclaration(
        name="lock_feature",
        description="Locks a facial region so it cannot be modified by subsequent edits.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "region": types.Schema(
                    type="STRING",
                    enum=[e.value for e in RegionEnum],
                    description="The facial region to lock."
                )
            },
            required=["region"]
        )
    ),
    types.FunctionDeclaration(
        name="unlock_feature",
        description="Unlocks a previously locked facial region.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "region": types.Schema(
                    type="STRING",
                    enum=[e.value for e in RegionEnum],
                    description="The facial region to unlock."
                )
            },
            required=["region"]
        )
    ),
    types.FunctionDeclaration(
        name="revert",
        description="Reverts a region or the whole composite to an earlier version.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "region": types.Schema(
                    type="STRING",
                    description="Region to revert, or 'whole_face'/'all' to revert the full face."
                ),
                "to_version": types.Schema(
                    type="INTEGER",
                    description="Target historical version number to revert to."
                )
            },
            required=["to_version"]
        )
    ),
    types.FunctionDeclaration(
        name="compare_versions",
        description="Presents a side-by-side comparison between two versions.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "a": types.Schema(
                    type="INTEGER",
                    description="First version number to compare."
                ),
                "b": types.Schema(
                    type="INTEGER",
                    description="Second version number to compare."
                )
            },
            required=["a", "b"]
        )
    ),
    types.FunctionDeclaration(
        name="ask_witness",
        description="Asks the witness an open-ended question for clarification without leading or suggesting features.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "question": types.Schema(
                    type="STRING",
                    description="Open-ended question asking what the witness recalls (never leading, never yes/no)."
                )
            },
            required=["question"]
        )
    ),
]

REFINEMENT_TOOL = types.Tool(function_declarations=REFINEMENT_FUNCTION_DECLARATIONS)


class RefinementAgent:
    """
    Refinement Agent using Gemini Function Calling.
    Translates witness chat into strict tool calls.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_TEXT_MODEL
        self.system_prompt = load_refiner_system_prompt()

    async def process_feedback(
        self,
        message: str,
        current_version: int = 1,
        locked_regions: Optional[List[str]] = None,
        available_versions: Optional[List[int]] = None,
        stated_features: Optional[Dict[str, Any]] = None,
    ) -> List[RefinementToolCall]:
        """
        Processes witness feedback and returns a list of validated tool calls.
        """
        locked_regions = locked_regions or []
        available_versions = available_versions or [current_version]

        # 1. Rule 8: Named real individuals blocklist
        celebrity_name = check_for_named_real_people(message)
        if celebrity_name:
            raise ValueError(
                f"Reference to real individual '{celebrity_name}' is strictly prohibited under Rule 8."
            )

        # 2. Check if running with real Gemini API or mock/unit test mode
        if not self.api_key or self.api_key.startswith("test_key"):
            return self._heuristic_fallback_parser(
                message, current_version, locked_regions, available_versions
            )

        # 3. Build prompt with context
        context_lines = [
            f"Current Version: {current_version}",
            f"Available Versions: {available_versions}",
            f"Currently Locked Regions: {locked_regions}",
        ]
        if stated_features:
            context_lines.append(f"Currently Stated Features: {json.dumps(stated_features)}")

        full_prompt = (
            f"CONTEXT:\n" + "\n".join(context_lines) + "\n\n"
            f"WITNESS INPUT: \"{message}\"\n\n"
            f"Translate this input into one or more strict tool calls. Do NOT output prose text."
        )

        try:
            client = gemini_client.get_client()
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                tools=[REFINEMENT_TOOL],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY"  # Force tool calling
                    )
                ),
                temperature=0.0
            )

            response = client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=config
            )

            tool_calls = self._extract_tool_calls_from_response(response, locked_regions)
            if not tool_calls:
                return self._heuristic_fallback_parser(
                    message, current_version, locked_regions, available_versions
                )
            return tool_calls

        except Exception as e:
            logger.warning(f"Gemini function calling failed or unavailable: {e}. Using heuristic parser.")
            return self._heuristic_fallback_parser(
                message, current_version, locked_regions, available_versions
            )

    def _extract_tool_calls_from_response(
        self,
        response: Any,
        locked_regions: List[str]
    ) -> List[RefinementToolCall]:
        """Parses function calls from Gemini API response candidates."""
        tool_calls: List[RefinementToolCall] = []

        if not hasattr(response, "candidates") or not response.candidates:
            return tool_calls

        candidate = response.candidates[0]
        if not hasattr(candidate, "content") or not candidate.content:
            return tool_calls

        parts = getattr(candidate.content, "parts", []) or []
        for part in parts:
            fn_call = getattr(part, "function_call", None)
            if fn_call is not None:
                name = fn_call.name
                args = dict(fn_call.args) if hasattr(fn_call, "args") and fn_call.args else {}

                parsed = self._parse_single_call(name, args, locked_regions)
                if parsed:
                    tool_calls.append(parsed)

        return tool_calls

    def _parse_single_call(
        self,
        name: str,
        args: Dict[str, Any],
        locked_regions: List[str]
    ) -> Optional[RefinementToolCall]:
        """Validates and instantiates a typed tool call."""
        try:
            if name == "edit_region":
                reg = RegionEnum(args["region"].lower())
                # Check locked constraint
                if reg.value in locked_regions:
                    return AskWitnessCall(
                        question=f"The {reg.value} is currently locked. Would you like to unlock it to make changes?"
                    )
                mag = MagnitudeEnum(args.get("magnitude", "moderate").lower())
                return EditRegionCall(
                    region=reg,
                    attribute=args.get("attribute", "general"),
                    direction_or_value=args.get("direction_or_value", "modified"),
                    magnitude=mag
                )

            elif name == "adjust_global":
                return AdjustGlobalCall(
                    age_delta_years=args.get("age_delta_years"),
                    weight=args.get("weight"),
                    expression=args.get("expression")
                )

            elif name == "lock_feature":
                reg = RegionEnum(args["region"].lower())
                return LockFeatureCall(region=reg)

            elif name == "unlock_feature":
                reg = RegionEnum(args["region"].lower())
                return UnlockFeatureCall(region=reg)

            elif name == "revert":
                return RevertCall(
                    region=args.get("region"),
                    to_version=int(args["to_version"])
                )

            elif name == "compare_versions":
                return CompareVersionsCall(
                    a=int(args["a"]),
                    b=int(args["b"])
                )

            elif name == "ask_witness":
                raw_q = args["question"]
                is_lead, _ = LeadingQuestionDetector.is_leading(raw_q)
                if is_lead:
                    raw_q = LeadingQuestionDetector.sanitize_to_open_ended(raw_q)
                return AskWitnessCall(question=raw_q)

        except Exception as e:
            logger.warning(f"Failed to parse tool call {name} with args {args}: {e}")

        return None

    def _heuristic_fallback_parser(
        self,
        message: str,
        current_version: int,
        locked_regions: List[str],
        available_versions: List[int]
    ) -> List[RefinementToolCall]:
        """
        Deterministic, rule-based fallback parser that translates common witness feedback
        into tools when API is offline, mocked, or in testing.
        """
        msg = message.lower().strip()
        calls: List[RefinementToolCall] = []

        # 1. Lock command: "lock the eyes", "the eyes are right/perfect/good"
        lock_match = re.search(r"\b(lock|keep|freeze|save)\s+(the\s+)?(\w+)\b", msg)
        perfect_match = re.search(r"\b(the\s+)?(\w+)\s+(is|are|look|looks)\s+(perfect|right|spot on|good|exact)\b", msg)

        if lock_match or perfect_match:
            raw_reg = (lock_match.group(3) if lock_match else perfect_match.group(2)).lower()
            for r in RegionEnum:
                if r.value == raw_reg or r.value.startswith(raw_reg) or raw_reg.startswith(r.value):
                    calls.append(LockFeatureCall(region=r))
                    return calls

        # 2. Unlock command
        unlock_match = re.search(r"\b(unlock|unfreeze)\s+(the\s+)?(\w+)\b", msg)
        if unlock_match:
            raw_reg = unlock_match.group(3).lower()
            for r in RegionEnum:
                if r.value == raw_reg or r.value.startswith(raw_reg):
                    calls.append(UnlockFeatureCall(region=r))
                    return calls

        # 3. Revert command: "go back to version 2", "revert nose to v1"
        revert_match = re.search(r"\b(revert|go back|restore)\s+(?:the\s+)?(\w+)?\s*(?:to\s+)?(?:version\s+|v)?(\d+)\b", msg)
        if revert_match:
            target_reg = revert_match.group(2)
            ver_num = int(revert_match.group(3))
            reg_val = None
            if target_reg:
                for r in RegionEnum:
                    if r.value == target_reg.lower():
                        reg_val = r.value
                        break
            calls.append(RevertCall(region=reg_val, to_version=ver_num))
            return calls

        # 4. Compare command: "compare v1 and v2"
        compare_match = re.search(r"\bcompare\s+(?:version\s+|v)?(\d+)\s+(?:and|with|to)\s+(?:version\s+|v)?(\d+)\b", msg)
        if compare_match:
            v_a = int(compare_match.group(1))
            v_b = int(compare_match.group(2))
            calls.append(CompareVersionsCall(a=v_a, b=v_b))
            return calls

        # 5. Global adjustments: age, expression, weight
        age_match = re.search(r"\b(make\s+(?:him|her|them)?\s*)?(\d+)\s*years?\s*(older|younger)\b", msg)
        if age_match:
            delta = int(age_match.group(2)) * (1 if age_match.group(3) == "older" else -1)
            calls.append(AdjustGlobalCall(age_delta_years=delta))
            return calls

        # 6. Specific region edit: "make the nose wider", "darker eyes", "longer hair"
        for r in RegionEnum:
            if r.value in msg:
                # Check if locked
                if r.value in locked_regions:
                    calls.append(AskWitnessCall(
                        question=f"The {r.value} is currently locked. Would you like to unlock it to make changes?"
                    ))
                    return calls

                # Magnitude
                mag = MagnitudeEnum.MODERATE
                if "slightly" in msg or "a bit" in msg or "subtle" in msg or "slight" in msg:
                    mag = MagnitudeEnum.SLIGHT
                elif "much" in msg or "significantly" in msg or "very" in msg or "strong" in msg:
                    mag = MagnitudeEnum.STRONG

                # Extract attribute & direction
                direction = "modified"
                attribute = "shape"
                if "wider" in msg:
                    direction = "wider"; attribute = "width"
                elif "narrower" in msg or "thinner" in msg:
                    direction = "narrower"; attribute = "width"
                elif "longer" in msg:
                    direction = "longer"; attribute = "length"
                elif "shorter" in msg:
                    direction = "shorter"; attribute = "length"
                elif "darker" in msg:
                    direction = "darker"; attribute = "color"
                elif "lighter" in msg:
                    direction = "lighter"; attribute = "color"
                elif "hooked" in msg or "aquiline" in msg:
                    direction = "hooked"; attribute = "bridge"
                elif "straight" in msg or "straighter" in msg:
                    direction = "straight"; attribute = "bridge"
                else:
                    direction = msg.replace(r.value, "").strip()

                calls.append(EditRegionCall(
                    region=r,
                    attribute=attribute,
                    direction_or_value=direction,
                    magnitude=mag
                ))
                return calls

        # 7. Ambiguous input: Ask an open-ended question
        calls.append(AskWitnessCall(
            question="Can you tell me more about what specific feature you'd like to adjust?"
        ))
        return calls


# Singleton instance
refinement_agent = RefinementAgent()
