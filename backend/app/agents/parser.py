import re
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Set

from app.schemas.face_spec import FaceSpec, FaceSpecPatch, FeatureAttribute
from app.schemas.gemini import GeminiTextRequest
from app.services.gemini_client import gemini_client

logger = logging.getLogger("parser_agent")

PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "parser.md"

CELEBRITY_NAMES: Set[str] = {
    "brad pitt", "tom cruise", "taylor swift", "leonardo dicaprio",
    "keanu reeves", "angelina jolie", "johnny depp", "barack obama",
    "donald trump", "morgan freeman", "george clooney", "will smith",
    "beyonce", "dwayne johnson", "elon musk", "ryan gosling",
    "chris hemsworth", "scarlett johansson", "robert downey", "robert downey jr",
    "bill gates", "kanye west", "kim kardashian", "jennifer aniston",
    "matt damon", "ben affleck", "al pacino", "robert de niro", "harrison ford"
}


class AdversarialCelebrityError(ValueError):
    """Raised when witness or input attempts to compare or anchor to a celebrity or famous person."""
    pass


def detect_celebrity_reference(text: str) -> Optional[str]:
    """
    Detects adversarial celebrity references or comparisons in witness statements.
    Forensic science standards strictly prohibit celebrity-anchored composite generation.
    """
    clean_text = text.lower()

    # Direct name match
    for celeb in CELEBRITY_NAMES:
        pattern = r"\b" + re.escape(celeb) + r"\b"
        if re.search(pattern, clean_text):
            return celeb

    # Structural comparison match: e.g. "make him look like <Name>", "looks like <Name>"
    sim_patterns = [
        r"(?:make\s+(?:him|her|them)?\s*look\s+like|looks?\s+(?:just\s+)?like|resembles?)\s+([a-zA-Z\s'-]+)",
        r"(?:famous\s+(?:actor|celebrity|person|singer|star)|celebrity)\s+([a-zA-Z\s'-]+)?"
    ]
    for pat in sim_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            captured = match.group(1).strip() if match.group(1) else "famous person"
            # If captured text contains common names or trigger words
            if any(c in captured.lower() for c in CELEBRITY_NAMES) or "celebrity" in text.lower() or "famous" in text.lower():
                return captured

    return None


def load_system_prompt() -> str:
    """Loads system prompt from app/agents/prompts/parser.md."""
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8").strip()
    return (
        "You extract facial descriptions from a crime witness into a structured patch. Rules: "
        "(a) record ONLY what the witness explicitly stated; leave everything else null; "
        "(b) never infer ethnicity, skin tone, age or gender unless stated; "
        "(c) convert hedging language into confidence between 0 and 1 (certain=0.9, 'I think'=0.6, 'maybe'=0.4, 'not sure'=0.25); "
        "(d) if the witness corrects an earlier statement, overwrite that field and keep the new utterance id as the source; "
        "(e) preserve the witness's own wording in a 'verbatim' field; (f) output must validate against the schema."
    )


class FaceSpecParserAgent:
    """
    Natural Language Parser Agent (Phase 2a).
    Translates witness utterances into structured FaceSpecPatch updates.
    """

    def __init__(self):
        self.system_prompt = load_system_prompt()

    async def parse_utterance(
        self,
        utterance: str,
        utterance_id: str,
        current_spec: Optional[FaceSpec] = None
    ) -> FaceSpecPatch:
        """
        Parses witness utterance into a FaceSpecPatch.
        1. Checks for adversarial celebrity references.
        2. Constructs contextual prompt including current FaceSpec state for corrections/negations.
        3. Calls Gemini text model with structured response_schema.
        4. Injects utterance_id as source and validates patch.
        """
        # 1. Adversarial celebrity check
        celebrity = detect_celebrity_reference(utterance)
        if celebrity:
            logger.warning(f"Adversarial celebrity reference rejected in utterance '{utterance_id}': {celebrity}")
            raise AdversarialCelebrityError(
                f"Adversarial description detected: reference to celebrity or famous person '{celebrity}' is strictly prohibited in forensic facial composites."
            )

        # 2. Build contextual prompt
        context_parts = [
            f"Witness Utterance ID: {utterance_id}",
            f"Witness Statement: \"{utterance}\""
        ]

        if current_spec is not None:
            # Provide current active stated features so corrections ('no, narrower than that') have context
            stated_features = {}
            for path, attr in current_spec.get_all_feature_paths().items():
                if attr.is_stated:
                    stated_features[path] = {
                        "value": attr.value,
                        "confidence": attr.confidence,
                        "verbatim": attr.verbatim
                    }
            if stated_features:
                context_parts.append(
                    f"Current Composite Spec (Active Features): {json.dumps(stated_features, indent=2)}"
                )

        user_prompt = (
            "\n\n".join(context_parts) + "\n\n"
            "Extract a JSON patch for this utterance following the rules strictly. "
            "For each stated feature, set value, confidence (0.0 to 1.0 reflecting hedging/certainty), "
            f"verbatim (exact witness words), and set source_utterance_ids to [\"{utterance_id}\"]. "
            "Leave all unmentioned features null. Do not infer unstated attributes."
        )

        request = GeminiTextRequest(
            prompt=user_prompt,
            system_instruction=self.system_prompt,
            temperature=0.0,
            response_schema=FaceSpecPatch
        )

        # 3. Call Gemini
        response = await gemini_client.generate_text(request)
        content = response.content.strip()

        # 4. Parse JSON into FaceSpecPatch
        try:
            patch_data = json.loads(content)
            patch = FaceSpecPatch.model_validate(patch_data)
        except Exception as e:
            logger.error(f"Failed to parse Gemini output as FaceSpecPatch: {e}, raw: {content}")
            raise ValueError(f"Parser failed to produce valid FaceSpecPatch: {str(e)}")

        # 5. Guarantee source_utterance_ids are stamped with current utterance_id
        self._ensure_utterance_ids(patch, utterance_id)
        return patch

    def _ensure_utterance_ids(self, patch: FaceSpecPatch, utterance_id: str):
        """Recursively ensures any populated FeatureAttribute includes the current utterance_id."""
        def _stamp(attr: Optional[FeatureAttribute]):
            if attr is not None and attr.value is not None:
                if not attr.source_utterance_ids:
                    attr.source_utterance_ids = [utterance_id]
                elif utterance_id not in attr.source_utterance_ids:
                    attr.source_utterance_ids.append(utterance_id)

        for _, patch_val in patch:
            if isinstance(patch_val, FeatureAttribute):
                _stamp(patch_val)
            elif patch_val is not None:
                for _, sub_val in patch_val:
                    if isinstance(sub_val, FeatureAttribute):
                        _stamp(sub_val)


parser_agent = FaceSpecParserAgent()
