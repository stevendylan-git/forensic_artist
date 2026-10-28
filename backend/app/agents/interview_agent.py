"""
Forensic Elicitation Agent & Cognitive Interview System Prompt
Rule 7: Agents must never lead the witness.
Rule 8: Never generate faces of named real people.
"""
from typing import Optional
import re

COGNITIVE_INTERVIEW_SYSTEM_PROMPT = """You are a certified forensic sketch interviewer conducting a cognitive interview with an eyewitness.
Your goal is to gather accurate details about a suspect's facial features without contaminating the witness's memory.

NON-NEGOTIABLE PROTOCOLS:
1. NEVER ASK LEADING QUESTIONS. Do not introduce features the witness hasn't mentioned (e.g., NEVER ask "Did he have a large nose?" or "Were his eyes blue?").
2. ALWAYS USE OPEN-ENDED ELICITATION (e.g., "Tell me about the suspect's nose shape", "What do you recall about the hair?").
3. DO NOT INVENT DETAILS. If the witness cannot recall a feature, reassure them that it is completely fine and preserve the feature as unknown.
4. BE RESPECTFUL, PATIENT, AND FORENSICALLY NEUTRAL.
"""

# Blocklist of high-profile known real individuals (Rule 8)
CELEBRITY_NAMES_PATTERN = re.compile(
    r"\b(elon musk|trump|biden|obama|brad pitt|tom cruise|leonardo dicaprio|keanu reeves|"
    r"taylor swift|beyonce|rihanna|modi|putin|zelenskyy|gandhi|einstein)\b",
    re.IGNORECASE
)


def check_for_named_real_people(text: str) -> Optional[str]:
    """
    Rule 8: Checks if the witness or prompt attempts to generate a named real person.
    Returns matched name if found, else None.
    """
    match = CELEBRITY_NAMES_PATTERN.search(text)
    if match:
        return match.group(0)
    return None
