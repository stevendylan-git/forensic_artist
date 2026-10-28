import os
import time
import copy
import uuid
import hashlib
import logging
import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from google.genai.errors import APIError

from app.config import settings
from app.schemas.face_spec import FaceSpec, FeatureAttribute
from app.schemas.gemini import GeminiImageRequest
from app.services.gemini_client import gemini_client
from app.services.storage import storage_service
from app.services.websocket import ws_manager
from app.models.case import GeneratedImage, CompositeSession
from app.models.audit import AuditAction, AuditOutcome
from app.audit.ledger import AuditLedger

logger = logging.getLogger("image_gen")


class GenerationSafetyBlockError(Exception):
    """
    Raised when an image generation request is blocked or refused by safety / content policies.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class GenerationResult(BaseModel):
    """Result of an initial composite image generation."""
    image_bytes: bytes
    image_id: str
    storage_path: Optional[str] = None
    prompt: str
    prompt_hash: str
    model_id: str
    version: int
    latency_ms: float
    estimated_cost: float
    status: str = "success"


class CandidateItem(BaseModel):
    """Result of an individual candidate variation within a recognition grid."""
    candidate_index: int
    variation_feature: str
    variation_value: str
    image_bytes: Optional[bytes] = None
    image_id: Optional[str] = None
    storage_path: Optional[str] = None
    prompt: str
    prompt_hash: str
    latency_ms: float = 0.0
    status: str  # "success", "safety_refusal", "failed"
    error: Optional[str] = None


class CandidateGridResult(BaseModel):
    """Result of generating a recognition grid of candidates."""
    case_id: Optional[str]
    feature: str
    total_requested: int
    total_successful: int
    total_failed: int
    candidates: List[CandidateItem]
    total_latency_ms: float
    total_estimated_cost: float


# Morphological Feature Variation Catalog for Candidate Recognition Grids (N=9 per feature)
FEATURE_VARIATIONS: Dict[str, List[str]] = {
    "nose": [
        "straight narrow bridge with refined tip",
        "aquiline hooked bridge with slightly drooping tip",
        "broad flat bridge with rounded bulbous tip",
        "upturned button nose with short bridge",
        "prominent roman bridge with angular defined tip",
        "wide flared nostrils with flat low bridge",
        "snub nose with slightly concave bridge",
        "crooked slightly asymmetric bridge with blunt tip",
        "classic straight greek nose of medium width",
    ],
    "nose.bridge": [
        "straight and flat nasal bridge",
        "high arched roman nasal bridge",
        "prominent hooked nasal bridge",
        "low depressed nasal bridge",
        "wide nasal bridge",
        "narrow sharp nasal bridge",
        "slightly crooked nasal bridge",
        "smooth concave nasal bridge",
        "broad rectangular nasal bridge",
    ],
    "nose.tip": [
        "upturned pointed nasal tip",
        "rounded bulbous nasal tip",
        "drooping downturned nasal tip",
        "flat blunt nasal tip",
        "defined chiseled nasal tip",
        "bifid cleft nasal tip",
        "broad fleshy nasal tip",
        "delicate narrow nasal tip",
        "neutral rounded nasal tip",
    ],
    "nose.width": [
        "very narrow slender nose",
        "narrow straight nose",
        "moderately narrow nose",
        "average standard width nose",
        "medium-wide nose",
        "broad nose with medium nostrils",
        "very wide nose with flared nostrils",
        "broad flat base nose",
        "wide triangular nose",
    ],
    "eyes": [
        "almond-shaped eyes with neutral spacing",
        "deep-set hooded eyes with heavy brow ridge",
        "round prominent wide-set eyes",
        "monolid upturned eyes",
        "narrow downturned eyes with close spacing",
        "large prominent eyes with thin eyelids",
        "heavy-lidded sleepy eyes",
        "sharp angular eyes with defined epicanthic fold",
        "standard symmetrical oval eyes",
    ],
    "eyes.shape": [
        "almond eye shape",
        "round circular eye shape",
        "hooded eye shape",
        "deep-set eye shape",
        "monolid eye shape",
        "upturned cat-eye shape",
        "downturned puppy eye shape",
        "narrow elongated eye shape",
        "prominent protruding eye shape",
    ],
    "eyes.color": [
        "dark brown irises",
        "light hazel irises",
        "deep blue irises",
        "light blue irises",
        "emerald green irises",
        "slate gray irises",
        "amber-brown irises",
        "dark nearly black irises",
        "green-gray mixed irises",
    ],
    "hair": [
        "short buzz cut hair",
        "side-parted neatly combed short hair",
        "textured wavy medium-length hair",
        "tightly curled cropped afro-textured hair",
        "slicked-back straight dark hair",
        "messy tousled crop hair",
        "receding M-shaped hairline with thin short hair",
        "crew cut with faded tapered sides",
        "straight jaw-length hair tucked behind ears",
    ],
    "hair.style": [
        "short buzz cut",
        "classic side part",
        "messy textured crop",
        "slicked back",
        "crew cut with fade",
        "curly textured mop",
        "straight layered crop",
        "undercut with short top",
        "spiky crew cut",
    ],
    "hair.color": [
        "jet black hair",
        "dark espresso brown hair",
        "medium chestnut brown hair",
        "light ash brown hair",
        "dirty blonde hair",
        "light golden blonde hair",
        "auburn reddish-brown hair",
        "salt-and-pepper graying hair",
        "silver gray hair",
    ],
    "jawline": [
        "sharp angular square jawline",
        "soft rounded gentle jawline",
        "heavy wide mandibular jawline",
        "narrow tapered V-shaped jawline",
        "chiseled defined firm jawline",
        "oval smoothly curved jawline",
        "slight double chin with soft jawline",
        "broad flat square jawline",
        "medium standard balanced jawline",
    ],
    "chin": [
        "prominent forward-projecting pointed chin",
        "cleft chin with central vertical dimple",
        "receding weak small chin",
        "broad flat square chin",
        "round soft fleshy chin",
        "angular chiseled chin",
        "dimpled rounded chin",
        "narrow sharp delicate chin",
        "balanced neutral standard chin",
    ],
    "eyebrows": [
        "thick bushy straight eyebrows",
        "thin arched sparse eyebrows",
        "angular peaked defined eyebrows",
        "straight horizontal low-set eyebrows",
        "rounded curved soft eyebrows",
        "heavy downward-sloping brows",
        "unibrow connecting slightly over nasal bridge",
        "tapered high-arch thin eyebrows",
        "medium natural textured eyebrows",
    ],
    "mouth_lips": [
        "thin upper and lower lips with straight mouth line",
        "full fleshy prominent lips",
        "wide mouth with defined cupid's bow",
        "downturned mouth corners with medium lips",
        "upturned subtle mouth corners with thin lips",
        "slightly uneven asymmetric lip line",
        "compact small narrow mouth with plump lips",
        "straight neutral closed lips of medium thickness",
        "flat pressed thin lips",
    ],
    "face_shape": [
        "oval symmetrical face shape",
        "square broad face shape with angular corners",
        "round circular face shape with full cheeks",
        "heart-shaped face with broad forehead and tapered chin",
        "oblong elongated rectangular face shape",
        "diamond face shape with prominent wide cheekbones",
        "pear-shaped face with wider lower jaw",
        "triangular tapered face shape",
        "standard symmetrical balanced face shape",
    ],
    "facial_hair": [
        "clean-shaven, smooth skin with no facial hair",
        "light 5 o'clock shadow stubble across jaw and upper lip",
        "heavy coarse stubble across entire lower face",
        "neatly trimmed short goatee on chin",
        "full dense short-trimmed beard and mustache",
        "classic neat mustache with clean-shaven cheeks",
        "chin strap beard along jawline",
        "circle beard surrounding mouth",
        "light mustache with subtle soul patch",
    ],
    "whole_face": [
        "balanced oval facial structure with neutral symmetrical features",
        "angular square face with prominent defined cheekbones and strong jaw",
        "soft rounded facial contour with gentle balanced proportions",
        "oblong elongated facial structure with high forehead and defined features",
        "heart-shaped face with broad upper structure and tapered jawline",
        "diamond facial architecture with wide cheekbones and refined chin",
        "athletic lean facial build with defined bone structure",
        "compact square-round face with distinct facial planes",
        "mature defined facial typology with balanced standard proportions",
    ],
}


def build_forensic_prompt(face_spec: FaceSpec, confidence_threshold: float = 0.5) -> str:
    """
    Builds a deterministic image generation prompt directly from the FaceSpec model.
    Rules:
    - Never uses free-form LLM hallucination.
    - Only includes attributes with confidence >= confidence_threshold and is_stated=True.
    - Explicitly describes unknown or low-confidence features neutrally ("ambiguous / unspecified").
    - Mandatory directives:
        - neutral frontal passport-style portrait
        - plain light-gray background
        - even lighting
        - 1:1 square aspect ratio
        - head-and-shoulders
        - no text, no watermarks
        - no accessories unless specified
        - "This is a synthetic, non-existent person."
    """
    def eval_attr(attr: FeatureAttribute, stated_desc: str, fallback_desc: str) -> str:
        if attr.is_stated and attr.confidence >= confidence_threshold:
            return stated_desc
        return fallback_desc

    features_list: List[str] = []

    # 1. Demographics & Presentation
    sex_desc = eval_attr(
        face_spec.sex_presentation,
        f"{face_spec.sex_presentation.value} presentation",
        "ambiguous / unspecified sex presentation"
    )
    features_list.append(sex_desc)

    age_desc = eval_attr(
        face_spec.age_range,
        f"estimated age {face_spec.age_range.value}",
        "ambiguous / unspecified age"
    )
    features_list.append(age_desc)

    skin_desc = eval_attr(
        face_spec.skin_tone,
        f"{face_spec.skin_tone.value} skin tone",
        "ambiguous / unspecified neutral skin tone"
    )
    features_list.append(skin_desc)

    # 2. Craniofacial & Bone Structure
    shape_desc = eval_attr(
        face_spec.face_shape,
        f"{face_spec.face_shape.value} face shape",
        "ambiguous / unspecified neutral face shape"
    )
    features_list.append(shape_desc)

    forehead_desc = eval_attr(
        face_spec.forehead,
        f"{face_spec.forehead.value} forehead",
        "unspecified neutral forehead"
    )
    features_list.append(forehead_desc)

    jaw_desc = eval_attr(
        face_spec.jawline,
        f"{face_spec.jawline.value} jawline",
        "unspecified neutral jawline"
    )
    features_list.append(jaw_desc)

    chin_desc = eval_attr(
        face_spec.chin,
        f"{face_spec.chin.value} chin",
        "unspecified neutral chin"
    )
    features_list.append(chin_desc)

    cheek_desc = eval_attr(
        face_spec.cheekbones,
        f"{face_spec.cheekbones.value} cheekbones",
        "unspecified neutral cheekbones"
    )
    features_list.append(cheek_desc)

    # 3. Eyebrows
    brows_desc = eval_attr(
        face_spec.eyebrows,
        f"{face_spec.eyebrows.value} eyebrows",
        "unspecified neutral eyebrows"
    )
    features_list.append(brows_desc)

    # 4. Eyes (shape, size, spacing, color)
    eyes_parts = []
    if face_spec.eyes.shape.is_stated and face_spec.eyes.shape.confidence >= confidence_threshold:
        eyes_parts.append(f"{face_spec.eyes.shape.value} shape")
    if face_spec.eyes.size.is_stated and face_spec.eyes.size.confidence >= confidence_threshold:
        eyes_parts.append(f"{face_spec.eyes.size.value} size")
    if face_spec.eyes.spacing.is_stated and face_spec.eyes.spacing.confidence >= confidence_threshold:
        eyes_parts.append(f"{face_spec.eyes.spacing.value} spacing")
    if face_spec.eyes.color.is_stated and face_spec.eyes.color.confidence >= confidence_threshold:
        eyes_parts.append(f"{face_spec.eyes.color.value} iris color")

    if eyes_parts:
        features_list.append(f"eyes with {', '.join(eyes_parts)}")
    else:
        features_list.append("ambiguous / unspecified neutral eyes with neutral iris color")

    # 5. Nose (bridge, tip, width, length)
    nose_parts = []
    if face_spec.nose.bridge.is_stated and face_spec.nose.bridge.confidence >= confidence_threshold:
        nose_parts.append(f"{face_spec.nose.bridge.value} bridge")
    if face_spec.nose.tip.is_stated and face_spec.nose.tip.confidence >= confidence_threshold:
        nose_parts.append(f"{face_spec.nose.tip.value} tip")
    if face_spec.nose.width.is_stated and face_spec.nose.width.confidence >= confidence_threshold:
        nose_parts.append(f"{face_spec.nose.width.value} width")
    if face_spec.nose.length.is_stated and face_spec.nose.length.confidence >= confidence_threshold:
        nose_parts.append(f"{face_spec.nose.length.value} length")

    if nose_parts:
        features_list.append(f"nose with {', '.join(nose_parts)}")
    else:
        features_list.append("ambiguous / unspecified neutral nose")

    # 6. Mouth & Lips
    mouth_desc = eval_attr(
        face_spec.mouth_lips,
        f"{face_spec.mouth_lips.value} mouth and lips",
        "ambiguous / unspecified neutral mouth with closed lips"
    )
    features_list.append(mouth_desc)

    # 7. Ears
    ears_desc = eval_attr(
        face_spec.ears,
        f"{face_spec.ears.value} ears",
        "unspecified neutral ears"
    )
    features_list.append(ears_desc)

    # 8. Hair & Facial Hair
    hair_parts = []
    if face_spec.hair.style.is_stated and face_spec.hair.style.confidence >= confidence_threshold:
        hair_parts.append(f"{face_spec.hair.style.value} style")
    if face_spec.hair.length.is_stated and face_spec.hair.length.confidence >= confidence_threshold:
        hair_parts.append(f"{face_spec.hair.length.value} length")
    if face_spec.hair.color.is_stated and face_spec.hair.color.confidence >= confidence_threshold:
        hair_parts.append(f"{face_spec.hair.color.value} color")
    if face_spec.hair.hairline.is_stated and face_spec.hair.hairline.confidence >= confidence_threshold:
        hair_parts.append(f"{face_spec.hair.hairline.value} hairline")

    if hair_parts:
        features_list.append(f"hair with {', '.join(hair_parts)}")
    else:
        features_list.append("ambiguous / unspecified neutral hair")

    facial_hair_desc = eval_attr(
        face_spec.facial_hair,
        f"facial hair: {face_spec.facial_hair.value}",
        "clean-shaven, no facial hair"
    )
    features_list.append(facial_hair_desc)

    # 9. Distinguishing Marks
    marks_parts = []
    if face_spec.distinguishing_marks.type.is_stated and face_spec.distinguishing_marks.type.confidence >= confidence_threshold:
        marks_parts.append(face_spec.distinguishing_marks.type.value)
    if face_spec.distinguishing_marks.location.is_stated and face_spec.distinguishing_marks.location.confidence >= confidence_threshold:
        marks_parts.append(f"at {face_spec.distinguishing_marks.location.value}")
    if marks_parts:
        features_list.append(f"distinguishing mark: {' '.join(marks_parts)}")
    else:
        features_list.append("no distinguishing marks, no scars, no tattoos")

    # 10. Accessories
    if face_spec.glasses_accessories.is_stated and face_spec.glasses_accessories.confidence >= confidence_threshold:
        features_list.append(f"accessories: {face_spec.glasses_accessories.value}")
    else:
        features_list.append("no glasses, no accessories, no jewelry")

    # 11. Expression
    expr_desc = eval_attr(
        face_spec.expression,
        f"facial expression: {face_spec.expression.value}",
        "neutral expression, relaxed mouth closed, looking straight ahead"
    )
    features_list.append(expr_desc)

    descriptors_str = "; ".join(features_list)

    # Assemble complete deterministic forensic prompt
    prompt = (
        f"A neutral frontal passport-style portrait of a synthetic, non-existent person. "
        f"Head-and-shoulders composition, 1:1 aspect ratio, centered in frame, facing directly forward. "
        f"Plain light-gray background, even diffused studio lighting with no harsh cast shadows. "
        f"Subject physical characteristics: {descriptors_str}. "
        f"Completely neutral facial expression, closed lips, direct eye contact with camera. "
        f"No text, no labels, no watermarks, no borders, no accessories unless specified. "
        f"This is a synthetic, non-existent person."
    )
    return prompt


def apply_feature_variation(face_spec: FaceSpec, feature: str, variation_val: str) -> FaceSpec:
    """
    Non-destructively clones face_spec and sets the specified feature to variation_val with confidence=1.0.
    Holds all other features strictly constant.
    """
    updated = copy.deepcopy(face_spec)
    parts = feature.split(".")

    if len(parts) == 1:
        target_obj = getattr(updated, parts[0], None)
        if isinstance(target_obj, FeatureAttribute):
            target_obj.value = variation_val
            target_obj.confidence = 1.0
        elif parts[0] == "nose":
            updated.nose.bridge.value = variation_val
            updated.nose.bridge.confidence = 1.0
        elif parts[0] == "eyes":
            updated.eyes.shape.value = variation_val
            updated.eyes.shape.confidence = 1.0
        elif parts[0] == "hair":
            updated.hair.style.value = variation_val
            updated.hair.style.confidence = 1.0
        elif parts[0] in ("mouth", "mouth_lips"):
            updated.mouth_lips.value = variation_val
            updated.mouth_lips.confidence = 1.0
        elif parts[0] == "whole_face":
            updated.face_shape.value = variation_val
            updated.face_shape.confidence = 1.0
    elif len(parts) == 2:
        parent_obj = getattr(updated, parts[0], None)
        if parent_obj is not None:
            child_attr = getattr(parent_obj, parts[1], None)
            if isinstance(child_attr, FeatureAttribute):
                child_attr.value = variation_val
                child_attr.confidence = 1.0

    return updated


async def generate_initial(
    face_spec: FaceSpec,
    case_id: Optional[str] = None,
    version: int = 1,
    utterance_id: Optional[str] = None,
    triggering_call: str = "generate_initial",
    db: Optional[AsyncSession] = None,
    confidence_threshold: float = 0.5,
    progress_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
) -> GenerationResult:
    """
    Generates an initial composite portrait image deterministically from FaceSpec.
    1. Builds deterministic prompt from FaceSpec (unknowns described neutrally).
    2. Streams WebSocket progress: queued -> generating -> done/failed.
    3. Executes Gemini image generation with safety block handling.
    4. Envelope encrypts image (AES-256-GCM with per-case DEK) and stores in object store.
    5. Creates an immutable version row (GeneratedImage & CompositeSession linking).
    6. Logs latency and cost.
    """
    image_id = str(uuid.uuid4())
    model_id = settings.GEMINI_IMAGE_MODEL

    # Stream 'queued' progress
    if progress_callback:
        await progress_callback({
            "status": "queued",
            "stage": "queued",
            "case_id": case_id,
            "version": version,
            "message": "Queued for generation"
        })
    elif case_id:
        await ws_manager.broadcast(case_id, {
            "type": "progress",
            "status": "queued",
            "stage": "queued",
            "case_id": case_id,
            "version": version,
            "message": "Queued for generation"
        })

    # Build deterministic prompt
    prompt = build_forensic_prompt(face_spec, confidence_threshold=confidence_threshold)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # Stream 'generating' progress
    if progress_callback:
        await progress_callback({
            "status": "generating",
            "stage": "generating",
            "case_id": case_id,
            "version": version,
            "message": "Rendering composite portrait via Gemini Imagen..."
        })
    elif case_id:
        await ws_manager.broadcast(case_id, {
            "type": "progress",
            "status": "generating",
            "stage": "generating",
            "case_id": case_id,
            "version": version,
            "message": "Rendering composite portrait via Gemini Imagen..."
        })

    start_perf = time.perf_counter()
    image_bytes: Optional[bytes] = None

    try:
        request = GeminiImageRequest(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            person_generation="ALLOW_ADULT"
        )
        response = await gemini_client.generate_image(request)

        if not response.image_bytes_list:
            raise GenerationSafetyBlockError(
                "Image generation produced 0 images. Request may have been blocked by safety filters.",
                details={"model_id": model_id, "prompt_hash": prompt_hash}
            )
        image_bytes = response.image_bytes_list[0]

    except APIError as api_err:
        err_msg = str(api_err)
        logger.error(f"Gemini API error during image generation [model={model_id}]: {err_msg}")
        is_safety = any(s in err_msg.lower() for s in ["safety", "policy", "blocked", "filter", "prohibited", "refusal"])
        if progress_callback:
            await progress_callback({"status": "failed", "stage": "failed", "case_id": case_id, "error": err_msg})
        elif case_id:
            await ws_manager.broadcast(case_id, {"type": "progress", "status": "failed", "stage": "failed", "case_id": case_id, "error": err_msg})

        if is_safety:
            raise GenerationSafetyBlockError(
                f"Generation declined by content safety policy: {err_msg}",
                details={"model_id": model_id, "prompt_hash": prompt_hash, "raw_error": err_msg}
            )
        raise

    except Exception as exc:
        if isinstance(exc, GenerationSafetyBlockError):
            if progress_callback:
                await progress_callback({"status": "failed", "stage": "failed", "case_id": case_id, "error": str(exc)})
            elif case_id:
                await ws_manager.broadcast(case_id, {"type": "progress", "status": "failed", "stage": "failed", "case_id": case_id, "error": str(exc)})
            raise
        logger.error(f"Unexpected image generation failure [model={model_id}]: {str(exc)}")
        if progress_callback:
            await progress_callback({"status": "failed", "stage": "failed", "case_id": case_id, "error": str(exc)})
        elif case_id:
            await ws_manager.broadcast(case_id, {"type": "progress", "status": "failed", "stage": "failed", "case_id": case_id, "error": str(exc)})
        raise

    latency_ms = (time.perf_counter() - start_perf) * 1000.0
    estimated_cost = settings.IMAGEN_COST_PER_IMAGE

    logger.info(
        f"Generated initial image successfully in {latency_ms:.2f}ms. "
        f"Estimated cost: ${estimated_cost:.4f}. Model: {model_id}. Prompt hash: {prompt_hash[:12]}"
    )

    storage_path = None
    if case_id and db:
        # 1. Envelope Encrypt & Store
        storage_path = await storage_service.store_image(
            case_id=case_id,
            image_id=image_id,
            image_bytes=image_bytes,
            db=db
        )

        # 2. Check if a CompositeSession row already exists for this case & version
        stmt = select(CompositeSession).where(
            CompositeSession.case_id == case_id,
            CompositeSession.version == version
        )
        res = await db.execute(stmt)
        session_row = res.scalar_one_or_none()
        composite_session_id = session_row.id if session_row else None

        # 3. Create immutable GeneratedImage version linking row
        img_record = GeneratedImage(
            id=image_id,
            case_id=case_id,
            composite_session_id=composite_session_id,
            version=version,
            image_path=storage_path,
            model_id=model_id,
            prompt=prompt,
            prompt_hash=prompt_hash,
            triggering_call=triggering_call,
            utterance_id=utterance_id,
            is_candidate=False,
            latency_ms=round(latency_ms, 2),
            estimated_cost=round(estimated_cost, 4)
        )
        db.add(img_record)
        await db.flush()

    # Stream 'done' progress
    if progress_callback:
        await progress_callback({
            "status": "done",
            "stage": "done",
            "case_id": case_id,
            "version": version,
            "image_id": image_id,
            "storage_path": storage_path,
            "latency_ms": round(latency_ms, 2),
            "estimated_cost": round(estimated_cost, 4),
            "message": "Image generated and encrypted successfully."
        })
    elif case_id:
        await ws_manager.broadcast(case_id, {
            "type": "progress",
            "status": "done",
            "stage": "done",
            "case_id": case_id,
            "version": version,
            "image_id": image_id,
            "storage_path": storage_path,
            "latency_ms": round(latency_ms, 2),
            "estimated_cost": round(estimated_cost, 4),
            "message": "Image generated and encrypted successfully."
        })

    return GenerationResult(
        image_bytes=image_bytes,
        image_id=image_id,
        storage_path=storage_path,
        prompt=prompt,
        prompt_hash=prompt_hash,
        model_id=model_id,
        version=version,
        latency_ms=round(latency_ms, 2),
        estimated_cost=round(estimated_cost, 4),
        status="success"
    )


async def generate_candidates(
    face_spec: FaceSpec,
    n: int = 9,
    vary: str = "nose",
    case_id: Optional[str] = None,
    version: int = 1,
    utterance_id: Optional[str] = None,
    triggering_call: str = "generate_candidates",
    db: Optional[AsyncSession] = None,
    confidence_threshold: float = 0.5,
    progress_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    custom_variations: Optional[List[str]] = None,
    base_image_bytes: Optional[bytes] = None,
) -> CandidateGridResult:
    """
    Generates N candidate composite variations (default N=9 for recognition grid),
    varying ONLY the requested feature while holding all others constant as specified.
    Concurrency bounded by asyncio.Semaphore. Gracefully handles partial failures.
    When base_image_bytes is provided and vary != 'whole_face', composites candidates so the
    rest of the face stays strictly identical.
    """
    clean_vary = vary.strip().lower()

    if custom_variations:
        selected_variations = custom_variations[:n]
    else:
        # Look up variation values
        variation_options = FEATURE_VARIATIONS.get(clean_vary)

        if not variation_options:
            # Fallback: check prefix (e.g. "nose.bridge" -> "nose")
            prefix = clean_vary.split(".")[0]
            variation_options = FEATURE_VARIATIONS.get(prefix)

        if not variation_options:
            # Generic variations if unlisted
            variation_options = [
                f"variation {i+1} of {clean_vary}" for i in range(n)
            ]

        # Select n variations
        selected_variations = variation_options[:n]

    if len(selected_variations) < n:
        # Pad if needed
        for i in range(len(selected_variations), n):
            selected_variations.append(f"alternative variation {i+1} of {clean_vary}")
            selected_variations.append(f"alternative variation {i+1} of {clean_vary}")

    # Stream 'queued' progress
    if progress_callback:
        await progress_callback({
            "status": "queued",
            "stage": "queued",
            "case_id": case_id,
            "feature": clean_vary,
            "total_requested": n,
            "message": f"Queued {n} candidate variations for feature '{clean_vary}'"
        })
    elif case_id:
        await ws_manager.broadcast(case_id, {
            "type": "progress",
            "status": "queued",
            "stage": "queued",
            "case_id": case_id,
            "feature": clean_vary,
            "total_requested": n,
            "message": f"Queued {n} candidate variations for feature '{clean_vary}'"
        })

    semaphore = asyncio.Semaphore(settings.IMAGE_GEN_CONCURRENCY)
    model_id = settings.GEMINI_IMAGE_MODEL
    overall_start = time.perf_counter()

    # Pre-fetch / generate case DEK once so concurrent workers do not race DB
    if case_id and db:
        from app.services.storage import get_or_create_case_dek
        await get_or_create_case_dek(case_id, db)

    completed_candidates: List[CandidateItem] = [None] * n  # type: ignore
    pending_db_records: List[GeneratedImage] = []
    completed_count = 0

    async def _render_single_candidate(idx: int, variation_val: str):
        nonlocal completed_count
        variant_spec = apply_feature_variation(face_spec, clean_vary, variation_val)
        prompt = build_forensic_prompt(variant_spec, confidence_threshold=confidence_threshold)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        candidate_id = str(uuid.uuid4())

        async with semaphore:
            req_start = time.perf_counter()
            try:
                request = GeminiImageRequest(
                    prompt=prompt,
                    number_of_images=1,
                    aspect_ratio="1:1",
                    person_generation="ALLOW_ADULT"
                )
                response = await gemini_client.generate_image(request)

                if not response.image_bytes_list:
                    raise GenerationSafetyBlockError(
                        f"Candidate {idx+1} produced 0 images. Blocked by safety filter.",
                        details={"model_id": model_id, "prompt_hash": prompt_hash}
                    )

                img_bytes = response.image_bytes_list[0]

                # If base image provided and not varying whole face, composite so the rest of the face stays identical
                if base_image_bytes and clean_vary != "whole_face":
                    try:
                        from app.services.locality import locality_service, to_png_bytes
                        composited_arr, loc_score, unverified = locality_service.composite(
                            prev_img=base_image_bytes,
                            new_img=img_bytes,
                            allowed_regions=[clean_vary]
                        )
                        img_bytes = to_png_bytes(composited_arr)
                    except Exception as comp_err:
                        logger.warning(f"Locality compositing fallback for candidate {idx+1}: {comp_err}")

                req_latency = (time.perf_counter() - req_start) * 1000.0

                storage_path = None
                if case_id:
                    storage_path = await storage_service.store_image(
                        case_id=case_id,
                        image_id=candidate_id,
                        image_bytes=img_bytes,
                        db=db
                    )
                    if db:
                        record = GeneratedImage(
                            id=candidate_id,
                            case_id=case_id,
                            version=version,
                            image_path=storage_path,
                            model_id=model_id,
                            prompt=prompt,
                            prompt_hash=prompt_hash,
                            triggering_call=triggering_call,
                            utterance_id=utterance_id,
                            is_candidate=True,
                            variation_feature=clean_vary,
                            variation_value=variation_val,
                            latency_ms=round(req_latency, 2),
                            estimated_cost=round(settings.IMAGEN_COST_PER_IMAGE, 4)
                        )
                        pending_db_records.append(record)

                item = CandidateItem(
                    candidate_index=idx,
                    variation_feature=clean_vary,
                    variation_value=variation_val,
                    image_bytes=img_bytes,
                    image_id=candidate_id,
                    storage_path=storage_path,
                    prompt=prompt,
                    prompt_hash=prompt_hash,
                    latency_ms=round(req_latency, 2),
                    status="success"
                )
                completed_candidates[idx] = item

            except Exception as e:
                req_latency = (time.perf_counter() - req_start) * 1000.0
                err_text = str(e)
                is_safety = (
                    isinstance(e, GenerationSafetyBlockError)
                    or any(s in err_text.lower() for s in ["safety", "policy", "blocked", "filter", "prohibited", "refusal"])
                )
                status_code = "safety_refusal" if is_safety else "failed"
                logger.warning(
                    f"Candidate variation {idx+1} ({variation_val}) failed [{status_code}]: {err_text}"
                )
                item = CandidateItem(
                    candidate_index=idx,
                    variation_feature=clean_vary,
                    variation_value=variation_val,
                    prompt=prompt,
                    prompt_hash=prompt_hash,
                    latency_ms=round(req_latency, 2),
                    status=status_code,
                    error=err_text
                )
                completed_candidates[idx] = item

            completed_count += 1
            # Stream partial progress
            prog_payload = {
                "type": "progress",
                "status": "generating",
                "stage": "generating",
                "case_id": case_id,
                "feature": clean_vary,
                "current": completed_count,
                "total": n,
                "message": f"Rendered candidate {completed_count}/{n} ({clean_vary}: {variation_val})"
            }
            if progress_callback:
                await progress_callback(prog_payload)
            elif case_id:
                await ws_manager.broadcast(case_id, prog_payload)

    # Launch all candidate generation tasks concurrently with gather
    tasks = [
        _render_single_candidate(i, val)
        for i, val in enumerate(selected_variations)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Flush all completed DB records safely after concurrency terminates
    if case_id and db and pending_db_records:
        for rec in pending_db_records:
            db.add(rec)
        await db.flush()

    total_latency_ms = (time.perf_counter() - overall_start) * 1000.0
    successful_candidates = [c for c in completed_candidates if c and c.status == "success"]
    failed_candidates = [c for c in completed_candidates if c and c.status != "success"]
    total_estimated_cost = len(successful_candidates) * settings.IMAGEN_COST_PER_IMAGE

    logger.info(
        f"Completed candidate grid for '{clean_vary}' ({len(successful_candidates)}/{n} successful) "
        f"in {total_latency_ms:.2f}ms. Cost: ${total_estimated_cost:.4f}"
    )

    # Stream 'done' progress
    done_payload = {
        "type": "progress",
        "status": "done",
        "stage": "done",
        "case_id": case_id,
        "feature": clean_vary,
        "total_requested": n,
        "total_successful": len(successful_candidates),
        "total_failed": len(failed_candidates),
        "total_latency_ms": round(total_latency_ms, 2),
        "total_estimated_cost": round(total_estimated_cost, 4),
        "message": f"Candidate grid completed: {len(successful_candidates)}/{n} successful."
    }
    if progress_callback:
        await progress_callback(done_payload)
    elif case_id:
        await ws_manager.broadcast(case_id, done_payload)

    return CandidateGridResult(
        case_id=case_id,
        feature=clean_vary,
        total_requested=n,
        total_successful=len(successful_candidates),
        total_failed=len(failed_candidates),
        candidates=completed_candidates,
        total_latency_ms=round(total_latency_ms, 2),
        total_estimated_cost=round(total_estimated_cost, 4)
    )
