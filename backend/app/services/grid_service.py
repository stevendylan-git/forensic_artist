import json
import logging
import uuid
import copy
from typing import List, Dict, Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.case import Case, CompositeSession, GeneratedImage
from app.models.audit import AuditAction, AuditOutcome
from app.audit.ledger import AuditLedger
from app.schemas.face_spec import FaceSpec
from app.schemas.case import CandidateItemResponse, CandidateGridResponse, GridSelectResponse
from app.services.image_gen import (
    FEATURE_VARIATIONS,
    apply_feature_variation,
    generate_candidates,
)
from app.services.storage import storage_service
from app.services.websocket import ws_manager

logger = logging.getLogger("grid_service")

# Extended morphological variations for "none_of_these" with greater spread
FEATURE_VARIATIONS_WIDE: Dict[str, List[str]] = {
    "whole_face": [
        "dramatically angular chiseled square facial structure with heavy jaw",
        "soft prominent circular face with very full rounded cheeks",
        "extremely elongated slender oblong facial structure with high forehead",
        "distinct heart-shaped face with wide temples and sharp pointed chin",
        "wide diamond facial structure with prominent lateral cheekbones",
        "compact broad rectangular face with strong mandibular definition",
        "asymmetric rugged athletic facial structure with strong brow",
        "tapered triangular face with narrow jaw and delicate features",
        "balanced classic symmetrical oval face with neutral proportions",
    ],
    "nose": [
        "dramatically hooked aquiline bridge with sharply drooping tip",
        "exceptionally broad flat bridge with large rounded bulbous tip",
        "severely narrow sharp upturned button nose with low bridge",
        "highly pronounced angular roman bridge with chiseled tip",
        "wide flared nostrils with very low depressed nasal bridge",
        "markedly crooked asymmetric nasal bridge with blunt offset tip",
        "delicate straight slender greek nose with narrow tip",
        "concave scooped snub nose with high upturned tip",
        "broad rectangular heavy bridge with fleshy tip",
    ],
    "eyes": [
        "very narrow tightly closed monolid eyes",
        "extremely wide-set large circular prominent eyes",
        "deep-set heavy-browed hooded sunken eyes",
        "sharp dramatically upturned angular cat eyes",
        "distinctly downturned sad-angled almond eyes",
        "close-set piercing narrow eyes with heavy lids",
        "prominent bulging large round eyes with thin lids",
        "asymmetric eyes with uneven eyelid fold depth",
        "standard symmetrical balanced neutral oval eyes",
    ],
    "hair": [
        "completely shaved buzz cut down to the scalp",
        "dense voluminous tightly coiled textured afro crop",
        "slicked-back straight oily dark hair",
        "wildly messy voluminous textured tousled hair",
        "severe receding widow's peak with thin side hair",
        "tight military crew cut with high skin fade",
        "straight jaw-length bob parted sharply down the middle",
        "heavy curly textured mop covering forehead",
        "neatly combed classic side part with pomade finish",
    ],
    "jawline": [
        "extremely sharp prominent square jaw with acute mandibular angle",
        "very soft rounded jawline fading smoothly into neck",
        "heavily developed massive wide mandibular jawline",
        "extremely narrow tapered sharp V-shaped jaw",
        "asymmetrical jawline with one side broader",
        "double chin with soft receding lower jawline",
        "chiseled athletic jaw with visible masseter muscles",
        "broad flat horizontal jawline",
        "classic balanced neutral curved jawline",
    ],
    "chin": [
        "dramatically forward-jutting pointed chin",
        "deeply indented cleft chin with prominent central groove",
        "severely receding weak micro-chin sloping back",
        "broad flat heavy square chin",
        "bulbous fleshy protruding round chin",
        "narrow sharp delicate pointed chin",
        "asymmetric slightly offset pointed chin",
        "dimpled soft compact round chin",
        "standard balanced neutral chin",
    ],
    "mouth": [
        "extremely thin slit-like lips with nearly absent vermilion border",
        "exceptionally full voluptuous fleshy prominent lips",
        "wide mouth with pronounced downturned corners",
        "compact narrow pursed mouth with bow-shaped upper lip",
        "asymmetrical uneven mouth line with tilted smile",
        "straight flat tightly pressed thin lips",
        "full lower lip with significantly thinner upper lip",
        "prominent cupid's bow with medium-full lips",
        "standard symmetrical balanced lips of medium thickness",
    ],
    "eyebrows": [
        "extremely thick unruly bushy dark eyebrows",
        "barely visible pencil-thin high-arched plucked eyebrows",
        "continuous unibrow joining fully across the nasal bridge",
        "sharp aggressive triangular peaked eyebrows",
        "completely flat straight horizontal low-set heavy brows",
        "steeply downward-sloping sad arched eyebrows",
        "patchy sparse uneven eyebrows",
        "sculpted clean angular arched eyebrows",
        "natural textured eyebrows of medium density",
    ],
}


def get_variations_for_feature(
    feature: str,
    action: str = "standard",
    blend_candidate_index: Optional[int] = None,
    spread: float = 1.0,
    base_val: Optional[str] = None
) -> List[str]:
    """
    Returns 9 candidate variation strings for a given feature and action mode.
    - standard: standard catalog variations.
    - none_of_these: greater morphological spread / divergent variations.
    - blend_toward: 9 fine-grained micro-variations around candidate #k.
    """
    clean_feat = feature.strip().lower()
    prefix = clean_feat.split(".")[0]

    # Mode 1: blend_toward around candidate #k
    if action == "blend_toward" and base_val:
        # Micro-variations centered tightly on base_val
        return [
            f"subtly narrower {base_val}",
            f"subtly wider {base_val}",
            f"moderately refined {base_val}",
            f"{base_val} with slightly softer contour",
            f"{base_val} with slightly more pronounced definition",
            f"{base_val} with subtle higher projection",
            f"{base_val} with subtle lower projection",
            f"classic balanced variation of {base_val}",
            f"{base_val} with delicate micro-adjustment",
        ]

    # Mode 2: none_of_these (greater spread)
    if action == "none_of_these" or spread > 1.2:
        wide_catalog = (
            FEATURE_VARIATIONS_WIDE.get(clean_feat)
            or FEATURE_VARIATIONS_WIDE.get(prefix)
        )
        if wide_catalog:
            return wide_catalog[:9]
        # Fallback amplified spread
        std_catalog = (
            FEATURE_VARIATIONS.get(clean_feat)
            or FEATURE_VARIATIONS.get(prefix)
            or [f"spread variation {i+1} of {clean_feat}" for i in range(9)]
        )
        return [f"prominently accentuated {v}" for v in std_catalog[:9]]

    # Mode 3: standard catalog
    std = (
        FEATURE_VARIATIONS.get(clean_feat)
        or FEATURE_VARIATIONS.get(prefix)
    )
    if not std:
        if clean_feat == "whole_face":
            std = [
                "balanced oval facial structure with neutral symmetrical features",
                "angular square face with prominent defined cheekbones and strong jaw",
                "soft rounded facial contour with gentle balanced proportions",
                "oblong elongated facial structure with high forehead and defined features",
                "heart-shaped face with broad upper structure and tapered jawline",
                "diamond facial architecture with wide cheekbones and refined chin",
                "athletic lean facial build with defined bone structure",
                "compact square-round face with distinct facial planes",
                "mature defined facial typology with balanced standard proportions",
            ]
        else:
            std = [f"variation {i+1} of {clean_feat}" for i in range(9)]

    return std[:9]


class GridService:
    """
    Orchestration service for 3x3 Candidate Recognition Grids (Phase 4).
    """

    @staticmethod
    async def generate_recognition_grid(
        case: Case,
        feature: str = "whole_face",
        action: str = "standard",
        blend_candidate_index: Optional[int] = None,
        spread: float = 1.0,
        confidence_threshold: float = 0.5,
        db: AsyncSession = None,  # type: ignore
    ) -> CandidateGridResponse:
        """
        Generates a 3x3 recognition grid (9 candidate variations) varying ONLY the target feature.
        When feature != 'whole_face', locality.composite guarantees the rest of the face stays identical.
        Supports 'none_of_these' (greater spread) and 'blend_toward #k' (smaller variation around #k).
        """
        clean_feature = feature.strip().lower()

        # 1. Fetch latest composite session
        stmt = (
            select(CompositeSession)
            .where(CompositeSession.case_id == case.id)
            .order_by(CompositeSession.version.desc())
        )
        res = await db.execute(stmt)
        latest_session = res.scalars().first()

        if latest_session:
            face_spec = FaceSpec.model_validate_json(latest_session.face_spec_json)
            current_version = latest_session.version
            base_image_bytes = None
            if latest_session.image_path:
                try:
                    base_image_bytes = await storage_service.retrieve_image(
                        case_id=case.id,
                        relative_storage_path=latest_session.image_path,
                        db=db
                    )
                except Exception as e:
                    logger.warning(f"Could not load base image for composite: {e}")
        else:
            face_spec = FaceSpec()
            current_version = 1
            base_image_bytes = None

        # 2. Determine base variation if blending toward candidate #k
        base_val = None
        if action == "blend_toward":
            target_idx = blend_candidate_index if blend_candidate_index is not None else 0
            # Look up candidate #k in DB
            cand_stmt = (
                select(GeneratedImage)
                .where(
                    GeneratedImage.case_id == case.id,
                    GeneratedImage.is_candidate == True
                )
                .order_by(GeneratedImage.created_at.desc())
            )
            cands_res = await db.execute(cand_stmt)
            recent_cands = cands_res.scalars().all()
            if recent_cands and 0 <= target_idx < len(recent_cands):
                base_val = recent_cands[target_idx].variation_value
            if not base_val:
                # Fallback to catalog item
                std_opts = get_variations_for_feature(clean_feature, action="standard")
                base_val = std_opts[target_idx % len(std_opts)]

        # 3. Compute 9 variation strings
        variation_strings = get_variations_for_feature(
            feature=clean_feature,
            action=action,
            blend_candidate_index=blend_candidate_index,
            spread=spread,
            base_val=base_val
        )

        # 4. Generate candidates via concurrent generation + locality compositing
        grid_result = await generate_candidates(
            face_spec=face_spec,
            n=9,
            vary=clean_feature,
            case_id=case.id,
            version=current_version,
            utterance_id=latest_session.utterance_id if latest_session else None,
            triggering_call=f"grid:{clean_feature}:{action}",
            db=db,
            confidence_threshold=confidence_threshold,
            custom_variations=variation_strings,
            base_image_bytes=base_image_bytes if clean_feature != "whole_face" else None
        )

        candidate_responses = [
            CandidateItemResponse(
                candidate_index=c.candidate_index,
                variation_feature=c.variation_feature,
                variation_value=c.variation_value,
                image_id=c.image_id,
                storage_path=c.storage_path,
                latency_ms=c.latency_ms,
                status=c.status,
                error=c.error
            )
            for c in grid_result.candidates
        ]

        return CandidateGridResponse(
            case_id=case.id,
            feature=clean_feature,
            action=action,
            spread=spread,
            total_candidates=len(candidate_responses),
            candidates=candidate_responses,
            total_latency_ms=grid_result.total_latency_ms,
            total_estimated_cost=grid_result.total_estimated_cost
        )

    @staticmethod
    async def select_candidate(
        case: Case,
        candidate_index: Optional[int],
        candidate_id: Optional[str],
        user_id: str,
        role: str,
        db: AsyncSession,
    ) -> GridSelectResponse:
        """
        Applies a candidate variation selected by the witness as a new composite version.
        Synchronizes FaceSpec, creates immutable CompositeSession, and logs to AuditLedger.
        """
        # 1. Fetch candidate image record
        candidate_record: Optional[GeneratedImage] = None
        if candidate_id:
            cand_stmt = select(GeneratedImage).where(
                GeneratedImage.id == candidate_id,
                GeneratedImage.case_id == case.id
            )
            res = await db.execute(cand_stmt)
            candidate_record = res.scalar_one_or_none()
        elif candidate_index is not None:
            cand_stmt = (
                select(GeneratedImage)
                .where(
                    GeneratedImage.case_id == case.id,
                    GeneratedImage.is_candidate == True
                )
                .order_by(GeneratedImage.created_at.desc())
            )
            res = await db.execute(cand_stmt)
            recent_cands = res.scalars().all()
            if 0 <= candidate_index < len(recent_cands):
                candidate_record = recent_cands[candidate_index]

        if not candidate_record:
            raise ValueError("Candidate image not found. Ensure candidate_id or candidate_index is valid.")

        # 2. Fetch latest session
        latest_stmt = (
            select(CompositeSession)
            .where(CompositeSession.case_id == case.id)
            .order_by(CompositeSession.version.desc())
        )
        res = await db.execute(latest_stmt)
        latest_session = res.scalars().first()

        if not latest_session:
            raise ValueError("No baseline composite session exists to apply candidate to.")

        face_spec = FaceSpec.model_validate_json(latest_session.face_spec_json)
        new_version = latest_session.version + 1

        # 3. Update FaceSpec with candidate variation
        updated_face_spec = apply_feature_variation(
            face_spec=face_spec,
            feature=candidate_record.variation_feature or "whole_face",
            variation_val=candidate_record.variation_value or "selected candidate"
        )

        # 4. Copy candidate image to a permanent version record in storage
        candidate_bytes = await storage_service.retrieve_image(
            case_id=case.id,
            relative_storage_path=candidate_record.image_path,
            db=db
        )
        new_img_id = str(uuid.uuid4())
        new_storage_path = await storage_service.store_image(
            case_id=case.id,
            image_id=new_img_id,
            image_bytes=candidate_bytes,
            db=db
        )

        # 5. Create new CompositeSession
        trigger_str = f"grid_select:{candidate_record.variation_feature}:{candidate_record.variation_value}"
        new_session = CompositeSession(
            id=str(uuid.uuid4()),
            case_id=case.id,
            version=new_version,
            face_spec_json=updated_face_spec.model_dump_json(),
            locked_regions_json=latest_session.locked_regions_json,
            image_path=new_storage_path,
            rendered_image_url=f"/api/v1/cases/{case.id}/images/{new_img_id}",
            model_id=candidate_record.model_id,
            prompt_hash=candidate_record.prompt_hash,
            triggering_call=trigger_str,
            locality_score=0.0,
            locality_unverified=False,
            utterance_id=latest_session.utterance_id
        )
        db.add(new_session)

        # 6. Create new GeneratedImage record (is_candidate=False)
        gen_img_rec = GeneratedImage(
            id=new_img_id,
            case_id=case.id,
            composite_session_id=new_session.id,
            version=new_version,
            image_path=new_storage_path,
            model_id=candidate_record.model_id,
            prompt=candidate_record.prompt,
            prompt_hash=candidate_record.prompt_hash,
            triggering_call=trigger_str,
            utterance_id=latest_session.utterance_id,
            is_candidate=False,
            variation_feature=candidate_record.variation_feature,
            variation_value=candidate_record.variation_value,
            latency_ms=candidate_record.latency_ms,
            estimated_cost=candidate_record.estimated_cost
        )
        db.add(gen_img_rec)

        # 7. Append audit log entry
        await AuditLedger.append_entry(
            session=db,
            action=AuditAction.CREATE_VERSION,
            resource=f"/api/v1/cases/{case.id}/grid/select",
            outcome=AuditOutcome.SUCCESS,
            user_id=user_id,
            role=role,
            case_id=case.id,
            details={
                "action": "grid_candidate_selected",
                "candidate_index": candidate_index,
                "candidate_id": candidate_record.id,
                "feature": candidate_record.variation_feature,
                "variation_value": candidate_record.variation_value,
                "new_version": new_version,
            }
        )
        await db.commit()

        # 8. Notify WebSocket clients
        await ws_manager.broadcast(case.id, {
            "type": "version_created",
            "case_id": case.id,
            "version": new_version,
            "triggering_call": trigger_str,
            "image_url": new_session.rendered_image_url
        })

        return GridSelectResponse(
            case_id=case.id,
            version=new_version,
            selected_candidate_index=candidate_index,
            feature=candidate_record.variation_feature or "whole_face",
            variation_value=candidate_record.variation_value or "",
            image_url=new_session.rendered_image_url,
            message=f"Applied candidate '{candidate_record.variation_value}' as Version {new_version}."
        )


grid_service = GridService()
