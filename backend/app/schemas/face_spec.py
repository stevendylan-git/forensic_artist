import copy
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict


class FeatureAttribute(BaseModel):
    """
    Morphological facial feature attribute.
    Missing = None with 'unknown' semantics.
    Rule 3: Unstated features remain None/'unknown'. Never invent attributes.
    """
    model_config = ConfigDict(populate_by_name=True)

    value: Optional[str] = Field(
        default=None,
        description="Explicitly stated descriptive attribute value, or None for unknown/unstated."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Witness confidence rating between 0.0 and 1.0."
    )
    source_utterance_ids: List[str] = Field(
        default_factory=list,
        description="IDs of witness utterances providing or modifying this attribute."
    )
    verbatim: Optional[str] = Field(
        default=None,
        description="Preserved exact witness phrasing."
    )
    locked: bool = Field(
        default=False,
        description="Whether this region is locked against subsequent modifications."
    )

    @property
    def is_stated(self) -> bool:
        return self.value is not None and self.value.strip().lower() not in {"", "unknown", "null", "none"}


class EyesSpec(BaseModel):
    """Granular eyes attributes: shape, size, spacing, color."""
    model_config = ConfigDict(populate_by_name=True)

    shape: FeatureAttribute = Field(default_factory=FeatureAttribute)
    size: FeatureAttribute = Field(default_factory=FeatureAttribute)
    spacing: FeatureAttribute = Field(default_factory=FeatureAttribute)
    color: FeatureAttribute = Field(default_factory=FeatureAttribute)


class NoseSpec(BaseModel):
    """Granular nose attributes: width, length, bridge, tip."""
    model_config = ConfigDict(populate_by_name=True)

    width: FeatureAttribute = Field(default_factory=FeatureAttribute)
    length: FeatureAttribute = Field(default_factory=FeatureAttribute)
    bridge: FeatureAttribute = Field(default_factory=FeatureAttribute)
    tip: FeatureAttribute = Field(default_factory=FeatureAttribute)


class HairSpec(BaseModel):
    """Granular hair attributes: length, style, color, hairline."""
    model_config = ConfigDict(populate_by_name=True)

    length: FeatureAttribute = Field(default_factory=FeatureAttribute)
    style: FeatureAttribute = Field(default_factory=FeatureAttribute)
    color: FeatureAttribute = Field(default_factory=FeatureAttribute)
    hairline: FeatureAttribute = Field(default_factory=FeatureAttribute)


class DistinguishingMarksSpec(BaseModel):
    """Granular distinguishing marks attributes: type, location."""
    model_config = ConfigDict(populate_by_name=True)

    type: FeatureAttribute = Field(default_factory=FeatureAttribute)
    location: FeatureAttribute = Field(default_factory=FeatureAttribute)


# Patch Sub-models where all fields are Optional for partial updates
class EyesPatch(BaseModel):
    shape: Optional[FeatureAttribute] = None
    size: Optional[FeatureAttribute] = None
    spacing: Optional[FeatureAttribute] = None
    color: Optional[FeatureAttribute] = None


class NosePatch(BaseModel):
    width: Optional[FeatureAttribute] = None
    length: Optional[FeatureAttribute] = None
    bridge: Optional[FeatureAttribute] = None
    tip: Optional[FeatureAttribute] = None


class HairPatch(BaseModel):
    length: Optional[FeatureAttribute] = None
    style: Optional[FeatureAttribute] = None
    color: Optional[FeatureAttribute] = None
    hairline: Optional[FeatureAttribute] = None


class DistinguishingMarksPatch(BaseModel):
    type: Optional[FeatureAttribute] = None
    location: Optional[FeatureAttribute] = None


class FaceSpecPatch(BaseModel):
    """
    FaceSpec Update Patch extracted from a single witness utterance.
    Only explicit statements are included; unmentioned attributes are null.
    """
    model_config = ConfigDict(populate_by_name=True)

    age_range: Optional[FeatureAttribute] = None
    sex_presentation: Optional[FeatureAttribute] = None
    skin_tone: Optional[FeatureAttribute] = None
    face_shape: Optional[FeatureAttribute] = None
    forehead: Optional[FeatureAttribute] = None
    eyebrows: Optional[FeatureAttribute] = None
    eyes: Optional[EyesPatch] = None
    nose: Optional[NosePatch] = None
    mouth_lips: Optional[FeatureAttribute] = Field(default=None, alias="mouth")
    jawline: Optional[FeatureAttribute] = None
    chin: Optional[FeatureAttribute] = None
    cheekbones: Optional[FeatureAttribute] = None
    ears: Optional[FeatureAttribute] = None
    hair: Optional[HairPatch] = None
    facial_hair: Optional[FeatureAttribute] = None
    glasses_accessories: Optional[FeatureAttribute] = Field(default=None, alias="accessories")
    distinguishing_marks: Optional[DistinguishingMarksPatch] = None
    expression: Optional[FeatureAttribute] = None
    notes: Optional[FeatureAttribute] = None


class FaceSpec(BaseModel):
    """
    Forensic Face Specification: Source of Truth for the composite pipeline.
    Covers:
      - age_range, sex_presentation, skin_tone, face_shape, forehead, eyebrows
      - eyes (shape, size, spacing, color)
      - nose (width, length, bridge, tip)
      - mouth/lips, jawline, chin, cheekbones, ears
      - hair (length, style, color, hairline)
      - facial_hair, glasses/accessories, distinguishing marks (type, location), expression, notes
    Every leaf field is {value: str|None, confidence: float 0-1, source_utterance_ids: list[str]}.
    Missing = None with 'unknown' semantics.
    """
    model_config = ConfigDict(populate_by_name=True)

    age_range: FeatureAttribute = Field(default_factory=FeatureAttribute)
    sex_presentation: FeatureAttribute = Field(default_factory=FeatureAttribute)
    skin_tone: FeatureAttribute = Field(default_factory=FeatureAttribute)
    face_shape: FeatureAttribute = Field(default_factory=FeatureAttribute)
    forehead: FeatureAttribute = Field(default_factory=FeatureAttribute)
    eyebrows: FeatureAttribute = Field(default_factory=FeatureAttribute)
    eyes: EyesSpec = Field(default_factory=EyesSpec)
    nose: NoseSpec = Field(default_factory=NoseSpec)
    mouth_lips: FeatureAttribute = Field(default_factory=FeatureAttribute, alias="mouth")
    jawline: FeatureAttribute = Field(default_factory=FeatureAttribute)
    chin: FeatureAttribute = Field(default_factory=FeatureAttribute)
    cheekbones: FeatureAttribute = Field(default_factory=FeatureAttribute)
    ears: FeatureAttribute = Field(default_factory=FeatureAttribute)
    hair: HairSpec = Field(default_factory=HairSpec)
    facial_hair: FeatureAttribute = Field(default_factory=FeatureAttribute)
    glasses_accessories: FeatureAttribute = Field(default_factory=FeatureAttribute, alias="accessories")
    distinguishing_marks: DistinguishingMarksSpec = Field(default_factory=DistinguishingMarksSpec)
    expression: FeatureAttribute = Field(default_factory=FeatureAttribute)
    notes: FeatureAttribute = Field(default_factory=FeatureAttribute)

    def get_all_feature_paths(self) -> Dict[str, FeatureAttribute]:
        """Flattens the spec into a map of dot-delimited field paths -> FeatureAttribute."""
        features: Dict[str, FeatureAttribute] = {}
        for field_name, val in self:
            if isinstance(val, FeatureAttribute):
                features[field_name] = val
            elif isinstance(val, (EyesSpec, NoseSpec, HairSpec, DistinguishingMarksSpec)):
                for sub_name, sub_val in val:
                    if isinstance(sub_val, FeatureAttribute):
                        features[f"{field_name}.{sub_name}"] = sub_val
        return features

    def get_locked_regions(self) -> List[str]:
        """Returns list of field paths currently locked against modification."""
        locked = []
        for path, attr in self.get_all_feature_paths().items():
            if attr.locked:
                locked.append(path)
        return locked

    def diff(self, other: Union["FaceSpec", Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Computes differences between self (before) and other (after).
        Returns a dict of changed paths with old and new values, confidence, and utterances.
        """
        if isinstance(other, dict):
            other = FaceSpec.model_validate(other)

        my_features = self.get_all_feature_paths()
        other_features = other.get_all_feature_paths()

        differences: Dict[str, Dict[str, Any]] = {}
        all_paths = set(my_features.keys()).union(set(other_features.keys()))

        for path in sorted(all_paths):
            attr_a = my_features.get(path, FeatureAttribute())
            attr_b = other_features.get(path, FeatureAttribute())

            # Detect differences in value, confidence, or verbatim
            if (
                attr_a.value != attr_b.value
                or attr_a.confidence != attr_b.confidence
                or attr_a.verbatim != attr_b.verbatim
                or attr_a.source_utterance_ids != attr_b.source_utterance_ids
            ):
                differences[path] = {
                    "before": {
                        "value": attr_a.value,
                        "confidence": attr_a.confidence,
                        "verbatim": attr_a.verbatim,
                        "source_utterance_ids": list(attr_a.source_utterance_ids),
                    },
                    "after": {
                        "value": attr_b.value,
                        "confidence": attr_b.confidence,
                        "verbatim": attr_b.verbatim,
                        "source_utterance_ids": list(attr_b.source_utterance_ids),
                    }
                }

        return differences

    def apply_patch(
        self,
        patch: Union[FaceSpecPatch, Dict[str, Any]],
        utterance_id: Optional[str] = None
    ) -> "FaceSpec":
        """
        Applies a FaceSpecPatch non-destructively to produce a new FaceSpec.
        Rules:
        - Never overwrites locked attributes.
        - Updates value, confidence, verbatim.
        - Sets source_utterance_ids to include utterance_id.
        """
        if isinstance(patch, dict):
            patch = FaceSpecPatch.model_validate(patch)

        # Deep copy self
        updated = copy.deepcopy(self)

        def _update_attribute(target: FeatureAttribute, source: FeatureAttribute, path: str):
            if target.locked:
                return  # Skip locked attributes

            target.value = source.value
            target.confidence = source.confidence
            if source.verbatim:
                target.verbatim = source.verbatim

            # Update utterance sources
            if source.source_utterance_ids:
                target.source_utterance_ids = list(source.source_utterance_ids)
            elif utterance_id:
                target.source_utterance_ids = [utterance_id]

        for field_name, patch_val in patch:
            if patch_val is None:
                continue

            target_val = getattr(updated, field_name, None)

            if isinstance(patch_val, FeatureAttribute) and isinstance(target_val, FeatureAttribute):
                _update_attribute(target_val, patch_val, field_name)

            elif isinstance(patch_val, (EyesPatch, NosePatch, HairPatch, DistinguishingMarksPatch)):
                if target_val is not None:
                    for sub_field, sub_patch_val in patch_val:
                        if sub_patch_val is not None and isinstance(sub_patch_val, FeatureAttribute):
                            sub_target_val = getattr(target_val, sub_field, None)
                            if isinstance(sub_target_val, FeatureAttribute):
                                _update_attribute(sub_target_val, sub_patch_val, f"{field_name}.{sub_field}")

        return updated

    def to_forensic_prompt(self) -> str:
        """
        Builds a deterministic rendering prompt based solely on stated features.
        Unstated features remain unknown and are rendered neutrally.
        """
        descriptions = []
        for path, attr in self.get_all_feature_paths().items():
            if attr.is_stated:
                clean_name = path.replace(".", " ").replace("_", " ")
                descriptions.append(f"{clean_name}: {attr.value} (confidence: {int(attr.confidence * 100)}%)")

        if not descriptions:
            return "A neutral, forensic police facial composite sketch, charcoal pencil style on textured paper, front-facing mugshot angle, unidentifiable face."

        features_str = "; ".join(descriptions)
        return (
            f"Forensic police composite sketch, high-contrast charcoal pencil drawing on textured paper, "
            f"front-facing mugshot angle, neutral expression, neutral lighting. Stated suspect features: {features_str}. "
            f"Leave any unspecified features ambiguous and neutral."
        )
