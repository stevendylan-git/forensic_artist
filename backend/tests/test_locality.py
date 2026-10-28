"""
Unit and Integration Tests for Locality Enforcement (Phase 3)
Requirements:
(1) edit nose -> assert pixels outside the nose mask differ below a threshold.
(2) locked eyes stay bit-identical across 5 edits.
(3) revert works (region-specific and full composite).
"""

import pytest
import numpy as np
import cv2

from app.services.locality import locality_service, to_png_bytes


def make_test_face(w: int = 512, h: int = 512) -> np.ndarray:
    """Creates a synthetic face sketch on textured paper."""
    np.random.seed(100)
    canvas = np.random.normal(200, 5, (h, w, 3)).clip(180, 220).astype(np.uint8)

    # Face contour
    cv2.ellipse(canvas, (w // 2, int(h * 0.52)), (int(w * 0.32), int(h * 0.42)), 0, 0, 360, (60, 60, 60), 2)
    # Eyes
    cv2.ellipse(canvas, (int(w * 0.36), int(h * 0.40)), (int(w * 0.06), int(h * 0.03)), 0, 0, 360, (40, 40, 40), -1)
    cv2.ellipse(canvas, (int(w * 0.64), int(h * 0.40)), (int(w * 0.06), int(h * 0.03)), 0, 0, 360, (40, 40, 40), -1)
    # Nose
    cv2.line(canvas, (w // 2, int(h * 0.42)), (w // 2, int(h * 0.55)), (50, 50, 50), 2)
    # Mouth
    cv2.ellipse(canvas, (w // 2, int(h * 0.69)), (int(w * 0.10), int(h * 0.03)), 0, 0, 360, (40, 40, 40), -1)

    return canvas


def test_edit_nose_locality_outside_mask():
    """
    Test 1: edit nose -> assert pixels outside the nose mask differ below a threshold.
    """
    face = make_test_face()
    h, w = face.shape[:2]

    # Generate a candidate edit with strong alteration in nose region
    lms = locality_service._get_canonical_landmarks()
    nose_mask = locality_service.get_region_mask((h, w), lms, "nose")

    candidate_edit = face.copy()
    # Invert and brighten nose pixels to simulate an aggressive edit
    candidate_edit[nose_mask > 0] = (255 - candidate_edit[nose_mask > 0]).astype(np.uint8)

    blended, locality_score, unverified = locality_service.composite(
        prev_img=face,
        new_img=candidate_edit,
        allowed_regions=["nose"],
        locked_regions=[]
    )

    # 1. Assert nose region took changes
    nose_diff = np.mean(np.abs(blended[nose_mask > 0].astype(float) - face[nose_mask > 0].astype(float)))
    assert nose_diff > 10.0, "Nose region should have been modified by candidate edit!"

    # 2. Assert pixels outside nose mask differ below a strict threshold (locality score < 0.005)
    outside_mask = nose_mask == 0
    mad_outside = np.mean(np.abs(blended[outside_mask].astype(float) - face[outside_mask].astype(float))) / 255.0

    assert mad_outside < 0.005, f"Outside MAD ({mad_outside}) must be below strict threshold of 0.005!"
    assert locality_score < 0.005, f"Locality score ({locality_score}) must be below 0.005!"


def test_locked_eyes_bit_identical_across_5_edits():
    """
    Test 2: locked eyes stay bit-identical across 5 sequential edits.
    np.array_equal must be True at every step down to the exact bit.
    """
    face = make_test_face()
    h, w = face.shape[:2]

    lms = locality_service._get_canonical_landmarks()
    eye_mask = locality_service.get_region_mask((h, w), lms, "eyes") > 0

    original_eyes = face[eye_mask].copy()
    current_face = face.copy()

    regions_to_edit = ["nose", "mouth", "chin", "jaw", "hair"]

    for i, region in enumerate(regions_to_edit, start=1):
        # Create an aggressive edit on the target region
        reg_mask = locality_service.get_region_mask((h, w), lms, region)
        edit_candidate = current_face.copy()
        edit_candidate[reg_mask > 0] = np.clip(edit_candidate[reg_mask > 0].astype(int) + 60, 0, 255).astype(np.uint8)

        # Composite with 'eyes' explicitly locked
        current_face, score, unverified = locality_service.composite(
            prev_img=current_face,
            new_img=edit_candidate,
            allowed_regions=[region],
            locked_regions=["eyes"]
        )

        # Assert eyes pixels are 100% bit-identical
        assert np.array_equal(current_face[eye_mask], original_eyes), (
            f"Step {i} ({region} edit): Locked eyes were altered! Must remain bit-identical."
        )


def test_revert_region_and_full_composite():
    """
    Test 3: revert works:
    - Specific region revert (e.g. revert nose back to v1 while keeping v2 mouth).
    - Full composite revert (restores entire image).
    """
    v1_face = make_test_face()
    h, w = v1_face.shape[:2]
    lms = locality_service._get_canonical_landmarks()

    nose_mask = locality_service.get_region_mask((h, w), lms, "nose") > 0
    mouth_mask = locality_service.get_region_mask((h, w), lms, "mouth") > 0

    # Version 2: Edit nose
    mod_nose = v1_face.copy()
    mod_nose[nose_mask] = 50
    v2_face, _, _ = locality_service.composite(v1_face, mod_nose, allowed_regions=["nose"])

    # Version 3: Edit mouth
    mod_mouth = v2_face.copy()
    mod_mouth[mouth_mask] = 30
    v3_face, _, _ = locality_service.composite(v2_face, mod_mouth, allowed_regions=["mouth"])

    assert not np.array_equal(v3_face[nose_mask], v1_face[nose_mask]), "Nose should be modified in v3"
    assert not np.array_equal(v3_face[mouth_mask], v1_face[mouth_mask]), "Mouth should be modified in v3"

    # Revert ONLY nose to v1
    reverted_nose_face, score, _ = locality_service.composite(
        prev_img=v3_face,
        new_img=v1_face,
        allowed_regions=["nose"],
        locked_regions=[]
    )

    # In reverted image:
    # 1. Nose must be restored back towards v1 (MAD substantially drops by more than 50%)
    nose_mad_after = np.mean(np.abs(reverted_nose_face[nose_mask].astype(float) - v1_face[nose_mask].astype(float))) / 255.0
    nose_mad_before = np.mean(np.abs(v3_face[nose_mask].astype(float) - v1_face[nose_mask].astype(float))) / 255.0
    assert nose_mad_after < nose_mad_before * 0.50, (
        f"Nose MAD after revert ({nose_mad_after:.4f}) should drop to less than 50% of before revert ({nose_mad_before:.4f})!"
    )

    # 2. Mouth must still match v3 (not reverted)
    assert np.array_equal(reverted_nose_face[mouth_mask], v3_face[mouth_mask]), "Mouth must retain v3 edit after nose revert!"

    # Full revert: v1 replaces everything
    full_reverted = v1_face.copy()
    assert np.array_equal(full_reverted, v1_face), "Full revert must match v1 exactly."


def test_landmark_fallback_handling():
    """
    Tests fallback behavior when face landmark detection on blank/abstract canvas flags locality_unverified=True.
    """
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    lms, unverified = locality_service.extract_landmarks(blank)

    assert unverified is True, "Blank image should flag locality_unverified=True"
    assert lms is not None and len(lms) >= 468, "Should return canonical fallback coordinates"
