"""
Locality Enforcement Benchmark Script (Phase 3)
Evaluates and benchmarks the deterministic locality engine:
1. Locality Score (Mean Absolute Difference outside allowed mask) across facial features.
2. Locked region bit-exact preservation across sequential edits.
3. Latency of landmark detection, affine alignment, and feathered compositing.
4. Generates an evidentiary benchmark table for demonstrations.
"""

import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import numpy as np
import cv2
from app.services.locality import locality_service, to_png_bytes


def create_synthetic_forensic_sketch(w: int = 512, h: int = 512) -> np.ndarray:
    """
    Creates a realistic synthetic forensic composite base sketch
    with face oval, eyes, nose, mouth, eyebrows, jaw, and paper texture.
    """
    # Textured light-gray background
    np.random.seed(42)
    noise = np.random.normal(210, 8, (h, w, 3)).clip(180, 240).astype(np.uint8)

    # Face contour oval
    cv2.ellipse(noise, (w // 2, int(h * 0.52)), (int(w * 0.32), int(h * 0.42)), 0, 0, 360, (70, 70, 70), 2)

    # Eyes
    left_eye_center = (int(w * 0.36), int(h * 0.40))
    right_eye_center = (int(w * 0.64), int(h * 0.40))
    cv2.ellipse(noise, left_eye_center, (int(w * 0.06), int(h * 0.03)), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(noise, right_eye_center, (int(w * 0.06), int(h * 0.03)), 0, 0, 360, (50, 50, 50), -1)

    # Eyebrows
    cv2.line(noise, (int(w * 0.28), int(h * 0.33)), (int(w * 0.42), int(h * 0.32)), (40, 40, 40), 3)
    cv2.line(noise, (int(w * 0.58), int(h * 0.32)), (int(w * 0.72), int(h * 0.33)), (40, 40, 40), 3)

    # Nose
    cv2.line(noise, (w // 2, int(h * 0.42)), (w // 2, int(h * 0.55)), (60, 60, 60), 2)
    cv2.ellipse(noise, (w // 2, int(h * 0.55)), (int(w * 0.05), int(h * 0.02)), 0, 0, 360, (60, 60, 60), 2)

    # Mouth
    cv2.ellipse(noise, (w // 2, int(h * 0.69)), (int(w * 0.10), int(h * 0.03)), 0, 0, 360, (50, 50, 50), -1)

    # Chin
    cv2.ellipse(noise, (w // 2, int(h * 0.82)), (int(w * 0.08), int(h * 0.04)), 0, 0, 180, (80, 80, 80), 2)

    return noise


def run_benchmark() -> None:
    print("\n" + "=" * 96)
    print("                FORENSIC LOCALITY ENFORCEMENT BENCHMARK (PHASE 3)")
    print("=" * 96)

    base_sketch = create_synthetic_forensic_sketch()
    h, w = base_sketch.shape[:2]

    # Features to benchmark
    features = ["nose", "mouth", "chin", "jaw", "eyebrows", "hair"]

    # Eye mask for locked verification
    lms = locality_service._get_canonical_landmarks()
    eye_mask = locality_service.get_region_mask((h, w), lms, "eyes") > 0

    print(f"{'Feature Tested':<16} | {'Mask Area':<10} | {'Locality Score':<15} | {'Outside MAD':<13} | {'Locked Eyes Bit-Exact':<22} | {'Latency':<9} | {'Status'}")
    print("-" * 96)

    all_passed = True

    for feat in features:
        # Create a modified image with significant alteration in the feature zone
        modified_sketch = base_sketch.copy()
        feat_mask = locality_service.get_region_mask((h, w), lms, feat)
        # Apply intense alteration inside feature zone (contrast shift + geometric lines)
        modified_sketch[feat_mask > 0] = (255 - modified_sketch[feat_mask > 0]).astype(np.uint8)

        start = time.perf_counter()
        blended, loc_score, unverified = locality_service.composite(
            prev_img=base_sketch,
            new_img=modified_sketch,
            allowed_regions=[feat],
            locked_regions=["eyes"]
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Calculate mask area
        mask_pct = (np.count_nonzero(feat_mask) / (h * w)) * 100.0

        # Verify locked eyes are 100% bit-identical
        eyes_identical = np.array_equal(blended[eye_mask], base_sketch[eye_mask])
        diff_count = np.count_nonzero(blended[eye_mask] != base_sketch[eye_mask])

        # Verify outside MAD
        outside_mad = locality_service.locality_score(base_sketch, blended, allowed_regions=[feat])
        status = "PASS" if (outside_mad < 0.005 and eyes_identical) else "FAIL"
        if status == "FAIL":
            all_passed = False

        eyes_str = "100% (0 diffs)" if eyes_identical else f"FAIL ({diff_count} diffs)"

        print(f"{feat:<16} | {mask_pct:6.1f}%    | {loc_score:<15.6f} | {outside_mad:<13.6f} | {eyes_str:<22} | {latency_ms:6.1f} ms | {status}")

    print("=" * 96)

    # 5-Edit Sequential Stress Test
    print("\n--- 5-Edit Sequential Locality Stress Test (Locked: 'eyes') ---")
    current_img = base_sketch.copy()
    sequential_seq = ["nose", "mouth", "chin", "jaw", "hair"]

    stress_passed = True
    for i, step_feat in enumerate(sequential_seq, start=1):
        step_mod = current_img.copy()
        step_mask = locality_service.get_region_mask((h, w), lms, step_feat)
        step_mod[step_mask > 0] = np.clip(step_mod[step_mask > 0].astype(int) + 40, 0, 255).astype(np.uint8)

        current_img, score, _ = locality_service.composite(
            prev_img=current_img,
            new_img=step_mod,
            allowed_regions=[step_feat],
            locked_regions=["eyes"]
        )

        step_identical = np.array_equal(current_img[eye_mask], base_sketch[eye_mask])
        if not step_identical:
            stress_passed = False

        print(f"Edit {i}: {step_feat:<10} -> Locked Eyes Bit-Identical: {step_identical} (Score: {score:.6f})")

    print("-" * 96)
    print(f"Overall Locality Engine Status: {'VERIFIED [ALL PASS]' if (all_passed and stress_passed) else 'FAILED'}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    run_benchmark()
