"""
Locality Enforcement & Compositing Service (Phase 3)
Deterministic service (no LLM) responsible for:
1. Detecting facial landmarks using MediaPipe FaceMesh / FaceLandmarker with canonical anthropometric fallbacks.
2. Generating feathered region masks for all facial features (hair, forehead, eyebrows, eyes, nose, mouth, jaw, chin, cheeks, ears, facial_hair, skin, whole_face).
3. Aligning candidate/new images to the reference/previous image via estimated 2D similarity transform.
4. Compositing modifications so ONLY allowed regions take pixels from new_img; locked and other regions are strictly preserved.
5. Computing locality score (mean absolute pixel difference outside allowed region mask).
6. Handling failure fallback (retry once, flag locality_unverified=True).
"""

import os
import io
import logging
import urllib.request
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("locality_service")

# Model configuration
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "models", "face_landmarker.task"
)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# MediaPipe 468/478 Landmark Indices per Region
FACEMESH_REGIONS = {
    "eyes": {
        "left": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "right": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    },
    "eyebrows": {
        "left": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "right": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    },
    "nose": [
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 278, 279,
        360, 420, 439, 438, 457, 48, 64, 240, 99, 102, 115, 129, 358, 331
    ],
    "mouth": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
        0, 37, 39, 40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
    ],
    "jaw": [
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378,
        379, 365, 397, 288, 361, 454
    ],
    "chin": [
        152, 148, 176, 149, 150, 136, 172, 377, 400, 378, 379, 365, 397,
        175, 199, 200, 18
    ],
    "cheeks": {
        "left": [116, 123, 147, 213, 138, 192, 214, 210, 211, 34, 143, 111, 117],
        "right": [345, 352, 376, 433, 367, 416, 434, 430, 431, 264, 372, 340, 346]
    },
    "forehead": [
        10, 338, 297, 332, 284, 251, 389, 356, 127, 162, 21, 54, 103, 67, 109
    ],
    "ears": {
        "left": [234, 127, 93, 137, 177, 215],
        "right": [454, 356, 323, 366, 401, 435]
    }
}


def to_numpy_rgb(image: Union[bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """Converts bytes, PIL image, or array into a standard RGB uint8 numpy array."""
    if isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image.copy()
    raise ValueError(f"Unsupported image type: {type(image)}")


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encodes an RGB numpy array to PNG bytes."""
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


class LocalityService:
    """
    Deterministic Locality Enforcement Service.
    Guarantees that edits are confined strictly to requested facial regions
    and that locked features remain 100% bit-identical.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._landmarker = None
        self._ensure_model_available()

    def _ensure_model_available(self):
        """Ensures the face landmarker model task file exists on disk."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            try:
                logger.info(f"Downloading face_landmarker model to {self.model_path}...")
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                logger.info("Face landmarker model downloaded successfully.")
            except Exception as e:
                logger.warning(f"Could not download face_landmarker model: {e}. Will use canonical fallback.")

    def _get_landmarker(self):
        """Lazy initialization of MediaPipe FaceLandmarker."""
        if self._landmarker is None and os.path.exists(self.model_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=self.model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=True,
                    num_faces=1
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                logger.warning(f"Failed to create MediaPipe FaceLandmarker: {e}")
                self._landmarker = None
        return self._landmarker

    def extract_landmarks(
        self,
        image: np.ndarray,
        retry_preprocess: bool = True
    ) -> Tuple[Optional[List[Tuple[float, float]]], bool]:
        """
        Extracts 468+ facial landmarks from an image.
        Returns:
            (landmarks, locality_unverified)
            landmarks: list of (x_norm, y_norm) in [0.0, 1.0], or None.
            locality_unverified: True if fallback heuristics had to be used.
        """
        h, w = image.shape[:2]
        landmarker = self._get_landmarker()

        if landmarker is not None:
            try:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                result = landmarker.detect(mp_img)

                if result.face_landmarks and len(result.face_landmarks) > 0:
                    pts = [(lm.x, lm.y) for lm in result.face_landmarks[0]]
                    return pts, False

                # Retry once with contrast enhancement (CLAHE) if detection missed
                if retry_preprocess:
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

                    mp_img_retry = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced)
                    retry_result = landmarker.detect(mp_img_retry)
                    if retry_result.face_landmarks and len(retry_result.face_landmarks) > 0:
                        pts = [(lm.x, lm.y) for lm in retry_result.face_landmarks[0]]
                        return pts, False

            except Exception as e:
                logger.warning(f"Landmark detection encountered error: {e}")

        # Fallback: Canonical anthropometric proportions
        logger.info("Using canonical anthropometric facial proportions fallback.")
        return self._get_canonical_landmarks(), True

    def _get_canonical_landmarks(self) -> List[Tuple[float, float]]:
        """
        Provides canonical 2D normalized face landmarks according to standard
        forensic proportions (centered face, eyes at y=0.40, nose at y=0.55, mouth at y=0.70).
        Total 468 coordinates populated so all index queries remain valid.
        """
        pts = [(0.5, 0.5)] * 478

        # Eyes
        for idx in FACEMESH_REGIONS["eyes"]["left"]:
            pts[idx] = (0.36, 0.40)
        for idx in FACEMESH_REGIONS["eyes"]["right"]:
            pts[idx] = (0.64, 0.40)

        # Eyebrows
        for idx in FACEMESH_REGIONS["eyebrows"]["left"]:
            pts[idx] = (0.36, 0.32)
        for idx in FACEMESH_REGIONS["eyebrows"]["right"]:
            pts[idx] = (0.64, 0.32)

        # Nose
        for idx in FACEMESH_REGIONS["nose"]:
            pts[idx] = (0.50, 0.54)

        # Mouth
        for idx in FACEMESH_REGIONS["mouth"]:
            pts[idx] = (0.50, 0.69)

        # Chin
        for idx in FACEMESH_REGIONS["chin"]:
            pts[idx] = (0.50, 0.83)

        # Jaw
        for i, idx in enumerate(FACEMESH_REGIONS["jaw"]):
            frac = i / max(1, len(FACEMESH_REGIONS["jaw"]) - 1)
            pts[idx] = (0.22 + 0.56 * frac, 0.50 + 0.35 * np.sin(np.pi * frac))

        # Cheeks
        for idx in FACEMESH_REGIONS["cheeks"]["left"]:
            pts[idx] = (0.30, 0.54)
        for idx in FACEMESH_REGIONS["cheeks"]["right"]:
            pts[idx] = (0.70, 0.54)

        # Forehead
        for idx in FACEMESH_REGIONS["forehead"]:
            pts[idx] = (0.50, 0.22)

        # Ears
        for idx in FACEMESH_REGIONS["ears"]["left"]:
            pts[idx] = (0.16, 0.44)
        for idx in FACEMESH_REGIONS["ears"]["right"]:
            pts[idx] = (0.84, 0.44)

        return pts

    def get_region_mask(
        self,
        shape: Tuple[int, int],
        landmarks: List[Tuple[float, float]],
        region: str,
        feather_radius: int = 15
    ) -> np.ndarray:
        """
        Generates an 8-bit binary mask (0 or 255) for the requested facial region.
        Regions supported:
        'hair', 'forehead', 'eyebrows', 'eyes', 'nose', 'mouth', 'jaw',
        'chin', 'cheeks', 'ears', 'facial_hair', 'skin', 'whole_face'.
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        norm_region = region.lower().replace("-", "_").strip()

        # Handle alias mappings
        if norm_region in {"mouth_lips", "lips"}:
            norm_region = "mouth"
        elif norm_region in {"jawline"}:
            norm_region = "jaw"
        elif norm_region in {"cheekbones"}:
            norm_region = "cheeks"
        elif norm_region in {"skin_tone"}:
            norm_region = "skin"
        elif norm_region in {"all", "wholeface", "face"}:
            norm_region = "whole_face"

        def _pts_to_px(indices: List[int]) -> np.ndarray:
            px = []
            for idx in indices:
                if idx < len(landmarks):
                    x, y = landmarks[idx]
                    px.append([int(x * w), int(y * h)])
            return np.array(px, dtype=np.int32)

        if norm_region == "whole_face":
            # Oval encompassing full face
            center_x = int(0.5 * w)
            center_y = int(0.52 * h)
            rx = int(0.32 * w)
            ry = int(0.42 * h)
            cv2.ellipse(mask, (center_x, center_y), (rx, ry), 0, 0, 360, 255, -1)
            return mask

        elif norm_region == "eyes":
            left_px = _pts_to_px(FACEMESH_REGIONS["eyes"]["left"])
            right_px = _pts_to_px(FACEMESH_REGIONS["eyes"]["right"])
            if len(left_px) >= 3:
                hull_left = cv2.convexHull(left_px)
                cv2.drawContours(mask, [hull_left], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.36 * w), int(0.40 * h)), (int(0.09 * w), int(0.06 * h)), 0, 0, 360, 255, -1)

            if len(right_px) >= 3:
                hull_right = cv2.convexHull(right_px)
                cv2.drawContours(mask, [hull_right], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.64 * w), int(0.40 * h)), (int(0.09 * w), int(0.06 * h)), 0, 0, 360, 255, -1)

            # Dilate slightly to cover full eyelid sockets
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(w * 0.02)), max(5, int(h * 0.02))))
            mask = cv2.dilate(mask, kernel)
            return mask

        elif norm_region == "eyebrows":
            left_px = _pts_to_px(FACEMESH_REGIONS["eyebrows"]["left"])
            right_px = _pts_to_px(FACEMESH_REGIONS["eyebrows"]["right"])
            if len(left_px) >= 3:
                hull_left = cv2.convexHull(left_px)
                cv2.drawContours(mask, [hull_left], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.36 * w), int(0.32 * h)), (int(0.10 * w), int(0.04 * h)), 0, 0, 360, 255, -1)

            if len(right_px) >= 3:
                hull_right = cv2.convexHull(right_px)
                cv2.drawContours(mask, [hull_right], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.64 * w), int(0.32 * h)), (int(0.10 * w), int(0.04 * h)), 0, 0, 360, 255, -1)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(w * 0.02)), max(5, int(h * 0.02))))
            mask = cv2.dilate(mask, kernel)
            return mask

        elif norm_region == "nose":
            nose_px = _pts_to_px(FACEMESH_REGIONS["nose"])
            if len(nose_px) >= 3:
                hull = cv2.convexHull(nose_px)
                cv2.drawContours(mask, [hull], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.50 * w), int(0.52 * h)), (int(0.08 * w), int(0.12 * h)), 0, 0, 360, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(w * 0.02)), max(5, int(h * 0.02))))
            mask = cv2.dilate(mask, kernel)
            return mask

        elif norm_region == "mouth":
            mouth_px = _pts_to_px(FACEMESH_REGIONS["mouth"])
            if len(mouth_px) >= 3:
                hull = cv2.convexHull(mouth_px)
                cv2.drawContours(mask, [hull], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.50 * w), int(0.69 * h)), (int(0.13 * w), int(0.07 * h)), 0, 0, 360, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(w * 0.02)), max(5, int(h * 0.02))))
            mask = cv2.dilate(mask, kernel)
            return mask

        elif norm_region == "chin":
            chin_px = _pts_to_px(FACEMESH_REGIONS["chin"])
            if len(chin_px) >= 3:
                hull = cv2.convexHull(chin_px)
                cv2.drawContours(mask, [hull], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.50 * w), int(0.83 * h)), (int(0.12 * w), int(0.08 * h)), 0, 0, 360, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(w * 0.02)), max(5, int(h * 0.02))))
            mask = cv2.dilate(mask, kernel)
            return mask

        elif norm_region == "jaw":
            jaw_px = _pts_to_px(FACEMESH_REGIONS["jaw"])
            if len(jaw_px) >= 3:
                # Draw a thick curve along the lower jaw perimeter
                for i in range(len(jaw_px) - 1):
                    cv2.line(mask, tuple(jaw_px[i]), tuple(jaw_px[i + 1]), 255, thickness=int(h * 0.08))
            else:
                cv2.ellipse(mask, (int(0.50 * w), int(0.78 * h)), (int(0.25 * w), int(0.14 * h)), 0, 0, 180, 255, int(h * 0.08))
            return mask

        elif norm_region == "cheeks":
            left_px = _pts_to_px(FACEMESH_REGIONS["cheeks"]["left"])
            right_px = _pts_to_px(FACEMESH_REGIONS["cheeks"]["right"])
            if len(left_px) >= 3:
                hull_left = cv2.convexHull(left_px)
                cv2.drawContours(mask, [hull_left], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.30 * w), int(0.54 * h)), (int(0.09 * w), int(0.08 * h)), 0, 0, 360, 255, -1)

            if len(right_px) >= 3:
                hull_right = cv2.convexHull(right_px)
                cv2.drawContours(mask, [hull_right], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.70 * w), int(0.54 * h)), (int(0.09 * w), int(0.08 * h)), 0, 0, 360, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(w * 0.02)), max(5, int(h * 0.02))))
            mask = cv2.dilate(mask, kernel)
            return mask

        elif norm_region == "forehead":
            forehead_px = _pts_to_px(FACEMESH_REGIONS["forehead"])
            if len(forehead_px) >= 3:
                hull = cv2.convexHull(forehead_px)
                cv2.drawContours(mask, [hull], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.50 * w), int(0.24 * h)), (int(0.22 * w), int(0.12 * h)), 0, 0, 360, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(w * 0.03)), max(5, int(h * 0.03))))
            mask = cv2.dilate(mask, kernel)
            return mask

        elif norm_region == "ears":
            left_px = _pts_to_px(FACEMESH_REGIONS["ears"]["left"])
            right_px = _pts_to_px(FACEMESH_REGIONS["ears"]["right"])
            if len(left_px) >= 3:
                hull_left = cv2.convexHull(left_px)
                cv2.drawContours(mask, [hull_left], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.15 * w), int(0.44 * h)), (int(0.06 * w), int(0.10 * h)), 0, 0, 360, 255, -1)

            if len(right_px) >= 3:
                hull_right = cv2.convexHull(right_px)
                cv2.drawContours(mask, [hull_right], -1, 255, -1)
            else:
                cv2.ellipse(mask, (int(0.85 * w), int(0.44 * h)), (int(0.06 * w), int(0.10 * h)), 0, 0, 360, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(w * 0.02)), max(5, int(h * 0.02))))
            mask = cv2.dilate(mask, kernel)
            return mask

        elif norm_region == "hair":
            # Hair covers the top of the skull and temple zones above forehead and outer ears
            hair_poly = np.array([
                [int(0.10 * w), int(0.45 * h)],
                [int(0.12 * w), int(0.20 * h)],
                [int(0.25 * w), int(0.06 * h)],
                [int(0.50 * w), int(0.02 * h)],
                [int(0.75 * w), int(0.06 * h)],
                [int(0.88 * w), int(0.20 * h)],
                [int(0.90 * w), int(0.45 * h)],
                [int(0.80 * w), int(0.26 * h)],
                [int(0.50 * w), int(0.18 * h)],
                [int(0.20 * w), int(0.26 * h)],
            ], dtype=np.int32)
            cv2.fillPoly(mask, [hair_poly], 255)
            return mask

        elif norm_region == "facial_hair":
            # Mustache area + chin + jaw perimeter
            cv2.ellipse(mask, (int(0.50 * w), int(0.63 * h)), (int(0.12 * w), int(0.04 * h)), 0, 0, 360, 255, -1)
            cv2.ellipse(mask, (int(0.50 * w), int(0.80 * h)), (int(0.18 * w), int(0.12 * h)), 0, 0, 360, 255, -1)
            return mask

        elif norm_region == "skin":
            # Full face minus eyes, eyebrows, mouth, and nostril cores
            cv2.ellipse(mask, (int(0.50 * w), int(0.52 * h)), (int(0.30 * w), int(0.38 * h)), 0, 0, 360, 255, -1)
            # Subtract eyes
            eyes_mask = self.get_region_mask(shape, landmarks, "eyes")
            mouth_mask = self.get_region_mask(shape, landmarks, "mouth")
            brows_mask = self.get_region_mask(shape, landmarks, "eyebrows")
            mask = cv2.subtract(mask, eyes_mask)
            mask = cv2.subtract(mask, mouth_mask)
            mask = cv2.subtract(mask, brows_mask)
            return mask

        else:
            logger.warning(f"Unrecognized region '{region}'. Defaulting to whole_face.")
            return self.get_region_mask(shape, landmarks, "whole_face")

    def align(
        self,
        prev_img: Union[bytes, np.ndarray],
        new_img: Union[bytes, np.ndarray]
    ) -> np.ndarray:
        """
        Estimates a similarity transform (translation, rotation, uniform scale)
        from corresponding facial landmarks and warps new_img onto prev_img canvas.
        """
        prev_arr = to_numpy_rgb(prev_img)
        new_arr = to_numpy_rgb(new_img)
        h, w = prev_arr.shape[:2]

        prev_lms, _ = self.extract_landmarks(prev_arr)
        new_lms, _ = self.extract_landmarks(new_arr)

        if not prev_lms or not new_lms:
            # Cannot align if landmarks missing, return resized new image
            if new_arr.shape[:2] != (h, w):
                return cv2.resize(new_arr, (w, h), interpolation=cv2.INTER_CUBIC)
            return new_arr

        # Key anchor landmarks: left eye center, right eye center, nose tip, mouth corners, chin
        anchor_indices = [
            # Left eye corners & pupil approx
            33, 133, 159, 145,
            # Right eye corners & pupil approx
            362, 263, 386, 374,
            # Nose bridge & tip
            168, 1, 4,
            # Mouth corners
            61, 291,
            # Chin
            152
        ]

        src_pts = []
        dst_pts = []
        for idx in anchor_indices:
            if idx < len(new_lms) and idx < len(prev_lms):
                src_pts.append([new_lms[idx][0] * w, new_lms[idx][1] * h])
                dst_pts.append([prev_lms[idx][0] * w, prev_lms[idx][1] * h])

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        # Estimate partial affine (similarity) transform
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=15.0
        )

        if M is not None:
            aligned = cv2.warpAffine(
                new_arr, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT
            )
            return aligned

        # Fallback if matrix estimation degenerate
        if new_arr.shape[:2] != (h, w):
            return cv2.resize(new_arr, (w, h), interpolation=cv2.INTER_CUBIC)
        return new_arr

    def composite(
        self,
        prev_img: Union[bytes, np.ndarray],
        new_img: Union[bytes, np.ndarray],
        allowed_regions: List[str],
        locked_regions: Optional[List[str]] = None,
        feather_radius: int = 15
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Feathered-mask blend so ONLY allowed regions take pixels from new_img;
        locked and all other regions stay strictly from prev_img.
        Returns:
            (blended_image_rgb, locality_score, locality_unverified)
        """
        prev_arr = to_numpy_rgb(prev_img)
        new_arr = to_numpy_rgb(new_img)
        h, w = prev_arr.shape[:2]

        locked_regions = locked_regions or []

        # Align new_img to prev_img frame
        aligned_new = self.align(prev_arr, new_arr)

        # Extract landmarks from prev_img
        landmarks, locality_unverified = self.extract_landmarks(prev_arr)

        # 1. Build union of allowed regions mask
        allowed_mask = np.zeros((h, w), dtype=np.uint8)
        for region in allowed_regions:
            rmask = self.get_region_mask((h, w), landmarks, region, feather_radius=feather_radius)
            allowed_mask = cv2.bitwise_or(allowed_mask, rmask)

        # 2. Build union of locked regions mask
        locked_mask = np.zeros((h, w), dtype=np.uint8)
        for region in locked_regions:
            lmask = self.get_region_mask((h, w), landmarks, region, feather_radius=feather_radius)
            locked_mask = cv2.bitwise_or(locked_mask, lmask)

        # 3. Allowed blend mask: allowed MINUS locked
        blend_binary = cv2.subtract(allowed_mask, locked_mask)

        # If nothing can be modified (all requested regions were locked), return prev_arr directly
        if cv2.countNonZero(blend_binary) == 0:
            return prev_arr.copy(), 0.0, locality_unverified

        # 4. Color-matched feathering
        # Apply Gaussian blur to create smooth alpha channel
        ksize = feather_radius * 2 + 1
        blurred = cv2.GaussianBlur(blend_binary.astype(np.float32) / 255.0, (ksize, ksize), feather_radius / 3.0)

        # Enforce zero modification on strictly locked features
        blurred[locked_mask > 0] = 0.0

        # Cut off extremely low alpha values outside the feathered margin
        blurred[blurred < 0.005] = 0.0

        alpha = np.dstack([blurred, blurred, blurred])

        # 5. Blend pixels: result = new * alpha + prev * (1 - alpha)
        blended = (aligned_new.astype(np.float32) * alpha + prev_arr.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        # Strict preservation: Any pixel where alpha is 0.0 or inside locked_mask is 100% bit-identical to prev_arr
        untouched_condition = (blurred == 0.0) | (locked_mask > 0)
        blended[untouched_condition] = prev_arr[untouched_condition]

        # 6. Calculate locality score (mean absolute pixel difference outside allowed mask)
        loc_score = self.locality_score(prev_arr, blended, allowed_regions=allowed_regions)

        return blended, loc_score, locality_unverified

    def locality_score(
        self,
        prev_img: Union[bytes, np.ndarray],
        final_img: Union[bytes, np.ndarray],
        allowed_regions: List[str]
    ) -> float:
        """
        Computes the locality metric:
        Mean absolute pixel difference outside the allowed region mask.
        Normalized to [0.0, 1.0]. 0.0 indicates 100% pixel-identical preservation.
        """
        prev_arr = to_numpy_rgb(prev_img)
        final_arr = to_numpy_rgb(final_img)
        h, w = prev_arr.shape[:2]

        landmarks, _ = self.extract_landmarks(prev_arr)

        # Union allowed masks
        allowed_mask = np.zeros((h, w), dtype=np.uint8)
        for region in allowed_regions:
            rmask = self.get_region_mask((h, w), landmarks, region)
            allowed_mask = cv2.bitwise_or(allowed_mask, rmask)

        # Evaluation zone: outside allowed mask
        outside_mask = allowed_mask == 0
        num_outside_px = np.count_nonzero(outside_mask)

        if num_outside_px == 0:
            return 0.0

        # Mean absolute pixel difference across all 3 RGB channels
        diff = np.abs(prev_arr[outside_mask].astype(np.float32) - final_arr[outside_mask].astype(np.float32))
        mean_abs_diff = np.mean(diff) / 255.0

        return float(round(mean_abs_diff, 6))


# Singleton instance
locality_service = LocalityService()
