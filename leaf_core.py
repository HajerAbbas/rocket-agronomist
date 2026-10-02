"""
leaf_core.py
============
(shared by leaf_service.py [n8n API] and app.py [Shiny app])
One HTTP endpoint that replaces 4 boxes of the old design
(Validate Inputs + Preprocess/OpenCV + Analyse Leaf + Python Logic):

    POST /analyze   (multipart form, field "file" = photo)
      1. image quality check   (blur, darkness)
      2. leaf detection        (is a green leaf in the frame? how big?)
      3. leaf segmentation     (mask of leaf pixels)
      4. nitrogen calculator   (rocket_n_calculator.estimate_nitrogen)

n8n calls this with one HTTP Request node.

Run:
    pip install fastapi uvicorn python-multipart opencv-python-headless numpy
    uvicorn leaf_service:app --host 0.0.0.0 --port 8000

Optional environment variables:
    CALIBRATION_PATH         JSON from fit_calibration()  -> N % output
    REFERENCE_FEATURES_PATH  JSON features of a well-fed reference plant
                             -> sufficiency index output
"""

from __future__ import annotations

import base64
import json
import os
from typing import Optional

import cv2
import numpy as np
from rocket_n_calculator import estimate_nitrogen

# ----------------------------- settings (tune on your own photos) -----------
MAX_SIDE = 1200            # resize longest side to this many pixels
BLUR_MIN = 60.0            # Laplacian variance below this = blurry
DARK_MAX = 0.15            # mean brightness below this = too dark
MIN_GREEN_FRACTION = 0.05  # at least 5% of frame must be leaf-coloured
MIN_LARGEST_REGION = 0.03  # biggest leaf region must be >= 3% of frame
SMALL_LEAF_WARN = 0.10     # leaf smaller than 10% of frame -> "move closer"

CALIBRATION_PATH = os.getenv("CALIBRATION_PATH")
REFERENCE_FEATURES_PATH = os.getenv("REFERENCE_FEATURES_PATH")


def _load_reference() -> Optional[dict]:
    if REFERENCE_FEATURES_PATH and os.path.exists(REFERENCE_FEATURES_PATH):
        with open(REFERENCE_FEATURES_PATH) as fh:
            return json.load(fh)
    return None


REFERENCE = _load_reference()
CALIBRATION = CALIBRATION_PATH if CALIBRATION_PATH and os.path.exists(CALIBRATION_PATH) else None


# ----------------------------- core functions -------------------------------
def decode_image(data: bytes) -> np.ndarray:
    """bytes -> RGB uint8, longest side <= MAX_SIDE."""
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:                      # e.g. iPhone HEIC -> try Pillow
        try:
            import io
            from PIL import Image, ImageOps
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError("file is not a readable image")
    h, w = bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def check_quality(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean() / 255.0)
    return {
        "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
        "blur_score": round(blur, 1), "is_blurry": blur < BLUR_MIN,
        "mean_brightness": round(bright, 3), "is_too_dark": bright < DARK_MAX,
    }


def detect_leaf(rgb: np.ndarray) -> (dict, np.ndarray):
    """
    Colour-based leaf detector:
      Excess Green (ExG = 2g - r - b) + Otsu threshold, combined with a
      green hue/saturation gate, cleaned with morphology; keeps connected
      regions >= 1% of the frame.
    Returns (detection info, boolean leaf mask).
    """
    f = rgb.astype(np.float32) / 255.0
    tot = f.sum(axis=2) + 1e-6
    r, g, b = f[..., 0] / tot, f[..., 1] / tot, f[..., 2] / tot
    exg = 2 * g - r - b

    exg_u8 = np.clip((exg + 0.2) / 0.8 * 255, 0, 255).astype(np.uint8)
    otsu_t, _ = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_exg = otsu_t / 255 * 0.8 - 0.2
    exg_mask = exg > max(otsu_exg, 0.05)       # absolute floor: Otsu alone
                                               # "finds" green in any image
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H: 0-179 in OpenCV
    hue_mask = (hsv[..., 0] >= 17) & (hsv[..., 0] <= 85) \
        & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 30)
    mask = (exg_mask & hue_mask).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    area_total = mask.size
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.01 * area_total]
    leaf = np.isin(labels, keep)

    green_frac = float(leaf.sum() / area_total)
    largest = max((stats[i, cv2.CC_STAT_AREA] for i in keep), default=0) / area_total
    detected = green_frac >= MIN_GREEN_FRACTION and largest >= MIN_LARGEST_REGION

    # erode edges so background colour does not bleed into measurements
    leaf_core = cv2.erode(leaf.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    info = {
        "leaf_detected": bool(detected),
        "leaf_score": round(min(1.0, green_frac / 0.25), 3),
        "green_fraction": round(green_frac, 4),
        "largest_region_fraction": round(float(largest), 4),
        "n_leaf_regions": len(keep),
    }
    return info, leaf_core


def _preview_jpeg_b64(rgb: np.ndarray, max_side: int = 720) -> str:
    h, w = rgb.shape[:2]
    sc = max_side / max(h, w)
    if sc < 1:
        rgb = cv2.resize(rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def overlay_jpeg_b64(rgb: np.ndarray, mask: np.ndarray) -> str:
    vis = rgb.copy()
    vis[mask] = (0.5 * vis[mask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    return _preview_jpeg_b64(vis)


def analyze_bytes(data: bytes, debug: bool = False,
                  reference_features: Optional[dict] = None) -> dict:
    out = {"decision": "retake", "reasons": [], "image_quality": None,
           "leaf_detection": None, "nitrogen": None}
    try:
        rgb = decode_image(data)
    except ValueError as e:
        out["reasons"].append(str(e))
        return out

    q = check_quality(rgb)
    out["image_quality"] = q
    det, mask = detect_leaf(rgb)
    out["leaf_detection"] = det

    if not det["leaf_detected"]:
        out["reasons"].append("no_leaf_detected")
    if q["is_blurry"]:
        out["reasons"].append("image_blurry")
    if q["is_too_dark"]:
        out["reasons"].append("image_too_dark")
    if det["leaf_detected"] and det["green_fraction"] < SMALL_LEAF_WARN:
        out["reasons"].append("leaf_too_small_move_closer")

    if det["leaf_detected"] and not q["is_blurry"] and not q["is_too_dark"]:
        n = estimate_nitrogen(rgb, mask, calibration=CALIBRATION,
                              reference_features=reference_features or REFERENCE)
        out["nitrogen"] = n
        if n.get("status") == "ok":
            out["decision"] = "ok"
        else:
            out["reasons"].append(f"nitrogen_{n.get('status')}")

    if debug:
        out["mask_overlay_jpeg_base64"] = overlay_jpeg_b64(rgb, mask)
        out["image_jpeg_base64"] = _preview_jpeg_b64(rgb)
    return out




def reference_features_from_bytes(data: bytes) -> Optional[dict]:
    """Features of a well-fertilized reference plant photo (or None if no leaf)."""
    from rocket_n_calculator import extract_features
    rgb = decode_image(data)
    det, mask = detect_leaf(rgb)
    if not det["leaf_detected"]:
        return None
    return extract_features(rgb, mask)["features"]
