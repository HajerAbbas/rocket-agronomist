"""
rocket_n_calculator.py
======================
Specialized nitrogen calculator for rocket / arugula (Eruca sativa, Diplotaxis
tenuifolia) from flash RGB photos. Designed as a deterministic tool inside an
agentic system: leaf segmentation is done upstream; this module receives the
image + a boolean leaf mask and returns a JSON-serializable result.

WHY THERE IS NO "HARD-CODED" N FORMULA
--------------------------------------
No peer-reviewed RGB->N equation exists for rocket. All published RGB methods
(lettuce, spinach, pakchoi, rice, maize, cotton...) are empirical regressions
calibrated per crop, camera and lighting. So this tool:

  1. Standardizes colour (gray card or colour checker) -> removes flash
     distance / exposure / white-balance differences.
  2. Rejects glare (specular flash reflections) and shadow pixels.
  3. Extracts the RGB features best supported in the literature:
       DGCI (Karcher & Richardson 2003), NGRDI, ExG, ExGR, GLI, VARI, MGRVI,
       RGBVI, chromatic r/g/b, CIELAB L*a*b*, hue.
  4. Estimates N in one of three modes:
       - "calibrated":            ridge regression fitted on YOUR lab data
                                  (Kjeldahl/Dumas leaf N vs. photos).
       - "reference_sufficiency": sufficiency index vs. a well-fertilized
                                  reference rocket plant photographed with the
                                  same setup (no lab data needed).
       - "index_only":            features only, no N claim.

Dependencies: numpy (required), Pillow (only for loading files / CLI).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, str]

# Literature-supported default feature subset for calibration. DGCI and a* are
# hue/chroma-based (more robust to residual exposure error), NGRDI is the
# strongest simple index in lettuce (r > 0.93 with N in romaine lettuce).
DEFAULT_CAL_FEATURES = ("dgci", "ngrdi", "lab_a")

# Sufficiency-index thresholds, borrowed from chlorophyll-meter practice
# (N-rich reference strip approach). Treat as starting points for rocket.
SI_SUFFICIENT = 0.95
SI_DEFICIENT = 0.90

MIN_VALID_PIXELS = 500


# =============================================================================
# Colour utilities
# =============================================================================
def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def linear_rgb_to_lab(lin: np.ndarray) -> np.ndarray:
    """Linear sRGB (N,3) in 0..1 -> CIELAB (N,3), D65 white."""
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """sRGB (N,3) 0..1 -> H in degrees [0,360), S, V in 0..1."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-12
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4
    h *= 60.0
    s = np.where(mx > 1e-12, d / np.maximum(mx, 1e-12), 0.0)
    return np.stack([h, s, mx], axis=1)


# =============================================================================
# Input handling
# =============================================================================
def _load_image(img: ArrayLike) -> np.ndarray:
    """Return float sRGB image HxWx3 in 0..1."""
    if isinstance(img, str):
        from PIL import Image  # lazy import
        img = np.asarray(Image.open(img).convert("RGB"))
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("image must be HxWx3 (RGB)")
    img = img[:, :, :3]
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float64) / 255.0
    img = img.astype(np.float64)
    return img / 255.0 if img.max() > 1.0 else img


def _load_mask(mask: ArrayLike, shape) -> np.ndarray:
    if isinstance(mask, str):
        from PIL import Image
        mask = np.asarray(Image.open(mask).convert("L"))
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask > 0
    if mask.shape != tuple(shape[:2]):
        raise ValueError(f"mask shape {mask.shape} != image shape {shape[:2]}")
    return mask


# =============================================================================
# Colour correction (critical for flash photos)
# =============================================================================
def color_correct(
    img: np.ndarray,
    gray_card_mask: Optional[ArrayLike] = None,
    gray_card_reflectance: float = 0.18,
    checker_measured_srgb: Optional[np.ndarray] = None,
    checker_reference_srgb: Optional[np.ndarray] = None,
) -> (np.ndarray, str):
    """
    Returns (corrected sRGB image 0..1, method name).

    - Colour checker (preferred): 3x3 matrix fitted by least squares in linear
      space from N measured patches -> reference patch values (both sRGB 0..255).
    - Gray card: per-channel gains in linear space so the card becomes neutral
      at its known reflectance (0.18 for an 18% card). Also normalizes exposure,
      which removes most of the flash-distance effect.
    """
    lin = srgb_to_linear(img)
    if checker_measured_srgb is not None and checker_reference_srgb is not None:
        meas = srgb_to_linear(np.asarray(checker_measured_srgb, float) / 255.0)
        ref = srgb_to_linear(np.asarray(checker_reference_srgb, float) / 255.0)
        if meas.shape[0] < 4:
            raise ValueError("need >= 4 colour-checker patches")
        M, *_ = np.linalg.lstsq(meas, ref, rcond=None)
        out = lin.reshape(-1, 3) @ M
        return linear_to_srgb(out.reshape(img.shape)), "color_checker_3x3"
    if gray_card_mask is not None:
        gm = _load_mask(gray_card_mask, img.shape)
        if gm.sum() < 50:
            raise ValueError("gray card mask too small (< 50 px)")
        card = np.median(lin[gm], axis=0)
        if np.any(card < 1e-4) or np.any(card > 0.98):
            raise ValueError("gray card is black or clipped; re-shoot")
        gains = gray_card_reflectance / card
        return linear_to_srgb(lin * gains), "gray_card"
    return img, "none"


# =============================================================================
# Feature extraction
# =============================================================================
def _pixel_quality(raw: np.ndarray, leaf: np.ndarray):
    """Glare (flash specular / sensor clipping) and shadow masks on RAW pixels."""
    hsv = rgb_to_hsv(raw.reshape(-1, 3)).reshape(raw.shape)
    clipped = (raw >= 250 / 255).any(axis=2)
    specular = (hsv[..., 2] > 0.92) & (hsv[..., 1] < 0.20)
    glare = (clipped | specular) & leaf
    shadow = (hsv[..., 2] < 0.08) & leaf
    return glare, shadow


def extract_features(
    image: ArrayLike,
    leaf_mask: ArrayLike,
    gray_card_mask: Optional[ArrayLike] = None,
    gray_card_reflectance: float = 0.18,
    checker_measured_srgb: Optional[np.ndarray] = None,
    checker_reference_srgb: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute colour features on valid leaf pixels.
    Use THIS function both for calibration photos and for prediction, so the
    pipeline is identical.
    """
    raw = _load_image(image)
    leaf = _load_mask(leaf_mask, raw.shape)
    n_leaf = int(leaf.sum())
    if n_leaf == 0:
        raise ValueError("leaf mask is empty")

    glare, shadow = _pixel_quality(raw, leaf)
    valid = leaf & ~glare & ~shadow
    n_valid = int(valid.sum())

    img, cc_method = color_correct(raw, gray_card_mask, gray_card_reflectance,
                                   checker_measured_srgb, checker_reference_srgb)

    px = img[valid]                         # (N,3) corrected sRGB 0..1
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    eps = 1e-9
    tot = R + G + B + eps
    r, g, b = R / tot, G / tot, B / tot

    hsv = rgb_to_hsv(px)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    lab = linear_rgb_to_lab(srgb_to_linear(px))

    # DGCI (Karcher & Richardson 2003): higher = darker green
    dgci = ((H - 60.0) / 60.0 + (1.0 - S) + (1.0 - V)) / 3.0

    exg = 2 * g - r - b
    exr = 1.4 * r - g
    vari_den = G + R - B
    vari_ok = np.abs(vari_den) > 0.02

    per_pixel = {
        "dgci": dgci,
        "ngrdi": (G - R) / (G + R + eps),
        "exg": exg,
        "exgr": exg - exr,
        "gli": (2 * G - R - B) / (2 * G + R + B + eps),
        "vari": np.where(vari_ok, (G - R) / np.where(vari_ok, vari_den, 1), np.nan),
        "mgrvi": (G ** 2 - R ** 2) / (G ** 2 + R ** 2 + eps),
        "rgbvi": (G ** 2 - B * R) / (G ** 2 + B * R + eps),
        "r_chrom": r, "g_chrom": g, "b_chrom": b,
        "lab_L": lab[:, 0], "lab_a": lab[:, 1], "lab_b": lab[:, 2],
        "hue_deg": H, "saturation": S, "brightness": V,
    }

    feats = {k: (float(np.nanmedian(v)) if n_valid else float("nan"))
             for k, v in per_pixel.items()}
    if n_valid:
        q25, q75 = np.percentile(dgci, [25, 75])
        feats["dgci_iqr"] = float(q75 - q25)          # patchiness / chlorosis spread
        feats["frac_yellowish"] = float(np.mean((H > 35) & (H < 75)))
        feats["frac_brownish"] = float(np.mean(H <= 35))
    quality = {
        "leaf_pixels": n_leaf,
        "valid_pixels": n_valid,
        "valid_fraction": round(n_valid / n_leaf, 4),
        "glare_fraction": round(float(glare.sum()) / n_leaf, 4),
        "shadow_fraction": round(float(shadow.sum()) / n_leaf, 4),
        "color_correction": cc_method,
    }
    return {"features": feats, "quality": quality}


# =============================================================================
# Calibration (fit on your own lab data)
# =============================================================================
@dataclass
class Calibration:
    features: List[str]
    mean: List[float]
    std: List[float]
    coef: List[float]
    intercept: float
    train_min: List[float]
    train_max: List[float]
    y_min: float
    y_max: float
    n_samples: int
    cv_rmse: float
    cv_r2: float
    ridge_alpha: float
    target: str = "leaf_N_percent_dry_weight"
    color_correction: str = "gray_card"
    notes: str = ""
    feature_p90: Dict[str, float] = field(default_factory=dict)

    def predict(self, feats: Dict) -> float:
        x = np.array([feats[f] for f in self.features], float)
        z = (x - np.array(self.mean)) / np.array(self.std)
        return float(self.intercept + z @ np.array(self.coef))

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @staticmethod
    def load(path: str) -> "Calibration":
        with open(path) as fh:
            return Calibration(**json.load(fh))


def _ridge_fit(X, y, alpha):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    k = Z.shape[1]
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(k), Z.T @ (y - y.mean()))
    return mu, sd, beta, float(y.mean())


def rank_features(feature_dicts: Sequence[Dict], n_values: Sequence[float]) -> List:
    """Pearson r of every feature vs. measured N (use to pick features)."""
    y = np.asarray(n_values, float)
    out = []
    for k in feature_dicts[0]:
        x = np.array([d.get(k, np.nan) for d in feature_dicts], float)
        ok = np.isfinite(x)
        if ok.sum() > 3 and np.std(x[ok]) > 0:
            out.append((k, float(np.corrcoef(x[ok], y[ok])[0, 1])))
    return sorted(out, key=lambda t: -abs(t[1]))


def fit_calibration(
    feature_dicts: Sequence[Dict],
    n_values: Sequence[float],
    features: Sequence[str] = DEFAULT_CAL_FEATURES,
    ridge_alpha: float = 1.0,
    color_correction: str = "gray_card",
    notes: str = "",
) -> Calibration:
    """
    feature_dicts: list of extract_features(...)["features"] for each sample.
    n_values:      lab-measured leaf N (% dry weight) for the same samples.
    Uses ridge regression + leave-one-out CV to report honest accuracy.
    """
    X = np.array([[d[f] for f in features] for d in feature_dicts], float)
    y = np.asarray(n_values, float)
    if len(y) < len(features) + 5:
        raise ValueError(f"need at least {len(features) + 5} samples; aim for 40+")
    if not np.all(np.isfinite(X)):
        raise ValueError("non-finite feature values; drop bad photos")

    preds = np.empty_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        mu, sd, beta, b0 = _ridge_fit(X[m], y[m], ridge_alpha)
        preds[i] = b0 + ((X[i] - mu) / sd) @ beta
    resid = y - preds
    cv_rmse = float(np.sqrt(np.mean(resid ** 2)))
    cv_r2 = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))

    mu, sd, beta, b0 = _ridge_fit(X, y, ridge_alpha)
    return Calibration(
        features=list(features), mean=mu.tolist(), std=sd.tolist(),
        coef=beta.tolist(), intercept=b0,
        train_min=X.min(axis=0).tolist(), train_max=X.max(axis=0).tolist(),
        y_min=float(y.min()), y_max=float(y.max()), n_samples=int(len(y)),
        cv_rmse=cv_rmse, cv_r2=cv_r2, ridge_alpha=ridge_alpha,
        color_correction=color_correction, notes=notes,
        feature_p90={f: float(np.percentile(X[:, j], 90))
                     for j, f in enumerate(features)},
    )


# =============================================================================
# Main tool entry point
# =============================================================================
def estimate_nitrogen(
    image: ArrayLike,
    leaf_mask: ArrayLike,
    calibration: Optional[Union[Calibration, str]] = None,
    reference_features: Optional[Dict] = None,
    gray_card_mask: Optional[ArrayLike] = None,
    gray_card_reflectance: float = 0.18,
    checker_measured_srgb: Optional[np.ndarray] = None,
    checker_reference_srgb: Optional[np.ndarray] = None,
) -> Dict:
    """
    Agent-facing calculator. Always returns a JSON-serializable dict and never
    raises for bad input (errors are reported in the result).
    """
    try:
        ex = extract_features(image, leaf_mask, gray_card_mask, gray_card_reflectance,
                              checker_measured_srgb, checker_reference_srgb)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}

    f, q = ex["features"], ex["quality"]
    warnings: List[str] = []
    result = {"status": "ok", "mode": "index_only", "nitrogen": None,
              "sufficiency": None, "confidence": "low",
              "quality": q, "features": _round(f), "warnings": warnings}

    # ---- quality gates ----
    if q["valid_pixels"] < MIN_VALID_PIXELS:
        result["status"] = "rejected"
        warnings.append(f"only {q['valid_pixels']} usable leaf pixels "
                        f"(< {MIN_VALID_PIXELS}); move closer or reduce glare")
        return result
    if q["glare_fraction"] > 0.15:
        warnings.append("high flash glare (>15% of leaf); angle the flash or diffuse it")
    if q["color_correction"] == "none":
        warnings.append("no colour reference in photo; results depend on flash "
                        "distance/exposure. Include an 18% gray card.")
    if f.get("frac_brownish", 0) > 0.10:
        warnings.append("brown/necrotic tissue detected; colour may reflect disease "
                        "or senescence, not only N")

    # ---- mode 1: calibrated ----
    if calibration is not None:
        cal = Calibration.load(calibration) if isinstance(calibration, str) else calibration
        if cal.color_correction != q["color_correction"]:
            warnings.append(f"calibration used '{cal.color_correction}' colour correction "
                            f"but this photo used '{q['color_correction']}'")
        n = cal.predict(f)
        extrap = []
        for j, name in enumerate(cal.features):
            lo, hi = cal.train_min[j], cal.train_max[j]
            margin = 0.10 * (hi - lo)
            if not (lo - margin <= f[name] <= hi + margin):
                extrap.append(name)
        if extrap:
            warnings.append(f"features outside calibration range: {extrap} (extrapolating)")
        if all(f[k] >= cal.feature_p90.get(k, math.inf) for k in cal.features if k != "lab_a"):
            warnings.append("very dark green canopy: RGB greenness saturates at high N, "
                            "so surplus N cannot be reliably distinguished from sufficient N")
        half = 1.96 * cal.cv_rmse
        result.update(mode="calibrated", nitrogen={
            "value": round(n, 3),
            "unit": "% dry weight (leaf N)",
            "approx_interval_95": [round(n - half, 3), round(n + half, 3)],
            "calibration_cv_r2": round(cal.cv_r2, 3),
            "calibration_cv_rmse": round(cal.cv_rmse, 3),
            "calibration_n_samples": cal.n_samples,
        })
        conf = "high"
        if extrap or cal.cv_r2 < 0.7 or q["valid_fraction"] < 0.7 or q["color_correction"] == "none":
            conf = "medium"
        if (extrap and cal.cv_r2 < 0.7) or cal.cv_r2 < 0.5 or cal.n_samples < 20:
            conf = "low"
        result["confidence"] = conf

    # ---- mode 2: sufficiency index vs reference plant ----
    if reference_features is not None:
        si = f["dgci"] / reference_features["dgci"] if reference_features["dgci"] else float("nan")
        if si >= 1.05:
            status = "greener_than_reference (check reference plant / lighting)"
        elif si >= SI_SUFFICIENT:
            status = "sufficient"
        elif si >= SI_DEFICIENT:
            status = "marginal (approaching deficiency)"
        else:
            status = "deficient"
        result["sufficiency"] = {
            "sufficiency_index_dgci": round(si, 3),
            "status": status,
            "thresholds": {"sufficient": SI_SUFFICIENT, "deficient": SI_DEFICIENT},
            "note": "thresholds borrowed from chlorophyll-meter practice; "
                    "validate for rocket",
        }
        if result["mode"] == "index_only":
            result["mode"] = "reference_sufficiency"
            result["confidence"] = "medium" if q["color_correction"] != "none" else "low"

    if result["mode"] == "index_only":
        warnings.append("no calibration or reference supplied: returning colour "
                        "indices only, no N value")
    return result


def _round(d: Dict, nd: int = 4) -> Dict:
    return {k: (round(v, nd) if isinstance(v, float) and math.isfinite(v) else None)
            for k, v in d.items()}


# Tool schema the agent framework can register (OpenAI/Anthropic style).
TOOL_SPEC = {
    "name": "rocket_nitrogen_calculator",
    "description": ("Estimate leaf nitrogen status of rocket/arugula from a flash RGB "
                    "photo and a leaf mask. Returns N % (if calibrated), a sufficiency "
                    "index (if a reference plant is given), colour indices, quality "
                    "flags and confidence."),
    "input_schema": {
        "type": "object",
        "properties": {
            "image_path": {"type": "string"},
            "leaf_mask_path": {"type": "string"},
            "gray_card_mask_path": {"type": "string"},
            "calibration_path": {"type": "string"},
            "reference_features_path": {"type": "string"},
        },
        "required": ["image_path", "leaf_mask_path"],
    },
}


def run_tool(args: Dict) -> Dict:
    """Thin wrapper matching TOOL_SPEC."""
    ref = None
    if args.get("reference_features_path"):
        with open(args["reference_features_path"]) as fh:
            ref = json.load(fh)
    return estimate_nitrogen(
        args["image_path"], args["leaf_mask_path"],
        calibration=args.get("calibration_path"),
        reference_features=ref,
        gray_card_mask=args.get("gray_card_mask_path"),
    )


# =============================================================================
# Self-test with synthetic data (python rocket_n_calculator.py --selftest)
# =============================================================================
def _synthetic_sample(n_pct, rng, size=96):
    """Fake leaf whose colour moves yellow-green -> dark green with N, under a
    random flash exposure, plus a gray card and a few glare pixels."""
    t = np.clip((n_pct - 2.9) / 2.4, 0, 1)
    pale, dark = np.array([150, 175, 70]), np.array([45, 95, 40])
    leaf_srgb = (pale + t * (dark - pale)) / 255.0
    exposure = rng.uniform(0.6, 1.4)                       # flash distance effect
    img = np.full((size, size, 3), 0.85)
    yy, xx = np.mgrid[:size, :size]
    leaf = (yy - 48) ** 2 / 40 ** 2 + (xx - 40) ** 2 / 30 ** 2 < 1
    card = (yy >= 5) & (yy < 25) & (xx >= 75) & (xx < 92)
    lin = srgb_to_linear(img)
    lin[leaf] = srgb_to_linear(leaf_srgb) + rng.normal(0, 0.004, (leaf.sum(), 3))
    lin[card] = 0.18
    lin = np.clip(lin * exposure, 0, 1)
    img = linear_to_srgb(lin)
    gl = leaf & (rng.random(leaf.shape) < 0.03)
    img[gl] = 1.0                                          # specular glare
    return (img * 255).astype(np.uint8), leaf, card


def _selftest():
    rng = np.random.default_rng(0)
    ns = rng.uniform(2.9, 5.3, 45)
    feats = []
    for n in ns:
        img, leaf, card = _synthetic_sample(n, rng)
        feats.append(extract_features(img, leaf, gray_card_mask=card)["features"])
    print("Top features by |r|:", [(k, round(r, 3)) for k, r in rank_features(feats, ns)[:6]])
    cal = fit_calibration(feats, ns, notes="synthetic self-test")
    print(f"LOOCV R2={cal.cv_r2:.3f}  RMSE={cal.cv_rmse:.3f} %N")

    img, leaf, card = _synthetic_sample(4.0, rng)
    ref_img, ref_leaf, ref_card = _synthetic_sample(5.2, rng)
    ref = extract_features(ref_img, ref_leaf, gray_card_mask=ref_card)["features"]
    res = estimate_nitrogen(img, leaf, calibration=cal, reference_features=ref,
                            gray_card_mask=card)
    print(json.dumps({k: res[k] for k in ("status", "mode", "nitrogen", "sufficiency",
                                          "confidence", "quality", "warnings")}, indent=2))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Rocket nitrogen calculator")
    p.add_argument("image", nargs="?")
    p.add_argument("mask", nargs="?")
    p.add_argument("--gray-card")
    p.add_argument("--calibration")
    p.add_argument("--reference-features")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest or not a.image:
        _selftest()
    else:
        print(json.dumps(run_tool({
            "image_path": a.image, "leaf_mask_path": a.mask,
            "gray_card_mask_path": a.gray_card, "calibration_path": a.calibration,
            "reference_features_path": a.reference_features}), indent=2))
