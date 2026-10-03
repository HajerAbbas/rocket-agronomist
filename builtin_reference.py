"""
builtin_reference.py - a default "healthy rocket" reference, so users don't
have to photograph their own reference plant.

How it is built
---------------
1. Search Wikimedia Commons for photos of rocket (Eruca sativa / vesicaria,
   Diplotaxis tenuifolia) with open licences only (CC0, public domain, CC BY,
   CC BY-SA; no NC/ND).
2. Run every photo through the SAME pipeline as user photos (leaf detection,
   glare removal, colour features).
3. Keep photos with a clear, healthy-looking green leaf area; drop outliers.
4. Store ONLY numbers (no images) + attribution in reference_builtin.json.

Why it is "low confidence"
--------------------------
Web photos are taken with other cameras, light and editing. Brightness-based
indices (DGCI) shift with exposure, so the comparison uses the CIELAB hue angle,
which barely changes with exposure and moves toward yellow when leaves lose
chlorophyll. White balance still differs between photos, so the reference is a
wide "typical healthy colour" band, not a precise target. A reference the user
photographs with their own phone and flash is always better.

Build it once and commit the JSON (recommended):
    python builtin_reference.py --out reference_builtin.json
If the JSON is missing, the app builds it in the background at start-up.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = os.getenv(
    "WIKIMEDIA_UA",
    "RocketAgronomist/1.0 (educational prototype; set WIKIMEDIA_UA to include your contact email)")
QUERIES = [
    "Eruca sativa leaves filetype:bitmap",
    "Eruca vesicaria leaves filetype:bitmap",
    "Eruca sativa plant filetype:bitmap",
    "Eruca vesicaria plant filetype:bitmap",
    "Diplotaxis tenuifolia leaves filetype:bitmap",
    "arugula plant filetype:bitmap",
    "rocket salad plant filetype:bitmap",
    "rucola filetype:bitmap",
]
EXCLUDE_WORDS = re.compile(
    r"flower|flowers|blossom|bl[uü]te|fleur|fiore|seed|seeds|pod|herbarium|drawing|illustration|"
    r"pizza|dish|food|bowl|recipe|plate|sandwich|market|bee|insect|butterfly|map|logo", re.I)
ALLOWED_LICENSE = re.compile(r"^(cc0|public domain|pd|cc[ -]by(-sa)?[ -]?\d|cc[ -]by(-sa)?$)", re.I)
BLOCKED_LICENSE = re.compile(r"nc|nd|fair use|non-free", re.I)

FEATURES = ("lab_hue", "dgci", "lab_chroma", "ngrdi", "g_chrom")
MIN_IMAGES = 8
VERSION = 1

APP_DIR = Path(__file__).parent
SHIPPED_PATH = APP_DIR / "reference_builtin.json"
CACHE_PATH = Path(os.getenv("BUILTIN_REFERENCE_CACHE", "/tmp/rocket_reference_builtin.json"))


# ----------------------------------------------------------------- HTTP
def _get(url: str, timeout: float = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_json(params: dict) -> dict:
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    return json.loads(_get(f"{API}?{q}"))


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


# ----------------------------------------------------------------- search
def find_candidates(max_per_query: int = 40, get_json: Callable = _get_json) -> list[dict]:
    """Openly licensed rocket photos on Commons (metadata only)."""
    seen, out = set(), []
    for q in QUERIES:
        try:
            d = get_json({"action": "query", "generator": "search", "gsrsearch": q,
                          "gsrnamespace": 6, "gsrlimit": max_per_query, "prop": "imageinfo",
                          "iiprop": "url|mime|extmetadata", "iiurlwidth": 1024})
        except Exception:  # noqa: BLE001 - one failed query should not stop the build
            continue
        for p in (d.get("query") or {}).get("pages", []):
            title = p.get("title", "")
            if title in seen or EXCLUDE_WORDS.search(title):
                continue
            ii = (p.get("imageinfo") or [{}])[0]
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            meta = ii.get("extmetadata") or {}
            lic = _strip_html((meta.get("LicenseShortName") or {}).get("value", ""))
            desc = _strip_html((meta.get("ImageDescription") or {}).get("value", ""))
            if not ALLOWED_LICENSE.search(lic) or BLOCKED_LICENSE.search(lic):
                continue
            if EXCLUDE_WORDS.search(desc[:300]):
                continue
            seen.add(title)
            out.append({
                "title": title,
                "page": ii.get("descriptionurl", ""),
                "image": ii.get("thumburl") or ii.get("url"),
                "license": lic,
                "author": _strip_html((meta.get("Artist") or {}).get("value", ""))[:120],
            })
    return out


# ----------------------------------------------------------------- measure
def measure_photo(data: bytes) -> Optional[dict]:
    """Same pipeline as user photos. Returns features or None if not a clear healthy leaf."""
    from leaf_core import decode_image, detect_leaf, check_quality
    from rocket_n_calculator import extract_features
    try:
        rgb = decode_image(data)
    except ValueError:
        return None
    q = check_quality(rgb)
    if q["is_too_dark"]:
        return None
    det, mask = detect_leaf(rgb)
    if not det["leaf_detected"] or det["green_fraction"] < 0.10:
        return None
    ex = extract_features(rgb, mask)
    f, qual = ex["features"], ex["quality"]
    if qual["valid_pixels"] < 2000:
        return None
    if f.get("frac_brownish", 0) > 0.05 or f.get("frac_yellowish", 0) > 0.30:
        return None                        # flowers, dry or damaged leaves
    return {k: float(f[k]) for k in FEATURES if f.get(k) is not None}


def _iqr_keep(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1, q3 = np.percentile(values, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return (values >= lo) & (values <= hi)


def summarize(records: list[dict]) -> dict:
    hue = np.array([r["features"]["lab_hue"] for r in records])
    dg = np.array([r["features"]["dgci"] for r in records])
    keep = _iqr_keep(hue) & _iqr_keep(dg)
    kept = [r for r, k in zip(records, keep) if k]
    stats = {}
    for name in FEATURES:
        v = np.array([r["features"][name] for r in kept if name in r["features"]])
        if len(v) == 0:
            continue
        p = np.percentile(v, [10, 25, 50, 75, 90])
        stats[name] = {"p10": p[0], "p25": p[1], "median": p[2], "p75": p[3], "p90": p[4],
                       "mean": float(v.mean()), "sd": float(v.std()),
                       "values": [round(float(x), 4) for x in sorted(v)]}
    return {
        "version": VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Wikimedia Commons (open licences only); only colour statistics are stored",
        "n_images": len(kept),
        "n_rejected_outliers": len(records) - len(kept),
        "stats": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in s.items()}
                  for k, s in stats.items()},
        "photos": [{k: r[k] for k in ("title", "page", "license", "author")} for r in kept],
    }


def build(max_images: int = 60, get_json: Callable = _get_json, get_bytes: Callable = _get,
          progress: Optional[Callable[[str], None]] = None, pause_s: float = 0.25) -> dict:
    say = progress or (lambda m: None)
    cands = find_candidates(get_json=get_json)
    if not cands:
        raise RuntimeError("could not reach Wikimedia Commons or found no openly licensed photos")
    say(f"{len(cands)} openly licensed candidate photos")
    records = []
    for i, c in enumerate(cands):
        if len(records) >= max_images:
            break
        try:
            data = get_bytes(c["image"])
        except Exception:  # noqa: BLE001
            continue
        feats = measure_photo(data)
        if feats and "lab_hue" in feats and "dgci" in feats:
            records.append({**c, "features": feats})
        if pause_s:
            time.sleep(pause_s)             # be polite to Wikimedia
        if i % 10 == 9:
            say(f"checked {i + 1} photos, {len(records)} usable")
    if len(records) < MIN_IMAGES:
        raise RuntimeError(f"only {len(records)} usable photos (need {MIN_IMAGES})")
    return summarize(records)


# ----------------------------------------------------------------- compare
def percentile_of(value: float, values: list[float]) -> float:
    v = np.asarray(values, float)
    return float((v <= value).mean() * 100) if len(v) else float("nan")


def apply_builtin(r: dict, ref: dict) -> dict:
    """Add a built-in-reference comparison to an analyze_bytes() result (in place)."""
    n = r.get("nitrogen")
    if not n or n.get("status") != "ok" or n.get("mode") != "index_only":
        return r                           # calibrated / own reference wins
    f = n.get("features", {})
    st = ref["stats"].get("lab_hue")
    if f.get("lab_hue") is None or not st:
        return r
    hue = f["lab_hue"]
    pct = percentile_of(hue, st["values"])
    if hue >= st["p25"]:
        status = "within_healthy"
    elif hue >= st["p10"]:
        status = "borderline"
    else:
        status = "below_healthy"
    n["mode"] = "builtin_reference"
    n["builtin"] = {"status": status, "leaf_hue": round(hue, 2), "percentile": round(pct, 1),
                    "p10": st["p10"], "p25": st["p25"], "median": st["median"], "p90": st["p90"],
                    "n_images": ref["n_images"],
                    "dgci_band": [ref["stats"]["dgci"]["p25"], ref["stats"]["dgci"]["p75"]]
                    if "dgci" in ref["stats"] else None}
    n["confidence"] = "low"
    n.setdefault("warnings", []).append(
        f"compared with {ref['n_images']} web photos of healthy rocket taken with other cameras "
        "and light; save your own reference plant for a reliable comparison")
    return r


# ----------------------------------------------------------------- manager
class BuiltinReference:
    """Loads the shipped JSON, or builds it once in the background."""

    def __init__(self):
        self._lock = threading.Lock()
        self.data: Optional[dict] = None
        self.state, self.detail = "off", ""
        self._load()

    def _load(self):
        for p in (SHIPPED_PATH, CACHE_PATH):
            try:
                d = json.loads(p.read_text())
                if d.get("version") == VERSION and d.get("n_images", 0) >= MIN_IMAGES:
                    self.data, self.state, self.detail = d, "ready", str(p.name)
                    return
            except (OSError, json.JSONDecodeError):
                continue

    def start_background_build(self):
        if self.data or os.getenv("BUILTIN_REFERENCE_AUTOBUILD", "1") == "0":
            return
        with self._lock:
            if self.state == "building":
                return
            self.state, self.detail = "building", "searching Wikimedia Commons"
        threading.Thread(target=self._build, daemon=True).start()

    def _build(self):
        try:
            d = build(progress=lambda m: setattr(self, "detail", m))
            for p in (SHIPPED_PATH, CACHE_PATH):
                try:
                    p.write_text(json.dumps(d, indent=1))
                    break
                except OSError:
                    continue
            self.data, self.state, self.detail = d, "ready", "built"
        except Exception as e:  # noqa: BLE001
            self.state, self.detail = "failed", str(e)[:200]

    def get(self) -> Optional[dict]:
        return self.data


BUILTIN = BuiltinReference()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the built-in healthy-rocket reference")
    ap.add_argument("--out", default=str(SHIPPED_PATH))
    ap.add_argument("--max", type=int, default=60)
    a = ap.parse_args()
    d = build(max_images=a.max, progress=print)
    Path(a.out).write_text(json.dumps(d, indent=1))
    h = d["stats"]["lab_hue"]
    print(f"saved {a.out}: {d['n_images']} photos, leaf hue median {h['median']:.1f} deg "
          f"(p10 {h['p10']:.1f}, p90 {h['p90']:.1f})")
