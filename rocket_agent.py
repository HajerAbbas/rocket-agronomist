"""
rocket_agent.py - the LangChain agent behind the app.

    photo ──► deterministic measurement (leaf_core)  ──► cached per image
                         │
    Gemini agent (create_agent) ── sees the photo, calls tools:
        analyze_leaf_photo     reads the cached measurement
        rocket_nitrogen_rules  thresholds + safety rules
        get_weather_forecast   OpenWeather 48 h summary
    ──► LeafReport (structured: verdict, status, confidence, reasoning, next steps)
    ──► safety_review (deterministic)

Every step is reported through an async `emit(event)` callback so the UI can
show the agent's progress live.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Awaitable, Callable, List, Literal, Optional

from pydantic import BaseModel, Field

from leaf_core import _preview_jpeg_b64, decode_image

MODEL = os.getenv("ROCKET_AGENT_MODEL", "google_genai:gemini-3.6-flash")
DEFAULT_LAT = float(os.getenv("FARM_LAT", "24.7136"))     # Riyadh
DEFAULT_LON = float(os.getenv("FARM_LON", "46.6753"))

Emit = Callable[[dict], Awaitable[None]]


# =============================================================================
# Per-session storage (photos, cached measurements, reference plant)
# =============================================================================
class _Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.images: dict[str, bytes] = {}
        self.analysis: dict[str, dict] = {}
        self.reference: dict[str, dict] = {}
        self.counter = defaultdict(int)

    def add_photo(self, thread_id: str, data: bytes, analysis: dict) -> str:
        with self.lock:
            self.counter[thread_id] += 1
            image_id = f"{thread_id}:{self.counter[thread_id]}"
            self.images[image_id] = data
            self.analysis[image_id] = analysis
        return image_id

    def forget(self, thread_id: str):
        with self.lock:
            for k in [k for k in self.images if k.startswith(thread_id + ":")]:
                self.images.pop(k, None)
                self.analysis.pop(k, None)
            self.reference.pop(thread_id, None)


STORE = _Store()


def set_reference(thread_id: str, feats: Optional[dict]):
    if feats is None:
        STORE.reference.pop(thread_id, None)
    else:
        STORE.reference[thread_id] = feats


def get_reference(thread_id: str) -> Optional[dict]:
    return STORE.reference.get(thread_id)


def slim_analysis(r: dict) -> dict:
    """What the agent sees: compact, no images."""
    n = r.get("nitrogen") or {}
    f = n.get("features") or {}
    return {
        "decision": r.get("decision"),
        "reasons": r.get("reasons"),
        "leaf_detection": r.get("leaf_detection"),
        "image_quality": r.get("image_quality"),
        "nitrogen": None if not n else {
            "mode": n.get("mode"), "nitrogen": n.get("nitrogen"),
            "sufficiency": n.get("sufficiency"), "confidence": n.get("confidence"),
            "warnings": n.get("warnings"),
            "key_features": {k: f.get(k) for k in
                             ("dgci", "ngrdi", "lab_a", "frac_yellowish", "frac_brownish")},
        },
    }


# =============================================================================
# Tools
# =============================================================================
def _make_tools():
    from langchain.tools import tool

    @tool
    def analyze_leaf_photo(image_id: str) -> str:
        """Read the measurement of a leaf photo: image quality, whether a leaf was
        detected, colour indices and nitrogen status. Call this first for every
        photo, using the image_id given in the message."""
        r = STORE.analysis.get(image_id)
        if r is None:
            return json.dumps({"error": f"unknown image_id {image_id}"})
        out = slim_analysis(r)
        out["reference_plant_registered"] = get_reference(image_id.rsplit(":", 1)[0]) is not None
        return json.dumps(out, ensure_ascii=False)

    @tool
    def rocket_nitrogen_rules() -> str:
        """Nitrogen interpretation thresholds and safety rules for rocket.
        Call before giving any nitrogen or fertilizer advice."""
        return json.dumps({
            "leaf_N_percent_dry_weight": {
                "deficient_below": 3.5, "high_above": 5.0,
                "note": "PLACEHOLDER thresholds; confirm with an agronomist"},
            "sufficiency_index": {"sufficient_at_or_above": 0.95, "deficient_below": 0.90},
            "safety": [
                "Never state an exact fertilizer dose; refer to the agronomist or product label.",
                "If confidence is low, do not recommend fertilizer; recommend a re-check.",
                "Do not fertilize if more than 10 mm of rain is forecast within 48 hours.",
                "Excess nitrogen raises leaf nitrate in rocket; avoid extra N close to harvest.",
                "Yellowing can also come from water stress, heat, disease, old leaves or other nutrients.",
            ],
        })

    @tool
    def get_weather_forecast(latitude: float = DEFAULT_LAT, longitude: float = DEFAULT_LON) -> str:
        """48-hour weather summary (rain, temperature, humidity) for the farm.
        Call before recommending fertilizer or irrigation. Defaults to Riyadh."""
        key = os.getenv("OPENWEATHER_API_KEY")
        if not key:
            return json.dumps({"available": False, "reason": "weather service not configured"})
        q = urllib.parse.urlencode({"lat": latitude, "lon": longitude, "units": "metric",
                                    "cnt": 16, "appid": key})
        try:
            with urllib.request.urlopen(
                    f"https://api.openweathermap.org/data/2.5/forecast?{q}", timeout=10) as resp:
                d = json.load(resp)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"available": False, "reason": str(e)})
        items = d.get("list", [])
        if not items:
            return json.dumps({"available": False, "reason": "empty forecast"})
        temps = [i["main"]["temp"] for i in items]
        return json.dumps({
            "available": True, "location": d.get("city", {}).get("name"),
            "hours_covered": 3 * len(items),
            "rain_total_mm": round(sum(i.get("rain", {}).get("3h", 0) for i in items), 1),
            "temp_max_c": max(temps), "temp_min_c": min(temps),
            "humidity_mean_pct": round(sum(i["main"]["humidity"] for i in items) / len(items)),
        })

    return [analyze_leaf_photo, rocket_nitrogen_rules, get_weather_forecast]


# =============================================================================
# Structured report = the agent's visible reasoning
# =============================================================================
class LeafReport(BaseModel):
    """Final report for the user. Always fill every field."""
    verdict: str = Field(description="One short headline, e.g. 'Leaf detected – nitrogen looks low'.")
    n_status: Literal["deficient", "marginal", "sufficient", "high", "unknown",
                      "retake", "not_rocket", "question"] = Field(
        description="'question' when answering a follow-up question without a new photo.")
    confidence: Literal["high", "medium", "low"]
    reasoning: List[str] = Field(
        description="2-5 short steps explaining how you reached the verdict, each citing concrete "
                    "evidence (a tool value, the weather, or what you saw in the photo).")
    next_steps: List[str] = Field(description="1-3 concrete actions for the farmer.")
    answer: str = Field(description="Friendly reply to the user, max ~6 lines.")


SYSTEM_PROMPT = """You are "Rocket Agronomist", an assistant that helps farmers check the
nitrogen status of rocket (arugula: Eruca sativa or Diplotaxis tenuifolia) from leaf photos.

For a new photo (the message contains [image_id: ...]):
1. Look at the photo yourself. If it is clearly not a leaf or not rocket, use n_status
   "not_rocket" and ask for a photo of one rocket leaf.
2. Call analyze_leaf_photo with that image_id. It is the ONLY source of numbers;
   never invent or change values.
   - decision "retake": use n_status "retake"; explain the reasons and how to fix them.
   - mode "calibrated": use nitrogen.nitrogen.value (% dry weight) and its interval.
   - mode "reference_sufficiency": use nitrogen.sufficiency.status.
   - mode "index_only": n_status "unknown"; the leaf was measured but nitrogen % needs
     calibration or a reference plant. Describe the greenness in plain words.
   - Respect nitrogen.confidence and all warnings.
3. Before nitrogen or fertilizer advice call rocket_nitrogen_rules and follow it.
   Before recommending fertilizer or irrigation call get_weather_forecast.
4. If you see brown spots, mildew, holes or wilting, say colour may reflect disease or
   stress rather than nitrogen.
5. Never give exact fertilizer doses.

For follow-up questions without a photo use n_status "question".

The reasoning field is shown to the user as "Why": short, factual, evidence-based steps.
Write every text field in the language requested in the message (Arabic or English)."""


# =============================================================================
# Agent construction
# =============================================================================
_AGENT = None
_MODEL_OVERRIDE = None
_LOCK = threading.Lock()


def set_model(model):
    """Inject a chat model (tests / other providers)."""
    global _MODEL_OVERRIDE, _AGENT
    _MODEL_OVERRIDE, _AGENT = model, None


def agent_status() -> tuple[bool, str]:
    if _MODEL_OVERRIDE is not None:
        return True, "custom model"
    if MODEL.startswith("google") and not os.getenv("GOOGLE_API_KEY"):
        return False, "GOOGLE_API_KEY is not set"
    return True, MODEL


def _agent():
    global _AGENT
    with _LOCK:
        if _AGENT is None:
            from langchain.agents import create_agent
            from langchain.agents.structured_output import ToolStrategy
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            serde = JsonPlusSerializer(allowed_msgpack_modules=[("rocket_agent", "LeafReport")])
            _AGENT = create_agent(
                model=_MODEL_OVERRIDE or MODEL,
                tools=_make_tools(),
                system_prompt=SYSTEM_PROMPT,
                response_format=ToolStrategy(LeafReport),
                checkpointer=InMemorySaver(serde=serde),
            )
        return _AGENT


# =============================================================================
# Safety review (deterministic)
# =============================================================================
_DOSE_RE = re.compile(r"\d+(\.\d+)?\s*(kg|g|ml|l|كجم|كغ|غرام|جرام|مل|لتر)\b\s*(/|per|لكل)", re.I)
_APPLY_RE = re.compile(
    r"(\b(apply|add|spread|give)\b[^.\n]{0,30}\b(fertili[sz]er|nitrogen|urea|npk)\b"
    r"|\bfertili[sz]e\b|سمّد|ضع سماد|أضف سماد|أضف السماد|أضف النيتروجين)", re.I)


def safety_review(report: dict, analysis: Optional[dict]) -> List[str]:
    text = " ".join([report.get("answer", "")] + report.get("next_steps", []))
    notes = []
    if _DOSE_RE.search(text):
        notes.append("dose")
    conf = ((analysis or {}).get("nitrogen") or {}).get("confidence")
    if conf == "low" and _APPLY_RE.search(text):
        notes.append("low_confidence")
    if analysis and analysis.get("decision") == "retake" and report.get("n_status") not in (
            "retake", "not_rocket"):
        notes.append("retake_ignored")
    return notes


# =============================================================================
# Running a turn with live step events
# =============================================================================
TOOL_LABELS = {
    "analyze_leaf_photo": ("read", "Reading the measurement", "قراءة نتائج القياس"),
    "rocket_nitrogen_rules": ("rules", "Checking nitrogen rules & safety limits",
                              "مراجعة قواعد النيتروجين وحدود السلامة"),
    "get_weather_forecast": ("weather", "Checking the 48-hour weather", "فحص الطقس لـ 48 ساعة"),
    "LeafReport": ("report", "Writing the report", "كتابة التقرير"),
}


def _summarize_tool(name: str, content: str, lang: str) -> str:
    try:
        d = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return ""
    ar = lang == "ar"
    if name == "get_weather_forecast":
        if not d.get("available"):
            return "الطقس غير متاح" if ar else "Weather unavailable"
        return (f"{d.get('location') or ''} مطر {d['rain_total_mm']} مم، {d['temp_min_c']:.0f}–{d['temp_max_c']:.0f}°م"
                if ar else
                f"{d.get('location') or ''} rain {d['rain_total_mm']} mm, {d['temp_min_c']:.0f}–{d['temp_max_c']:.0f} °C")
    if name == "analyze_leaf_photo":
        n = d.get("nitrogen") or {}
        return f"decision={d.get('decision')}, mode={n.get('mode')}, confidence={n.get('confidence')}"
    if name == "rocket_nitrogen_rules":
        return "5 safety rules, thresholds 3.5–5.0 % N" if not ar else "5 قواعد سلامة، حدود 3.5–5.0٪"
    return ""


async def run_turn(thread_id: str, text: str, lang: str = "en",
                   image_id: Optional[str] = None, emit: Optional[Emit] = None) -> dict:
    """One user turn. Returns {"report": dict|None, "safety": [...], "error": str|None}."""
    from langchain.messages import HumanMessage

    async def _emit(**ev):
        if emit:
            await emit(ev)

    ar = lang == "ar"
    content = [{"type": "text", "text": (
        f"[Reply language: {'Arabic' if ar else 'English'}]\n"
        + (f"[image_id: {image_id}] " if image_id else "") + (text or ""))}]
    if image_id and image_id in STORE.images:
        try:
            prev = _preview_jpeg_b64(decode_image(STORE.images[image_id]), max_side=768)
            content.append({"type": "image", "base64": prev, "mime_type": "image/jpeg"})
        except ValueError:
            pass

    await _emit(id="think", group="agent", status="running",
                label="الوكيل يفحص الصورة ويخطط" if ar and image_id else
                "الوكيل يقرأ سؤالك" if ar else
                "Agent looks at the photo and plans" if image_id else "Agent reads your question")
    t0 = time.perf_counter()
    report, think_done = None, False
    try:
        async for chunk in _agent().astream(
                {"messages": [HumanMessage(content=content)]},
                {"configurable": {"thread_id": thread_id}, "recursion_limit": 14},
                stream_mode="updates"):
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                for m in upd.get("messages", []) or []:
                    if not think_done and m.type == "ai":
                        think_done = True
                        await _emit(id="think", group="agent", status="done",
                                    ms=int((time.perf_counter() - t0) * 1000))
                    if m.type == "ai":
                        for tc in getattr(m, "tool_calls", []) or []:
                            key, en, a = TOOL_LABELS.get(tc["name"], (tc["name"], tc["name"], tc["name"]))
                            await _emit(id=tc["id"] or key, group="agent", status="running",
                                        label=a if ar else en)
                    elif m.type == "tool":
                        status = "done" if getattr(m, "status", "success") != "error" else "error"
                        await _emit(id=m.tool_call_id, group="agent", status=status,
                                    detail=_summarize_tool(m.name, m.content, lang))
                if upd.get("structured_response") is not None:
                    sr = upd["structured_response"]
                    report = sr.model_dump() if hasattr(sr, "model_dump") else dict(sr)
    except Exception as e:  # noqa: BLE001
        await _emit(id="think", group="agent", status="error", detail=type(e).__name__)
        return {"report": None, "safety": [], "error": f"{type(e).__name__}: {e}"}

    analysis = STORE.analysis.get(image_id) if image_id else None
    safety = safety_review(report or {}, analysis)
    await _emit(id="safety", group="agent", status="warn" if safety else "done",
                label="مراجعة السلامة" if ar else "Safety review",
                detail=(", ".join(safety) if safety else ("لا توجد مخاوف" if ar else "No issues found")))
    return {"report": report, "safety": safety, "error": None if report else "no report"}
