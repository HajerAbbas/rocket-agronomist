"""
rocket_agent.py
===============
Agentic rocket-leaf nitrogen assistant built with LangChain (v1, create_agent).

Architecture
------------
            user (photo / text)
                    |
          +-------------------+        sees the photo itself (vision)
          |   LLM agent        |  <---  + memory per conversation (checkpointer)
          |  (Gemini Flash)    |
          +-------------------+
            | decides which tools to call, in which order
   +--------+-----------+------------------+---------------------+
   |                    |                  |                     |
 analyze_leaf_photo  rocket_nitrogen_   get_weather_forecast  register_reference_
 (OpenCV + calculator) rules            (OpenWeather)         plant

The measurement stays deterministic (leaf_core.py + rocket_n_calculator.py).
The LLM only interprets results, combines information and talks to the user.

Setup
-----
    pip install -r requirements.txt
    export GOOGLE_API_KEY=...            # Gemini (Google AI Studio)
    export OPENWEATHER_API_KEY=...       # optional, enables weather tool
    export ROCKET_AGENT_MODEL=google_genai:gemini-3.6-flash   # optional override
    python rocket_agent.py path/to/leaf.jpg
"""

from __future__ import annotations

import base64
import json
import os
import threading
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from typing import Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from leaf_core import (_preview_jpeg_b64, analyze_bytes, decode_image,
                       reference_features_from_bytes)

MODEL = os.getenv("ROCKET_AGENT_MODEL", "google_genai:gemini-3.6-flash")
DEFAULT_LAT, DEFAULT_LON = 24.7136, 46.6753      # Riyadh; override per user

# ---------------------------------------------------------------------------
# Image store: tools receive a short image_id instead of raw bytes, so images
# never pass through the LLM as tool arguments. image_id = "<thread>:<n>".
# ---------------------------------------------------------------------------
_IMAGES: dict[str, bytes] = {}
_REFERENCE: dict[str, dict] = {}          # thread_id -> reference features
_LAST_ANALYSIS: dict[str, dict] = {}      # thread_id -> last full analysis (for UIs)
_COUNTER = defaultdict(int)
_LOCK = threading.Lock()


def _thread_of(image_id: str) -> str:
    return image_id.rsplit(":", 1)[0]


def store_image(thread_id: str, data: bytes) -> str:
    with _LOCK:
        _COUNTER[thread_id] += 1
        image_id = f"{thread_id}:{_COUNTER[thread_id]}"
        _IMAGES[image_id] = data
    return image_id


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def analyze_leaf_photo(image_id: str) -> str:
    """Measure a leaf photo. Checks image quality (blur, darkness), detects whether a
    green leaf is present, segments it, and computes colour indices and nitrogen status.
    Call this FIRST for every new photo, using the image_id given in the message.
    Returns JSON with decision ('ok' or 'retake'), reasons, leaf_detection,
    image_quality and nitrogen results."""
    data = _IMAGES.get(image_id)
    if data is None:
        return json.dumps({"error": f"unknown image_id {image_id}"})
    ref = _REFERENCE.get(_thread_of(image_id))
    result = analyze_bytes(data, debug=True, reference_features=ref)
    _LAST_ANALYSIS[_thread_of(image_id)] = result
    slim = {k: v for k, v in result.items() if not k.endswith("_base64")}
    slim["reference_plant_registered"] = ref is not None
    return json.dumps(slim, ensure_ascii=False)


@tool
def register_reference_plant(image_id: str) -> str:
    """Save a photo as the well-fertilized REFERENCE plant for this conversation.
    Use only when the user says the photo shows a healthy, well-fertilized plant to
    compare against. Later analyses will then include a sufficiency index."""
    data = _IMAGES.get(image_id)
    if data is None:
        return json.dumps({"error": f"unknown image_id {image_id}"})
    try:
        feats = reference_features_from_bytes(data)
    except ValueError as e:
        return json.dumps({"saved": False, "error": str(e)})
    if feats is None:
        return json.dumps({"saved": False, "error": "no leaf detected in reference photo"})
    _REFERENCE[_thread_of(image_id)] = feats
    return json.dumps({"saved": True, "reference_dgci": round(feats["dgci"], 4)})


@tool
def rocket_nitrogen_rules() -> str:
    """Interpretation thresholds and safety rules for rocket nitrogen. Call this
    before giving any nitrogen or fertilizer advice."""
    return json.dumps({
        "leaf_N_percent_dry_weight": {
            "deficient_below": 3.5, "high_above": 5.0,
            "note": "PLACEHOLDER thresholds; confirm with an agronomist for your variety"},
        "sufficiency_index": {"sufficient_at_or_above": 0.95, "deficient_below": 0.90},
        "safety": [
            "Never state an exact fertilizer dose; refer to the agronomist or product label.",
            "If confidence is low, do not recommend fertilizer; recommend a re-check.",
            "Avoid fertilizing if more than 10 mm of rain is forecast within 48 hours.",
            "Excess nitrogen raises leaf nitrate in rocket; avoid extra N close to harvest.",
            "Yellowing can also come from water stress, disease, old leaves or other nutrients.",
        ],
    })


@tool
def get_weather_forecast(latitude: float = DEFAULT_LAT, longitude: float = DEFAULT_LON) -> str:
    """48-hour weather summary for the farm location (rain, temperature, humidity).
    Call before recommending fertilizer or irrigation. Defaults to Riyadh if the user
    has not given a location."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return json.dumps({"available": False, "reason": "weather service not configured"})
    q = urllib.parse.urlencode({"lat": latitude, "lon": longitude, "units": "metric",
                                "cnt": 16, "appid": key})
    try:
        with urllib.request.urlopen(f"https://api.openweathermap.org/data/2.5/forecast?{q}",
                                    timeout=10) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"available": False, "reason": str(e)})
    items = d.get("list", [])
    rain = sum(i.get("rain", {}).get("3h", 0) for i in items)
    temps = [i["main"]["temp"] for i in items] or [None]
    hums = [i["main"]["humidity"] for i in items] or [None]
    return json.dumps({
        "available": True, "location": d.get("city", {}).get("name"),
        "hours_covered": 3 * len(items), "rain_total_mm": round(rain, 1),
        "temp_max_c": max(temps), "temp_min_c": min(temps),
        "humidity_mean_pct": round(sum(hums) / len(hums)) if hums[0] is not None else None,
    })


TOOLS = [analyze_leaf_photo, register_reference_plant, rocket_nitrogen_rules,
         get_weather_forecast]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are "Rocket Agronomist", an assistant that helps farmers check the
nitrogen status of rocket (arugula: Eruca sativa or Diplotaxis tenuifolia) from leaf photos.

Each photo message starts with an [image_id: ...] tag; pass that exact id to tools. For every new photo:
1. Look at the image yourself. If it is clearly not a leaf, or not rocket, say so and ask for a
   new photo of one rocket leaf (you may still call analyze_leaf_photo to explain why).
2. If the user says the photo is a healthy, well-fertilized reference plant, call
   register_reference_plant and confirm. Otherwise call analyze_leaf_photo with the image_id.
3. If decision is "retake", explain the reasons in simple words and how to fix them. Stop there.
4. If decision is "ok", interpret the nitrogen results. Never invent or change numbers.
   - mode "calibrated": report nitrogen.nitrogen.value (% dry weight) and its interval.
   - mode "reference_sufficiency": report nitrogen.sufficiency.status.
   - mode "index_only": say the leaf was measured but nitrogen % needs calibration; describe
     greenness in plain words and suggest sending a photo of a well-fertilized reference plant.
   - Respect nitrogen.confidence and every warning.
5. Before nitrogen or fertilizer advice, call rocket_nitrogen_rules and follow its safety rules.
   Before recommending fertilizer or irrigation, call get_weather_forecast (use the user's
   location if they gave one).
6. If you see brown spots, mildew, holes or wilting, say colour may reflect disease or stress
   rather than nitrogen, and suggest checking that first.
7. Never give exact fertilizer doses. When unsure, recommend a re-check or a local agronomist.

Reply style: short and phone-friendly (max ~8 lines): verdict first, then key numbers with
confidence, then 1-3 concrete next steps. Reply in the user's language; default to Arabic.
"""


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------
def _message_text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


# ---------------------------------------------------------------------------
# Safety review: deterministic check on every final reply (the "Safety Gate")
# ---------------------------------------------------------------------------
import re

_DOSE_RE = re.compile(r"\d+(\.\d+)?\s*(kg|g|ml|l|كجم|كغ|غرام|جرام|مل|لتر)\b\s*(/|per|لكل)", re.I)
_APPLY_RE = re.compile(r"(apply|add|fertili[sz]e|سمّد|سمد|أضف|ضع سماد)", re.I)


def safety_review(reply: str, analysis: Optional[dict]) -> str:
    notes = []
    if _DOSE_RE.search(reply):
        notes.append("⚠️ تأكد من الكمية مع المهندس الزراعي أو ملصق المنتج. / "
                     "Confirm any amount with your agronomist or the product label.")
    conf = ((analysis or {}).get("nitrogen") or {}).get("confidence")
    if conf == "low" and _APPLY_RE.search(reply):
        notes.append("⚠️ الثقة منخفضة: أعد الفحص قبل التسميد. / "
                     "Low confidence: re-check before fertilizing.")
    return reply + ("\n\n" + "\n".join(notes) if notes else "")


class RocketAgent:
    def __init__(self, model=MODEL, checkpointer=None):
        self.agent = create_agent(
            model=model,
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer or InMemorySaver(),
        )

    def _run(self, thread_id: str, content) -> str:
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            {"configurable": {"thread_id": thread_id}, "recursion_limit": 12},
        )
        return safety_review(_message_text(result["messages"][-1]),
                             _LAST_ANALYSIS.get(thread_id))

    def send_photo(self, thread_id: str, image_bytes: bytes, caption: str = "") -> str:
        """Store the photo, show a downscaled copy to the LLM, let the agent work."""
        image_id = store_image(thread_id, image_bytes)
        try:
            preview = _preview_jpeg_b64(decode_image(image_bytes), max_side=768)
        except ValueError:
            preview = None
        text = f"[image_id: {image_id}] {caption or 'Please check this leaf.'}"
        content = [{"type": "text", "text": text}]
        if preview:
            content.append({"type": "image", "base64": preview, "mime_type": "image/jpeg"})
        return self._run(thread_id, content)

    def send_text(self, thread_id: str, text: str) -> str:
        return self._run(thread_id, text)

    @staticmethod
    def last_analysis(thread_id: str) -> Optional[dict]:
        return _LAST_ANALYSIS.get(thread_id)


# ---------------------------------------------------------------------------
# CLI: python rocket_agent.py leaf.jpg  (then chat; empty line to quit)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    bot = RocketAgent()
    tid = str(uuid.uuid4())
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as fh:
            print("\nAgent:", bot.send_photo(tid, fh.read(), " ".join(sys.argv[2:])))
    while True:
        try:
            q = input("\nYou: ").strip()
        except EOFError:
            break
        if not q:
            break
        print("\nAgent:", bot.send_text(tid, q))
