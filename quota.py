"""
quota.py - stay inside the Gemini free tier (or any per-minute/per-day API limit).

Google checks three limits at the same time, per project and per model:
    RPM  requests per minute
    TPM  input tokens per minute
    RPD  requests per day (resets at midnight US Pacific time)
Exceeding any one returns HTTP 429.

This module tracks all three locally, BEFORE each call:
  * RPM / TPM full  -> wait until the 60-second window frees up (if short enough)
  * RPD used up     -> mark the model exhausted until the Pacific-midnight reset
and classifies real 429 errors AFTER a call (Google's counters are the truth):
  * per-minute 429  -> respect Google's retryDelay, then retry or switch model
  * per-day 429     -> mark the model exhausted, switch to the next model

Limits for the models this app uses are built in (KNOWN_LIMITS, copied from
AI Studio > Rate limits). Change them per model with
    QUOTA_LIMITS='{"gemini-3.5-flash-lite": {"rpm": 15, "rpd": 500, "tpm": 250000}}'
and for any other model with QUOTA_RPM / QUOTA_RPD / QUOTA_TPM.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
MAX_WAIT_S = float(os.getenv("QUOTA_MAX_WAIT_S", "45"))      # longest we keep a user waiting
SAFETY = float(os.getenv("QUOTA_SAFETY", "0.9"))             # use 90% of each limit


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Free-tier limits copied from this project's AI Studio "Rate limits" page.
# Override per model with QUOTA_LIMITS, or for unknown models with QUOTA_RPM/RPD/TPM.
KNOWN_LIMITS = {
    "gemini-3.5-flash-lite": {"rpm": 15, "tpm": 250_000, "rpd": 500},
    "gemini-3.1-flash-lite": {"rpm": 15, "tpm": 250_000, "rpd": 500},
    "gemini-3.8-flash":      {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-3.7-flash":      {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-3.6-flash":      {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-3.5-flash":      {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-2.5-flash-lite": {"rpm": 10, "tpm": 250_000, "rpd": 20},
}
DEFAULT_LIMITS = {"rpm": _env_int("QUOTA_RPM", 5), "rpd": _env_int("QUOTA_RPD", 20),
                  "tpm": _env_int("QUOTA_TPM", 250_000)}
try:
    PER_MODEL = json.loads(os.getenv("QUOTA_LIMITS", "{}"))
except json.JSONDecodeError:
    PER_MODEL = {}


def short_name(model: str) -> str:
    """'google_genai:gemini-3.6-flash' -> 'gemini-3.6-flash'"""
    return model.split(":", 1)[-1]


def next_reset() -> datetime:
    """Gemini daily quotas reset at midnight Pacific time."""
    now = datetime.now(PACIFIC)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def human_wait(seconds: float, lang: str = "en") -> str:
    seconds = max(0, int(seconds))
    h, m = seconds // 3600, (seconds % 3600) // 60
    if lang == "ar":
        return f"{h} س {m} د" if h else (f"{m} د" if m else f"{seconds} ث")
    return f"{h} h {m} min" if h else (f"{m} min" if m else f"{seconds} s")


class QuotaUnavailable(Exception):
    """No model can take the request now. kind: 'day' or 'minute'."""

    def __init__(self, kind: str, wait_s: float, detail: str = ""):
        super().__init__(f"{kind} quota: wait {wait_s:.0f}s {detail}")
        self.kind, self.wait_s, self.detail = kind, wait_s, detail


@dataclass
class _ModelState:
    limits: dict
    requests: deque = field(default_factory=deque)        # timestamps, last 60 s
    tokens: deque = field(default_factory=deque)          # (timestamp, tokens), last 60 s
    day_count: int = 0
    day_key: str = ""
    blocked_until: float = 0.0                            # from a per-minute 429
    exhausted_until: Optional[datetime] = None            # from RPD


class QuotaGuard:
    """Process-wide (all users share one API key, so they share the quota)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._models: dict[str, _ModelState] = {}

    # ----------------------------------------------------------------- state
    def _state(self, model: str) -> _ModelState:
        name = short_name(model)
        if name not in self._models:
            lim = dict(DEFAULT_LIMITS)
            lim.update(KNOWN_LIMITS.get(name, {}))
            lim.update(PER_MODEL.get(name, {}))
            self._models[name] = _ModelState(limits=lim)
        st = self._models[name]
        today = datetime.now(PACIFIC).strftime("%Y-%m-%d")
        if st.day_key != today:                              # new Pacific day
            st.day_key, st.day_count, st.exhausted_until = today, 0, None
        now = time.time()
        while st.requests and now - st.requests[0] >= 60:
            st.requests.popleft()
        while st.tokens and now - st.tokens[0][0] >= 60:
            st.tokens.popleft()
        return st

    # ----------------------------------------------------------------- checks
    def wait_needed(self, model: str, est_tokens: int) -> tuple[float, str]:
        """Seconds to wait before this call is safe, and which limit causes it.
        Returns (inf, 'rpd') when the daily quota is used up."""
        with self._lock:
            st = self._state(model)
            now = time.time()
            lim = st.limits
            if st.exhausted_until and datetime.now(PACIFIC) < st.exhausted_until:
                return float("inf"), "rpd"
            if st.day_count >= int(lim["rpd"] * SAFETY + 0.5) and lim["rpd"] > 0:
                st.exhausted_until = next_reset()
                return float("inf"), "rpd"
            waits = []
            if st.blocked_until > now:
                waits.append((st.blocked_until - now, "retry"))
            rpm_cap = max(1, int(lim["rpm"] * SAFETY))
            if len(st.requests) >= rpm_cap:
                waits.append((60 - (now - st.requests[len(st.requests) - rpm_cap]) + 0.5, "rpm"))
            tpm_cap = lim["tpm"] * SAFETY
            used = sum(t for _, t in st.tokens)
            if used + est_tokens > tpm_cap and st.tokens:
                # wait until enough old tokens leave the window
                need, freed = used + est_tokens - tpm_cap, 0
                for ts, t in st.tokens:
                    freed += t
                    if freed >= need:
                        waits.append((60 - (now - ts) + 0.5, "tpm"))
                        break
                else:
                    waits.append((60.5, "tpm"))
            if not waits:
                return 0.0, ""
            return max(waits)

    def reserve(self, model: str, est_tokens: int):
        """Count a call as soon as it is sent (so parallel users see it)."""
        with self._lock:
            st = self._state(model)
            now = time.time()
            st.requests.append(now)
            st.tokens.append((now, est_tokens))
            st.day_count += 1

    def settle(self, model: str, est_tokens: int, actual_tokens: Optional[int]):
        """Replace the estimate with Google's reported token count."""
        if actual_tokens is None:
            return
        with self._lock:
            st = self._state(model)
            for i in range(len(st.tokens) - 1, -1, -1):
                ts, t = st.tokens[i]
                if t == est_tokens:
                    st.tokens[i] = (ts, int(actual_tokens))
                    break

    def on_rate_limit(self, model: str, kind: str, retry_s: Optional[float]):
        with self._lock:
            st = self._state(model)
            if kind == "day":
                st.exhausted_until = next_reset()
            else:
                st.blocked_until = time.time() + (retry_s if retry_s else 60)

    def status(self, models: Optional[list] = None) -> dict:
        with self._lock:
            out = {}
            names = [short_name(m) for m in models] if models else list(self._models)
            for name in names:
                st = self._state(name)
                out[name] = {
                    "rpm_used": len(st.requests), "rpm": st.limits["rpm"],
                    "tpm_used": sum(t for _, t in st.tokens), "tpm": st.limits["tpm"],
                    "rpd_used": st.day_count, "rpd": st.limits["rpd"],
                    "exhausted": bool(st.exhausted_until and datetime.now(PACIFIC) < st.exhausted_until),
                    "reset_in_s": (next_reset() - datetime.now(PACIFIC)).total_seconds(),
                }
            return out


GUARD = QuotaGuard()


# --------------------------------------------------------------------- 429s
_RETRY_RE = re.compile(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", re.I)
_RETRY_IN_RE = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I)


def classify_rate_limit(err: BaseException) -> tuple[str, Optional[float]]:
    """Return ('day' | 'minute', retry_seconds) from a 429 error.

    Google puts the violated quota in the error details, e.g.
      quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier      -> day
      quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier   -> minute
      quotaId: GenerateContentInputTokensPerModelPerMinute-FreeTier   -> minute
    and a RetryInfo block with retryDelay: '23s'.
    """
    text = str(err) + " " + json.dumps(getattr(err, "details", None) or "", default=str)
    retry = None
    m = _RETRY_RE.search(text) or _RETRY_IN_RE.search(text)
    if m:
        retry = float(m.group(1))
    low = text.lower()
    if "perday" in low or "per_day" in low or "requests_per_day" in low or "daily" in low:
        return "day", retry
    return "minute", retry


def is_rate_limit(err: BaseException) -> bool:
    try:
        from langchain_core.exceptions import ModelRateLimitError
        if isinstance(err, ModelRateLimitError):
            return True
    except ImportError:
        pass
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    return code == 429 or "429" in str(err)[:40] or "RESOURCE_EXHAUSTED" in str(err)


def estimate_tokens(messages, system_prompt: str = "", tools_overhead: int = 700) -> int:
    """Rough input-token estimate: ~4 characters per token, ~260 tokens per image
    (a 768 px image is one tile for Gemini)."""
    chars, images = len(system_prompt), 0
    for m in messages:
        c = getattr(m, "content", "")
        if isinstance(c, str):
            chars += len(c)
        else:
            for b in c:
                if isinstance(b, dict):
                    if b.get("type") in ("image", "image_url"):
                        images += 1
                    else:
                        chars += len(str(b.get("text", "")))
        for tc in getattr(m, "tool_calls", []) or []:
            chars += len(json.dumps(tc.get("args", {}), default=str))
    return int(chars / 4) + images * 260 + tools_overhead
