"""
Rocket Agronomist - Shiny for Python app.

  1  Camera (live leaf check in the browser) or upload
  2  Measurement (Python, instant)  +  AI agent (LangChain + Gemini, streamed steps)
  3  Follow-up chat with memory

Run:  shiny run app.py
Env:  GOOGLE_API_KEY (required for the AI agent), OPENWEATHER_API_KEY (optional)
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from shiny import App, reactive, render, ui

import rocket_agent as ra
from explain import T as XT, STATUS_WORD, measurement_reasoning, measurement_steps, suff_key
from leaf_core import _preview_jpeg_b64, analyze_bytes, decode_image, reference_features_from_bytes

WWW = Path(__file__).parent / "www"
DGCI_MIN, DGCI_MAX = 0.30, 0.75

TXT = {
    "en": {
        "why_measure": "How the measurement was read",
        "agent_title": "AI agent assessment",
        "agent_wait": "The agent is reviewing the photo, the measurement and the weather…",
        "agent_off": "AI agent is offline ({why}). The measurement above still works.",
        "agent_err": "The AI agent could not finish ({err}). The measurement above is still valid.",
        "why": "Why", "next": "Next steps", "conf": "Confidence",
        "conf_v": {"high": "High", "medium": "Medium", "low": "Low"},
        "safety": {"dose": "Check any amount with your agronomist or the product label.",
                   "low_confidence": "Confidence is low: re-check before fertilizing.",
                   "retake_ignored": "The measurement asked for a retake; treat this advice with caution."},
        "greenness": "Leaf greenness", "paler": "Paler", "darker": "Darker green",
        "you": "This leaf", "ref": "Reference",
        "leaf_area": "Leaf area", "sharpness": "Sharpness", "glare": "Glare", "ok": "OK",
        "blurry": "Blurry", "n_label": "Nitrogen", "cal_needed": "Needs calibration",
        "tap": "Tap the photo to switch between the original and the detected leaf.",
        "ref_hint": "Saved on this device. Now take or upload a photo of the plant you want to check.",
        "empty": "Take a photo of one rocket leaf to start. The steps and the reasoning will appear here.",
        "tips": "Tips", "retake_tips": {
            "no_leaf_detected": "Point the camera at a single rocket leaf.",
            "image_blurry": "Hold the phone steady and tap the screen to focus.",
            "image_too_dark": "Turn on the flash or move to better light.",
            "leaf_too_small_move_closer": "Move closer so the leaf fills the frame.",
            "nitrogen_rejected": "Tilt the phone slightly to avoid flash glare."},
        "greet": "Hi! After you analyze a leaf you can ask me follow-up questions here, "
                 "for example: *Should I fertilize before the weekend?*",
        "offline_chat": "The AI agent is offline, so I can't answer questions right now.",
    },
    "ar": {
        "why_measure": "كيف تمت قراءة القياس",
        "agent_title": "تقييم الوكيل الذكي",
        "agent_wait": "الوكيل يراجع الصورة والقياس والطقس…",
        "agent_off": "الوكيل الذكي غير متصل ({why}). القياس أعلاه يعمل.",
        "agent_err": "تعذّر على الوكيل الإكمال ({err}). القياس أعلاه ما زال صالحًا.",
        "why": "لماذا", "next": "الخطوات التالية", "conf": "الثقة",
        "conf_v": {"high": "عالية", "medium": "متوسطة", "low": "منخفضة"},
        "safety": {"dose": "تأكد من أي كمية مع المهندس الزراعي أو ملصق المنتج.",
                   "low_confidence": "الثقة منخفضة: أعد الفحص قبل التسميد.",
                   "retake_ignored": "القياس طلب إعادة التصوير؛ تعامل مع هذه النصيحة بحذر."},
        "greenness": "درجة اخضرار الورقة", "paler": "أفتح", "darker": "أخضر داكن",
        "you": "هذه الورقة", "ref": "المرجع",
        "leaf_area": "مساحة الورقة", "sharpness": "الوضوح", "glare": "الانعكاس", "ok": "جيد",
        "blurry": "غير واضح", "n_label": "النيتروجين", "cal_needed": "يحتاج معايرة",
        "tap": "اضغط على الصورة للتبديل بين الأصلية والورقة المكتشفة.",
        "ref_hint": "تم الحفظ على هذا الجهاز. الآن التقط أو ارفع صورة للنبات الذي تريد فحصه.",
        "empty": "التقط صورة لورقة جرجير واحدة للبدء. ستظهر الخطوات والتفسير هنا.",
        "tips": "نصائح", "retake_tips": {
            "no_leaf_detected": "وجّه الكاميرا نحو ورقة جرجير واحدة.",
            "image_blurry": "ثبّت الهاتف واضغط على الشاشة للتركيز.",
            "image_too_dark": "شغّل الفلاش أو انتقل إلى إضاءة أفضل.",
            "leaf_too_small_move_closer": "اقترب حتى تملأ الورقة الإطار.",
            "nitrogen_rejected": "أمِل الهاتف قليلًا لتجنب انعكاس الفلاش."},
        "greet": "مرحبًا! بعد تحليل الورقة يمكنك طرح أسئلة متابعة هنا، مثل: *هل أسمّد قبل نهاية الأسبوع؟*",
        "offline_chat": "الوكيل الذكي غير متصل، لذلك لا يمكنني الإجابة الآن.",
    },
}

ICON_SHUTTER = '<span class="shutter-ring"></span>'
ICON_SWITCH = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h11l-3-3M20 17H9l3 3" '
               'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ICON_FLASH = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2 4 14h7l-1 8 9-12h-7z" '
              'fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>')
ICON_UPLOAD = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m0 0-5 5m5-5 5 5M4 20h16" '
               'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')

CAMERA_HTML = f"""
<div id="cam" class="cam" data-state="off">
  <div class="cam-stage">
    <video id="cam-video" playsinline muted autoplay></video>
    <img id="cam-still" alt="" hidden>
    <div class="cam-guide" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
    <div class="cam-off" id="cam-off">
      <p data-i18n="cam_off_text">Point your camera at one rocket leaf. The app checks live whether a leaf is in view.</p>
      <button type="button" class="btn-primary" id="cam-start" data-i18n="cam_start">Start camera</button>
      <p class="cam-off-small" id="cam-unsupported" hidden data-i18n="cam_unsupported">Camera not available here. Use Upload instead.</p>
    </div>
    <div class="cam-hud">
      <div class="cam-status" id="cam-status" role="status" aria-live="polite"></div>
      <div class="cam-meter" aria-hidden="true"><span id="cam-meter-fill"></span></div>
    </div>
    <div class="cam-busy" id="cam-busy" hidden><span class="spinner"></span><span data-i18n="analyzing">Analyzing…</span></div>
  </div>

  <div class="cam-controls cam-live-controls">
    <button type="button" class="btn-icon" id="cam-switch" title="Switch camera">{ICON_SWITCH}<span data-i18n="switch">Switch</span></button>
    <button type="button" class="shutter" id="cam-capture" aria-label="Capture photo" disabled>{ICON_SHUTTER}</button>
    <button type="button" class="btn-icon" id="cam-flash" title="Flash" hidden>{ICON_FLASH}<span data-i18n="flash">Flash</span></button>
    <label class="btn-icon" id="cam-upload-label">{ICON_UPLOAD}<span data-i18n="upload">Upload</span>
      <input type="file" id="cam-file" accept="image/*" hidden></label>
  </div>
  <div class="cam-still-controls">
    <button type="button" class="btn-primary btn-wide" id="cam-analyze" data-i18n="analyze">Analyze leaf</button>
    <div class="still-row">
      <button type="button" class="btn-ghost" id="cam-retake" data-i18n="retake">Retake</button>
      <label class="btn-ghost" id="cam-upload2-label"><span data-i18n="upload_another">Upload another</span>
        <input type="file" id="cam-file2" accept="image/*" hidden></label>
      <button type="button" class="btn-ghost btn-ref" id="cam-save-ref" data-i18n="save_ref">Save as reference</button>
    </div>
  </div>
  <label class="auto"><input type="checkbox" id="cam-auto"> <span data-i18n="auto">Auto-capture when the leaf is steady</span></label>
</div>
"""

REF_PANEL_HTML = """
<div id="ref-panel" class="ref-panel" data-has="0">
  <div class="ref-head"><h3 data-i18n="ref_title">Reference plant</h3>
    <span class="ref-device" data-i18n="ref_device">Saved on this device</span></div>
  <div class="ref-empty"><p data-i18n="ref_none">No reference yet. Photograph a healthy, well-fertilized
    plant and tap “Save as reference”. Later photos are compared with it.</p></div>
  <div class="ref-saved">
    <img id="ref-thumb" alt="">
    <div class="ref-info"><p class="ref-line" id="ref-line"></p>
      <button type="button" class="btn-link" id="ref-remove" data-i18n="ref_remove">Remove reference</button></div>
  </div>
</div>
"""

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;700&display=swap"),
        ui.tags.link(rel="stylesheet", href="styles.css"),
        ui.tags.title("Rocket Agronomist"),
    ),
    ui.tags.header(
        ui.div(
            ui.h1("Rocket Agronomist", class_="brand", **{"data-i18n": "brand"}),
            ui.p("Check rocket leaf nitrogen with your phone camera", class_="tagline",
                 **{"data-i18n": "tagline"}),
        ),
        ui.div(
            ui.tags.button("English", type="button", class_="lang-btn is-on", **{"data-lang": "en"}),
            ui.tags.button("العربية", type="button", class_="lang-btn", **{"data-lang": "ar"}),
            class_="lang", role="group", **{"aria-label": "Language"},
        ),
        class_="topbar",
    ),
    ui.div(
        # ---------------- column 1: capture ----------------
        ui.tags.section(
            ui.h2(ui.span("1", class_="num"), ui.span("Take a photo", **{"data-i18n": "s1"})),
            ui.HTML(CAMERA_HTML),
            ui.div(
                ui.input_text("note", None, placeholder="Optional note, e.g. older leaves look yellow",
                              width="100%"),
                class_="options",
            ),
            ui.HTML(REF_PANEL_HTML),
            class_="panel capture",
        ),
        # ---------------- column 2: results ----------------
        ui.tags.section(
            ui.h2(ui.span("2", class_="num"), ui.span("Results & reasoning", **{"data-i18n": "s2"})),
            ui.div(
                ui.div(ui.h3("Model steps", **{"data-i18n": "steps"}),
                       ui.span("", id="steps-title", class_="steps-title"), class_="steps-head"),
                ui.tags.ol(id="steps", class_="steps"),
                class_="steps-box",
            ),
            ui.output_ui("measurement"),
            ui.output_ui("agent_report"),
            ui.h2(ui.span("3", class_="num"), ui.span("Ask a follow-up", **{"data-i18n": "s3"}),
                  class_="h2-chat"),
            ui.chat_ui("chat", greeting=TXT["en"]["greet"] + "\n\n" + TXT["ar"]["greet"],
                       height="380px", width="100%", drawer=False, show_history=False,
                       allow_attachments=False,
                       placeholder="Ask about this leaf, fertilizer or weather…"),
            class_="panel results",
        ),
        class_="layout",
    ),
    ui.tags.footer(ui.p("Research prototype. Estimates from leaf colour, not a lab test.",
                        **{"data-i18n": "disclaimer"})),
    ui.tags.script(src="app.js"),
    class_="shell",
)


# =============================================================================
# Rendering helpers
# =============================================================================
def _gauge(dgci, ref_dgci, t):
    def pos(v):
        return max(0.0, min(100.0, (v - DGCI_MIN) / (DGCI_MAX - DGCI_MIN) * 100))
    marks = [ui.div(ui.span(t["you"], class_="mark-label"), class_="mark mark-you",
                    style=f"inset-inline-start:{pos(dgci):.1f}%")]
    if ref_dgci is not None:
        marks.append(ui.div(ui.span(t["ref"], class_="mark-label"), class_="mark mark-ref",
                            style=f"inset-inline-start:{pos(ref_dgci):.1f}%"))
    return ui.div(ui.h4(t["greenness"]), ui.div(ui.div(class_="track"), *marks, class_="gauge"),
                  ui.div(ui.span(t["paler"]), ui.span(t["darker"]), class_="gauge-ends"),
                  class_="gauge-block")


def _chip(label, value, kind=""):
    return ui.div(ui.span(label, class_="chip-l"), ui.span(value, class_="chip-v"), class_=f"chip {kind}")


def _shot(r, t):
    a, b = r.get("image_jpeg_base64"), r.get("mask_overlay_jpeg_base64")
    if not a:
        return None
    det = (r.get("leaf_detection") or {}).get("leaf_detected")
    src_a = f"data:image/jpeg;base64,{a}"
    if det and b:
        src_b = f"data:image/jpeg;base64,{b}"
        return ui.tags.figure(
            ui.tags.img(src=src_b, class_="shot shot-toggle", alt=t["tap"],
                        **{"data-a": src_a, "data-b": src_b, "tabindex": "0", "role": "button"}),
            ui.tags.figcaption(t["tap"]), class_="shot-fig")
    return ui.tags.figure(ui.tags.img(src=src_a, class_="shot", alt=""), class_="shot-fig")


# =============================================================================
# Server
# =============================================================================
def server(input, output, session):
    tid = session.id
    current = reactive.value(None)       # dict: analysis, image_id, kind ("leaf"|"reference")
    agent_wanted = reactive.value(False)
    chat = ui.Chat("chat")

    def lang():
        try:
            v = input.lang()
        except Exception:  # noqa: BLE001  (not sent yet)
            v = None
        return v if v in ("en", "ar") else "en"

    async def emit(ev: dict):
        await session.send_custom_message("step", ev)

    @reactive.extended_task
    async def agent_job(image_id: str, note: str, lang_: str):
        return await ra.run_turn(tid, note or "Please check this leaf.", lang_, image_id, emit)

    # ---------------- photo submitted from the camera component -------------
    @reactive.effect
    @reactive.event(input.photo_submit)
    async def _on_photo():
        payload = input.photo_submit()
        if not payload or "data" not in payload:
            return
        L = lang()
        xt = XT[L]
        data = base64.b64decode(payload["data"].split(",", 1)[-1])
        is_ref = bool(payload.get("as_reference"))
        agent_job.cancel()
        agent_wanted.set(False)

        await session.send_custom_message("steps_reset", {
            "title": xt["step_ref"] if is_ref else ("تحليل صورة" if L == "ar" else "Photo analysis")})
        await emit(dict(id="quality", group="measure", status="running", label=xt["step_quality"]))

        ref = ra.get_reference(tid)
        r = await asyncio.to_thread(analyze_bytes, data, True, None if is_ref else ref)
        for ev in measurement_steps(r, L):
            await emit(ev)

        if is_ref:
            feats = await asyncio.to_thread(reference_features_from_bytes, data) \
                if (r.get("leaf_detection") or {}).get("leaf_detected") else None
            if feats:
                ra.set_reference(tid, feats)
                thumb = await asyncio.to_thread(lambda: _preview_jpeg_b64(decode_image(data), 240))
                await session.send_custom_message("reference_saved", {
                    "features": {k: v for k, v in feats.items() if isinstance(v, (int, float))},
                    "thumb": thumb})
                await emit(dict(id="ref", group="measure", status="done", label=xt["step_ref"],
                                detail=xt["ref_saved"].format(d=f"{feats['dgci']:.3f}")))
            else:
                await emit(dict(id="ref", group="measure", status="error", label=xt["step_ref"],
                                detail=xt["ref_failed"]))
                await session.send_custom_message("reference_failed", {})
            current.set({"analysis": r, "image_id": None, "kind": "reference", "saved": bool(feats)})
        else:
            image_id = ra.STORE.add_photo(tid, data, r)
            current.set({"analysis": r, "image_id": image_id, "kind": "leaf", "ref": ref})
            ok, why = ra.agent_status()
            if ok:
                agent_wanted.set(True)
                agent_job.invoke(image_id, input.note(), L)
            else:
                await emit(dict(id="think", group="agent", status="skip",
                                label="AI agent" if L == "en" else "الوكيل الذكي", detail=why))
        await session.send_custom_message("analysis_done", {"decision": r.get("decision")})

    # ---------------- outputs ----------------
    @reactive.effect
    @reactive.event(input.stored_reference, ignore_none=False)
    def _restore_reference():
        """The browser keeps the reference (localStorage) and sends it on every page load."""
        try:
            payload = input.stored_reference()
        except Exception:  # noqa: BLE001
            payload = None
        feats = (payload or {}).get("features") if isinstance(payload, dict) else None
        clean = None
        if isinstance(feats, dict):
            clean = {k: float(v) for k, v in feats.items()
                     if isinstance(v, (int, float)) and abs(float(v)) < 1e6}
            if not (0.0 < clean.get("dgci", -1) < 1.5):
                clean = None
        ra.set_reference(tid, clean)

    @render.ui
    def measurement():
        L = lang()
        t, xt = TXT[L], XT[L]
        c = current()
        if c is None:
            return ui.div(ui.p(t["empty"]), class_="empty")
        r = c["analysis"]
        det = r.get("leaf_detection") or {}
        q = r.get("image_quality") or {}
        n = r.get("nitrogen") or {}
        ref = c.get("ref")

        if c["kind"] == "reference":
            title, kind = (xt["v_refsaved"], "ok") if c["saved"] else (xt["ref_failed"], "bad")
            body = [ui.p(t["ref_hint"], class_="hint")] if c["saved"] else []
            return ui.div(ui.div(ui.h3(title), class_=f"verdict verdict-{kind}"), *body,
                          _shot(r, t), class_="card")

        if r["decision"] != "ok":
            title = xt["v_noleaf"] if not det.get("leaf_detected") else xt["v_retake"]
            tips = [t["retake_tips"].get(x) for x in dict.fromkeys(r.get("reasons", []))]
            tips = [x for x in tips if x]
            return ui.div(
                ui.div(ui.h3(title), class_="verdict verdict-bad"),
                ui.tags.ul(*[ui.tags.li(x) for x in tips], class_="tips-list") if tips else None,
                _shot(r, t),
                ui.tags.details(ui.tags.summary(t["why_measure"]),
                                ui.tags.ol(*[ui.tags.li(x) for x in measurement_reasoning(r, ref, L)],
                                           class_="why-list")),
                class_="card")

        f = n.get("features", {})
        nq = n.get("quality", {})
        if n.get("mode") == "calibrated" and n.get("nitrogen"):
            n_val, n_kind = f"{n['nitrogen']['value']:.2f}%", ""
        elif n.get("sufficiency"):
            k = suff_key(n["sufficiency"]["status"])
            n_val = STATUS_WORD[L][k]
            n_kind = {"deficient": "bad", "marginal": "warn"}.get(k, "ok")
        else:
            n_val, n_kind = t["cal_needed"], "muted"

        return ui.div(
            ui.div(ui.h3(xt["v_leaf"]), class_="verdict verdict-ok"),
            ui.div(
                _chip(t["leaf_area"], f"{det['green_fraction'] * 100:.0f}%"),
                _chip(t["sharpness"], t["ok"] if not q["is_blurry"] else t["blurry"]),
                _chip(t["glare"], f"{nq.get('glare_fraction', 0) * 100:.1f}%"),
                _chip(t["n_label"], n_val, n_kind),
                class_="chips"),
            _gauge(f["dgci"], ref["dgci"] if ref else None, t) if f.get("dgci") is not None else None,
            _shot(r, t),
            ui.tags.details(ui.tags.summary(t["why_measure"]),
                            ui.tags.ol(*[ui.tags.li(x) for x in measurement_reasoning(r, ref, L)],
                                       class_="why-list"),
                            open=True),
            class_="card")

    @render.ui
    def agent_report():
        L = lang()
        t = TXT[L]
        c = current()
        if c is None or c["kind"] != "leaf":
            return None
        ok, why = ra.agent_status()
        if not ok:
            return ui.div(ui.p(t["agent_off"].format(why=why)), class_="card card-muted")
        if not agent_wanted():
            return None
        st = agent_job.status()
        if st == "running":
            return ui.div(ui.h3(t["agent_title"]),
                          ui.div(ui.span(class_="spinner"), ui.span(t["agent_wait"]), class_="waiting"),
                          class_="card agent")
        if st == "error":
            return ui.div(ui.p(t["agent_err"].format(err="error")), class_="card card-muted")
        if st != "success":
            return None
        res = agent_job.result()
        rep = res.get("report")
        if not rep:
            return ui.div(ui.p(t["agent_err"].format(err=res.get("error") or "no report")),
                          class_="card card-muted")
        conf = rep.get("confidence", "low")
        status_kind = {"deficient": "bad", "retake": "bad", "not_rocket": "bad",
                       "marginal": "warn", "high": "warn"}.get(rep.get("n_status"), "ok")
        return ui.div(
            ui.div(ui.span(t["agent_title"], class_="eyebrow"), ui.h3(rep.get("verdict", "")),
                   class_=f"verdict verdict-{status_kind}"),
            ui.div(ui.span(t["conf"]),
                   ui.div(*[ui.span(class_="seg on" if i < {"low": 1, "medium": 2, "high": 3}[conf] else "seg")
                            for i in range(3)], class_="segs"),
                   ui.span(t["conf_v"][conf], class_="conf-v"), class_="conf"),
            ui.p(rep.get("answer", ""), class_="answer"),
            ui.div(ui.h4(t["why"]), ui.tags.ol(*[ui.tags.li(x) for x in rep.get("reasoning", [])],
                                               class_="why-list"), class_="why"),
            ui.div(ui.h4(t["next"]), ui.tags.ul(*[ui.tags.li(x) for x in rep.get("next_steps", [])],
                                                class_="next-list"), class_="next"),
            *[ui.p("⚠️ " + t["safety"][s], class_="safety") for s in res.get("safety", [])],
            class_="card agent")

    # ---------------- follow-up chat ----------------
    @chat.on_user_submit
    async def _(user_input: str):
        L = lang()
        ok, _why = ra.agent_status()
        if not ok:
            await chat.append_message(TXT[L]["offline_chat"])
            return
        await session.send_custom_message("steps_reset", {
            "title": "سؤال متابعة" if L == "ar" else "Follow-up question"})
        res = await ra.run_turn(tid, user_input, L, None, emit)
        rep = res.get("report")
        if not rep:
            await chat.append_message(TXT[L]["agent_err"].format(err=res.get("error")))
            return
        why = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(rep.get("reasoning", [])))
        msg = rep.get("answer", "")
        if why:
            msg += f"\n\n**{TXT[L]['why']}:**\n{why}"
        for s in res.get("safety", []):
            msg += f"\n\n⚠️ {TXT[L]['safety'][s]}"
        await chat.append_message(msg)

    @session.on_ended
    def _cleanup():
        ra.STORE.forget(tid)


app = App(app_ui, server, static_assets=WWW)
