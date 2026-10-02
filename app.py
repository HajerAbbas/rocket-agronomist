"""
Rocket Agronomist - Shiny chat app for the LangChain agent.

Run locally:
    export GOOGLE_API_KEY=...          (and optionally OPENWEATHER_API_KEY)
    shiny run app.py --reload
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from shiny import App, reactive, render, ui

from rocket_agent import RocketAgent

bot = RocketAgent()
CSS = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")

GREETING = (
    "مرحبًا! أرسل صورة لورقة جرجير واحدة (مع الفلاش) وسأفحص مستوى النيتروجين.\n\n"
    "Hi! Send a flash photo of one rocket leaf and I'll check its nitrogen status. "
    "You can also send a photo of a well-fertilized plant as a reference."
)

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.link(rel="stylesheet",
                     href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;700&display=swap"),
        ui.tags.style(CSS),
        ui.tags.title("Rocket Agronomist"),
    ),
    ui.div(
        ui.div(ui.h1("Rocket Agronomist", class_="brand"),
               ui.p("مساعد الجرجير الزراعي", class_="brand-ar", lang="ar", dir="rtl"),
               class_="masthead"),
        class_="topbar",
    ),
    ui.div(
        ui.div(
            ui.input_file("photo", "Leaf photo / صورة الورقة", accept=["image/*"],
                          button_label="Take or choose photo", placeholder="No photo yet"),
            ui.input_checkbox("is_reference",
                              "This is a healthy, well-fertilized reference plant / نبات مرجعي"),
            ui.input_text("caption", "Note for the assistant (optional) / ملاحظة",
                          placeholder="e.g. older leaves look yellow"),
            ui.output_ui("analysis_panel"),
            class_="capture",
        ),
        ui.div(ui.chat_ui("chat", messages=[GREETING], height="70vh"), class_="chatcol"),
        class_="layout",
    ),
    ui.tags.footer(ui.p("Research prototype. Estimates from leaf colour, not a lab test. "
                        "نموذج بحثي، النتائج تقديرية وليست تحليلًا مخبريًا.")),
    class_="shell",
)


def server(input, output, session):
    chat = ui.Chat("chat")
    tid = session.id
    analysis_tick = reactive.value(0)

    async def ask(fn, *args):
        if not os.getenv("GOOGLE_API_KEY") and "google" in os.getenv(
                "ROCKET_AGENT_MODEL", "google_genai"):
            return "The assistant is not configured (missing GOOGLE_API_KEY)."
        try:
            return await asyncio.to_thread(fn, tid, *args)
        except Exception as e:  # noqa: BLE001
            return f"Sorry, something went wrong: {type(e).__name__}. Please try again."

    @chat.on_user_submit
    async def _(user_input: str):
        await chat.append_message(await ask(bot.send_text, user_input))

    @reactive.effect
    @reactive.event(input.photo)
    async def _():
        info = input.photo()
        if not info:
            return
        data = Path(info[0]["datapath"]).read_bytes()
        is_ref = input.is_reference()
        caption = input.caption().strip()
        if is_ref:
            caption = ("This photo is my healthy, well-fertilized REFERENCE plant. " + caption).strip()
        await chat.append_message(
            {"role": "user", "content": f"📷 {'Reference photo' if is_ref else 'Leaf photo'} sent"
                                        + (f": {caption}" if caption and not is_ref else "")})
        reply = await ask(bot.send_photo, data, caption)
        await chat.append_message(reply)
        analysis_tick.set(analysis_tick() + 1)
        ui.update_checkbox("is_reference", value=False)
        ui.update_text("caption", value="")

    @render.ui
    def analysis_panel():
        analysis_tick()
        r = RocketAgent.last_analysis(tid)
        if not r or not r.get("mask_overlay_jpeg_base64"):
            return ui.div(
                ui.h3("How to take the photo"),
                ui.tags.ul(ui.tags.li("One leaf, filling most of the frame"),
                           ui.tags.li("Flash on, phone 20–30 cm away"),
                           ui.tags.li("Plain background, no direct sunlight")),
                class_="tips")
        det = r.get("leaf_detection") or {}
        ok = det.get("leaf_detected")
        return ui.div(
            ui.h3("Last measured photo"),
            ui.tags.img(src=f"data:image/jpeg;base64,{r['mask_overlay_jpeg_base64']}",
                        class_="shot", alt="Detected leaf area"),
            ui.p(("Leaf detected" if ok else "No leaf detected")
                 + f", leaf covers {det.get('green_fraction', 0)*100:.0f}% of the frame",
                 class_="kpi-sub"),
            class_="last-shot",
        )


app = App(app_ui, server)
