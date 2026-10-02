# Rocket Agronomist

Check the nitrogen status of rocket (arugula) leaves with a phone camera.

## How it works
1. **Camera** (www/app.js): live preview with an in-browser leaf check
   (green-pixel coverage, brightness, steadiness), optional flash/torch,
   auto-capture, or upload.
2. **Measurement** (leaf_core.py, rocket_n_calculator.py): image quality, leaf
   detection, colour indices, nitrogen estimate. Instant, no AI.
3. **AI agent** (rocket_agent.py): LangChain + Gemini. Looks at the photo, calls
   tools (measurement, nitrogen rules, weather), returns a structured report with
   verdict, confidence, reasoning and next steps. A deterministic safety review
   checks every reply.
4. **Model steps**: every step above is streamed live to the timeline.

## Files
    app.py                 page layout + server logic
    rocket_agent.py        agent, tools, structured report, safety review
    explain.py             measurement steps + plain-language reasoning
    leaf_core.py           OpenCV leaf detection and quality checks
    rocket_n_calculator.py colour features + nitrogen estimate
    www/app.js             camera, live leaf check, steps timeline, language switch
    www/styles.css         styling
    requirements.txt

## Run locally
    pip install -r requirements.txt
    export GOOGLE_API_KEY=...          # without it the measurement still works
    export OPENWEATHER_API_KEY=...     # optional
    shiny run app.py

The camera needs HTTPS (or localhost). On Connect Cloud the app is served over HTTPS.

## Deploy (Posit Connect Cloud)
Push the whole folder, including www/, to a public GitHub repo. Publish app.py
as a Shiny app and add GOOGLE_API_KEY / OPENWEATHER_API_KEY as secret variables.
