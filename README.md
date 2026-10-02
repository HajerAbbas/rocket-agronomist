# Rocket Agronomist (LangChain agent)

Files
- rocket_agent.py        LangChain agent: tools, system prompt, memory, safety review
- app.py + styles.css    Shiny chat app so people can try it
- leaf_core.py           leaf detection + quality checks (OpenCV)
- rocket_n_calculator.py colour features + nitrogen estimate

## Run
    pip install -r requirements.txt
    export GOOGLE_API_KEY=...             # Google AI Studio
    export OPENWEATHER_API_KEY=...        # optional
    export ROCKET_AGENT_MODEL=google_genai:gemini-3.6-flash   # optional
    shiny run app.py                      # web chat
    python rocket_agent.py leaf.jpg       # or terminal chat

## Use another model
Any LangChain provider string works, e.g.
    ROCKET_AGENT_MODEL=anthropic:claude-sonnet-4-6   (pip install langchain-anthropic)

## Publish
Posit Connect Cloud: push this folder to GitHub, publish app.py, and add
GOOGLE_API_KEY (and OPENWEATHER_API_KEY) as secret environment variables.
Never commit API keys.

## Production notes
- InMemorySaver forgets chats on restart; swap for PostgresSaver.
- Images are kept in memory; move to disk or object storage for many users.
