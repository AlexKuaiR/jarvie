# Jarvis — ambient dorm assistant

Personal AI assistant for a dorm room. Not a wake-word → STT → LLM → TTS clone: the
differentiators are the sleep-aware scheduler and the literature graph. Protect those
when making design tradeoffs.

## Architecture (planned; update as it becomes real)
- Voice layer: Pipecat pipeline, managed speech-to-speech audio
- Edge (Raspberry Pi): always-on local wake-word + speaker-ID pre-filter, GPIO silent alarm
- Tool layer: typed function-calling tools, one module per capability
- Sleep: transformer classifying sleep stages from physiological signals
- Scheduler: sleep debt + circadian phase vs. calendar density
- Literature graph: Zotero local SQLite + Semantic Scholar citations

## Layout (repo root is where Claude runs)
- `server/bot.py`       Pipecat entrypoint (the voice pipeline)
- `server/jarvis/`      Jarvis modules (sleep, scheduler, literature, hardware, tools)
- `server/scratch/`     throwaway experiments; not production, don't import from it
- `server/pyproject.toml`, `server/uv.lock`   dependencies, managed with uv
- `server/Dockerfile`, `server/pcc-deploy.toml`   Pipecat Cloud deploy config
- `server/.venv/`       interpreter + packages; never edit, never commit
- `server/CLAUDE.md`    server-specific notes; it stacks with this file, so don't repeat it here

## Commands (run from repo root; call the venv directly, do not "activate")
- Test: `server/.venv/bin/python -m pytest server -q`
- Lint: `server/.venv/bin/ruff check server`
- Format: `server/.venv/bin/ruff format server`
- Run the bot: `cd server && uv run bot.py`
- Add a dependency: `cd server && uv add <package>` (ask first; this edits pyproject and uv.lock)

## Conventions
- Python 3.11+, type hints on public functions, small modules over big ones
- Every hardware touchpoint (GPIO, mic, sensors) goes behind an interface with a
  simulated implementation, so everything runs and tests on a laptop
- Long-running processes: explicit config, structured logging, clean shutdown,
  no bare `except`
- Never read or write `.env` files (root or `server/`); secrets come from
  environment variables only
- Never touch GPIO, deploy to the Pi, or deploy to Pipecat Cloud without asking first
- Don't edit the Dockerfile or pcc-deploy.toml unless asked

## Working style
- The author's background is research notebooks, not production Python. When
  introducing a pattern (asyncio, dependency injection, packaging), add a
  2-3 line comment on why, and prefer the simplest thing that works
- For non-trivial features: propose a short plan first, wait for approval, then build
  in small steps with tests
- Don't refactor unrelated code or add dependencies without saying so

## Current focus
- Wake-word activation + real-time voice layer on Pipecat
- Sleep-stage transformer training (still experimental, not yet in `server/jarvis/`)
