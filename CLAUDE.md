# Jarvis — ambient dorm assistant

Personal AI assistant for a dorm room. Not a wake-word → STT → LLM → TTS clone: the
differentiators are the sleep-aware scheduler and the literature graph. Protect those
when making design tradeoffs.

## Architecture
Current (exists in `server/`):
- Voice layer: Pipecat cascade pipeline in `server/bot.py`: Deepgram STT → OpenAI LLM
  (`OPENAI_MODEL`, default `gpt-4.1`) → Cartesia TTS, with Silero VAD and local smart-turn v3
- Wake word: server-side phrase match on the STT transcript ("jarvis",
  `WakePhraseUserTurnStartStrategy`); transports are Daily or SmallWebRTC
- Tool layer: typed function-calling tools in `server/jarvis/tools.py` (`ALL_TOOLS`):
  Google Calendar events plus deadline add/list/update/mute/complete
- Storage: SQLite at `~/jarvis-data/jarvis.db` (`server/jarvis/db.py`)

Planned (not built yet; move to Current as it becomes real):
- Edge (Raspberry Pi): always-on local wake-word + speaker-ID pre-filter, GPIO silent alarm
- Sleep: transformer classifying sleep stages from physiological signals (the `sleep`
  table in `db.py` is a placeholder and is unused)
- Scheduler: sleep debt + circadian phase vs. calendar density
- Literature graph: Zotero local SQLite + Semantic Scholar citations

## Layout (repo root is where Claude runs)
- `server/bot.py`       Pipecat entrypoint (the voice pipeline)
- `server/jarvis/`      Jarvis modules: `calendar` (Google Calendar), `deadlines`, `db`
                        (SQLite), `config` (timezone + `~/jarvis-data` paths), `tools`
                        (LLM tools). Sleep, scheduler, literature and hardware modules are planned
- `server/scratch/`     throwaway experiments; not production, don't import from it
- `server/pyproject.toml`, `server/uv.lock`   dependencies, managed with uv
- `server/Dockerfile`, `server/pcc-deploy.toml`   Pipecat Cloud deploy config
- `server/.venv/`       interpreter + packages; never edit, never commit

## Commands (run from repo root; call the venv directly, do not "activate")
- Test: `server/.venv/bin/python -m pytest server -q` (no tests exist yet; pytest
  exits 5 with "no tests collected")
- Lint: `server/.venv/bin/ruff check server` (ruff selects only `I`, so this checks
  import order only)
- Format: `server/.venv/bin/ruff format server`
- Run the bot: `cd server && uv run bot.py`
- Add a dependency: `cd server && uv add <package>` (ask first; this edits pyproject and uv.lock)

## Conventions
- Python 3.10+ (`requires-python` in pyproject; the venv runs 3.12), type hints on public functions, small modules over big ones
- Every hardware touchpoint (GPIO, mic, sensors) goes behind an interface with a
  simulated implementation, so everything runs and tests on a laptop
- Long-running processes: explicit config, structured logging, clean shutdown,
  no bare `except`
- Never read or write `.env` files (root or `server/`). Where secrets come from today:
  - API keys: `server/bot.py` calls `load_dotenv(override=True)`, so values in
    `server/.env` win over variables already set in the shell
  - Google OAuth: `credentials.json` and `token.json` in `~/jarvis-data/`
    (`server/jarvis/config.py`); Claude is denied reads there
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
- Wake-word activation + real-time voice layer on Pipecat (wake word is currently
  server-side; the Pi edge detector is planned)
- Sleep-stage transformer training (still experimental, not yet in `server/jarvis/`)
