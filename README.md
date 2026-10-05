# Jarvie

An ambient voice assistant for a dorm room, built on [Pipecat](https://github.com/pipecat-ai/pipecat).

The goal isn't another wake word → speech-to-text → LLM → text-to-speech clone. What sets Jarvie
apart are two planned pieces: a **sleep-aware scheduler** that weighs sleep debt and circadian phase
against how packed your calendar is, and a **literature graph** built from your Zotero library and
its citations. Today Jarvie handles voice, your calendar and academic deadlines; the rest is on the
roadmap below.

> **Note:** voice activation currently only responds to **"Jarvis"**. The wake word will be updated
> to match the project's new name, Jarvie, in a later release.

## What works today

- **Voice conversation** with a spoken wake word ("Jarvis"). Deepgram speech-to-text → OpenAI
  (`gpt-4.1` by default) → Cartesia text-to-speech, with Silero voice-activity detection and
  Pipecat's local smart-turn model deciding when you've finished speaking.
- **Calendar**: "What do I have tomorrow?" reads your Google Calendar (read-only).
- **Deadlines**: tell Jarvie about exams, assignments and applications and it tracks them in a
  local SQLite database. You can add, list, update, complete and mute them by voice.
- **Startup briefing**: when you connect, Jarvie mentions the most urgent deadlines due this week.
  Say "got it" to mute one for 3 days. Anything due today or tomorrow is always mentioned, even if
  muted.

## Roadmap

### Planned features
- **Sleep-stage model**: a transformer that classifies sleep stages from physiological signals
  (wearable data). Training is experimental and lives outside the bot for now.
- **Sleep-aware scheduler**: balances sleep debt and circadian phase against calendar density, so
  study sessions and reminders fit around how rested you actually are.
- **Literature graph**: your local Zotero library (SQLite) plus Semantic Scholar citations, so you
  can ask about papers, authors and what cites what.
- **Raspberry Pi edge device**: an always-on local wake-word detector with speaker identification,
  so only your voice wakes Jarvie and no audio leaves the room until it does. Also a GPIO-driven
  silent alarm (e.g. light or vibration instead of sound, so a roommate isn't woken).

### Claude Code integration (voice-driven development)
A longer-term idea: update Jarvie itself by voice. You'd say something like *"Jarvis, add a tool
that tells me when the library closes"*, and Jarvie would hand the request to
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) running on this repo.

Sketch of how it could work:
1. A `code_change` tool in Jarvie captures the spoken request and confirms it back to you.
2. It starts Claude Code headless (`claude -p "<request>"`, or the Claude Agent SDK) in the repo,
   on a new git branch, with a restricted set of allowed tools.
3. Claude Code makes the change and runs the test suite and linter. The repo's `CLAUDE.md` and
   `.claude/settings.json` already define the conventions and the permission rules it works within.
4. Jarvie reads back a short spoken summary: what changed, and whether the tests passed.
5. Nothing is merged, pushed or deployed until you approve it, by voice or at your laptop.

Guardrails this needs before it's safe: speaker-ID gating (so a roommate or a video playing can't
trigger code changes), work on a branch only, an explicit spoken confirmation before any commit or
push, and no access to secrets, GPIO or deploys.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

1. **Install dependencies**
   ```bash
   cd server
   uv sync
   ```
   On Linux, also install PortAudio for audio I/O: `sudo apt install libportaudio2`
   (macOS and Windows get it bundled with `sounddevice`).

2. **Set API keys** as environment variables, or in `server/.env` (git-ignored; start from
   `server/.env.example`):

   | Variable | Needed for |
   | --- | --- |
   | `DEEPGRAM_API_KEY` | speech-to-text |
   | `OPENAI_API_KEY` | the LLM |
   | `OPENAI_MODEL` | optional, defaults to `gpt-4.1` |
   | `CARTESIA_API_KEY` | text-to-speech |
   | `CARTESIA_VOICE_ID` | optional, picks the voice |

   Values in `server/.env` override variables already set in your shell.

3. **Connect Google Calendar**: create an OAuth client ("Desktop app") in Google Cloud with the
   Calendar API enabled, and save its JSON as `~/jarvie-data/credentials.json`. The first calendar
   request opens a browser to sign in; the token is saved to `~/jarvie-data/token.json`.

4. **Run the bot**
   ```bash
   uv run bot.py                     # SmallWebRTC, open the printed local URL
   uv run bot.py --transport daily   # Daily
   ```
   Say "Jarvis" to start talking.

Local data (the deadlines database and Google credentials) lives in `~/jarvie-data/`, outside the
repo.

## Development

From the repo root:

```bash
server/.venv/bin/python -m pytest server -q   # tests (Google and the real database are faked)
server/.venv/bin/ruff check server            # lint (import order)
server/.venv/bin/ruff format server           # format
```

`CLAUDE.md` holds the project conventions for working on this repo with Claude Code.

## Project structure

```
.
├── CLAUDE.md                # Conventions for Claude Code
├── .claude/settings.json    # Claude Code permissions and hooks
└── server/
    ├── bot.py               # Pipecat voice pipeline (entry point)
    ├── jarvie/
    │   ├── calendar.py      # Google Calendar access
    │   ├── deadlines.py     # Deadline tracking (add/list/update/mute/complete)
    │   ├── db.py            # SQLite schema and connection
    │   ├── config.py        # Timezone and ~/jarvie-data paths
    │   └── tools.py         # Tools the LLM can call
    ├── tests/               # pytest suite
    ├── scratch/             # Throwaway experiments, not imported by the bot
    ├── pyproject.toml, uv.lock
    └── Dockerfile, pcc-deploy.toml   # Pipecat Cloud deploy config
```

## Deploying to Pipecat Cloud

The project includes Pipecat Cloud config, but it isn't deployable as-is: the `Dockerfile` copies
only `bot.py` (not `jarvie/`), and the calendar needs files from `~/jarvie-data/`, which won't exist
in the container. See the
[Pipecat Cloud docs](https://docs.pipecat.ai/deployment/pipecat-cloud/introduction) once those are
sorted out.

## Learn more

- [Pipecat documentation](https://docs.pipecat.ai/)
- [Pipecat examples](https://github.com/pipecat-ai/pipecat-examples)
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code)

## License

[MIT](LICENSE)
