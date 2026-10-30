"""Compare two ways of waking Jarvie for stage 1 ("jarvie, wake up", ...):

    always-on  bot.py runs all day; Deepgram transcribes everything and the bot wakes when the
               transcript contains a stage-1 phrase.
    daemon     only the local wake-word model listens; when it fires, it starts bot.py, which then
               runs until an idle timeout.
    daemon + confirm
               as daemon, but before greeting, the bot checks Deepgram's transcript of the wake audio
               for a real stage-1 phrase and goes straight back to sleep if there isn't one. Filters
               near-misses ("jacky wake up", "wakie wake") the model can't, without retraining. The
               daemon must keep the last ~2 s of audio and hand it to the bot, because the phrase was
               spoken before the bot started.

Two questions, answered separately:
    accuracy   on the same labeled clips: Deepgram transcript matching vs the local model
    cost       per day: speech-to-text minutes and LLM tokens for each design, from usage assumptions

Stage 2 (the short "jarvie" while the bot is already awake) is the same in both designs, so it's
left out.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from ..wake_phrases import JARVIE_SPELLINGS, STAGE1_PHRASES

# Stage-1 phrases and the spellings Deepgram may use for "jarvie" (not a dictionary word) live in
# jarvie/wake_phrases.py, shared with bot.py. "Jarvis" is not one of them: it must not wake Jarvie.
# Matching is done on a normalized transcript: lowercase, punctuation removed, spaces collapsed.
FOLLOW_WORD_SPELLINGS = {"wakie": ["wakie", "wakey", "wacky", "waky", "wake"]}


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def phrase_pattern(
    phrases: list[str] = STAGE1_PHRASES, jarvie_spellings: list[str] = JARVIE_SPELLINGS
) -> re.Pattern:
    """One regex matching any stage-1 phrase, with any spelling of "jarvie" (and "wakie").

    Optional comma/pause between the name and the command is already removed by normalize().
    """
    alternatives = []
    for phrase in phrases:
        words = normalize(phrase).split()
        parts = []
        for w in words:
            if w == "jarvie":
                parts.append("(?:" + "|".join(map(re.escape, jarvie_spellings)) + ")")
            elif w in FOLLOW_WORD_SPELLINGS:
                parts.append("(?:" + "|".join(map(re.escape, FOLLOW_WORD_SPELLINGS[w])) + ")")
            else:
                parts.append(re.escape(w))
        alternatives.append(r"\s".join(parts))
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b")


def transcript_detects(transcript: str, pattern: re.Pattern | None = None) -> bool:
    return bool((pattern or phrase_pattern()).search(normalize(transcript)))


# --- Deepgram transcription (cached) -------------------------------------------------------------

Transcriber = Callable[[bytes], str]


def deepgram_transcriber(api_key: str, model: str = "nova-3-general") -> Transcriber:
    """Prerecorded transcription with the same model the bot streams with."""
    from deepgram import DeepgramClient

    client = DeepgramClient(api_key=api_key)

    def transcribe(audio: bytes) -> str:
        response = client.listen.v1.media.transcribe_file(
            request=audio, model=model, smart_format=True, punctuate=True
        )
        channels = getattr(getattr(response, "results", None), "channels", None) or []
        if not channels or not channels[0].alternatives:
            return ""
        return channels[0].alternatives[0].transcript or ""

    return transcribe


def transcribe_clips(files: list[Path], transcribe: Transcriber, cache_dir: Path) -> dict[str, str]:
    """Transcript per file name. Cached by file content, so each clip is only sent once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in files:
        audio = f.read_bytes()
        cached = cache_dir / f"{hashlib.sha1(audio).hexdigest()}.json"
        if cached.exists():
            out[f.name] = json.loads(cached.read_text())["transcript"]
            continue
        transcript = transcribe(audio)
        cached.write_text(json.dumps({"file": f.name, "transcript": transcript}))
        out[f.name] = transcript
    return out


def add_labels(scored: pd.DataFrame) -> pd.DataFrame:
    """Split pace out of `condition` ("loud-fast") and similarity out of `kind` ("neg-similar"),
    as recorded by scratch/record.py. Missing pace means normal; untiered negatives are "unknown"."""
    df = scored.copy()
    cond = df.condition.str.split("-", n=1, expand=True).reindex(columns=[0, 1])
    df["condition"], df["pace"] = cond[0], cond[1].fillna("normal")
    kind = df.kind.str.split("-", n=1, expand=True).reindex(columns=[0, 1])
    df["kind"] = kind[0]
    df["similarity"] = kind[1].where(df.kind == "neg").fillna("")
    df.loc[(df.kind == "neg") & (df.similarity == ""), "similarity"] = "unknown"
    return df


def compare_detections(
    clips: pd.DataFrame,
    transcripts: dict[str, str],
    threshold: float,
    pattern: re.Pattern | None = None,
) -> pd.DataFrame:
    """Per-clip decisions of both designs.

    `clips` needs columns file, kind, pace, similarity, max_score (as built by the eval notebook).
    """
    pattern = pattern or phrase_pattern()
    df = clips[["file", "kind", "pace", "similarity", "max_score"]].copy()
    df["transcript"] = df.file.map(transcripts).fillna("")
    df["always_on_fires"] = df.transcript.map(lambda t: transcript_detects(t, pattern))
    df["daemon_fires"] = df.max_score >= threshold
    # Two-stage: the daemon wakes the bot, the bot only stays awake if the transcript has the phrase.
    df["confirmed_fires"] = df.daemon_fires & df.always_on_fires
    return df


def estimate_confirmation(decisions: pd.DataFrame) -> dict[str, float]:
    """From labeled clips: how often a false daemon start would survive the transcript check, and
    the two-stage recall on normal-pace positives. Feed these into Assumptions."""
    neg_fired = decisions[(decisions.kind == "neg") & decisions.daemon_fires]
    pos = decisions[(decisions.kind == "pos") & (decisions.pace == "normal")]
    return {
        "confirm_pass_rate": float(neg_fired.always_on_fires.mean()) if len(neg_fired) else 0.0,
        "confirmed_recall": float(pos.confirmed_fires.mean()) if len(pos) else 0.0,
    }


def accuracy_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """Recall on positives (per pace) and firing rate on negatives (per similarity tier)."""
    rows = []
    pos = decisions[decisions.kind == "pos"]
    for pace in ["all", *sorted(pos.pace.unique())]:
        d = pos if pace == "all" else pos[pos.pace == pace]
        if len(d):
            rows.append(
                {
                    "group": f"recall, {pace} pace",
                    "n": len(d),
                    "always_on": d.always_on_fires.mean(),
                    "daemon": d.daemon_fires.mean(),
                    "daemon_confirmed": d.confirmed_fires.mean(),
                }
            )
    neg = decisions[decisions.kind == "neg"]
    for tier in ["all", *sorted(neg.similarity.unique())]:
        d = neg if tier == "all" else neg[neg.similarity == tier]
        if len(d):
            rows.append(
                {
                    "group": f"false triggers, {tier}",
                    "n": len(d),
                    "always_on": d.always_on_fires.mean(),
                    "daemon": d.daemon_fires.mean(),
                    "daemon_confirmed": d.confirmed_fires.mean(),
                }
            )
    table = pd.DataFrame(rows).set_index("group")
    table["daemon - always_on"] = table.daemon - table.always_on
    return table


def disagreements(decisions: pd.DataFrame) -> pd.DataFrame:
    """Clips where exactly one design fires: the ones worth listening to."""
    d = decisions[decisions.always_on_fires != decisions.daemon_fires]
    return d[
        [
            "file",
            "kind",
            "pace",
            "similarity",
            "max_score",
            "always_on_fires",
            "daemon_fires",
            "transcript",
        ]
    ].sort_values(["kind", "file"])


# --- Token usage from bot logs -------------------------------------------------------------------

# Pipecat logs one line per LLM call when enable_usage_metrics=True (bot.py), e.g.
# "OpenAILLMService#0 prompt tokens: 1834, completion tokens: 52"
_USAGE_LINE = re.compile(r"prompt tokens: (\d+), completion tokens: (\d+)")


def parse_token_usage(log_text: str) -> pd.DataFrame:
    """Prompt/completion tokens per LLM call found in a bot log (`uv run bot.py 2>&1 | tee bot.log`)."""
    rows = [
        {"prompt_tokens": int(p), "completion_tokens": int(c)}
        for p, c in _USAGE_LINE.findall(log_text)
    ]
    return pd.DataFrame(rows, columns=["prompt_tokens", "completion_tokens"])


# --- Cost model ----------------------------------------------------------------------------------


@dataclass
class Assumptions:
    """Everything the cost comparison depends on. Prices are inputs: check them before trusting $."""

    listening_hours_per_day: float = 12.0
    activations_per_day: float = 5.0  # times you actually want Jarvie
    conversation_minutes: float = 1.5  # talking time per real activation
    idle_timeout_minutes: float = 2.0  # daemon: bot stops after this long without speech

    # LLM calls and tokens. Each bot start greets you (briefing + "introduce yourself").
    greeting_prompt_tokens: float = 1800
    greeting_completion_tokens: float = 60
    calls_per_activation: float = 3  # your turns plus tool-call round trips
    prompt_tokens_per_call: float = (
        2200  # system prompt + ~1.5k tokens of tool definitions + history
    )
    completion_tokens_per_call: float = 60

    # False activations per hour of listening. Unknown until there are hours of background audio;
    # sweep them (sweep_false_activations) rather than trusting a single value.
    always_on_false_per_hour: float = 0.05
    daemon_false_per_hour: float = 0.3
    daemon_recall: float = 0.765  # from wakeword_eval.ipynb (normal pace)
    always_on_recall: float = 1.0  # from the accuracy comparison

    # Two-stage (daemon + transcript confirmation); estimate_confirmation() fills these from clips.
    confirm_seconds: float = 5.0  # Deepgram time per start to transcribe the buffered wake audio
    confirm_pass_rate: float = 0.3  # share of false daemon starts whose transcript still matches
    confirmed_recall: float = 0.765  # daemon recall x transcript recall on what the daemon caught

    # Prices (USD). Deepgram Nova-3 streaming list price is $0.0077/min ($0.0048 promotional in 2026);
    # GPT-4.1 is $2 / $8 per million input / output tokens.
    stt_per_minute: float = 0.0077
    llm_input_per_million: float = 2.0
    llm_output_per_million: float = 8.0

    def with_measured_tokens(self, usage: pd.DataFrame) -> Assumptions:
        """Use average tokens per call from a parsed bot log (first call = greeting)."""
        if usage.empty:
            return self
        rest = usage.iloc[1:] if len(usage) > 1 else usage
        return replace(
            self,
            greeting_prompt_tokens=float(usage.prompt_tokens.iloc[0]),
            greeting_completion_tokens=float(usage.completion_tokens.iloc[0]),
            prompt_tokens_per_call=float(rest.prompt_tokens.mean()),
            completion_tokens_per_call=float(rest.completion_tokens.mean()),
        )


def daily_costs(a: Assumptions) -> pd.DataFrame:
    """Speech-to-text minutes, LLM tokens and dollars per day for each design."""

    def llm(prompt: float, completion: float) -> float:
        return prompt / 1e6 * a.llm_input_per_million + completion / 1e6 * a.llm_output_per_million

    convo_prompt = a.calls_per_activation * a.prompt_tokens_per_call
    convo_completion = a.calls_per_activation * a.completion_tokens_per_call

    # Always-on: Deepgram bills every minute it's streaming, speech or not. The bot greets once
    # when it starts; a false wake gets one LLM reply.
    ao_false = a.always_on_false_per_hour * a.listening_hours_per_day
    ao = {
        "stt_minutes": a.listening_hours_per_day * 60,
        "prompt_tokens": a.greeting_prompt_tokens
        + a.activations_per_day * convo_prompt
        + ao_false * a.prompt_tokens_per_call,
        "completion_tokens": a.greeting_completion_tokens
        + a.activations_per_day * convo_completion
        + ao_false * a.completion_tokens_per_call,
        "false_activations": ao_false,
        "attempts_per_activation": 1 / a.always_on_recall if a.always_on_recall else float("inf"),
    }

    # Daemon: Deepgram and the LLM only run while the bot is up. Every start (real or false) costs
    # a greeting and keeps the bot up until the idle timeout; missed attempts are free but repeated.
    d_false = a.daemon_false_per_hour * a.listening_hours_per_day
    starts = a.activations_per_day + d_false
    dm = {
        "stt_minutes": a.activations_per_day * (a.conversation_minutes + a.idle_timeout_minutes)
        + d_false * a.idle_timeout_minutes,
        "prompt_tokens": starts * a.greeting_prompt_tokens + a.activations_per_day * convo_prompt,
        "completion_tokens": starts * a.greeting_completion_tokens
        + a.activations_per_day * convo_completion,
        "false_activations": d_false,
        "attempts_per_activation": 1 / a.daemon_recall if a.daemon_recall else float("inf"),
    }

    # Daemon + confirmation: every start streams a few seconds to Deepgram first. Only starts whose
    # transcript has a stage-1 phrase greet you and wait for the idle timeout; the rest go straight
    # back to sleep at the cost of those few seconds.
    passed = d_false * a.confirm_pass_rate
    rejected = d_false - passed
    confirm_min = a.confirm_seconds / 60
    cf = {
        "stt_minutes": a.activations_per_day
        * (confirm_min + a.conversation_minutes + a.idle_timeout_minutes)
        + passed * (confirm_min + a.idle_timeout_minutes)
        + rejected * confirm_min,
        "prompt_tokens": (a.activations_per_day + passed) * a.greeting_prompt_tokens
        + a.activations_per_day * convo_prompt,
        "completion_tokens": (a.activations_per_day + passed) * a.greeting_completion_tokens
        + a.activations_per_day * convo_completion,
        "false_activations": passed,
        "attempts_per_activation": 1 / a.confirmed_recall if a.confirmed_recall else float("inf"),
    }

    table = pd.DataFrame({"always_on": ao, "daemon": dm, "daemon_confirmed": cf}).T
    table["stt_usd"] = table.stt_minutes * a.stt_per_minute
    table["llm_usd"] = [llm(p, c) for p, c in zip(table.prompt_tokens, table.completion_tokens)]
    table["total_usd"] = table.stt_usd + table.llm_usd
    table["total_tokens"] = table.prompt_tokens + table.completion_tokens
    return table


def savings(a: Assumptions, design: str = "daemon") -> dict[str, float]:
    """What `design` ("daemon" or "daemon_confirmed") saves compared with the always-on bot."""
    t = daily_costs(a)
    ao, dm = t.loc["always_on"], t.loc[design]
    return {
        "usd_per_day_saved": ao.total_usd - dm.total_usd,
        "usd_per_month_saved": 30 * (ao.total_usd - dm.total_usd),
        "pct_cost_saved": 100 * (1 - dm.total_usd / ao.total_usd) if ao.total_usd else 0.0,
        "stt_minutes_per_day_saved": ao.stt_minutes - dm.stt_minutes,
        "tokens_per_day_saved": ao.total_tokens - dm.total_tokens,
    }


def sweep_false_activations(
    a: Assumptions, values: list[float], design: str = "daemon"
) -> pd.DataFrame:
    """How the comparison changes with the daemon's (unknown) false activations per hour."""
    rows = []
    for v in values:
        s = savings(replace(a, daemon_false_per_hour=v), design)
        rows.append({"daemon_false_per_hour": v, **s})
    return pd.DataFrame(rows).set_index("daemon_false_per_hour")


def assumptions_table(a: Assumptions) -> pd.Series:
    return pd.Series(asdict(a))
