import pandas as pd
import pytest

from jarvie.eval import gating


@pytest.mark.parametrize(
    "transcript",
    ["Jarvie, wake up!", "jarvy help me", "Hey Jarvie, wacky wacky.", "ok jarvee wake up now",
     "Jarvie, I need some help"],
)
def test_stage1_phrases_detected_with_stt_spellings(transcript):
    assert gating.transcript_detects(transcript)


@pytest.mark.parametrize(
    "transcript",
    ["wake up", "Jarvis.", "Jacky, wake up.", "I need some help", "jarvis what's the weather",
     "harvey help me", "Jarvis, wake up!", "jarvis help me", ""],
)
def test_other_speech_not_detected(transcript):
    assert not gating.transcript_detects(transcript)


def test_add_labels_splits_pace_and_similarity():
    scored = pd.DataFrame({
        "file": ["a", "b", "c"],
        "kind": ["pos", "neg-very-similar", "neg"],
        "condition": ["loud-fast", "loud-normal", "quiet"],
        "max_score": [0.9, 0.2, 0.1],
    })
    df = gating.add_labels(scored)
    assert list(df.kind) == ["pos", "neg", "neg"]
    assert list(df.pace) == ["fast", "normal", "normal"]
    assert list(df.similarity) == ["", "very-similar", "unknown"]
    assert list(df.condition) == ["loud", "loud", "quiet"]


def test_transcripts_cached_by_content(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "a.wav").write_bytes(b"audio-a")
    (clips / "b.wav").write_bytes(b"audio-b")
    calls = []

    def fake(audio: bytes) -> str:
        calls.append(audio)
        return f"text for {audio.decode()}"

    files = sorted(clips.glob("*.wav"))
    first = gating.transcribe_clips(files, fake, tmp_path / "cache")
    second = gating.transcribe_clips(files, fake, tmp_path / "cache")
    assert first == second == {"a.wav": "text for audio-a", "b.wav": "text for audio-b"}
    assert len(calls) == 2  # second run came from the cache


def test_accuracy_and_disagreements():
    clips = pd.DataFrame({
        "file": ["p1", "p2", "n1", "n2"],
        "kind": ["pos", "pos", "neg", "neg"],
        "pace": ["normal", "fast", "normal", "normal"],
        "similarity": ["", "", "similar", "not-similar"],
        "max_score": [0.9, 0.1, 0.8, 0.0],
    })
    transcripts = {"p1": "Jarvie, wake up.", "p2": "Jarvie wake up", "n1": "Jacky wake up", "n2": "hi"}
    decisions = gating.compare_detections(clips, transcripts, threshold=0.5)
    table = gating.accuracy_table(decisions)
    assert table.loc["recall, all pace", "always_on"] == 1.0
    assert table.loc["recall, all pace", "daemon"] == 0.5
    assert table.loc["false triggers, similar", "daemon"] == 1.0
    assert table.loc["false triggers, similar", "always_on"] == 0.0
    assert list(gating.disagreements(decisions).file) == ["n1", "p2"]


def test_parse_token_usage_from_pipecat_log():
    log = """
    2026-10-13 10:00:01 | DEBUG | OpenAILLMService#0 prompt tokens: 1834, completion tokens: 52
    2026-10-13 10:00:09 | DEBUG | something else
    2026-10-13 10:00:12 | DEBUG | OpenAILLMService#0 prompt tokens: 2101, completion tokens: 40, cache read input tokens: 1536
    """
    usage = gating.parse_token_usage(log)
    assert usage.values.tolist() == [[1834, 52], [2101, 40]]
    a = gating.Assumptions().with_measured_tokens(usage)
    assert (a.greeting_prompt_tokens, a.prompt_tokens_per_call) == (1834, 2101)
    assert gating.parse_token_usage("no usage here").empty


def test_daily_costs_by_hand():
    a = gating.Assumptions(
        listening_hours_per_day=10, activations_per_day=2, conversation_minutes=1,
        idle_timeout_minutes=2, greeting_prompt_tokens=1000, greeting_completion_tokens=0,
        calls_per_activation=1, prompt_tokens_per_call=1000, completion_tokens_per_call=0,
        always_on_false_per_hour=0.1, daemon_false_per_hour=0.5, daemon_recall=0.5,
        stt_per_minute=0.01, llm_input_per_million=1.0, llm_output_per_million=0.0,
    )
    t = gating.daily_costs(a)
    # always-on: 600 STT minutes; 1 greeting + 2 conversations + 1 false wake = 4,000 prompt tokens
    assert t.loc["always_on", "stt_minutes"] == 600
    assert t.loc["always_on", "prompt_tokens"] == 4000
    assert t.loc["always_on", "total_usd"] == pytest.approx(6.0 + 0.004)
    # daemon: 2 x (1 + 2) + 5 false x 2 = 16 minutes; 7 starts x 1000 + 2 x 1000 = 9,000 tokens
    assert t.loc["daemon", "stt_minutes"] == 16
    assert t.loc["daemon", "prompt_tokens"] == 9000
    assert t.loc["daemon", "attempts_per_activation"] == 2
    s = gating.savings(a)
    assert s["usd_per_day_saved"] == pytest.approx(6.004 - (0.16 + 0.009))
    assert s["stt_minutes_per_day_saved"] == 584


def test_false_activation_sweep_is_monotonic():
    sweep = gating.sweep_false_activations(gating.Assumptions(), [0.0, 1.0, 5.0])
    assert sweep.usd_per_day_saved.is_monotonic_decreasing


def test_confirmation_filters_near_misses():
    clips = pd.DataFrame({
        "file": ["p1", "p2", "n_jacky", "n_jarvis", "n_quiet"],
        "kind": ["pos", "pos", "neg", "neg", "neg"],
        "pace": ["normal"] * 5,
        "similarity": ["", "", "very-similar", "similar", "not-similar"],
        "max_score": [0.9, 0.95, 0.97, 0.96, 0.1],
    })
    transcripts = {"p1": "Jarvie, wake up.", "p2": "Jarvy help me", "n_jacky": "Jacky, wake up.",
                   "n_jarvis": "Jarvis, wake up.", "n_quiet": "okay"}
    decisions = gating.compare_detections(clips, transcripts, threshold=0.5)
    assert list(decisions.confirmed_fires) == [True, True, False, False, False]
    est = gating.estimate_confirmation(decisions)
    assert est == {"confirm_pass_rate": 0.0, "confirmed_recall": 1.0}  # jacky and jarvis both rejected
    table = gating.accuracy_table(decisions)
    assert table.loc["false triggers, very-similar", "daemon_confirmed"] == 0.0


def test_confirmation_costs_by_hand():
    a = gating.Assumptions(
        listening_hours_per_day=10, activations_per_day=2, conversation_minutes=1,
        idle_timeout_minutes=2, greeting_prompt_tokens=1000, greeting_completion_tokens=0,
        calls_per_activation=1, prompt_tokens_per_call=1000, completion_tokens_per_call=0,
        daemon_false_per_hour=0.5, confirm_seconds=6, confirm_pass_rate=0.2, confirmed_recall=0.5,
        stt_per_minute=0.01, llm_input_per_million=1.0, llm_output_per_million=0.0,
    )
    row = gating.daily_costs(a).loc["daemon_confirmed"]
    # 5 false starts: 1 passes (confirm + idle), 4 rejected (confirm only); 2 real (confirm + talk + idle)
    assert row.stt_minutes == pytest.approx(2 * (0.1 + 1 + 2) + 1 * (0.1 + 2) + 4 * 0.1)
    assert row.prompt_tokens == 3 * 1000 + 2 * 1000  # greetings for 2 real + 1 passed, 2 conversations
    assert row.false_activations == pytest.approx(1)
    assert gating.savings(a, "daemon_confirmed")["usd_per_day_saved"] > gating.savings(a)["usd_per_day_saved"]


@pytest.mark.parametrize("transcript", ["Jarvie, are you up?", "hey jarvy are you up", "Jarvie... are you up"])
def test_are_you_up_is_a_stage1_phrase(transcript):
    assert gating.transcript_detects(transcript)


@pytest.mark.parametrize("transcript", ["You up?", "Are you up?", "Jarvis, are you up?", "Jacky, are you up?"])
def test_are_you_up_near_misses_are_not(transcript):
    assert not gating.transcript_detects(transcript)
