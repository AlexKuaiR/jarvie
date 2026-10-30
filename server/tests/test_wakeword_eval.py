"""Tests for the wake-word evaluation helpers.

No real ONNX model is needed: FakeSession stands in for onnxruntime's InferenceSession
and scores each window as the mean of its values, and identity_featurize turns each audio
sample into a one-value "embedding frame", so expected scores are easy to compute.
One smoke test uses the real model and is skipped when it isn't available.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from jarvie.eval import wakeword

SR = 16000


class FakeSession:
    """Mimics InferenceSession.run(): one score per window = mean of the window's samples."""

    def __init__(self):
        self.input_shapes = []

    def run(self, output_names, feeds):
        (chunk,) = feeds.values()
        self.input_shapes.append(chunk.shape)
        return [np.array([[chunk.mean()]], dtype=np.float32)]


def identity_featurize(audio):
    """Stand-in for the real featurizer: one 1-dim frame per sample, shape (n, 1)."""
    return audio[:, None]


def write_wav(path, samples):
    # FLOAT so samples read back exactly; the default 16-bit PCM would round 0.8 to 0.79998.
    sf.write(path, np.asarray(samples, dtype=np.float32), SR, subtype="FLOAT")
    return path


# --- event_count -------------------------------------------------------------


def test_event_count_empty_and_below_threshold():
    assert wakeword.event_count([], 0.5) == 0
    assert wakeword.event_count([0.1, 0.2, 0.4], 0.5) == 0


def test_event_count_collapses_consecutive_frames():
    # One utterance spanning three overlapping windows is one false accept.
    assert wakeword.event_count([0.1, 0.9, 0.9, 0.9, 0.1], 0.5) == 1


def test_event_count_separate_runs_are_separate_events():
    assert wakeword.event_count([0.9, 0.9, 0.1, 0.8, 0.1, 0.7], 0.5) == 3


def test_event_count_threshold_is_inclusive():
    assert wakeword.event_count([0.5], 0.5) == 1


# --- score_clip --------------------------------------------------------------


def test_score_clip_slides_window_over_whole_clip(tmp_path):
    # 10 samples, window 4, hop 2 -> windows start at 0, 2, 4, 6 (the last one ends at 10).
    path = write_wav(tmp_path / "clip.wav", np.arange(10) / 10)
    session = FakeSession()

    result = wakeword.score_clip(
        session, path, "audio", window=4, hop=2, featurize=identity_featurize, pad_seconds=0
    )

    assert result["scores"] == pytest.approx([0.15, 0.35, 0.55, 0.75])
    assert result["max"] == pytest.approx(0.75)
    # 3-D batch (1, window, embedding_dim), as the real model requires
    assert all(shape == (1, 4, 1) for shape in session.input_shapes)


def test_score_clip_exactly_one_window_long(tmp_path):
    path = write_wav(tmp_path / "clip.wav", [0.5] * 4)
    result = wakeword.score_clip(
        FakeSession(), path, "audio", window=4, hop=2, featurize=identity_featurize, pad_seconds=0
    )
    assert result["scores"] == pytest.approx([0.5])


def test_score_clip_shorter_than_window(tmp_path):
    path = write_wav(tmp_path / "clip.wav", [0.5] * 3)
    result = wakeword.score_clip(
        FakeSession(), path, "audio", window=4, hop=2, featurize=identity_featurize, pad_seconds=0
    )
    assert result["scores"] == []
    assert result["max"] == 0.0


def test_score_clip_reports_duration_in_seconds(tmp_path):
    path = write_wav(tmp_path / "clip.wav", np.zeros(SR * 3))
    result = wakeword.score_clip(
        FakeSession(), path, "audio", window=SR, hop=SR, featurize=identity_featurize, pad_seconds=0
    )
    assert result["duration"] == pytest.approx(3.0)


def test_score_clip_pads_with_silence_by_default(tmp_path):
    # 1 s of silence on each side (as livekit's predict_clip does) adds windows before and after
    # the clip, so a phrase at the very end can still line up with a window's end.
    path = write_wav(tmp_path / "clip.wav", [0.8] * 8)
    unpadded = wakeword.score_clip(
        FakeSession(), path, "audio", 4, 4, identity_featurize, pad_seconds=0
    )
    padded = wakeword.score_clip(FakeSession(), path, "audio", 4, 4, identity_featurize)
    assert len(unpadded["scores"]) == 2
    assert len(padded["scores"]) == (SR + 8 + SR) // 4
    assert padded["max"] == pytest.approx(0.8)  # the clip's own windows are still scored
    assert padded["duration"] == unpadded["duration"] == pytest.approx(8 / SR)  # unpadded length


def test_score_clip_rejects_wrong_sample_rate(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(8000, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="16000"):
        wakeword.score_clip(
            FakeSession(), path, "audio", 4, 2, featurize=identity_featurize, pad_seconds=0
        )


MODEL = Path(__file__).resolve().parents[1] / "models" / "jarvie_wake.onnx"


@pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")
def test_real_model_and_featurizer_smoke(tmp_path):
    # End-to-end with the real classifier and livekit-wakeword's bundled feature models.
    featurize = wakeword.make_featurizer()
    session, name, window = wakeword.load_model(MODEL)
    path = write_wav(tmp_path / "silence.wav", np.zeros(SR * 2))

    result = wakeword.score_clip(session, path, name, window, 1, featurize=featurize)

    assert window == 16
    assert result["scores"]
    assert all(0.0 <= s <= 1.0 for s in result["scores"])


@pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")
def test_featurizer_matches_livekit_wakeword(tmp_path):
    # The eval must preprocess audio exactly like the trainer. livekit's WakeWordModel scores
    # the last 16 embeddings of whatever audio it's given; our last window covers the same
    # frames, so the two scores must agree.
    from livekit.wakeword import WakeWordModel

    rng = np.random.default_rng(0)
    t = np.arange(SR * 2) / SR
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (t > 0.8) + rng.normal(0, 0.01, t.size)
    path = write_wav(tmp_path / "tone.wav", audio)

    session, name, window = wakeword.load_model(MODEL)
    ours = wakeword.score_clip(
        session, path, name, window, 1, featurize=wakeword.make_featurizer(), pad_seconds=0
    )

    reference = WakeWordModel(models=[str(MODEL)])
    (theirs,) = reference.predict(audio.astype(np.float32)).values()

    assert ours["scores"][-1] == pytest.approx(theirs, abs=1e-4)


# --- score_all ---------------------------------------------------------------


def test_score_all_parses_names_and_skips_malformed(tmp_path):
    write_wav(tmp_path / "pos_desk_quiet_wake_001.wav", [0.8] * 8)
    write_wav(tmp_path / "neg_far_loud_sleep_012.wav", [0.1] * 8)
    write_wav(tmp_path / "badname.wav", [0.5] * 8)

    df = wakeword.score_all(
        FakeSession(),
        tmp_path,
        "audio",
        window=4,
        hop=4,
        featurize=identity_featurize,
        pad_seconds=0,
    )

    assert list(df.file) == ["neg_far_loud_sleep_012.wav", "pos_desk_quiet_wake_001.wav"]
    neg = df.iloc[0]
    assert (neg.kind, neg.position, neg.condition, neg.phrase, neg["index"]) == (
        "neg",
        "far",
        "loud",
        "sleep",
        12,
    )
    assert df.iloc[1].max_score == pytest.approx(0.8)


def test_score_all_empty_folder(tmp_path):
    assert wakeword.score_all(
        FakeSession(),
        tmp_path,
        "audio",
        window=4,
        hop=4,
        featurize=identity_featurize,
        pad_seconds=0,
    ).empty


# --- sweeps ------------------------------------------------------------------


def eval_frame() -> pd.DataFrame:
    """Two positives and 30 minutes of negatives for "wake", plus a "sleep" row to filter out."""
    return pd.DataFrame(
        [
            {
                "kind": "pos",
                "phrase": "wake",
                "max_score": 0.9,
                "duration": 3.0,
                "frame_scores": [0.9],
            },
            {
                "kind": "pos",
                "phrase": "wake",
                "max_score": 0.4,
                "duration": 3.0,
                "frame_scores": [0.4],
            },
            {
                "kind": "neg",
                "phrase": "wake",
                "max_score": 0.6,
                "duration": 1800.0,
                "frame_scores": [0.6, 0.6, 0.1, 0.3],
            },
            {
                "kind": "pos",
                "phrase": "sleep",
                "max_score": 0.0,
                "duration": 3.0,
                "frame_scores": [0.0],
            },
        ]
    )


def test_sweeps_recall_and_false_accepts_per_hour():
    curve = wakeword.sweeps(eval_frame(), np.array([0.2, 0.5, 0.95]), phrase="wake")

    assert list(curve.recall) == pytest.approx([1.0, 0.5, 0.0])
    assert list(curve.frr) == pytest.approx([0.0, 0.5, 1.0])
    # threshold 0.2: [0.6, 0.6] is one event and [0.3] another; 0.5: one event; 0.95: none.
    assert list(curve.false_events) == [2, 1, 0]
    # 30 minutes of negatives is 0.5 hours, so events per hour is double the count.
    assert list(curve.neg_hours) == pytest.approx([0.5] * 3)
    assert list(curve.fpph) == pytest.approx([4.0, 2.0, 0.0])


def test_sweeps_without_negatives_raises():
    only_positives = eval_frame().query("kind == 'pos'")
    with pytest.raises(ValueError, match="no negative clips"):
        wakeword.sweeps(only_positives, np.array([0.5]), phrase="wake")


def test_sweeps_ignores_the_other_phrase():
    # The "sleep" positive scores 0.0; if it leaked in, recall at 0.2 would be 2/3.
    curve = wakeword.sweeps(eval_frame(), np.array([0.2]), phrase="wake")
    assert curve.recall.iloc[0] == pytest.approx(1.0)


# --- recall by position and condition ---------------------------------------


def test_by_condition_reports_recall_per_position_and_condition():
    df = pd.DataFrame(
        [
            {
                "kind": "pos",
                "phrase": "wake",
                "position": "desk",
                "condition": "quiet",
                "max_score": 0.9,
            },
            {
                "kind": "pos",
                "phrase": "wake",
                "position": "desk",
                "condition": "quiet",
                "max_score": 0.8,
            },
            {
                "kind": "pos",
                "phrase": "wake",
                "position": "desk",
                "condition": "loud",
                "max_score": 0.9,
            },
            {
                "kind": "pos",
                "phrase": "wake",
                "position": "desk",
                "condition": "loud",
                "max_score": 0.2,
            },
            # Negatives must not count towards recall.
            {
                "kind": "neg",
                "phrase": "wake",
                "position": "desk",
                "condition": "loud",
                "max_score": 0.1,
            },
        ]
    )

    table = wakeword.by_position_and_condition(df, threshold=0.5, phrase="wake")

    # Worst condition first: that's the one to fix.
    assert list(table.index) == [("desk", "loud"), ("desk", "quiet")]
    assert list(table.recall) == pytest.approx([0.5, 1.0])
    assert list(table.n) == [2, 2]


# --- operating_point ---------------------------------------------------------


def test_operating_point_picks_best_recall_within_budget():
    curve = pd.DataFrame(
        {
            "threshold": [0.3, 0.5, 0.7],
            "recall": [0.99, 0.95, 0.80],
            "fpph": [2.0, 0.05, 0.0],
        }
    )
    best = wakeword.operating_point(curve, max_fpph=0.1)
    assert best.threshold == 0.5


def test_operating_point_raises_when_nothing_fits_budget():
    curve = pd.DataFrame({"threshold": [0.3], "recall": [0.99], "fpph": [2.0]})
    with pytest.raises(ValueError):
        wakeword.operating_point(curve, max_fpph=0.1)
