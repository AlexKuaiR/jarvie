import numpy as np
import pandas as pd
import pytest

from jarvie.eval import compare


def test_eer_separable_and_overlapping():
    t, rate = compare.eer(np.array([0.9, 0.8]), np.array([0.1, 0.2]))
    assert rate == 0.0 and 0.2 < t <= 0.8
    t, rate = compare.eer(np.array([0.9, 0.3]), np.array([0.5, 0.1]))  # one miss, one false fire
    assert rate == pytest.approx(0.5)
    assert np.isnan(compare.eer(np.array([]), np.array([0.1]))[1])


def clip_rows(model, rows):
    return [
        {"model": model, "kind": k, "pace": p, "similarity": s, "max_score": v}
        for k, p, s, v in rows
    ]


def test_summarize_per_model():
    rows = clip_rows(
        "good",
        [
            ("pos", "normal", "", 0.9),
            ("pos", "normal", "", 0.8),
            ("pos", "fast", "", 0.7),
            ("pos", "slow", "", 0.1),
            ("neg", "normal", "not-similar", 0.0),
            ("neg", "normal", "similar", 0.2),
        ],
    ) + clip_rows(
        "bad",
        [
            ("pos", "normal", "", 0.4),
            ("pos", "normal", "", 0.6),
            ("pos", "fast", "", 0.1),
            ("pos", "slow", "", 0.1),
            ("neg", "normal", "not-similar", 0.0),
            ("neg", "normal", "similar", 0.9),
        ],
    )
    s = compare.summarize(pd.DataFrame(rows))
    assert list(s.index) == ["good", "bad"]
    assert s.loc["good", "eer_normal"] == 0.0
    assert s.loc["good", "recall_normal"] == 1.0
    assert s.loc["good", "recall_slow"] == 0.0
    assert s.loc["good", "fires_similar"] == 0.0
    assert s.loc["bad", "eer_normal"] > 0
    assert np.isnan(s.loc["good", "fires_very-similar"])  # no clips in that tier


def test_apply_labels_excludes_and_relabels():
    scores = pd.DataFrame(
        {
            "file": ["help.wav", "wakie_wake.wav", "jacky.wav"],
            "kind": ["pos", "neg", "neg"],
            "pace": ["normal"] * 3,
            "similarity": ["", "similar", "very-similar"],
            "max_score": [0.1, 0.9, 0.9],
        }
    )
    df = compare.apply_labels(scores, exclude={"help.wav"}, as_positive={"wakie_wake.wav"})
    assert list(df.file) == ["wakie_wake.wav", "jacky.wav"]
    assert list(df.kind) == ["pos", "neg"]
    assert df.similarity.iloc[0] == ""


def test_discover_test_sets(tmp_path):
    (tmp_path / "audio/wake").mkdir(parents=True)
    (tmp_path / "audio-speakers/sam/wake").mkdir(parents=True)
    (tmp_path / "audio-speakers/kim/sleep").mkdir(parents=True)
    sets = compare.discover_test_sets("wake", tmp_path)
    assert sets == {"me": tmp_path / "audio/wake", "sam": tmp_path / "audio-speakers/sam/wake"}


def test_check_leakage(tmp_path):
    test = tmp_path / "audio/wake"
    train = tmp_path / "real/v1/me/wake"
    test.mkdir(parents=True)
    train.mkdir(parents=True)
    (test / "a.wav").write_bytes(b"one")
    (test / "manifest.csv").write_text("file,text\na.wav,x\n")
    (train / "b.wav").write_bytes(b"two")
    compare.check_leakage({"me": test}, tmp_path / "real")  # no overlap: fine
    (train / "copy.wav").write_bytes(b"one")
    with pytest.raises(ValueError, match="also appear in training"):
        compare.check_leakage({"me": test}, tmp_path / "real")


def test_general_summary_weights_speakers_equally():
    def rows(speaker, model, pos_scores, neg_scores):
        return [
            {
                "speaker": speaker,
                "model": model,
                "kind": "pos",
                "pace": "normal",
                "similarity": "",
                "max_score": v,
            }
            for v in pos_scores
        ] + [
            {
                "speaker": speaker,
                "model": model,
                "kind": "neg",
                "pace": "normal",
                "similarity": "similar",
                "max_score": v,
            }
            for v in neg_scores
        ]

    scores = pd.DataFrame(
        rows("me", "m", [0.9] * 4, [0.1] * 4)
        + rows("sam", "m", [0.9] * 20, [0.1] * 20)  # many clips, perfect
        + rows("kim", "m", [0.9, 0.1], [0.9, 0.1])  # few clips, half wrong
    )
    by_speaker = compare.summarize_by_speaker(scores)
    assert set(by_speaker.index.get_level_values("speaker")) == {"me", "sam", "kim"}
    general = compare.general_summary(scores)
    assert general.loc["m", "eer_normal"] == pytest.approx((0.0 + 0.5) / 2)  # mean over sam and kim
    assert compare.general_summary(scores[scores.speaker == "me"]).empty


def test_discover_name_test_sets(tmp_path):
    (tmp_path / "audio/name").mkdir(parents=True)
    (tmp_path / "audio-speakers/sam/name").mkdir(parents=True)
    assert set(compare.discover_test_sets("name", tmp_path)) == {"me", "sam"}


def gate_rows(model, pos_scores, everyday_frames):
    """Positives (max score only) for `scores`; everyday clips (per-window scores) for `everyday`."""
    scores = [
        {"model": model, "kind": "pos", "pace": "normal", "similarity": "", "max_score": v}
        for v in pos_scores
    ] + [  # summarize() needs normal-pace negatives to pick its threshold
        {"model": model, "kind": "neg", "pace": "normal", "similarity": "similar", "max_score": 0.1}
    ]
    everyday = [
        {"model": model, "duration": 1800.0, "frame_scores": frames} for frames in everyday_frames
    ]
    return scores, everyday


def test_gate_summary_counts_separate_fires_per_hour():
    # Two 30-min clips = 1 hour. At 0.5 the first clip fires twice (the adjacent 0.6s are one
    # opening; the 0.9 comes after the gate has closed again), the second once: 3 per hour.
    first = [0.6, 0.6] + [0.1] * 40 + [0.9]
    s, e = gate_rows("m", [0.9, 0.7, 0.6], [first, [0.0, 0.55, 0.0]])
    gate = compare.gate_summary(pd.DataFrame(s), pd.DataFrame(e), max_fpph=1.0)
    row = gate.loc["m"]
    assert row.everyday_hours == pytest.approx(1.0)
    # Lowest threshold within 1 fire/hour is above 0.6, so only the 0.9 and 0.7 positives pass.
    assert 0.6 < row.gate_threshold <= 0.7
    assert row.gate_recall == pytest.approx(2 / 3)
    assert row.gate_fpph == pytest.approx(1.0)
    budget3 = compare.gate_summary(pd.DataFrame(s), pd.DataFrame(e), max_fpph=3.0).loc["m"]
    # Full recall needs 0.6; there the second clip (peak 0.55) no longer fires: 2 per hour.
    assert budget3.gate_recall == 1.0 and budget3.gate_threshold == pytest.approx(0.6)
    assert budget3.gate_fpph == pytest.approx(2.0)
    assert budget3.fpph_at_threshold == pytest.approx(3.0)  # summarize()'s threshold is lower


def test_gate_summary_without_everyday_audio_or_within_budget():
    s, e = gate_rows("m", [0.9], [[1.0, 0.0, 1.0]])
    no_audio = compare.gate_summary(pd.DataFrame(s), pd.DataFrame()).loc["m"]
    assert no_audio.everyday_hours == 0 and np.isnan(no_audio.gate_recall)
    # Scores of 1.0 fire at every threshold, so even 0 fires/hour can't be met.
    strict = compare.gate_summary(pd.DataFrame(s), pd.DataFrame(e), max_fpph=0.0).loc["m"]
    assert np.isnan(strict.gate_threshold) and strict.fpph_at_threshold > 0


def test_score_everyday_empty_without_recordings(tmp_path):
    assert compare.score_everyday([], tmp_path).empty
    (tmp_path / "audio/wake").mkdir(parents=True)
    (tmp_path / "audio/wake/pos_desk_quiet-normal_wake_000.wav").write_bytes(b"x")
    assert compare.score_everyday([], tmp_path).empty  # wake clips aren't everyday speech
    (tmp_path / "audio/wake/neg-not-similar_desk_loud-normal_wake_000.wav").write_bytes(b"x")
    assert compare.score_everyday([], tmp_path).empty  # older clip: not in a manifest.csv


def test_gate_fires_counts_gate_openings():
    hold = round(compare.GATE_HOLD_SECONDS / compare.FRAME_SECONDS)
    assert compare.gate_fires([0.9, 0.1, 0.9], 0.5) == 1  # second fire while the gate is open
    assert compare.gate_fires([0.9] + [0.1] * hold + [0.9], 0.5) == 2
    # Stuck above the threshold: one opening per hold, not a single long "fire".
    assert compare.gate_fires([0.9] * (3 * hold), 0.5) == 3
