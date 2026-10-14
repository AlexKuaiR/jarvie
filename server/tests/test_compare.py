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
    return [{"model": model, "kind": k, "pace": p, "similarity": s, "max_score": v} for k, p, s, v in rows]


def test_summarize_per_model():
    rows = clip_rows("good", [
        ("pos", "normal", "", 0.9), ("pos", "normal", "", 0.8), ("pos", "fast", "", 0.7),
        ("pos", "slow", "", 0.1),
        ("neg", "normal", "not-similar", 0.0), ("neg", "normal", "similar", 0.2),
    ]) + clip_rows("bad", [
        ("pos", "normal", "", 0.4), ("pos", "normal", "", 0.6), ("pos", "fast", "", 0.1),
        ("pos", "slow", "", 0.1),
        ("neg", "normal", "not-similar", 0.0), ("neg", "normal", "similar", 0.9),
    ])
    s = compare.summarize(pd.DataFrame(rows))
    assert list(s.index) == ["good", "bad"]
    assert s.loc["good", "eer_normal"] == 0.0
    assert s.loc["good", "recall_normal"] == 1.0
    assert s.loc["good", "recall_slow"] == 0.0
    assert s.loc["good", "fires_similar"] == 0.0
    assert s.loc["bad", "eer_normal"] > 0
    assert np.isnan(s.loc["good", "fires_very-similar"])  # no clips in that tier


def test_apply_labels_excludes_and_relabels():
    scores = pd.DataFrame({
        "file": ["help.wav", "wakie_wake.wav", "jacky.wav"],
        "kind": ["pos", "neg", "neg"], "pace": ["normal"] * 3,
        "similarity": ["", "similar", "very-similar"], "max_score": [0.1, 0.9, 0.9],
    })
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
    (train / "b.wav").write_bytes(b"two")
    compare.check_leakage({"me": test}, tmp_path / "real")  # no overlap: fine
    (train / "copy.wav").write_bytes(b"one")
    with pytest.raises(ValueError, match="also appear in training"):
        compare.check_leakage({"me": test}, tmp_path / "real")


def test_general_summary_weights_speakers_equally():
    def rows(speaker, model, pos_scores, neg_scores):
        return [{"speaker": speaker, "model": model, "kind": "pos", "pace": "normal", "similarity": "",
                 "max_score": v} for v in pos_scores] + \
               [{"speaker": speaker, "model": model, "kind": "neg", "pace": "normal",
                 "similarity": "similar", "max_score": v} for v in neg_scores]
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
