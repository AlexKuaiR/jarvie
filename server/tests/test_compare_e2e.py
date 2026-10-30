"""End to end: compare.evaluate on a fake ~/jarvie-data laid out the way record_session.py writes it.

No model files or mic needed: the featurizer and the model are fakes in which a clip's loudness is
its score, so loud clips "fire" and quiet ones don't.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from jarvie.eval import compare, wakeword
from training import record_session as rs

FRAME = 1280  # one 80 ms embedding frame at 16 kHz


class LoudnessModel:
    """Stands in for an onnxruntime session: the score of a window is its loudest frame."""

    def run(self, outputs, inputs):
        (chunk,) = inputs.values()
        return [np.array([[float(chunk.max())]])]


def fake_featurizer(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // FRAME
    return np.abs(audio[: n * FRAME]).reshape(n, FRAME).mean(axis=1, keepdims=True)


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    monkeypatch.setattr(wakeword, "load_model", lambda path: (LoudnessModel(), "x", 4))
    monkeypatch.setattr(wakeword, "make_featurizer", lambda: fake_featurizer)
    models = tmp_path / "models"
    models.mkdir()
    for name in ["jarvie_name__B__base", "jarvie_name__B+real-v1__base"]:
        (models / f"{name}.onnx").write_bytes(b"")
    return sorted(models.glob("jarvie_name*.onnx"))


def write(folder, label, index, level, seconds=2.0, bursts=(), listed=True):
    """A clip at a constant loudness, optionally with 0.5 s loud bursts at the given seconds."""
    audio = np.full(int(seconds * rs.SR), level, dtype=np.float32)
    for at in bursts:
        audio[int(at * rs.SR) : int((at + 0.5) * rs.SR)] = 0.9
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{label}_{index:03d}.wav"
    sf.write(folder / name, audio, rs.SR)
    if listed:  # record_session.py logs every take in manifest.csv; older clips have no row
        manifest = folder / "manifest.csv"
        with manifest.open("a") as f:
            f.write(
                ("" if manifest.exists() and manifest.stat().st_size else "file,text\n")
                + f"{name},x\n"
            )


def record(data, speaker, section, position, condition, levels):
    """Test clips for one prompt section, named and placed exactly as record_session.py does."""
    s = rs.parse_prompts(rs.PROMPTS_FILE.read_text())[section]
    folder = rs.output_dir("test", s.mode, "v1", speaker).relative_to(rs.DATA)
    for i, level in enumerate(levels):
        write(data / folder, rs.label(s, position, condition), i, level)


def test_not_ready_lists_what_is_missing(tmp_path):
    with pytest.raises(ValueError, match="no jarvie_name.*models") as e:
        compare.evaluate("name", [], tmp_path)
    assert "no test recordings" in str(e.value)
    # Only older clips (no manifest.csv) still means no test recordings.
    write(tmp_path / "audio" / "name", "pos_desk_loud-normal_name", 0, 0.8, listed=False)
    missing = compare.missing_inputs("name", [Path("m.onnx")], tmp_path)
    assert len(missing) == 1 and "no test recordings" in missing[0]


def test_evaluate_name_models_end_to_end(fake_models, tmp_path):
    data = tmp_path / "jarvie-data"
    record(data, "me", "name-normal", "desk", "quiet", [0.8, 0.7, 0.6])
    record(data, "me", "name-fast", "desk", "quiet", [0.8, 0.2])
    record(data, "me", "neg-name-very-similar", "desk", "quiet", [0.1, 0.3])
    record(data, "sam", "name-normal", "desk", "quiet", [0.8])
    record(data, "sam", "neg-name-similar", "desk", "quiet", [0.1])
    # A 3-minute everyday recording, in audio/wake/ as record_session.py puts it, with 3 bursts.
    everyday = rs.parse_prompts(rs.PROMPTS_FILE.read_text())["neg-everyday"]
    write(
        data / "audio" / "wake",
        rs.label(everyday, "desk", "room"),
        0,
        level=0.05,
        seconds=180,
        bursts=(30, 90, 150),
    )
    # Old short not-similar test clip in audio/wake/: not everyday audio, so not in the Gate table.
    write(data / "audio" / "wake", rs.label(everyday, "desk", "loud"), 0, 0.05, 2.0)

    # Older clips without a manifest row (scratch/record.py) are ignored everywhere.
    write(data / "audio" / "name", "pos_desk_loud-normal_name", 0, level=0.0, listed=False)
    write(data / "audio" / "wake", "neg-not-similar_desk_old-normal_wake", 0, 0.9, 60, listed=False)

    result = compare.evaluate("name", fake_models, data, max_fpph=30.0)
    assert "pos_desk_loud-normal_name_000.wav" not in set(result.scores.file)

    assert list(result.personal.index) == [p.stem for p in fake_models]
    me = result.personal.iloc[0]
    assert 0.2 < me.threshold <= 0.6  # separates the 0.6+ positives from the ~0.3 negative
    assert me.eer_normal == 0 and me.recall_normal == 1.0 and me.recall_fast == 0.5
    assert not result.general.empty  # sam's test clips
    assert set(result.by_speaker.index.get_level_values("speaker")) == {"me", "sam"}
    # The everyday recording is only in the Gate table, not counted as one negative clip.
    assert not result.scores.file.str.startswith("neg-not-similar").any()
    gate = result.gate.iloc[0]
    assert gate.everyday_hours == pytest.approx(180 / 3600)
    # 3 bursts in 3 minutes = 60 per hour at any threshold below the bursts' loudness.
    assert gate.fpph_at_threshold == pytest.approx(60.0)
    assert gate.gate_recall == 0  # 30/hour is only met by thresholds that catch nothing...
    relaxed = compare.evaluate("name", fake_models, data, max_fpph=60.0).gate.iloc[0]
    assert relaxed.gate_recall == 1.0  # ...but 60 does, with every positive caught


def test_evaluate_wake_keeps_everyday_out_of_the_main_tables(fake_models, tmp_path):
    data = tmp_path / "jarvie-data"
    record(data, "me", "wake-normal", "desk", "quiet", [0.8, 0.7])
    record(data, "me", "neg-very-similar", "desk", "quiet", [0.1, 0.2])
    everyday = rs.parse_prompts(rs.PROMPTS_FILE.read_text())["neg-everyday"]
    write(data / "audio" / "wake", rs.label(everyday, "desk", "room"), 0, 0.05, 60, bursts=(30,))
    # A short clip with the same label (older test sets have these) is an ordinary negative.
    write(data / "audio" / "wake", rs.label(everyday, "desk", "loud"), 0, 0.05, 2.0)
    result = compare.evaluate("wake", fake_models, data)
    assert len(result.everyday) == len(fake_models)  # scored once, reused for the Gate table
    assert result.everyday.duration.min() == pytest.approx(60.0)
    # The short clip stays in the main tables; the 60 s recording doesn't.
    assert result.personal.iloc[0]["fires_not-similar"] == 0.0
    assert result.scores[result.scores.similarity == "not-similar"].duration.max() < 3
    assert result.gate.iloc[0].fpph_at_threshold == pytest.approx(60.0)


def test_evaluate_refuses_test_clips_that_are_in_training(fake_models, tmp_path):
    data = tmp_path / "jarvie-data"
    record(data, "me", "name-normal", "desk", "quiet", [0.8])
    train = data / "real" / "v1" / "me" / "name"
    train.mkdir(parents=True)
    clip = next((data / "audio" / "name").glob("*.wav"))
    (train / clip.name).write_bytes(clip.read_bytes())
    with pytest.raises(ValueError, match="also appear in training"):
        compare.evaluate("name", fake_models, data)
