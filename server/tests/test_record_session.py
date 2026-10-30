from pathlib import Path

import numpy as np
import pytest

from jarvie.eval import gating
from training import record_session as rs

PROMPTS = Path(rs.__file__).with_name("recording_prompts.txt")


def test_real_prompt_file_parses():
    sections = rs.parse_prompts(PROMPTS.read_text())
    assert {
        "wake-normal",
        "wake-fast",
        "wake-slow",
        "neg-very-similar",
        "neg-similar",
        "neg-everyday",
        "sleep-normal",
    } <= set(sections)
    assert all(s.kind in {"pos", "neg"} for s in sections.values())
    assert sections["neg-everyday"].minutes == 5
    assert len(sections["wake-normal"].lines) >= 20
    # negatives never contain an actual stage-1 wake phrase (that would teach it to ignore them)
    pattern = gating.phrase_pattern(jarvie_spellings=["jarvie"])
    for s in sections.values():
        if s.kind == "neg" and s.mode == "wake":
            for line in s.lines:
                spoken = line.split("(")[0]
                assert not gating.transcript_detects(spoken, pattern), (s.name, line)


def test_parse_prompts_headers_and_errors():
    text = """
    # comment
    [a] kind=pos pace=fast mode=sleep
    jarvie, go to sleep (fast)
    [b] kind=neg similarity=similar
    wake up
    """
    sections = rs.parse_prompts(text)
    assert sections["a"].pace == "fast" and sections["a"].mode == "sleep"
    assert sections["b"].similarity == "similar" and sections["b"].mode == "wake"
    assert sections["b"].lines == ["wake up"]
    with pytest.raises(ValueError, match="similarity"):
        rs.parse_prompts("[c] kind=neg\nwake up")
    with pytest.raises(ValueError, match="before any"):
        rs.parse_prompts("orphan line")


def test_labels_match_the_eval_naming():
    sections = rs.parse_prompts(PROMPTS.read_text())
    lbl = rs.label(sections["neg-very-similar"], "desk", "quiet")
    assert lbl == "neg-very-similar_desk_quiet-normal_wake"
    # the eval's parser reads it back to the right labels
    import pandas as pd

    stem = f"{lbl}_000".split("_")
    df = gating.add_labels(pd.DataFrame([{"kind": stem[0], "condition": stem[2], "max_score": 0}]))
    assert (df.kind[0], df.similarity[0], df.pace[0]) == ("neg", "very-similar", "normal")
    with pytest.raises(SystemExit):
        rs.label(sections["wake-normal"], "far_away", "quiet")  # underscores would break the name


def test_output_dirs_keep_test_and_train_apart():
    assert rs.output_dir("test", "wake", "v1") == rs.DATA / "audio" / "wake"
    assert rs.output_dir("test", "wake", "v1", "sam") == rs.DATA / "audio-speakers" / "sam" / "wake"
    assert rs.output_dir("train", "wake", "v1") == rs.DATA / "real" / "v1" / "me" / "wake"
    assert rs.output_dir("train", "sleep", "v1", "sam") == rs.DATA / "real" / "v1" / "sam" / "sleep"
    with pytest.raises(SystemExit):
        rs.output_dir("train", "wake", "v1", "Sam Smith")


def test_check_take():
    t = np.arange(int(4 * rs.SR)) / rs.SR
    noise = np.random.default_rng(0).normal(0, 0.002, t.size)
    speech = noise + 0.3 * np.sin(2 * np.pi * 200 * t) * ((t > 0.8) & (t < 2.0))
    assert rs.check_take(speech.astype("float32")) == []
    assert any(
        "cut off" in p
        for p in rs.check_take(
            (noise + 0.3 * np.sin(2 * np.pi * 200 * t) * (t > 2.5)).astype("float32")
        )
    )
    assert any("clipped" in p for p in rs.check_take(np.clip(speech * 5, -1, 1).astype("float32")))
    assert any("quiet" in p for p in rs.check_take((speech * 0.05).astype("float32")))
    assert any("no clear speech" in p for p in rs.check_take(noise.astype("float32")))


def test_save_names_and_manifest(tmp_path):
    section = rs.parse_prompts("[x] kind=pos pace=normal\njarvie, wake up")["x"]
    args = type(
        "A", (), {"position": "desk", "condition": "quiet", "split": "train", "speaker": "me"}
    )()
    audio = np.zeros(rs.SR, dtype="float32")
    p1 = rs.save(audio, tmp_path, "pos_desk_quiet-normal_wake", "jarvie, wake up", section, args)
    p2 = rs.save(audio, tmp_path, "pos_desk_quiet-normal_wake", "jarvie, wake up", section, args)
    assert (p1.name, p2.name) == (
        "pos_desk_quiet-normal_wake_000.wav",
        "pos_desk_quiet-normal_wake_001.wav",
    )
    import csv

    rows = list(csv.DictReader((tmp_path / "manifest.csv").open()))
    assert len(rows) == 2
    assert rows[0]["text"] == "jarvie, wake up" and rows[0]["speaker"] == "me"


def test_name_sections_for_the_stage2_model():
    from training.real_data import says_jarvie

    sections = rs.parse_prompts(PROMPTS.read_text())
    name = {n: s for n, s in sections.items() if s.mode == "name"}
    assert {
        "name-normal",
        "name-fast",
        "name-slow",
        "neg-name-very-similar",
        "neg-name-similar",
    } <= set(name)
    for s in name.values():
        for line in s.lines:
            # positives say the name (and nothing after it); negatives never do
            assert says_jarvie(line) == (s.kind == "pos"), (s.name, line)
    # "jarvis" appears only as something that must NOT trigger
    for s in sections.values():
        if any("jarvis" in line.lower() for line in s.lines):
            assert s.kind == "neg", s.name


def fake_session(monkeypatch, replies):
    """Run the recorder without a mic: scripted replies to input(), and saves logged, not written."""
    saved = []
    answers = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(rs, "beep", lambda *a, **k: None)
    monkeypatch.setattr(rs, "record", lambda seconds: np.zeros(rs.SR, dtype="float32"))
    monkeypatch.setattr(rs, "check_take", lambda audio: [])

    def save(audio, folder, lbl, text, *rest):
        saved.append(text)
        return Path(f"{lbl}_{len(saved):03d}.wav")

    monkeypatch.setattr(rs, "save", save)
    return saved


SECTIONS = rs.parse_prompts("[a] kind=pos pace=normal mode=name\nl1\nl2\nl3\nl4\nl5\n")


def session_args(shuffle=False):
    return type(
        "A",
        (),
        dict(
            split="train",
            set="v1",
            speaker="me",
            position="desk",
            condition="quiet",
            repeats=1,
            shuffle=shuffle,
            minutes=None,
        ),
    )()


def test_quit_and_resume_records_every_take_once_even_shuffled(monkeypatch):
    # Record two takes (Enter to record, Enter to keep), then quit.
    saved = fake_session(monkeypatch, ["", "", "", "", "q"])
    stopped = rs.run_section(SECTIONS["a"], session_args(shuffle=True))
    assert stopped == 2 and len(saved) == 2
    # Resume with --start 2: the same shuffled order, so the other three lines follow.
    resumed = fake_session(monkeypatch, [""] * 6)
    assert rs.run_section(SECTIONS["a"], session_args(shuffle=True), start=stopped) is None
    assert sorted(saved + resumed) == ["l1", "l2", "l3", "l4", "l5"]


def test_quit_stops_the_whole_command(monkeypatch, capsys, tmp_path):
    prompts = tmp_path / "p.txt"
    prompts.write_text(
        "[a] kind=pos pace=normal mode=name\nl1\nl2\n[b] kind=pos pace=normal mode=name\nl3\n"
    )
    saved = fake_session(monkeypatch, ["", "", "q"])
    monkeypatch.setattr("sys.argv", ["record_session.py", "a", "b", "--prompts", str(prompts)])
    rs.main()
    assert saved == ["l1"]  # section b never started
    assert "Resume with: a b --start 1" in capsys.readouterr().out


def test_recorded_takes_reach_training_and_eval(tmp_path, monkeypatch):
    """Round trip: the folders and manifests record_session.py writes are the ones the training
    loader (real_data) and the eval (compare) read, for every split, speaker and model."""
    from jarvie.eval import compare
    from training import real_data

    monkeypatch.setattr(rs, "DATA", tmp_path)
    sections = rs.parse_prompts(PROMPTS.read_text())
    rng = np.random.default_rng(0)

    def take(section, split, speaker="me", condition="quiet", text=None):
        s = sections[section]
        args = type(
            "A",
            (),
            {"position": "desk", "condition": condition, "split": split, "speaker": speaker},
        )()
        folder = rs.output_dir(split, s.mode, "v1", speaker)
        lbl = rs.label(s, "desk", condition)
        audio = (rng.standard_normal(rs.SR) * 0.1).astype("float32")  # distinct takes
        return rs.save(audio, folder, lbl, text or s.lines[0], s, args)

    take("wake-normal", "train")
    take("name-normal", "train")
    take("neg-name-very-similar", "train")
    take("neg-everyday", "train", condition="talk")
    take("name-normal", "train", speaker="sam")
    test_name = take("name-normal", "test")
    test_everyday = take("neg-everyday", "test", condition="room")
    take("name-normal", "test", speaker="kim")

    # Training: real/v1/<speaker>/<mode>/, which is what gets uploaded to RunPod.
    clips = real_data.collect(tmp_path / "real" / "v1")
    assert real_data.summary(clips) == {
        "me": {"wake/pos": 1, "name/pos": 1, "name/neg": 1, "wake/neg": 1},
        "sam": {"name/pos": 1},
    }
    assert all(c.text for c in clips)  # manifest text found for every take
    pos, neg = real_data.training_roles(clips, "name")
    assert len(pos) == 2 and {c.mode for c in neg} == {"name", "wake"}  # everyday counts too

    # Eval: audio/<mode>/ and audio-speakers/<speaker>/<mode>/, only manifest-listed takes.
    sets = compare.discover_test_sets("name", tmp_path)
    assert sets == {"me": test_name.parent, "kim": tmp_path / "audio-speakers" / "kim" / "name"}
    assert wakeword_recorded(sets["me"]) == {test_name.name}
    assert compare.missing_inputs("name", [Path("m.onnx")], tmp_path) == []
    compare.check_leakage(sets, tmp_path / "real")  # test takes aren't in training
    # Everyday test audio lands in audio/wake/, where the Gate table looks for it.
    assert test_everyday.parent == tmp_path / "audio" / "wake"
    assert any(
        Path(f).match(compare.EVERYDAY_PATTERN) for f in wakeword_recorded(test_everyday.parent)
    )


def wakeword_recorded(folder):
    from jarvie.eval import wakeword

    return wakeword.recorded_files(folder)
