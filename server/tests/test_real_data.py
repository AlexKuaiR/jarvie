import random

import numpy as np
import pytest
import soundfile as sf

from training import real_data as rd

SR = rd.SR


def phrase_clip(seconds=4.0, start=1.0, end=2.2, level=0.3):
    t = np.arange(int(seconds * SR)) / SR
    noise = np.random.default_rng(0).normal(0, 0.002, t.size)
    return (noise + level * np.sin(2 * np.pi * 220 * t) * ((t > start) & (t < end))).astype(
        "float32"
    )


def test_speech_bounds():
    start, end = rd.speech_bounds(phrase_clip())
    assert start == pytest.approx(1.0, abs=0.03) and end == pytest.approx(2.2, abs=0.03)
    assert rd.speech_bounds(np.zeros(SR, dtype="float32")) is None


def test_trim_positive_ends_just_after_the_phrase():
    for seed in range(5):
        out = rd.trim_positive(phrase_clip(), random.Random(seed))
        assert len(out) == int(rd.CLIP_SECONDS * SR)
        _, end = rd.speech_bounds(out)
        # phrase end sits 0-200 ms before the clip's end (plus a frame of slack)
        assert rd.CLIP_SECONDS - 0.22 <= end <= rd.CLIP_SECONDS + 0.001


def test_trim_positive_pads_when_the_phrase_starts_at_the_beginning():
    out = rd.trim_positive(phrase_clip(start=0.0, end=0.6), random.Random(0))
    assert len(out) == int(rd.CLIP_SECONDS * SR)
    assert np.all(out[: int(0.9 * SR)] == 0)  # zero-padded before the recording started


def test_trim_negative_is_centred():
    out = rd.trim_negative(phrase_clip(start=1.0, end=2.0))
    s, e = rd.speech_bounds(out)
    assert (s + e) / 2 == pytest.approx(rd.CLIP_SECONDS / 2, abs=0.05)


def test_chunk_long_skips_silence():
    speech = phrase_clip(seconds=10, start=2, end=8)
    chunks = rd.chunk_long(speech)
    assert chunks and all(len(c) == int(rd.CLIP_SECONDS * SR) for c in chunks)
    assert len(chunks) < len(range(0, 10 * SR - 2 * SR + 1, SR))  # silent stretches dropped


def test_speaker_perturb_changes_speed_within_range():
    audio = phrase_clip()
    for seed in range(10):
        out = rd.speaker_perturb(audio, random.Random(seed))
        assert 1 / 1.12 - 0.01 <= len(out) / len(audio) <= 1 / 0.88 + 0.01


def make_uploads(tmp_path):
    up = tmp_path / "uploads"
    files = {
        "me/wake/pos_desk_quiet-normal_wake_000.wav": phrase_clip(),
        "me/wake/neg-very-similar_desk_quiet-normal_wake_000.wav": phrase_clip(),
        "me/sleep/pos_bed_quiet-normal_sleep_000.wav": phrase_clip(),
        "sam/wake/pos_far_quiet-fast_wake_000.wav": phrase_clip(),
        "me/wake/neg-not-similar_desk_quiet-normal_wake_001.wav": phrase_clip(
            seconds=10, start=1, end=9
        ),
        "me/wake/silent_take.wav": np.zeros(SR, dtype="float32"),  # not a labeled name: ignored
    }
    for rel, audio in files.items():
        (up / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(up / rel, audio, SR)
    (up / "me/wake/manifest.csv").write_text("file,text\n")
    return up


def test_collect_summary_and_roles(tmp_path):
    clips = rd.collect(make_uploads(tmp_path))
    assert len(clips) == 5
    assert rd.summary(clips) == {
        "me": {"wake/pos": 1, "wake/neg": 2, "sleep/pos": 1},
        "sam": {"wake/pos": 1},
    }
    pos, neg = rd.training_roles(clips, "wake")
    assert len(pos) == 2  # me + sam wake positives
    assert len(neg) == 3  # wake negatives + the sleep phrase (cross-negative)
    pos_s, neg_s = rd.training_roles(clips, "sleep")
    assert len(pos_s) == 1 and len(neg_s) == 4


def test_fingerprint_changes_with_recordings(tmp_path):
    up = make_uploads(tmp_path)
    before = rd.fingerprint(rd.collect(up))
    sf.write(up / "me/wake/pos_desk_quiet-normal_wake_001.wav", phrase_clip(), SR)
    assert rd.fingerprint(rd.collect(up)) != before


def test_write_copies(tmp_path):
    clips = rd.collect(make_uploads(tmp_path))
    pos, neg = rd.training_roles(clips, "wake")
    end, silent = rd.write_copies(
        pos, "pos", tmp_path / "out", repeat=5, speaker_aug=True, seed="s"
    )
    files = sorted((tmp_path / "out").glob("clip_*.wav"))
    assert len(files) == 2 * 5 and end == rd.REAL_INDEX_START + 10 and silent == 0
    assert all(sf.info(str(f)).frames == int(rd.CLIP_SECONDS * SR) for f in files)
    # the long everyday-speech recording becomes many clips, written once (no repeats)
    end2, _ = rd.write_copies(neg, "neg", tmp_path / "neg", repeat=5, speaker_aug=False, seed="s")
    n_neg = len(list((tmp_path / "neg").glob("clip_*.wav")))
    assert n_neg == end2 - rd.REAL_INDEX_START and n_neg > 2 * 5


def test_spellings_fallback_matches_the_shared_list():
    # real_data.py keeps a copy for RunPod, where the jarvie package isn't uploaded.
    import re
    from pathlib import Path

    from jarvie.wake_phrases import JARVIE_SPELLINGS

    source = Path(rd.__file__).read_text()
    fallback = re.search(r"except ImportError:.*?JARVIE_SPELLINGS = (\[.*?\])", source, re.S).group(
        1
    )
    assert eval(fallback) == JARVIE_SPELLINGS


def make_name_uploads(tmp_path):
    up = tmp_path / "uploads"
    rows = {
        "me/name/pos_desk_quiet-normal_name_000.wav": "jarvie",
        "me/name/neg-very-similar_desk_quiet-normal_name_000.wav": "jarvis",
        "me/wake/pos_desk_quiet-normal_wake_000.wav": "jarvie, wake up",
        "me/wake/neg-similar_desk_quiet-normal_wake_000.wav": "hey jarvie",
        "me/wake/neg-very-similar_desk_quiet-normal_wake_001.wav": "jacky, wake up",
    }
    for rel, text in rows.items():
        path = up / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, phrase_clip(), SR)
        manifest = path.parent / "manifest.csv"
        new = not manifest.exists()
        with manifest.open("a") as f:
            if new:
                f.write("file,text\n")
            f.write(f'{path.name},"{text}"\n')
    return up


def test_name_model_roles(tmp_path):
    clips = rd.collect(make_name_uploads(tmp_path))
    assert {c.mode for c in clips} == {"name", "wake"}
    assert next(c for c in clips if c.mode == "name" and c.kind == "pos").text == "jarvie"
    pos, neg = rd.training_roles(clips, "name")
    assert [c.text for c in pos] == ["jarvie"]
    # negatives that don't say the name; "hey jarvie" and the stage-1 phrase are left out
    assert sorted(c.text for c in neg) == ["jacky, wake up", "jarvis"]
    # for stage 1, the name alone is a negative (it must not start the bot)
    wake_pos, wake_neg = rd.training_roles(clips, "wake")
    assert [c.text for c in wake_pos] == ["jarvie, wake up"]
    assert "jarvie" in [c.text for c in wake_neg]
