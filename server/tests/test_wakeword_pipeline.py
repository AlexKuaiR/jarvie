"""Tests for the cached training pipeline, with a fake livekit that only writes the files each
real stage would write. They check what gets rerun and what gets reused."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training import wakeword_pipeline as wp


class FakeAugmentor:
    def apply_rir(self, audio, p=0.5):
        return audio * 10.0  # like livekit: reverb that makes the clip much louder

    def mix_with_background(self, audio, snr_db_range=(5.0, 15.0)):
        return audio + 0.5


ORIGINAL_RIR = FakeAugmentor.apply_rir  # captured before any patching


def fake_livekit(calls: Counter, seen: dict) -> SimpleNamespace:
    augment_module = SimpleNamespace(AudioAugmentor=FakeAugmentor)

    class Config(SimpleNamespace):
        @property
        def model_output_dir(self) -> Path:
            return Path(self.output_dir) / self.model_name

    def run_generate(cfg):
        calls["generate", cfg.model_name] += 1
        for split in wp.SPLITS:
            d = cfg.model_output_dir / split
            d.mkdir(parents=True, exist_ok=True)
            for i in range(30):
                (d / f"clip_{i:06d}.wav").write_bytes(b"clean")

    def run_augment(cfg):
        calls["augment", cfg.model_name] += 1
        seen["augmentation", cfg.model_name] = cfg.augmentation
        seen["patched"] = augment_module.AudioAugmentor.apply_rir is not ORIGINAL_RIR
        for split in wp.SPLITS:
            for clip in list((cfg.model_output_dir / split).glob("clip_??????.wav")):
                clip.with_name(clip.stem + "_r0.wav").write_bytes(b"augmented")

    def run_extraction(cfg):
        calls["extract", cfg.model_name] += 1
        for name in wp.FEATURE_FILES:
            np.save(cfg.model_output_dir / name, np.zeros((1, 16, 96)))

    def run_train(cfg):
        calls["train", cfg.model_name] += 1
        seen["steps", cfg.model_name] = getattr(cfg, "steps", None)
        (cfg.model_output_dir / f"{cfg.model_name}.pt").write_bytes(b"weights")

    def run_export(cfg):
        path = cfg.model_output_dir / f"{cfg.model_name}.onnx"
        path.write_bytes(b"onnx")
        return path

    def run_eval(cfg, path):
        return {"aut": 0.1, "fpph": 0.5, "recall": 0.9, "optimal_threshold": 0.4}

    return SimpleNamespace(
        WakeWordConfig=Config, run_generate=run_generate, run_augment=run_augment,
        run_extraction=run_extraction, run_train=run_train, run_export=run_export,
        run_eval=run_eval, augment_module=augment_module,
    )


@pytest.fixture
def setup(tmp_path):
    calls, seen = Counter(), {}
    lk = fake_livekit(calls, seen)
    (tmp_path / "data/backgrounds").mkdir(parents=True)
    (tmp_path / "data/backgrounds/noise.wav").write_bytes(b"x")

    def make(**changes):
        settings = dict(
            root=tmp_path,
            phrases={"jarvie_wake": ["jarvie, wake up"], "jarvie_sleep": ["jarvie, go to sleep"]},
            negatives={"jarvie_wake": ["wake up"], "jarvie_sleep": ["go to sleep"]},
            generation={"n_samples": 100, "length_scales": [0.8, 1.0], "tts_batch_size": 50},
            training={"steps": 1000, "model": {"model_size": "medium"}},
            keep_augmented_audio=5,
            livekit=lk,
        )
        settings.update(changes)
        return wp.Pipeline(**settings)

    return make, calls, seen, tmp_path


def count(calls: Counter, stage: str) -> int:
    return sum(n for (s, _), n in calls.items() if s == stage)


def test_second_run_reuses_every_stage(setup):
    make, calls, _, _ = setup
    make().run("A")
    make().run("A")  # a new Pipeline, as in a new session on the same volume
    assert count(calls, "generate") == 2  # once per model, never again
    assert count(calls, "augment") == 2
    assert count(calls, "train") == 2


def test_variant_b_reuses_speech_and_patches_only_its_augmentation(setup):
    make, calls, seen, _ = setup
    pipe = make()
    pipe.run("A")
    assert seen["patched"] is False
    pipe.run("B")
    assert seen["patched"] is True
    assert count(calls, "generate") == 2  # B didn't regenerate speech
    assert count(calls, "augment") == 4
    assert count(calls, "train") == 4
    # livekit's originals are back after B's augmentation
    assert pipe.lk.augment_module.AudioAugmentor.apply_rir is ORIGINAL_RIR


def test_training_override_only_retrains(setup):
    make, calls, seen, root = setup
    pipe = make()
    pipe.run("B")
    pipe.run("B", trial="short", steps=50)
    assert count(calls, "generate") == 2
    assert count(calls, "augment") == 2
    assert count(calls, "train") == 4
    assert seen["steps", "jarvie_wake"] == 50
    assert (root / "models/jarvie_wake__B__short.onnx").exists()


def test_tts_batch_size_does_not_invalidate_speech(setup):
    make, calls, _, _ = setup
    make().run("A")
    make(generation={"n_samples": 100, "length_scales": [0.8, 1.0], "tts_batch_size": 25}).run("A")
    assert count(calls, "generate") == 2


def test_changing_phrases_regenerates_that_model_only(setup):
    make, calls, _, _ = setup
    make().run("A")
    make(phrases={"jarvie_wake": ["jarvie, wake up", "jarvie, help me"],
                  "jarvie_sleep": ["jarvie, go to sleep"]}).run("A")
    assert calls["generate", "jarvie_wake"] == 2
    assert calls["generate", "jarvie_sleep"] == 1


def test_variant_c_needs_room_noise_then_uses_it(setup):
    make, _, seen, root = setup
    pipe = make()
    with pytest.raises(RuntimeError, match="prepare_noise"):
        pipe.run("C")
    (root / "noise/room").mkdir(parents=True)
    (root / "noise/room/dorm.wav").write_bytes(b"x")
    pipe.run("C")
    assert seen["augmentation", "jarvie_wake"]["background_paths"] == [str(root / "noise/room")]


def test_interrupted_stage_reruns(setup):
    make, calls, _, root = setup
    make().run("A", models=["jarvie_wake"])
    train_dir = next((root / "stages/train").glob("jarvie_wake-*"))
    (train_dir / "_done.json").unlink()  # as if the pod stopped before training finished
    (train_dir / "jarvie_wake/jarvie_wake.pt").unlink()
    make().run("A", models=["jarvie_wake"])
    assert calls["train", "jarvie_wake"] == 2
    assert calls["generate", "jarvie_wake"] == 1


def test_results_and_pruning(setup):
    make, _, _, root = setup
    pipe = make()
    pipe.run("A")
    pipe.run("B", trial="t2", steps=10)
    df = pipe.results()
    assert list(df.variant) == ["A", "A", "B", "B"]
    assert set(df.trial) == {"base", "t2"}
    aug_dir = next((root / "stages/aug").glob("jarvie_wake-*")) / "jarvie_wake/positive_train"
    assert len(list(aug_dir.glob("*_r0.wav"))) == 5  # keep_augmented_audio
    assert len(list(aug_dir.glob("clip_??????.wav"))) == 30  # clean clips untouched


def test_augmentation_patch_keeps_loudness_and_prevents_clipping():
    rng = np.random.default_rng(0)
    audio = (0.5 * rng.standard_normal(16000)).astype(np.float32)
    module = SimpleNamespace(AudioAugmentor=FakeAugmentor)
    aug = FakeAugmentor()
    patch = {"noise_prob": 1.0, "snr_db": [10, 25], "rir_prob": 1.0, "max_peak": 0.95}
    with wp._augmentation_patch(module, patch):
        reverbed = module.AudioAugmentor.apply_rir(aug, audio)
        mixed = module.AudioAugmentor.mix_with_background(aug, reverbed)
    assert np.sqrt(np.mean(reverbed**2)) == pytest.approx(np.sqrt(np.mean(audio**2)), rel=1e-5)
    assert np.abs(mixed).max() == pytest.approx(0.95)
    assert module.AudioAugmentor.apply_rir is ORIGINAL_RIR  # restored


def test_settings_key_is_order_independent():
    assert wp.settings_key({"a": 1, "b": [1, 2]}) == wp.settings_key({"b": [1, 2], "a": 1})
    assert wp.settings_key({"a": 1}) != wp.settings_key({"a": 2})


def test_adding_room_noise_reruns_c_augmentation(setup):
    make, calls, _, root = setup
    (root / "noise/room").mkdir(parents=True)
    (root / "noise/room/dorm.wav").write_bytes(b"x")
    pipe = make()
    pipe.run("C", models=["jarvie_wake"])
    pipe.run("C", models=["jarvie_wake"])
    assert calls["augment", "jarvie_wake"] == 1  # unchanged noise: cached
    (root / "noise/room/music.wav").write_bytes(b"more noise")
    pipe.run("C", models=["jarvie_wake"])
    assert calls["augment", "jarvie_wake"] == 2  # new recording: augmentation reruns
    assert calls["generate", "jarvie_wake"] == 1  # speech synthesis still reused


def test_shared_data_dir(setup):
    make, _, _, root = setup
    smoke = make(root=root / "smoke", shared_data=root / "data")
    assert smoke.data_dir == root / "data"
    assert not (root / "smoke/data").exists()


# --- CPU budget and resuming augmentation ------------------------------------


def test_available_cpus_reads_cgroup_quota(tmp_path, monkeypatch):
    v2 = tmp_path / "cpu.max"
    monkeypatch.setattr(wp, "CGROUP_V2_CPU_MAX", v2)
    monkeypatch.setattr(wp, "CGROUP_V1_QUOTA", tmp_path / "missing_quota")
    monkeypatch.setattr(wp, "CGROUP_V1_PERIOD", tmp_path / "missing_period")

    v2.write_text("1200000 100000\n")  # a 12-CPU quota on a 128-core host
    assert wp.available_cpus() == 12

    v2.write_text("max 100000\n")  # no quota: fall back to the CPUs we're allowed to run on
    assert wp.available_cpus() >= 1

    v2.unlink()  # cgroup v1
    (tmp_path / "quota").write_text("800000")
    (tmp_path / "period").write_text("100000")
    monkeypatch.setattr(wp, "CGROUP_V1_QUOTA", tmp_path / "quota")
    monkeypatch.setattr(wp, "CGROUP_V1_PERIOD", tmp_path / "period")
    assert wp.available_cpus() == 8


def test_limit_threads_respects_existing_settings(monkeypatch):
    monkeypatch.setattr(wp, "available_cpus", lambda: 12)
    for var in wp.THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")  # a user override wins
    assert wp.limit_threads() == 12
    import os

    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "12"


def test_interrupted_extraction_does_not_redo_augmentation(setup):
    make, calls, _, root = setup
    pipe = make()
    real_extraction = pipe.lk.run_extraction

    def interrupted(cfg):
        raise KeyboardInterrupt  # the user stops the cell during feature extraction

    pipe.lk.run_extraction = interrupted
    with pytest.raises(KeyboardInterrupt):
        pipe.run("A", models=["jarvie_wake"])
    assert calls["augment", "jarvie_wake"] == 1

    pipe.lk.run_extraction = real_extraction
    pipe.run("A", models=["jarvie_wake"])
    assert calls["augment", "jarvie_wake"] == 1  # not redone
    assert calls["extract", "jarvie_wake"] == 1
    aug_dir = next((root / "stages/aug").glob("jarvie_wake-*"))
    assert (aug_dir / "_done.json").exists()


def test_augmentation_finished_before_markers_existed_is_detected(setup):
    # The run on the pod: augmentation finished, extraction interrupted, and the code that
    # wrote _augmented.json didn't exist yet.
    make, calls, _, root = setup
    pipe = make()
    real_extraction = pipe.lk.run_extraction  # shared with later make() pipelines; restore it
    pipe.lk.run_extraction = lambda cfg: (_ for _ in ()).throw(KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        pipe.run("A", models=["jarvie_wake"])
    pipe.lk.run_extraction = real_extraction
    aug_dir = next((root / "stages/aug").glob("jarvie_wake-*"))
    (aug_dir / "_augmented.json").unlink()

    make().run("A", models=["jarvie_wake"])  # new session, fresh pipeline
    assert calls["augment", "jarvie_wake"] == 1


def test_incomplete_augmentation_is_redone(setup):
    make, calls, _, root = setup
    pipe = make()
    real_extraction = pipe.lk.run_extraction  # shared with later make() pipelines; restore it
    pipe.lk.run_extraction = lambda cfg: (_ for _ in ()).throw(KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        pipe.run("A", models=["jarvie_wake"])
    pipe.lk.run_extraction = real_extraction
    aug_dir = next((root / "stages/aug").glob("jarvie_wake-*"))
    (aug_dir / "_augmented.json").unlink()
    next((aug_dir / "jarvie_wake/negative_train").glob("*_r0.wav")).unlink()  # one clip missing

    make().run("A", models=["jarvie_wake"])
    assert calls["augment", "jarvie_wake"] == 2
