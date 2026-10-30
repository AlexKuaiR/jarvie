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

    class Trainer:  # stand-in for livekit's WakeWordTrainer, used for warm starts
        def __init__(self, cfg):
            self.cfg = cfg
            self.model = SimpleNamespace(
                load_state_dict=lambda state: seen.__setitem__("init_state", state)
            )

        def train(self):
            calls["train", self.cfg.model_name] += 1
            seen["warm_started", self.cfg.model_name] = "init_state" in seen

        def save(self, path):
            Path(path).write_bytes(b"fine-tuned")

    return SimpleNamespace(
        WakeWordConfig=Config,
        run_generate=run_generate,
        run_augment=run_augment,
        run_extraction=run_extraction,
        run_train=run_train,
        run_export=run_export,
        run_eval=run_eval,
        augment_module=augment_module,
        WakeWordTrainer=Trainer,
        load_weights=lambda path: {"weights_from": str(path)},
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
    make(
        phrases={
            "jarvie_wake": ["jarvie, wake up", "jarvie, help me"],
            "jarvie_sleep": ["jarvie, go to sleep"],
        }
    ).run("A")
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


# --- GPU memory release between stages ---------------------------------------


def test_release_gpu_memory_without_cuda():
    wp.release_gpu_memory("test")  # no torch / no GPU here: must not fail


def test_gpu_memory_released_after_each_stage_that_ran(setup, monkeypatch):
    make, _, _, _ = setup
    released = []
    monkeypatch.setattr(wp, "release_gpu_memory", released.append)
    pipe = make()
    pipe.run("A")
    assert released == [
        "gen jarvie_wake",
        "aug jarvie_wake [A]",
        "train jarvie_wake [A/base]",
        "gen jarvie_sleep",
        "aug jarvie_sleep [A]",
        "train jarvie_sleep [A/base]",
    ]
    released.clear()
    pipe.run("A")  # everything cached: nothing ran, nothing to release
    assert released == []


def test_gpu_memory_released_even_when_a_stage_fails(setup, monkeypatch):
    make, _, _, _ = setup
    released = []
    monkeypatch.setattr(wp, "release_gpu_memory", released.append)
    pipe = make()
    real_train = pipe.lk.run_train
    pipe.lk.run_train = lambda cfg: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    with pytest.raises(RuntimeError):
        pipe.run("A", models=["jarvie_wake"])
    pipe.lk.run_train = real_train
    assert released[-1] == "train jarvie_wake [A/base]"


# --- real recordings, synthetic share, warm start (M2-M4) ---------------------------------


def add_real_recordings(root: Path, n: int = 3) -> None:
    import soundfile as sf

    t = np.arange(4 * 16000) / 16000
    clip = (0.3 * np.sin(2 * np.pi * 220 * t) * ((t > 1) & (t < 2.2))).astype("float32")
    for speaker, mode, kind in [
        ("me", "wake", "pos"),
        ("me", "wake", "neg-very-similar"),
        ("me", "sleep", "pos"),
    ]:
        folder = root / "real/uploads/v1" / speaker / mode
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            sf.write(folder / f"{kind}_desk_quiet-normal_{mode}_{i:03d}.wav", clip, 16000)


def train_split(root: Path, model: str, label_glob: str, split: str) -> list[str]:
    aug = [
        d
        for d in (root / "stages/aug").glob(f"{model}-*")
        if label_glob in (d / "settings.json").read_text()
    ]
    assert len(aug) == 1, aug
    return sorted(p.name for p in (aug[0] / model / split).glob("clip_??????.wav"))


def test_m2_adds_real_copies_and_reuses_speech(setup):
    make, calls, _, root = setup
    add_real_recordings(root)
    pipe = make()
    pipe.run("B")
    pipe.run("B", real="v1", real_repeat=4)
    assert count(calls, "generate") == 2  # speech reused
    pos = train_split(root, "jarvie_wake", '"real": "v1"', "positive_train")
    real_pos = [n for n in pos if n >= "clip_900000.wav"]
    assert len(pos) - len(real_pos) == 30  # all synthetic clips
    assert len(real_pos) == 3 * 4  # 3 wake positives x 4 copies
    neg = train_split(root, "jarvie_wake", '"real": "v1"', "negative_train")
    assert (
        len([n for n in neg if n >= "clip_900000.wav"]) == (3 + 3) * 4
    )  # near-misses + sleep phrases
    assert (root / "models/jarvie_wake__B+real-v1__base.onnx").exists()


def test_m3_real_only_keeps_synthetic_test_splits(setup):
    make, _, _, root = setup
    add_real_recordings(root)
    make().run("B", real="v1", synthetic=False, real_repeat=2, models=["jarvie_wake"])
    pos = train_split(root, "jarvie_wake", "synthetic_share", "positive_train")
    assert pos and all(n >= "clip_900000.wav" for n in pos)  # no synthetic training clips
    assert (
        len(train_split(root, "jarvie_wake", "synthetic_share", "positive_test")) == 30
    )  # validation kept
    assert (root / "models/jarvie_wake__B+real-v1-nosyn__base.onnx").exists()


def test_synthetic_share_keeps_a_fixed_fraction(setup):
    make, _, _, root = setup
    add_real_recordings(root)
    make().run("B", real="v1", synthetic_share=0.2, real_repeat=1, models=["jarvie_wake"])
    pos = train_split(root, "jarvie_wake", "synthetic_share", "positive_train")
    assert len([n for n in pos if n < "clip_900000.wav"]) == 6  # 20% of 30


def test_m4_fine_tunes_from_b(setup):
    make, calls, seen, root = setup
    add_real_recordings(root)
    pipe = make()
    with pytest.raises(RuntimeError, match="no finished training run"):
        pipe.run("B", trial="ft", real="v1", init_from=("B", "base"), models=["jarvie_wake"])
    pipe.run("B", models=["jarvie_wake"])
    pipe.run(
        "B",
        trial="ft",
        real="v1",
        synthetic_share=0.1,
        init_from=("B", "base"),
        steps=50,
        models=["jarvie_wake"],
    )
    assert seen["warm_started", "jarvie_wake"] is True
    assert "jarvie_wake.pt" in seen["init_state"]["weights_from"]
    assert (root / "models/jarvie_wake__B+real-v1-syn10__ft.onnx").exists()


def test_new_options_dont_change_existing_cache_keys(setup):
    make, calls, _, root = setup
    make().run("B")
    make().run("B", real=None, synthetic=True)  # explicit defaults: same stages
    assert count(calls, "augment") == 2 and count(calls, "train") == 2


def test_adding_recordings_reruns_only_real_runs(setup):
    make, calls, _, root = setup
    add_real_recordings(root, n=2)
    pipe = make()
    pipe.run("B", models=["jarvie_wake"])
    pipe.run("B", real="v1", real_repeat=1, models=["jarvie_wake"])
    add_real_recordings(root, n=3)  # one more recording per label
    pipe.run("B", models=["jarvie_wake"])
    pipe.run("B", real="v1", real_repeat=1, models=["jarvie_wake"])
    assert calls["augment", "jarvie_wake"] == 3  # B once, real run twice


def test_bad_data_options(setup):
    make, _, _, _ = setup
    with pytest.raises(ValueError, match="needs real"):
        make().run("B", synthetic=False)
    with pytest.raises(ValueError, match="between 0 and 1"):
        make().run("B", synthetic_share=1.5)
