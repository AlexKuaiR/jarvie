"""Cached, resumable training pipeline for the Jarvie wake/sleep models (livekit-wakeword).

Built for a RunPod pod with a network volume mounted at /workspace (works on Colab too).
livekit-wakeword runs everything for one model inside a single folder; this module splits
that into three cached stages, each in its own folder named after a hash of the settings
that affect it:

    gen    speech synthesis              <- phrases, negatives, sample counts, speeds
    aug    augmentation + features       <- gen + augmentation variant (A, B, C, ...)
    train  training, export, evaluation  <- aug + training settings (a "trial")

Each stage links in the previous stage's outputs and writes a _done.json marker when it
finishes. Re-running with unchanged settings skips straight to the cached result, in this
session or a later one; changing a setting reruns only that stage and the ones after it.
So A and B share one speech-synthesis run, and tuning trials share one augmentation run.

Layout under the workspace root (default /workspace/wakeword):

    data/                    livekit setup: Piper voice, ACAV100M features, RIRs, MUSAN noise
    noise/uploads/<set>/     your raw background recordings (any audio format)
    noise/<set>/             the same, converted to 16 kHz mono WAV
    stages/{gen,aug,train}/  cached stage folders
    models/                  exported .onnx files, named <model>__<variant>__<trial>.onnx
    results.csv              one row per finished model: settings and metrics
"""

from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def available_cpus() -> int:
    """CPUs this process may actually use.

    Inside a container (e.g. a RunPod pod), os.cpu_count() and nproc report the host machine's
    cores (128) even when the pod is limited to a dozen by a cgroup CPU quota. Sizing thread
    pools by the host count oversubscribes the real CPUs badly (feature extraction ran at
    ~1 clip/s), so read the quota first.
    """
    try:
        quota, period = CGROUP_V2_CPU_MAX.read_text().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    try:
        quota, period = int(CGROUP_V1_QUOTA.read_text()), int(CGROUP_V1_PERIOD.read_text())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS has no sched_getaffinity
        return os.cpu_count() or 1


def limit_threads() -> int:
    """Size numpy / torch / OpenMP thread pools to the real CPU budget (unless already set).

    These libraries read the variables once, when they're first imported, so this runs at the
    top of this module, before numpy is imported. In the notebook, import this module before
    torch, and restart the kernel for a change to take effect.
    """
    n = available_cpus()
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))
    return n


limit_threads()

import numpy as np  # noqa: E402  (after limit_threads, so numpy's thread pool is sized correctly)

try:  # imported as training.wakeword_pipeline (tests) or as a plain module next to it (RunPod)
    from . import real_data
except ImportError:
    import real_data  # type: ignore[no-redef]

SPLITS = [
    "positive_train",
    "positive_test",
    "negative_train",
    "negative_test",
    "background_train",
    "background_test",
]
FEATURE_FILES = [
    "positive_features_train.npy",
    "positive_features_test.npy",
    "negative_features_train.npy",
    "negative_features_test.npy",
    "background_noise_features_train.npy",
    "background_noise_features_test.npy",
]
CLEAN_CLIP = re.compile(r"^clip_\d{6}\.wav$")
AUGMENTED_CLIP = re.compile(r"^clip_\d{6}_r\d+\.wav$")
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}

# Gentler, level-safe augmentation used by B and C (see _augmentation_patch).
REALISTIC = {"noise_prob": 0.7, "snr_db": [10.0, 25.0], "rir_prob": 0.3, "max_peak": 0.95}

DEFAULT_AUGMENTATIONS: dict[str, dict[str, Any]] = {
    # A: livekit's own augmentation, untouched (noise on every clip at 5-15 dB, reverb that can clip).
    "A": {"patch": None, "rounds": 1, "backgrounds": ["musan"]},
    # B: noise on 70% of clips at 10-25 dB, reverb on 30% at the same loudness, no clipping.
    "B": {"patch": REALISTIC, "rounds": 1, "backgrounds": ["musan"]},
    # C: like B, but the background noise is your own room (upload to noise/uploads/room/).
    "C": {"patch": REALISTIC, "rounds": 1, "backgrounds": ["room"]},
}

# Settings that change what speech synthesis produces (anything else in `generation` is ignored
# for the cache key, e.g. tts_batch_size only changes speed).
_GEN_KEYS = [
    "n_samples",
    "n_samples_val",
    "n_background_samples",
    "n_background_samples_val",
    "length_scales",
    "noise_scales",
    "noise_scale_ws",
    "slerp_weights",
    "max_speakers",
]


def settings_key(settings: Any) -> str:
    """Short, stable hash of a settings structure; names a cached stage folder."""
    blob = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _load_livekit() -> SimpleNamespace:
    """Import livekit-wakeword only when a stage actually runs (it pulls in torch etc.)."""
    import onnxruntime as ort
    import torch
    from livekit.wakeword import (
        WakeWordConfig,
        run_augment,
        run_eval,
        run_export,
        run_extraction,
        run_generate,
        run_train,
    )
    from livekit.wakeword.data import augment as augment_module
    from livekit.wakeword.training.trainer import WakeWordTrainer

    cpus = available_cpus()
    # Feature extraction runs two tiny ONNX models once per clip; a few threads is fastest for
    # such small work, and more only adds synchronization overhead.
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = min(cpus, 8)
    sess_options.inter_op_num_threads = 1
    torch.set_num_threads(cpus)
    print(
        f"threads: {cpus} CPUs available; onnxruntime {sess_options.intra_op_num_threads}, torch {cpus}"
    )
    return SimpleNamespace(
        WakeWordConfig=WakeWordConfig,
        run_generate=run_generate,
        run_augment=run_augment,
        run_extraction=lambda cfg: run_extraction(cfg, sess_options),
        run_train=run_train,
        run_export=run_export,
        run_eval=run_eval,
        augment_module=augment_module,
        WakeWordTrainer=WakeWordTrainer,
        load_weights=lambda path: torch.load(path, map_location="cpu"),
    )


def _link_files(
    src_dir: Path, dst_dir: Path, pattern: re.Pattern | None = None, names: list[str] | None = None
) -> int:
    """Hard-link (or copy, if links aren't supported) files from src_dir into dst_dir.

    Hard links make a second folder that shares the same disk blocks, so reusing a stage's
    output costs almost no space or time on the network volume.
    """
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        if (pattern and not pattern.match(f.name)) or (names is not None and f.name not in names):
            continue
        target = dst_dir / f.name
        if target.exists():
            continue
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)
        count += 1
    return count


@contextlib.contextmanager
def _augmentation_patch(augment_module, patch: dict | None) -> Iterator[None]:
    """Temporarily replace livekit's reverb and noise steps with level-safe versions.

    run_augment() builds an AudioAugmentor and calls these methods for every clip, so patching
    the class changes every round. The originals are always restored afterwards.
    """
    if patch is None:
        yield
        return
    cls = augment_module.AudioAugmentor
    original_rir, original_mix = cls.apply_rir, cls.mix_with_background

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    def apply_rir(self, audio, p=0.5):
        # Reverb with probability rir_prob, rescaled to the input's loudness (livekit's isn't).
        out = original_rir(self, audio, p=patch["rir_prob"])
        if out is audio:
            return audio
        r_in, r_out = rms(audio), rms(out)
        return (out * (r_in / r_out)).astype(np.float32) if r_out > 0 else out

    def mix_with_background(self, audio, snr_db_range=(5.0, 15.0)):
        # Noise on noise_prob of clips at a gentler SNR, then scale down anything that would clip
        # (this is the last step before livekit writes 16-bit WAV, which hard-clips above 1.0).
        if random.random() < patch["noise_prob"]:
            audio = original_mix(self, audio, snr_db_range=tuple(patch["snr_db"]))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > patch["max_peak"]:
            audio = audio * (patch["max_peak"] / peak)
        return np.asarray(audio, dtype=np.float32)

    cls.apply_rir, cls.mix_with_background = apply_rir, mix_with_background
    try:
        yield
    finally:
        cls.apply_rir, cls.mix_with_background = original_rir, original_mix


def release_gpu_memory(label: str) -> None:
    """Free GPU memory left over from a stage that just ran, and report how much was freed.

    Every stage runs in the same Python process, so the previous stage's model, optimizer and GPU
    tensors can still be holding memory when the next one starts (jarvie_sleep's speech synthesis
    ran out of memory right after jarvie_wake's training). gc.collect() frees objects that are
    only kept alive by reference cycles; empty_cache() then hands PyTorch's freed blocks back to
    the GPU. empty_cache() on its own only releases memory nothing is using anymore.
    """
    try:
        import torch
    except ImportError:  # no torch (e.g. the Mac test environment): nothing to free on a GPU
        gc.collect()
        return
    if not torch.cuda.is_available():
        gc.collect()
        return
    before = torch.cuda.memory_allocated() / 1e9
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() / 1e9
    print(f"  gpu   after {label}: {before:.1f} -> {after:.1f} GB allocated")


@contextlib.contextmanager
def _releasing_gpu(label: str) -> Iterator[None]:
    """Run a stage's work, then release GPU memory even if it failed or was interrupted."""
    try:
        yield
    finally:
        release_gpu_memory(label)


@dataclass
class Pipeline:
    """Everything that defines a training run; edit the fields, then call run()."""

    root: Path
    phrases: dict[str, list[str]]
    negatives: dict[str, list[str]]
    generation: dict[str, Any]
    training: dict[str, Any]
    augmentations: dict[str, dict[str, Any]] = field(
        default_factory=lambda: json.loads(json.dumps(DEFAULT_AUGMENTATIONS))
    )
    keep_augmented_audio: int = (
        20  # augmented WAVs kept per split for listening; the rest are deleted
    )
    shared_data: Path | None = (
        None  # use another pipeline's downloaded data (e.g. for a smoke test)
    )
    livekit: SimpleNamespace | None = None  # injected in tests; loaded lazily otherwise

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for d in ("noise/uploads", "real/uploads", "stages", "models"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --- paths ------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return Path(self.shared_data) if self.shared_data else self.root / "data"

    @property
    def lk(self) -> SimpleNamespace:
        if self.livekit is None:
            self.livekit = _load_livekit()
        return self.livekit

    def _background_dirs(self, names: list[str]) -> list[str]:
        dirs = []
        for name in names:
            d = self.data_dir / "backgrounds" if name == "musan" else self.root / "noise" / name
            if not any(d.rglob("*.wav")):
                hint = (
                    "run setup_data()"
                    if name == "musan"
                    else f"upload recordings to {self.root / 'noise/uploads' / name} "
                    f"and run prepare_noise({name!r})"
                )
                raise RuntimeError(f"no background audio in {d}: {hint}")
            dirs.append(str(d))
        return dirs

    # --- one-time setup ---------------------------------------------------

    def setup_data(self) -> None:
        """Download livekit's training data once (~16 GB); skipped when already on the volume."""
        marker = self.data_dir / "_setup_done.json"
        if marker.exists():
            print(f"data: already set up in {self.data_dir}")
            return
        subprocess.run(["livekit-wakeword", "setup", "--data-dir", str(self.data_dir)], check=True)
        marker.write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}))

    def prepare_noise(self, name: str = "room") -> float:
        """Convert recordings in noise/uploads/<name>/ to 16 kHz mono WAV in noise/<name>/.

        livekit reads background files without resampling, so a 44.1 kHz phone recording would
        otherwise be mixed in at the wrong speed and pitch. Re-running only converts files that
        were added or changed since last time. Returns minutes of audio available.
        """
        import librosa
        import soundfile as sf

        src, dst = self.root / "noise" / "uploads" / name, self.root / "noise" / name
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
        if not files:
            raise RuntimeError(
                f"no recordings in {src} (any of: {', '.join(sorted(AUDIO_SUFFIXES))})"
            )
        dst.mkdir(parents=True, exist_ok=True)
        total_s, converted = 0.0, 0
        for f in files:
            out = dst / f"{f.relative_to(src).with_suffix('').as_posix().replace('/', '__')}.wav"
            if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
                total_s += sf.info(str(out)).duration
                continue
            y, _ = librosa.load(str(f), sr=16000, mono=True)
            sf.write(str(out), y, 16000, subtype="FLOAT")
            total_s += len(y) / 16000
            converted += 1
        print(
            f"noise {name!r}: {len(files)} file(s), {converted} newly converted, "
            f"{total_s / 60:.1f} min -> {dst}"
        )
        return total_s / 60

    def real_clips(self, name: str) -> list:
        """Your recordings in real/uploads/<name>/<speaker>/<mode>/ (record_session.py --split train)."""
        uploads = self.root / "real" / "uploads" / name
        clips = real_data.collect(uploads)
        if not clips:
            raise RuntimeError(
                f"no recordings in {uploads}/<speaker>/<wake|sleep>/ "
                "(upload ~/jarvie-data/real/<name>/ from the Mac here)"
            )
        return clips

    def prepare_real(self, name: str = "v1") -> dict:
        """Check the uploaded recordings and print what's there, per speaker and label."""
        counts = real_data.summary(self.real_clips(name))
        for speaker, c in counts.items():
            print(
                f"real {name!r}, speaker {speaker}: "
                + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
            )
        return counts

    # --- stages -----------------------------------------------------------

    def _config(
        self,
        model: str,
        output_dir: Path,
        augmentation: dict | None = None,
        training: dict | None = None,
    ):
        return self.lk.WakeWordConfig(
            model_name=model,
            target_phrases=self.phrases[model],
            custom_negative_phrases=self.negatives.get(model, []),
            data_dir=str(self.data_dir),
            output_dir=str(output_dir),
            **self.generation,
            **({"augmentation": augmentation} if augmentation else {}),
            **(training or {}),
        )

    def _stage(self, kind: str, model: str, settings: dict) -> tuple[Path, Path, bool]:
        """Return (stage folder, model folder inside it, already done?) for these settings."""
        stage_dir = self.root / "stages" / kind / f"{model}-{settings_key(settings)}"
        model_dir = stage_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str))
        return stage_dir, model_dir, (stage_dir / "_done.json").exists()

    @staticmethod
    def _finish(stage_dir: Path, started: float, extra: dict | None = None) -> None:
        info = {
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "minutes": round((time.time() - started) / 60, 1),
            **(extra or {}),
        }
        (stage_dir / "_done.json").write_text(json.dumps(info, indent=2, default=str))

    def generate(self, model: str) -> tuple[Path, str]:
        settings = {
            "phrases": self.phrases[model],
            "negatives": self.negatives.get(model, []),
            **{k: self.generation[k] for k in _GEN_KEYS if k in self.generation},
        }
        stage_dir, _, done = self._stage("gen", model, settings)
        key = settings_key(settings)
        if done:
            print(f"  gen   {model}: cached ({stage_dir.name})")
            return stage_dir, key
        print(f"  gen   {model}: running ({stage_dir.name})")
        started = time.time()
        with _releasing_gpu(f"gen {model}"):
            self.lk.run_generate(self._config(model, stage_dir))  # livekit resumes partial splits
        self._finish(stage_dir, started)
        return stage_dir, key

    def _noise_fingerprint(self, names: list[str]) -> dict[str, str]:
        """Hash of each of your noise folders' files, so adding recordings reruns that augmentation.

        ("musan" is livekit's fixed download and doesn't need one.)
        """
        out = {}
        for name in names:
            if name != "musan":
                files = sorted((self.root / "noise" / name).rglob("*.wav"))
                out[name] = settings_key([(f.name, f.stat().st_size) for f in files])
        return out

    def augment(
        self,
        model: str,
        variant: str,
        gen_dir: Path,
        gen_key: str,
        data: dict | None = None,
        label: str | None = None,
    ) -> tuple[Path, str]:
        """Augmentation + features. `data` (only set when it differs from the default of "all
        synthetic, no real recordings") chooses the training clips; see run()."""
        spec = self.augmentations[variant]
        label = label or variant
        settings = {
            "gen": gen_key,
            "variant": variant,
            **spec,
            "noise": self._noise_fingerprint(spec["backgrounds"]),
        }
        real_clips: list = []
        if data:
            if data.get("real"):
                real_clips = self.real_clips(data["real"])
                data = {**data, "fingerprint": real_data.fingerprint(real_clips)}
            settings["data"] = data
        stage_dir, model_dir, done = self._stage("aug", model, settings)
        key = settings_key(settings)
        if done:
            print(f"  aug   {model} [{label}]: cached ({stage_dir.name})")
            return stage_dir, key
        print(f"  aug   {model} [{label}]: running ({stage_dir.name})")
        started = time.time()
        for split in SPLITS:
            if data and split in ("positive_train", "negative_train"):
                self._link_synthetic_share(
                    gen_dir / model / split,
                    model_dir / split,
                    data.get("synthetic_share", 1.0),
                    seed=key,
                )
            else:  # test and background splits always stay synthetic (livekit's validation)
                _link_files(gen_dir / model / split, model_dir / split, pattern=CLEAN_CLIP)
        if real_clips:
            self._add_real_copies(model, model_dir, real_clips, data, seed=key)
        augmentation = {
            "rounds": spec["rounds"],
            "background_paths": self._background_dirs(spec["backgrounds"]),
            "rir_paths": [str(self.data_dir / "rirs")],
        }
        cfg = self._config(model, stage_dir, augmentation=augmentation)
        # Augmentation and feature extraction can each take hours, and the stage's _done marker
        # is only written after both. Record augmentation separately, so a stopped pod or an
        # interrupted cell doesn't redo it.
        augmented_marker = stage_dir / "_augmented.json"
        with _releasing_gpu(f"aug {model} [{label}]"):
            if augmented_marker.exists() or self._augmentation_complete(model_dir, spec["rounds"]):
                print(
                    f"  aug   {model} [{label}]: augmentation already complete, extracting features"
                )
            else:
                with _augmentation_patch(self.lk.augment_module, spec.get("patch")):
                    self.lk.run_augment(cfg)
            augmented_marker.write_text(
                json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")})
            )
            self.lk.run_extraction(cfg)
        self._prune_augmented_audio(model_dir)
        self._finish(stage_dir, started)
        return stage_dir, key

    @staticmethod
    def _link_synthetic_share(src: Path, dst: Path, share: float, seed: str) -> None:
        """Link all (share=1), none (0) or a fixed random fraction of the synthetic clips."""
        if share <= 0 or not src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            return
        names = sorted(p.name for p in src.iterdir() if CLEAN_CLIP.match(p.name))
        if share < 1:
            rng = random.Random(f"{seed}:{src.name}")
            names = sorted(rng.sample(names, max(1, round(len(names) * share))))
        _link_files(src, dst, names=names)

    def _add_real_copies(
        self, model: str, model_dir: Path, clips: list, data: dict, seed: str
    ) -> None:
        """Trim your recordings to the training format and write `repeat` copies of each into the
        training folders (each copy is then augmented differently)."""
        mode = model.rsplit("_", 1)[-1]  # jarvie_wake -> wake, jarvie_sleep -> sleep
        positives, negatives = real_data.training_roles(clips, mode)
        repeat, speaker_aug = data.get("repeat", 20), data.get("speaker_aug", True)
        end_pos, silent_pos = real_data.write_copies(
            positives, "pos", model_dir / "positive_train", repeat, speaker_aug, seed
        )
        end_neg, silent_neg = real_data.write_copies(
            negatives, "neg", model_dir / "negative_train", repeat, speaker_aug, seed
        )
        start = real_data.REAL_INDEX_START
        print(
            f"  real  {model}: {len(positives)} positive and {len(negatives)} negative recordings -> "
            f"{end_pos - start} + {end_neg - start} training clips"
            + (
                f" ({silent_pos + silent_neg} without detectable speech skipped)"
                if silent_pos + silent_neg
                else ""
            )
        )

    @staticmethod
    def _augmentation_complete(model_dir: Path, rounds: int) -> bool:
        """True if every split already has `rounds` augmented versions of each clean clip.

        Covers runs interrupted before the _augmented.json marker existed.
        """
        found_any = False
        for split in SPLITS:
            d = model_dir / split
            if not d.exists():
                continue
            names = [p.name for p in d.iterdir()]
            clean = sum(1 for n in names if CLEAN_CLIP.match(n))
            augmented = sum(1 for n in names if AUGMENTED_CLIP.match(n))
            if clean == 0:
                continue
            found_any = True
            if augmented < clean * rounds:
                return False
        return found_any

    def _prune_augmented_audio(self, model_dir: Path) -> None:
        """Training only reads the extracted features, so most augmented WAVs can go."""
        for split in SPLITS:
            clips = sorted(
                p for p in (model_dir / split).glob("*.wav") if AUGMENTED_CLIP.match(p.name)
            )
            for p in clips[self.keep_augmented_audio :]:
                p.unlink()

    def _trained_weights(self, model: str, label: str, trial: str) -> Path:
        """The .pt of the most recent finished training run for (variant label, trial)."""
        matches = []
        for done in (self.root / "stages" / "train").glob(f"{model}-*/_done.json"):
            info = json.loads(done.read_text())
            if info.get("variant") == label and info.get("trial") == trial:
                matches.append((info.get("finished", ""), done.parent / model / f"{model}.pt"))
        if not matches:
            raise RuntimeError(
                f"no finished training run for {model} [{label}/{trial}] to start from; "
                f"run pipe.run({label!r}, trial={trial!r}) first"
            )
        return max(matches)[1]

    def _warm_start_train(self, cfg, init_weights: Path, model_dir: Path, model: str) -> None:
        """livekit's training loop, but starting from existing weights instead of random ones.

        WakeWordTrainer builds a fresh model in __init__ and trains it in train(); loading the
        weights in between turns the same 3-phase schedule into fine-tuning.
        """
        trainer = self.lk.WakeWordTrainer(cfg)
        try:
            trainer.model.load_state_dict(self.lk.load_weights(init_weights))
        except RuntimeError as e:  # different model_type / model_size
            raise RuntimeError(
                f"can't start from {init_weights}: it has a different architecture "
                f"(keep model settings identical when fine-tuning). {e}"
            ) from e
        trainer.train()
        trainer.save(model_dir / f"{model}.pt")

    def train(
        self,
        model: str,
        variant: str,
        trial: str,
        aug_dir: Path,
        aug_key: str,
        training: dict,
        label: str | None = None,
        init_from: tuple[str, str] | None = None,
    ) -> dict:
        label = label or variant
        settings = {"aug": aug_key, **training}
        init_weights = None
        if init_from:
            init_weights = self._trained_weights(model, *init_from)
            settings["init"] = init_weights.parent.parent.name  # that run's stage folder
        stage_dir, model_dir, done = self._stage("train", model, settings)
        if done:
            print(f"  train {model} [{label}/{trial}]: cached ({stage_dir.name})")
            return json.loads((stage_dir / "_done.json").read_text())
        print(
            f"  train {model} [{label}/{trial}]: running ({stage_dir.name})"
            + (f", starting from {init_from[0]}/{init_from[1]}" if init_from else "")
        )
        started = time.time()
        _link_files(aug_dir / model, model_dir, names=FEATURE_FILES)
        cfg = self._config(model, stage_dir, training=training)
        # Train, export and evaluate are resumable individually: a finished .pt isn't retrained.
        with _releasing_gpu(f"train {model} [{label}/{trial}]"):
            if not (model_dir / f"{model}.pt").exists():
                if init_weights:
                    self._warm_start_train(cfg, init_weights, model_dir, model)
                else:
                    self.lk.run_train(cfg)
            onnx_path = Path(self.lk.run_export(cfg))
            metrics = self.lk.run_eval(cfg, onnx_path)
        exported = self.root / "models" / f"{model}__{label}__{trial}.onnx"
        shutil.copy2(onnx_path, exported)
        result = {
            "model": model,
            "variant": label,
            "trial": trial,
            "onnx": str(exported),
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        self._finish(stage_dir, started, result)
        self._append_result(result, training)
        return result

    # --- entry points -----------------------------------------------------

    def run(
        self,
        variant: str,
        trial: str = "base",
        models: list[str] | None = None,
        *,
        real: str | None = None,
        real_repeat: int = 20,
        speaker_aug: bool = True,
        synthetic: bool = True,
        synthetic_share: float | None = None,
        init_from: tuple[str, str] | None = None,
        **training_overrides: Any,
    ) -> list[dict]:
        """Run (or reuse) every stage for one augmentation variant and one training trial.

        Training data (each option only reruns augmentation and training; speech is reused):
            real="v1"            add your recordings from real/uploads/v1/ (M2)
            real_repeat=20       copies per recording, each augmented differently
            speaker_aug=True     speed-perturb copies so your voice also stands for other voices
            synthetic=False      leave the synthetic clips out of training (M3, real-only)
            synthetic_share=0.1  keep only this fraction of synthetic clips (e.g. M4's rehearsal set)
        Training (only reruns training):
            init_from=("B", "base")  fine-tune that finished model instead of training from scratch (M4)
            any livekit training setting: steps=..., learning_rate=..., max_negative_weight=...

        Examples:
            pipe.run("B")                                              # M1 synthetic
            pipe.run("B", real="v1")                                   # M2 synthetic + real
            pipe.run("B", real="v1", synthetic=False)                  # M3 real only
            pipe.run("B", trial="ft", real="v1", synthetic_share=0.1,  # M4 fine-tune B on real
                     init_from=("B", "base"), steps=5000, learning_rate=1e-5)
        """
        if variant not in self.augmentations:
            raise KeyError(f"unknown variant {variant!r}; known: {', '.join(self.augmentations)}")
        share = 0.0 if not synthetic else synthetic_share
        if share is not None and not 0 <= share <= 1:
            raise ValueError("synthetic_share must be between 0 and 1")
        if real is None and share == 0:
            raise ValueError("no training clips: synthetic=False needs real=...")

        # Only non-default options go into the cache key, so existing runs stay cached.
        data: dict = {}
        if real:
            data.update(real=real, repeat=real_repeat, speaker_aug=speaker_aug)
        if share is not None and share < 1:
            data["synthetic_share"] = share
        label = variant + (f"+real-{real}" if real else "")
        if share == 0:
            label += "-nosyn"
        elif share is not None and share < 1:
            label += f"-syn{round(share * 100)}"

        training = {**self.training, **training_overrides}
        results = []
        for model in models or list(self.phrases):
            print(f"=== {model}: {label}, trial {trial} ===")
            gen_dir, gen_key = self.generate(model)
            aug_dir, aug_key = self.augment(model, variant, gen_dir, gen_key, data or None, label)
            results.append(
                self.train(
                    model,
                    variant,
                    trial,
                    aug_dir,
                    aug_key,
                    training,
                    label=label,
                    init_from=init_from,
                )
            )
        return results

    def _append_result(self, result: dict, training: dict) -> None:
        path = self.root / "results.csv"
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            **result,
            "training": json.dumps(training, sort_keys=True, default=str),
        }
        new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)

    def results(self):
        """All finished runs as a DataFrame (newest last)."""
        import pandas as pd

        path = self.root / "results.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def disk_usage(self) -> dict[str, str]:
        """Size of each top-level folder, to keep an eye on the network volume."""
        out = {}
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            out[d.name] = f"{size / 1e9:.1f} GB"
        return out
