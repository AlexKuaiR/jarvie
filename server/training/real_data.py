"""Turn real recordings (from record_session.py) into livekit-wakeword training clips.

livekit trains on 2.0 s clips at 16 kHz: positives end 0-200 ms after the phrase, negatives are
centred, and every file in a training folder named clip_NNNNNN.wav gets augmented (noise, reverb)
and turned into features. This module trims real recordings into that format and writes
`repeat` copies of each, so each copy is augmented differently and the real data isn't drowned
out by 20,000 synthetic clips.

Expected upload layout (as record_session.py writes it, under --split train):
    <uploads>/<speaker>/<mode>/<kind>[-<similarity>]_<position>_<condition>-<pace>_<mode>_<NNN>.wav
where mode is "wake" or "sleep" and kind is "pos" or "neg".
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
CLIP_SECONDS = 2.0  # livekit's augmentation.clip_duration: 16 embeddings
END_JITTER = 0.2  # livekit places positives 0-200 ms before the clip's end
LONG_SECONDS = 6.0  # recordings longer than this are everyday speech, cut into many clips
REAL_INDEX_START = 900_000  # clip_9NNNNN.wav: well above synthetic indices (< n_samples)


@dataclass(frozen=True)
class RealClip:
    path: Path
    speaker: str
    mode: str  # "wake" or "sleep": which phrase family it was recorded for
    kind: str  # "pos" or "neg"
    similarity: str  # "" for positives


def collect(uploads: Path) -> list[RealClip]:
    """All recordings under <uploads>/<speaker>/<mode>/, labeled from their file names."""
    clips = []
    for path in sorted(uploads.glob("*/*/*.wav")):
        speaker, mode = path.parent.parent.name, path.parent.name
        kind_field = path.stem.split("_")[0]
        kind, _, similarity = kind_field.partition("-")
        if kind not in ("pos", "neg") or mode not in ("wake", "sleep"):
            continue
        clips.append(RealClip(path, speaker, mode, kind, similarity))
    return clips


def fingerprint(clips: list[RealClip]) -> str:
    """Changes whenever a recording is added, removed or re-recorded (part of the cache key)."""
    items = [(str(c.path.relative_to(c.path.parents[2])), c.path.stat().st_size) for c in clips]
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:10]


def summary(clips: list[RealClip]) -> dict[str, dict[str, int]]:
    """Counts per speaker and label, e.g. {"me": {"wake/pos": 120, "wake/neg": 140, ...}}."""
    out: dict[str, dict[str, int]] = {}
    for c in clips:
        key = f"{c.mode}/{c.kind}"
        out.setdefault(c.speaker, {}).setdefault(key, 0)
        out[c.speaker][key] += 1
    return out


def training_roles(clips: list[RealClip], model_mode: str) -> tuple[list[RealClip], list[RealClip]]:
    """(positives, negatives) for one model. The other model's phrases are negatives too, so
    "jarvie, go to sleep" can't wake the bot, mirroring the synthetic cross-negatives."""
    positives = [c for c in clips if c.mode == model_mode and c.kind == "pos"]
    negatives = [c for c in clips if not (c.mode == model_mode and c.kind == "pos")]
    return positives, negatives


# --- audio processing ------------------------------------------------------------------------


def load(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = resample(audio, sr / SR)
    return audio


def resample(audio: np.ndarray, factor: float) -> np.ndarray:
    """Linear-interpolation resampling: factor > 1 makes the clip shorter (faster, higher)."""
    n = max(1, int(round(len(audio) / factor)))
    return np.interp(np.arange(n) * factor, np.arange(len(audio)), audio).astype(np.float32)


def speech_bounds(audio: np.ndarray, sr: int = SR) -> tuple[float, float] | None:
    """(start, end) in seconds of the speech, by frame energy relative to the noise floor."""
    frame = int(0.02 * sr)
    if len(audio) < frame:
        return None
    rms = np.array([np.sqrt(np.mean(audio[i:i + frame] ** 2)) for i in range(0, len(audio) - frame + 1, frame)])
    active = np.where(rms > max(np.percentile(rms, 10) * 4, rms.max() * 0.15, 1e-4))[0]
    if len(active) == 0:
        return None
    return active[0] * frame / sr, (active[-1] + 1) * frame / sr


def _window(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    """audio[start:start+length], zero-padded where the window runs past either end."""
    out = np.zeros(length, dtype=np.float32)
    src_lo, src_hi = max(start, 0), min(start + length, len(audio))
    if src_hi > src_lo:
        out[src_lo - start:src_hi - start] = audio[src_lo:src_hi]
    return out


def trim_positive(audio: np.ndarray, rng: random.Random, sr: int = SR) -> np.ndarray | None:
    """2.0 s ending 0-200 ms after the phrase ends, like livekit's synthetic positives."""
    bounds = speech_bounds(audio, sr)
    if bounds is None:
        return None
    end = int((bounds[1] + rng.uniform(0, END_JITTER)) * sr)
    length = int(CLIP_SECONDS * sr)
    return _window(audio, end - length, length)


def trim_negative(audio: np.ndarray, sr: int = SR) -> np.ndarray | None:
    """2.0 s centred on the speech."""
    bounds = speech_bounds(audio, sr)
    if bounds is None:
        return None
    centre = int((bounds[0] + bounds[1]) / 2 * sr)
    length = int(CLIP_SECONDS * sr)
    return _window(audio, centre - length // 2, length)


def chunk_long(audio: np.ndarray, sr: int = SR, hop_seconds: float = 1.0) -> list[np.ndarray]:
    """Cut a long everyday-speech recording into overlapping 2.0 s clips, skipping silent ones."""
    length, hop = int(CLIP_SECONDS * sr), int(hop_seconds * sr)
    chunks = []
    for start in range(0, max(len(audio) - length, 0) + 1, hop):
        chunk = audio[start:start + length]
        if len(chunk) == length and speech_bounds(chunk, sr) is not None:
            chunks.append(chunk.astype(np.float32))
    return chunks


def speaker_perturb(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Speed perturbation (x0.88-1.12): shifts pitch and vocal-tract length together, a standard
    way to make one speaker sound like several. Applied to the full recording, before trimming."""
    return resample(audio, rng.uniform(0.88, 1.12))


# --- writing training copies -----------------------------------------------------------------


def write_copies(clips: list[RealClip], role: str, dest: Path, repeat: int, speaker_aug: bool,
                 seed: str, start_index: int = REAL_INDEX_START) -> tuple[int, int]:
    """Write `repeat` training clips per recording into `dest` as clip_9NNNNN.wav.

    role is "pos" or "neg" (how this model should treat these recordings). Long recordings are
    cut into many clips and written once each (they're already many examples). Returns
    (next free index, number of recordings with no detectable speech).
    """
    import soundfile as sf

    dest.mkdir(parents=True, exist_ok=True)
    index, silent = start_index, 0
    for clip in clips:
        rng = random.Random(f"{seed}:{clip.path.name}:{clip.speaker}")
        audio = load(clip.path)
        if len(audio) / SR > LONG_SECONDS:
            for chunk in chunk_long(audio):
                sf.write(str(dest / f"clip_{index:06d}.wav"), chunk, SR)
                index += 1
            continue
        for copy in range(repeat):
            source = speaker_perturb(audio, rng) if speaker_aug and copy > 0 else audio
            out = trim_positive(source, rng) if role == "pos" else trim_negative(source)
            if out is None:
                silent += 1
                break
            sf.write(str(dest / f"clip_{index:06d}.wav"), out, SR)
            index += 1
    return index, silent
