"""Evaluate the livekit-wakeword wake/sleep classifiers on recorded clips.

The classifiers don't take raw audio. They take 16 consecutive 96-dim speech embeddings
(one every 80 ms, each summarizing ~0.76 s of audio), produced by livekit-wakeword's
feature models (the same openWakeWord-compatible mel spectrogram + speech embedding):
16 kHz audio -> mel spectrogram -> embedding. So scoring is two steps: featurize the
clip into embedding frames, then slide the classifier's window over those frames.

The featurizer must preprocess audio exactly like training did: livekit-wakeword feeds the
mel model float audio in [-1, 1]. (openWakeWord's own featurizer uses int16-scale audio,
which shifts the features and the scores, so it is not interchangeable.)

Typical use:
    session, name, window = load_model("models/jarvie_wake.onnx")
    featurize = make_featurizer()
    df = score_all(session, audio_dir, name, window, hop=1, featurize=featurize)
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf

SR = 16000  # the feature models expect 16 kHz mono audio

# Turns a clip's samples into embedding frames, shape (n_frames, 96).
Featurizer = Callable[[np.ndarray], np.ndarray]


def load_model(path: str | Path) -> tuple[ort.InferenceSession, str, int]:
    """Load a classifier. Returns (session, input name, window length in embedding frames)."""
    session = ort.InferenceSession(str(path))
    inp = session.get_inputs()[0]
    return session, inp.name, inp.shape[1]  # input is [batch, frames, 96]; frames = 16 here


def make_featurizer() -> Featurizer:
    """Build the real audio -> embedding-frames function from livekit-wakeword's bundled
    feature models, the same code path its training uses.

    Passed into score_clip/score_all as an argument (rather than called inside them) so
    tests can swap in a trivial fake and run without model files.
    """
    from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
    from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path

    mel_frontend = MelSpectrogramFrontend(get_mel_model_path())
    speech_embedding = SpeechEmbedding(get_embedding_model_path())

    def featurize(audio: np.ndarray) -> np.ndarray:
        # float32 in [-1, 1], exactly what sf.read gives and what training used
        mel = mel_frontend(audio.astype(np.float32))       # (1, mel_frames, 32)
        return speech_embedding.extract_embeddings(mel)[0]  # (n_frames, 96)

    return featurize


def score_clip(
    session, path: Path, input_name: str, window: int, hop: int, featurize: Featurizer,
    pad_seconds: float = 1.0,
) -> dict:
    """Score one clip: max score, the per-window scores, and its duration in seconds.

    window and hop are in embedding frames (80 ms each): window=16, hop=1 scores the
    clip every 80 ms. Each embedding covers ~0.76 s of audio, so a 16-frame window spans
    ~1.96 s (0.76 + 15 x 0.08).

    pad_seconds of silence are added before and after the clip, as livekit's own
    predict_clip does. The model was trained with each phrase ending right at the end of its
    window, so a phrase that ends at the very end of a recording otherwise has no room to line
    up; padding also lets a phrase at the very start get a full window. The reported duration
    is the unpadded clip's.
    """
    audio, sr = sf.read(path, dtype="float32")
    if sr != SR:
        raise ValueError(f"expected {SR} Hz audio, got {sr} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo -> mono

    duration = len(audio) / sr
    if pad_seconds > 0:
        pad = np.zeros(int(pad_seconds * sr), dtype=audio.dtype)
        audio = np.concatenate([pad, audio, pad])

    frames = featurize(audio)
    scores = []
    # + 1 so the last full window (ending exactly at the last frame) is scored too.
    for start in range(0, len(frames) - window + 1, hop):
        chunk = frames[start:start + window]
        # The model takes a batch: (1, window, embedding_dim)
        out = session.run(None, {input_name: chunk[None].astype(np.float32)})
        scores.append(float(out[0].squeeze()))
    return {
        "max": max(scores) if scores else 0.0,
        "scores": scores,
        "duration": duration,
    }

def score_all(
    session, audio_dir: Path, input_name: str, window: int, hop: int, featurize: Featurizer,
    pad_seconds: float = 1.0,
) -> pd.DataFrame:
    """Score every wav in a directory. One row per file."""
    rows = []
    skipped = []
    for path in sorted(audio_dir.glob("*.wav")):
        parts = path.stem.split("_")
        if len(parts) != 5:
            print(f"skipping malformed name: {path.name}")
            skipped.append(path.name)
            continue
        try:
            result = score_clip(session, path, input_name, window, hop, featurize, pad_seconds)
        except Exception as e:                      # bad sample rate, corrupt file
            skipped.append(f"{path.name} ({e})")
            continue

        rows.append({
            "file": path.name,
            "kind": parts[0],
            "position": parts[1],
            "condition": parts[2],
            "phrase": parts[3],
            "index": int(parts[4]),
            "duration": result["duration"],
            "max_score": result["max"],
            "frame_scores": result["scores"],
        })
    if skipped:
        print(f"skipped {len(skipped)} file(s):")
        for s in skipped[:10]:
            print(f"  {s}")
    return pd.DataFrame(rows)

def event_count(scores: list[float], threshold: float) -> int:
    """Count detections, collapsing consecutive above-threshold frames.
 
    With overlapping windows one utterance produces several adjacent
    frames above threshold. Those are one false accept, not several.
    """

    events = 0
    previous = False
    for s in scores:
        if s >= threshold and not previous:
            # activated
            events += 1
        previous = s >= threshold
    return events

def sweeps(df: pd.DataFrame, thresholds: np.ndarray, phrase: str) -> pd.DataFrame:
    """Recall and false accepts per hour at each threshold.
 
    Pass phrase="wake" or "sleep" — the two models have different
    operating points and averaging them is meaningless.
    """
    df = df[df.phrase == phrase]
    pos = df[df.kind == 'pos']
    neg = df[df.kind == 'neg']
    neg_hours = neg.duration.sum() / 3600
    if neg_hours == 0:
        raise ValueError(f"no negative clips for phrase {phrase!r}; can't compute false accepts")

    metrics = []
    for i in thresholds:
        # Fraction of positive clips detected: the mean of a True/False column.
        recall = float((pos.max_score >= i).mean())
        events = sum(event_count(frame_scores, i) for frame_scores in neg.frame_scores) # false positives
        metrics.append({
            "threshold": i,
            "recall": recall,
            "frr": 1 - recall,
            "false_events": events,
            "fpph": events / neg_hours,
            "neg_hours": neg_hours,
        })
    return pd.DataFrame(metrics)

def by_position_and_condition(df: pd.DataFrame, threshold: float, phrase: str) -> pd.DataFrame:
    """Recall per recording position and condition at a fixed threshold.
 
    This is the table that says what to fix: if recall is 97% at the desk
    and 61% with music, the problem is noise, not the model.
    """
    df = df[df.phrase == phrase]
    pos = df[df.kind == 'pos'].copy()
    pos['found'] = pos.max_score >= threshold
    return pos.groupby(["position", "condition"]).agg(recall = ("found", "mean"),
                                                   n = ("found", "size")).sort_values("recall")

def operating_point(curve: pd.DataFrame, max_fpph: float = 0.1) -> pd.Series:
    """Highest-recall threshold that stays under a false-accept budget."""
    viable = curve[curve.fpph <= max_fpph]
    if viable.empty:
        raise ValueError(f"no threshold achieves fpph <= {max_fpph}")
    return viable.loc[viable.recall.idxmax()]