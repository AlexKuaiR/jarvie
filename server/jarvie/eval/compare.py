"""Compare several wake-word models on the same real recordings, per speaker.

For each model: score every clip once, then summarize with the same definitions as
wakeword_eval.ipynb, so the numbers are directly comparable:

- primary set = normal-pace positives vs normal-pace negatives; its equal-error-rate (EER)
  threshold is that model's operating threshold
- fast / slow recall at that threshold (speaking pace the model must also handle)
- how often each similarity tier of negatives fires at that threshold
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import gating, wakeword

THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)
DATA = Path.home() / "jarvie-data"
OWNER = "me"  # record_session.py's default speaker: you


def eer(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, float]:
    """(threshold, rate) where the share of positives missed ~= the share of negatives that fire."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan"), float("nan")
    frr = np.array([(pos_scores < t).mean() for t in THRESHOLDS])
    far = np.array([(neg_scores >= t).mean() for t in THRESHOLDS])
    i = int(np.argmin(np.abs(frr - far)))
    return float(THRESHOLDS[i]), float((frr[i] + far[i]) / 2)


def apply_labels(
    scores: pd.DataFrame, exclude: set[str] = frozenset(), as_positive: set[str] = frozenset()
) -> pd.DataFrame:
    """Leave out clips that don't fit the comparison, and relabel clips that really are wake phrases.

    exclude:     e.g. clips saying a phrase some model wasn't trained on ("jarvie, I need some help")
    as_positive: negatives that are actually acceptable wake phrases ("jarvie wakie wake", and
                 "jarvis wake up" if Jarvis should wake it too)
    """
    df = scores[~scores.file.isin(exclude)].copy()
    flip = df.file.isin(as_positive)
    df.loc[flip, "kind"] = "pos"
    df.loc[flip, "similarity"] = ""
    return df.reset_index(drop=True)


def score_models(model_paths: list[Path], audio_dir: Path) -> pd.DataFrame:
    """One row per (model, clip): labels plus the clip's max score under that model."""
    featurize = wakeword.make_featurizer()
    frames = []
    for path in model_paths:
        session, name, window = wakeword.load_model(path)
        scored = gating.add_labels(
            wakeword.score_all(session, audio_dir, name, window, 1, featurize)
        )
        scored.insert(0, "model", path.stem)
        frames.append(scored.drop(columns="frame_scores"))
    return pd.concat(frames, ignore_index=True)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per model with its operating threshold, recall per pace and firing per tier."""
    rows = []
    for model, d in scores.groupby("model", sort=False):
        pos, neg = d[d.kind == "pos"], d[d.kind == "neg"]
        pos_n, neg_n = pos[pos.pace == "normal"], neg[neg.pace == "normal"]
        threshold, eer_rate = eer(pos_n.max_score.to_numpy(), neg_n.max_score.to_numpy())
        row = {"model": model, "threshold": threshold, "eer_normal": eer_rate}
        _, row["eer_all_paces"] = eer(pos.max_score.to_numpy(), neg.max_score.to_numpy())
        for pace in ["normal", "fast", "slow"]:
            p = pos[pos.pace == pace]
            row[f"recall_{pace}"] = (p.max_score >= threshold).mean() if len(p) else np.nan
        for tier in ["not-similar", "similar", "very-similar"]:
            n = neg_n[neg_n.similarity == tier]
            row[f"fires_{tier}"] = (n.max_score >= threshold).mean() if len(n) else np.nan
        row["mean_pos_score"] = pos_n.max_score.mean()
        row["mean_neg_score"] = neg_n.max_score.mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def counts(scores: pd.DataFrame) -> pd.Series:
    """How many clips back each number (per model the clips are the same)."""
    first = scores[scores.model == scores.model.iloc[0]]
    labels = np.where(
        first.kind == "pos", "pos_" + first.pace, "neg_" + first.pace + "_" + first.similarity
    )
    return pd.Series(labels).value_counts().sort_index()


# --- several speakers --------------------------------------------------------------------------


def discover_test_sets(mode: str = "wake", data: Path = DATA) -> dict[str, Path]:
    """Test folders as record_session.py writes them: yours in audio/<mode>/, other people's in
    audio-speakers/<speaker>/<mode>/."""
    sets = {}
    if (data / "audio" / mode).is_dir():
        sets[OWNER] = data / "audio" / mode
    for folder in sorted((data / "audio-speakers").glob(f"*/{mode}")):
        sets[folder.parent.name] = folder
    return sets


def _digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def check_leakage(test_sets: dict[str, Path], train_root: Path = DATA / "real") -> None:
    """Refuse to evaluate on a recording that is also in a training set (identical file content):
    the model would have heard it during training, and the results would be meaningless."""
    train = {_digest(p) for p in train_root.rglob("*.wav")} if train_root.exists() else set()
    leaked = [
        f"{speaker}/{p.name}"
        for speaker, folder in test_sets.items()
        for p in sorted(folder.glob("*.wav"))
        if _digest(p) in train
    ]
    if leaked:
        raise ValueError(
            f"{len(leaked)} test clip(s) also appear in training data under {train_root}: "
            + ", ".join(leaked[:10])
        )


def score_test_sets(model_paths: list[Path], test_sets: dict[str, Path]) -> pd.DataFrame:
    """score_models for every speaker's test folder, with a `speaker` column."""
    frames = [
        score_models(model_paths, folder).assign(speaker=speaker)
        for speaker, folder in test_sets.items()
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_by_speaker(scores: pd.DataFrame) -> pd.DataFrame:
    """summarize() separately for each speaker (rows: speaker, model)."""
    parts = {speaker: summarize(d) for speaker, d in scores.groupby("speaker", sort=False)}
    return pd.concat(parts, names=["speaker", "model"])


def general_summary(scores: pd.DataFrame, owner: str = OWNER) -> pd.DataFrame:
    """How models do on voices other than yours: each other speaker's summary, averaged so every
    speaker counts equally (a speaker with many clips doesn't dominate). Empty without other speakers."""
    others = scores[scores.speaker != owner]
    if others.empty:
        return pd.DataFrame()
    return summarize_by_speaker(others).groupby(level="model", sort=False).mean(numeric_only=True)
