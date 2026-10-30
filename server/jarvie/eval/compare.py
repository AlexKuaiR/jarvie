"""Compare several wake-word models on the same real recordings, per speaker.

For each model: score every clip once, then summarize with the same definitions as
wakeword_eval.ipynb, so the numbers are directly comparable:

- primary set = normal-pace positives vs normal-pace negatives; its equal-error-rate (EER)
  threshold is that model's operating threshold
- fast / slow recall at that threshold (speaking pace the model must also handle)
- how often each similarity tier of negatives fires at that threshold

gate_summary adds false fires per hour of everyday speech, for the stage-2 gate
(training/STAGE2_GATE.md). evaluate() runs the whole comparison; compare_models.ipynb calls it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fnmatch import fnmatch
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


def score_models(model_paths: list[Path], audio_dir: Path, pattern: str = "*.wav") -> pd.DataFrame:
    """One row per (model, clip): labels, the clip's max score under that model, and its per-window
    scores (`frame_scores`, needed to count separate fires in long recordings)."""
    featurize = wakeword.make_featurizer()
    frames = []
    for path in model_paths:
        session, name, window = wakeword.load_model(path)
        scored = gating.add_labels(
            wakeword.score_all(session, audio_dir, name, window, 1, featurize, pattern=pattern)
        )
        scored.insert(0, "model", path.stem)
        frames.append(scored)
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


def score_test_sets(
    model_paths: list[Path], test_sets: dict[str, Path], pattern: str = "*.wav"
) -> pd.DataFrame:
    """score_models for every speaker's test folder, with a `speaker` column."""
    frames = [
        score_models(model_paths, folder, pattern).assign(speaker=speaker)
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


# --- false fires per hour (the stage-2 gate) ---------------------------------------------------

# record_session.py's neg-everyday recordings: long takes of ordinary speech. They're labelled
# mode=wake, so they're in audio/wake/ whichever model family is being compared.
EVERYDAY_PATTERN = "neg-not-similar_*.wav"
FRAME_SECONDS = 0.08  # one score per embedding frame (hop=1)
GATE_HOLD_SECONDS = 3.0  # how long one fire keeps the gate open for the transcript check


def gate_fires(
    scores: list[float], threshold: float, hold_seconds: float = GATE_HOLD_SECONDS
) -> int:
    """How many times the gate would open: a fire holds it open for `hold_seconds`, and firing in
    that time doesn't count again. Unlike wakeword.event_count, a model stuck above the threshold
    keeps counting (once per hold), so an always-open gate can't look like one false fire."""
    hold = max(1, round(hold_seconds / FRAME_SECONDS))
    fires, next_allowed = 0, 0
    for i, s in enumerate(scores):
        if s >= threshold and i >= next_allowed:
            fires += 1
            next_allowed = i + hold
    return fires


def score_everyday(
    model_paths: list[Path], data: Path = DATA, train_root: Path = DATA / "real"
) -> pd.DataFrame:
    """Score your everyday-speech test recordings under every model. Empty if there are none."""
    folder = data / "audio" / "wake"
    if not any(folder.glob(EVERYDAY_PATTERN)):
        return pd.DataFrame()
    check_leakage({OWNER: folder}, train_root)
    return score_test_sets(model_paths, {OWNER: folder}, EVERYDAY_PATTERN)


def gate_summary(
    scores: pd.DataFrame, everyday: pd.DataFrame, max_fpph: float = 5.0
) -> pd.DataFrame:
    """One row per model: false fires per hour of everyday speech at its summarize() threshold, and
    the best recall (all paces) any threshold reaches within `max_fpph` false fires per hour.
    Fires are counted as gate openings (gate_fires).

    A gate in front of Deepgram wants high recall at a low threshold rather than a balanced EER:
    the transcript check rejects false fires, but a missed "jarvie" is simply ignored. NaN where
    there's no everyday audio, or no threshold stays within the budget.
    """
    rows = []
    for model, threshold in summarize(scores).threshold.items():
        pos = scores[(scores.model == model) & (scores.kind == "pos")]
        ev = everyday[everyday.model == model] if not everyday.empty else everyday
        hours = ev.duration.sum() / 3600 if len(ev) else 0.0
        row = {
            "model": model,
            "everyday_hours": hours,
            "threshold": threshold,
            "fpph_at_threshold": np.nan,
            "gate_threshold": np.nan,
            "gate_recall": np.nan,
            "gate_fpph": np.nan,
        }
        if hours > 0:

            def fpph(t: float) -> float:
                return sum(gate_fires(s, t) for s in ev.frame_scores) / hours

            if not np.isnan(threshold):
                row["fpph_at_threshold"] = fpph(threshold)
            if len(pos):
                curve = pd.DataFrame(
                    {
                        "threshold": THRESHOLDS,
                        "recall": [(pos.max_score >= t).mean() for t in THRESHOLDS],
                        "fpph": [fpph(t) for t in THRESHOLDS],
                    }
                )
                viable = curve[curve.fpph <= max_fpph]
                if not viable.empty:
                    # The highest threshold that still reaches the best recall: the same recall
                    # with fewer chances of a false fire.
                    best = viable[viable.recall == viable.recall.max()].iloc[-1]
                    row["gate_threshold"] = best.threshold
                    row["gate_recall"] = best.recall
                    row["gate_fpph"] = best.fpph
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# --- the whole comparison (what compare_models.ipynb runs) -------------------------------------


@dataclass
class Evaluation:
    """Every table compare_models.ipynb shows. `scores` has one row per (model, clip), without the
    long everyday recordings (those are only in `everyday`, used by the Gate table)."""

    scores: pd.DataFrame
    everyday: pd.DataFrame
    counts: pd.DataFrame
    personal: pd.DataFrame
    general: pd.DataFrame
    by_speaker: pd.DataFrame
    gate: pd.DataFrame


def missing_inputs(mode: str, model_paths: list[Path], data: Path = DATA) -> list[str]:
    """What still has to exist before evaluate() can run for `mode`. Empty when ready."""
    missing = []
    if not model_paths:
        missing.append(f"no jarvie_{mode}*.onnx models in server/models/ (train them, then copy)")
    if not any((data / "audio" / mode).glob("*.wav")):
        missing.append(
            f"no test recordings in {data / 'audio' / mode}/ (record_session.py ... --split test)"
        )
    return missing


def evaluate(
    mode: str,
    model_paths: list[Path],
    data: Path = DATA,
    exclude: set[str] = frozenset(),
    as_positive: set[str] = frozenset(),
    max_fpph: float = 5.0,
) -> Evaluation:
    """Score every model on every test set for `mode` and build all the tables.

    Refuses to run (ValueError, listing what's missing) until models and your test recordings
    exist, and refuses test clips that are also in training data (check_leakage).
    """
    missing = missing_inputs(mode, model_paths, data)
    if missing:
        raise ValueError("not ready to evaluate:\n  " + "\n  ".join(missing))
    test_sets = discover_test_sets(mode, data)
    check_leakage(test_sets, data / "real")
    scores = apply_labels(score_test_sets(model_paths, test_sets), exclude, as_positive)

    # A 30-min everyday recording would count as one negative clip in the EER and fires_* columns
    # (and almost surely fire), so it's kept out of those and only used per hour, in the Gate table.
    is_everyday = scores.file.map(lambda f: fnmatch(f, EVERYDAY_PATTERN))
    if mode == "wake":  # already scored: everyday recordings are in audio/wake/
        everyday = scores[is_everyday & (scores.speaker == OWNER)].reset_index(drop=True)
    else:
        everyday = score_everyday(model_paths, data, data / "real")
    scores = scores[~is_everyday].reset_index(drop=True)

    personal_scores = scores[scores.speaker == OWNER]
    return Evaluation(
        scores=scores,
        everyday=everyday,
        counts=scores.groupby("speaker").apply(counts).unstack(0).fillna(0).astype(int),
        personal=summarize(personal_scores),
        general=general_summary(scores),
        by_speaker=summarize_by_speaker(scores),
        gate=gate_summary(personal_scores, everyday, max_fpph),
    )
