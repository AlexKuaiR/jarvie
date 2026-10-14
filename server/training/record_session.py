"""Guided recording of wake-word training and test clips from recording_prompts.txt.

    uv run python training/record_session.py --list
    uv run python training/record_session.py wake-normal --position desk --condition quiet
    uv run python training/record_session.py neg-very-similar --position far --condition quiet --split test

For each line of the chosen section it shows what to say, beeps, records 4 s, checks the take
(clipped, too quiet, possibly cut off, silent) and lets you keep it, redo it or play it back.

Files use the same naming as the eval and training code expect,
    <kind>[-<similarity>]_<position>_<condition>-<pace>_<mode>_<index>.wav
and every take is logged with the exact text said in manifest.csv next to the clips:
    --split train  ->  ~/jarvie-data/real/v1/<speaker>/<mode>/   (for training; --set changes "v1")
    --split test   ->  ~/jarvie-data/audio/<mode>/               (your test set: what the eval scores)
                       ~/jarvie-data/audio-speakers/<speaker>/<mode>/  (other people's test sets)
--speaker defaults to "me" (you). Use a short name per person, e.g. --speaker sam.
Never record the same session into both: the test set must stay unseen by training.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000
TAKE_SECONDS = 4.0  # 3 s cut off phrases that started late
PROMPTS_FILE = Path(__file__).with_name("recording_prompts.txt")
DATA = Path.home() / "jarvie-data"
FIELD_OK = re.compile(r"^[a-z0-9]+$")  # position/condition become filename fields: no _ or -


@dataclass
class Section:
    name: str
    kind: str  # "pos" or "neg"
    pace: str
    mode: str  # "wake" or "sleep": which model, and which folder
    similarity: str | None = None
    minutes: float | None = None  # set for long everyday-speech recordings
    lines: list[str] = field(default_factory=list)


def parse_prompts(text: str) -> dict[str, Section]:
    """Parse recording_prompts.txt: [name] key=value ... headers, one take per line, # comments."""
    sections: dict[str, Section] = {}
    current: Section | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = re.match(r"^\[([\w-]+)\]\s*(.*)$", line)
        if header:
            attrs = dict(part.split("=", 1) for part in header.group(2).split())
            current = Section(
                name=header.group(1), kind=attrs["kind"], pace=attrs.get("pace", "normal"),
                mode=attrs.get("mode", "wake"), similarity=attrs.get("similarity"),
                minutes=float(attrs["minutes"]) if "minutes" in attrs else None,
            )
            if current.kind == "neg" and not current.similarity:
                raise ValueError(f"[{current.name}]: negatives need similarity=")
            sections[current.name] = current
        elif current is None:
            raise ValueError(f"prompt line before any [section]: {line!r}")
        else:
            current.lines.append(line)
    return sections


def label(section: Section, position: str, condition: str) -> str:
    """kind[-similarity]_position_condition-pace_mode, the format record.py and the eval use."""
    for name, value in (("position", position), ("condition", condition)):
        if not FIELD_OK.match(value):
            raise SystemExit(f"--{name} must be lowercase letters/digits only (it's a filename field), got {value!r}")
    kind = f"neg-{section.similarity}" if section.kind == "neg" else "pos"
    return f"{kind}_{position}_{condition}-{section.pace}_{section.mode}"


def output_dir(split: str, mode: str, dataset: str, speaker: str = "me") -> Path:
    if not FIELD_OK.match(speaker):
        raise SystemExit(f"--speaker must be lowercase letters/digits only, got {speaker!r}")
    if split == "test":
        return DATA / "audio" / mode if speaker == "me" else DATA / "audio-speakers" / speaker / mode
    return DATA / "real" / dataset / speaker / mode


def next_index(folder: Path, lbl: str) -> int:
    pattern = re.compile(rf"{re.escape(lbl)}_(\d+)\.wav")
    found = [int(m.group(1)) for p in folder.glob(f"{lbl}_*.wav") if (m := pattern.fullmatch(p.name))]
    return max(found, default=-1) + 1


def check_take(audio: np.ndarray, sr: int = SR) -> list[str]:
    """Problems worth re-recording for. Empty list = looks fine."""
    problems = []
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak >= 0.99:
        problems.append("clipped (too loud or too close to the mic)")
    if peak < 0.03:
        problems.append("very quiet (speak up or move closer)")
    frame = int(0.02 * sr)
    rms = np.array([np.sqrt(np.mean(audio[i:i + frame] ** 2)) for i in range(0, len(audio) - frame, frame)])
    if rms.size:
        floor = np.percentile(rms, 10)
        speech = rms > max(floor * 4, rms.max() * 0.15, 1e-4)
        if not speech.any() or speech.sum() < 5:
            problems.append("no clear speech detected")
        elif speech[-10:].any():  # speech in the last 0.2 s
            problems.append("speech runs to the end: phrase may be cut off")
    return problems


# --- audio I/O (sounddevice imported lazily so tests don't need an audio device) ---------------


def beep(freq: float = 880.0, seconds: float = 0.15) -> None:
    import sounddevice as sd

    t = np.arange(int(seconds * SR)) / SR
    sd.play((0.2 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))).astype("float32"), SR)
    sd.wait()


def record(seconds: float) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def play(audio: np.ndarray) -> None:
    import sounddevice as sd

    sd.play(audio, SR)
    sd.wait()


def save(audio: np.ndarray, folder: Path, lbl: str, text: str, section: Section, args) -> Path:
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{lbl}_{next_index(folder, lbl):03d}.wav"
    sf.write(path, audio, SR)
    manifest = folder / "manifest.csv"
    new = not manifest.exists()
    with manifest.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "text", "section", "kind", "similarity", "pace", "position",
                        "condition", "speaker", "split", "seconds", "recorded_at"])
        w.writerow([path.name, text, section.name, section.kind, section.similarity or "", section.pace,
                    args.position, args.condition, getattr(args, "speaker", "me"), args.split,
                    round(len(audio) / SR, 2), time.strftime("%Y-%m-%d %H:%M:%S")])
    return path


# --- session ---------------------------------------------------------------------------------


def run_long(section: Section, folder: Path, lbl: str, args) -> None:
    minutes = args.minutes or section.minutes or 5
    print(f"\n{section.lines[0] if section.lines else ''}")
    input(f"Recording {minutes:g} min of everyday speech in one take. Press Enter to start...")
    beep()
    print("Recording... (talk normally; it stops by itself)")
    audio = record(minutes * 60)
    path = save(audio, folder, lbl, f"everyday speech, {minutes:g} min", section, args)
    print(f"saved {path.name}")


def run_section(section: Section, args) -> None:
    folder = output_dir(args.split, section.mode, args.set, args.speaker)
    lbl = label(section, args.position, args.condition)
    print(f"\n=== [{section.name}] -> {folder}  ({lbl}_NNN.wav) ===")
    print(f"speaker: {args.speaker}, position: {args.position}, condition: {args.condition}, split: {args.split}")
    if section.minutes is not None:
        run_long(section, folder, lbl, args)
        return

    takes = [line for line in section.lines for _ in range(args.repeats)]
    if args.shuffle:
        np.random.default_rng().shuffle(takes)
    takes = takes[args.start:]
    kept = 0
    for n, text in enumerate(takes, start=args.start + 1):
        while True:
            print(f"\n[{n}/{len(takes) + args.start}]  say:  {text}")
            reply = input("  Enter = record, s = skip, q = quit: ").strip().lower()
            if reply == "q":
                print(f"\nstopped; kept {kept} take(s). Resume with --start {n - 1}")
                return
            if reply == "s":
                break
            beep()
            print("  recording...", flush=True)
            audio = record(TAKE_SECONDS)
            problems = check_take(audio)
            print("  " + ("; ".join(f"WARNING: {p}" for p in problems) if problems else "looks fine"))
            while True:
                choice = input("  Enter = keep, r = redo, p = play back, s = discard and skip: ").strip().lower()
                if choice == "p":
                    play(audio)
                    continue
                break
            if choice == "r":
                continue
            if choice == "s":
                break
            path = save(audio, folder, lbl, text, section, args)
            kept += 1
            print(f"  saved {path.name}")
            break
    print(f"\n[{section.name}] done: kept {kept} take(s) in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sections", nargs="*", help="section names from the prompts file (in order)")
    parser.add_argument("--list", action="store_true", help="list sections and how many takes each has")
    parser.add_argument("--position", default="desk", help="where you are: desk, far, bed, ... (letters only)")
    parser.add_argument("--condition", default="quiet", help="background: quiet, music, fan, talk, ... (letters only)")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--speaker", default="me", help='who is speaking: "me" (you) or a short name per person')
    parser.add_argument("--set", default="v1", help="training dataset name (folder under ~/jarvie-data/real/)")
    parser.add_argument("--repeats", type=int, default=1, help="say each line this many times")
    parser.add_argument("--shuffle", action="store_true", help="mix the order of the lines")
    parser.add_argument("--start", type=int, default=0, help="skip this many takes (to resume)")
    parser.add_argument("--minutes", type=float, help="override the length of long recordings")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_FILE)
    args = parser.parse_args()

    sections = parse_prompts(args.prompts.read_text())
    if args.list or not args.sections:
        for s in sections.values():
            what = f"{s.minutes:g} min continuous" if s.minutes is not None else f"{len(s.lines)} lines"
            tier = f", {s.similarity}" if s.similarity else ""
            print(f"  {s.name:20s} {s.kind}{tier}, {s.pace}, {s.mode}: {what}")
        return
    unknown = [n for n in args.sections if n not in sections]
    if unknown:
        raise SystemExit(f"unknown section(s): {', '.join(unknown)} (see --list)")
    for name in args.sections:
        run_section(sections[name], args)


if __name__ == "__main__":
    main()
