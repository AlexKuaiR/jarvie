"""Progress of every training-pipeline stage, read from the files on the volume.

Run on the pod, in a terminal:
    python /workspace/wakeword/monitor.py               # print once
    watch -n 60 python /workspace/wakeword/monitor.py   # refresh every minute (Ctrl+C to leave)

It only reads files, so it's safe to run while training and doesn't depend on the notebook's
browser connection.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/wakeword")
SPLITS = [
    "positive_train", "positive_test",
    "negative_train", "negative_test",
    "background_train", "background_test",
]
CLEAN = re.compile(r"^clip_\d{6}\.wav$")
AUGMENTED = re.compile(r"^clip_\d{6}_r\d+\.wav$")


def count(folder: Path, pattern: re.Pattern) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.glob("*.wav") if pattern.match(f.name))


def show(kind: str, stage_dir: Path) -> None:
    model = stage_dir.name.rsplit("-", 1)[0]
    mdir = stage_dir / model
    settings_file = stage_dir / "settings.json"
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    variant = f"  [variant {settings['variant']}]" if "variant" in settings else ""
    done = "DONE" if (stage_dir / "_done.json").exists() else "running/pending"
    print(f"{kind:5s} {stage_dir.name}{variant}  {done}")

    if kind == "gen":
        for split in SPLITS:
            print(f"    {split:17s} {count(mdir / split, CLEAN):6d} clips")
    elif kind == "aug":
        augmented = sum(count(mdir / split, AUGMENTED) for split in SPLITS)
        features = len(list(mdir.glob("*_features_*.npy")))
        print(f"    augmented clips: {augmented}   feature files: {features}/6")
    else:
        for metrics_file in mdir.glob("*_metrics.json"):
            try:
                last = json.loads(metrics_file.read_text())[-1]
                short = {k: round(v, 3) if isinstance(v, float) else v for k, v in last.items()}
                print("    last log:", short)
            except Exception:
                print("    metrics file being written")
        has_pt = "yes" if any(mdir.glob("*.pt")) else "no"
        has_onnx = "yes" if any(mdir.glob("*.onnx")) else "no"
        print(f"    model .pt: {has_pt}   .onnx: {has_onnx}")


def main() -> None:
    for kind in ["gen", "aug", "train"]:
        for stage_dir in sorted((ROOT / "stages" / kind).glob("*")):
            show(kind, stage_dir)
    models = sorted(p.name for p in (ROOT / "models").glob("*.onnx"))
    print("models:", models or "none yet")


if __name__ == "__main__":
    main()
