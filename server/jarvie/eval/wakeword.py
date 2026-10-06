import onnxruntime as ort
from pathlib import Path
import soundfile as sf
import pandas as pd

def load_model(path: str | Path) -> ort.InferenceSession:
    """Load imported wake word model"""
    session = ort.InferenceSession(str(path))
    input = session.get_inputs()[0]
    return session, input.name, input.shape

def score_clip(session, path: Path, input_name: str, window: int, hop: int) -> dict:
    """Score a singular audio file, returns max score and the frame scores"""
    
    audio, sr = sf.read(path, dtype="float32")
    scores = []
    for start in range(0, len(audio) - window, hop):
        chunk = audio[start:start + window]
        out = session.run(None, {input_name: chunk.reshape(1, -1)})
        scores.append(float(out[0].squeeze()))
    return {
        "max": max(scores) if scores else 0.0,
        "scores": scores,
        "duration": len(audio) / sr
    }

def score_all(session, audio_dir: Path) -> pd.DataFrame:
    """Score every wav in a directory. One row per file."""
    return