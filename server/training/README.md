# Wake-word training and evaluation

How Jarvie's custom wake/sleep models are recorded, trained (livekit-wakeword on RunPod) and
evaluated on real voice recordings, plus a log of results and the problems hit along the way.
The bot itself doesn't use these models yet; it still matches "jarvis" in the speech-to-text
transcript. They're for the planned Raspberry Pi edge detector.

## Files

| Path | What it is |
|---|---|
| `server/training/wakeword_pipeline.py` | Cached, resumable training pipeline around livekit-wakeword |
| `server/training/train_wakeword_runpod.ipynb` | The notebook to run on RunPod; all settings in cell 3 |
| `server/training/monitor.py` | Terminal progress view of every stage, independent of the notebook |
| `server/training/real_data.py` | Trims your recordings to the 2 s training format and writes oversampled copies |
| `server/training/record_session.py`, `recording_prompts.txt` | Guided recording of training/test clips from a prompt list |
| `server/training/train_wakeword_colab.ipynb` | Earlier Colab version; superseded by the RunPod notebook |
| `server/jarvie/eval/wakeword.py` | Scores recorded clips with a trained model; recall / false accepts / thresholds |
| `server/jarvie/eval/wakeword_eval.ipynb` | Evaluation on real recordings (run on the Mac) |
| `server/jarvie/eval/gating.py`, `gating_eval.ipynb` | Always-on bot vs wake-word daemon: accuracy (Deepgram transcript matching vs the local model) and cost per day |
| `server/scratch/record.py` | Records labeled clips into `~/jarvie-data/audio/<mode>/` |
| `server/models/` | The `.onnx` models currently being evaluated (not committed) |
| `server/tests/test_wakeword_pipeline.py`, `test_wakeword_eval.py` | Tests for both |

## Workflow

1. **Record** real clips on the Mac with `scratch/record.py` (naming below).
2. **Train** on RunPod with `train_wakeword_runpod.ipynb` (variants A, B, C and tuning trials).
3. **Download** the `.onnx` files, copy the one to test into `server/models/` as
   `jarvie_wake.onnx` / `jarvie_sleep.onnx`.
4. **Evaluate** with `jarvie/eval/wakeword_eval.ipynb`: restart the kernel, Run All.

### Recording (`scratch/record.py`)

```
uv run python scratch/record.py pos_desk_quiet_wake wake 15 --pace fast
uv run python scratch/record.py neg_desk_loud_wake wake 10 --pace normal --similarity similar
```

Clip names: `<kind>[-<similarity>]_<position>_<condition>-<pace>_<phrase>_<index>.wav`, e.g.
`neg-very-similar_desk_loud-normal_wake_014.wav`. Pace (`fast`/`normal`/`slow`) is folded into the
condition and similarity (`not-similar`/`similar`/`very-similar`, negatives only) into the kind, so
names keep 5 underscore fields. Numbering continues from the highest existing index per label.
Record with macOS Mic Mode on **Standard** (Voice Isolation strips background noise), and press
Enter, wait ~0.5 s, then speak, so the key click isn't part of the phrase.

## Training pipeline (`wakeword_pipeline.py`)

livekit-wakeword runs every step for a model inside one folder. The pipeline splits that into
three cached stages, each in its own folder named after a hash of the settings that affect it:

| Stage | Contents | Reruns only when you change |
|---|---|---|
| `gen` | speech synthesis (Piper TTS) | phrases, negatives, sample counts, `length_scales` |
| `aug` | augmentation, then feature extraction | the variant's augmentation settings, or your noise recordings |
| `train` | train, export to ONNX, evaluate | training settings (a "trial") |

Each stage hard-links the previous stage's outputs and writes `_done.json` when finished;
augmentation also writes `_augmented.json`, so an interruption during feature extraction doesn't
redo it. Re-running unchanged settings (in any session, on any GPU) returns cached results.
`tts_batch_size` is excluded from the key (it only changes speed).

```python
pipe.run("A")                                                   # everything, once
pipe.run("B")                                                   # reuses A's speech
pipe.run("B", trial="lr3e-4", steps=20000, learning_rate=3e-4)  # training only
pipe.prepare_noise("room"); pipe.run("C")                       # your room's noise
```

Layout on the RunPod network volume (`/workspace/wakeword/`): `data/` (livekit setup, ~16 GB,
downloaded once), `noise/uploads/<set>/` and `noise/<set>/` (16 kHz copies), `stages/`, `models/`
(`<model>__<variant>__<trial>.onnx`), `results.csv`.

### Augmentation variants

livekit's default augmentation adds background noise to **every** clip at 5-15 dB SNR, and applies
reverb by convolving with a room response normalized to peak 1.0 with no loudness correction.
Simulated on a real clip, that reverb raised the peak 4-8x (up to ~17x with a long room response);
livekit writes 16-bit WAV, which hard-clips everything above 1.0. Rounds also stack: each round
re-augments the previous one.

| Variant | Noise | Reverb | Clipping | Background audio |
|---|---|---|---|---|
| A | every clip, 5-15 dB | 50%, louder | yes | MUSAN |
| B | 70% of clips, 10-25 dB | 30%, same loudness | prevented (peak 0.95) | MUSAN |
| C | as B | as B | prevented | your room (`noise/uploads/room/`) |

All use 1 round. B and C patch livekit's `AudioAugmentor.apply_rir` / `mix_with_background` only
while their augmentation runs, then restore the originals.

### Model / phrase decisions

- Phrases stay under ~2 s even when slow: livekit trains on 2.0 s windows and crops the *start* of
  longer clips, so "jarvie, I need some help" (slow) lost "jarvie". Replaced by "jarvie, help me".
- `custom_negative_phrases` list the phrases without "jarvie" ("wake up", "help me", ...) and
  sound-alikes; livekit's automatic negatives only swap sounds within single words. Each model also
  gets the other model's phrases as negatives.
- `length_scales` widened to 0.65-1.2 for fast/slow speech; `model_size: medium` for now (`small`
  for the Pi); `n_samples` 20,000, `steps` 80,000, `max_negative_weight` 3000,
  `target_fp_per_hour` 0.1.

## Your own recordings and the four model families

Synthetic-only models reach ~98% recall on synthetic test speech but not on a real voice, and real
near-misses ("jacky wake up") still fire. So real recordings are added to training:

| Family | Training data | Call |
|---|---|---|
| M1 | synthetic (B) | `pipe.run("B")` |
| M2 | synthetic + your recordings | `pipe.run("B", real="v1")` |
| M3 | your recordings only (+ livekit's real background speech as general negatives) | `pipe.run("B", real="v1", synthetic=False)` |
| M4 | B fine-tuned on your recordings (+10% synthetic rehearsal) | `pipe.run("B", trial="ft", real="v1", synthetic_share=0.1, init_from=("B", "base"), steps=5000, learning_rate=1e-5)` |

- **Recording**: `uv run python training/record_session.py <section> --position ... --condition ...`
  with the lines in `recording_prompts.txt`. `--split train` (default) goes to
  `~/jarvie-data/real/v1/<speaker>/<mode>/`; `--split test` to `~/jarvie-data/audio/<mode>/` (you) or
  `~/jarvie-data/audio-speakers/<speaker>/<mode>/` (others). `--speaker` names who's speaking.
- **Into training** (`real_data.py`): each recording is trimmed to 2.0 s (positives end 0-200 ms after the
  phrase, negatives centred, long everyday speech cut into many clips), written `real_repeat` times
  (default 20) as `clip_9NNNNN.wav`, and augmented with the synthetic clips. `speaker_aug` speed-perturbs
  copies (x0.88-1.12) so one voice stands in for several. The other model's phrases count as negatives.
  Aim for real data at ~20-30% of each class; more distinct recordings beat more copies.
- **Warm start** (M4): livekit's `WakeWordTrainer` builds the model, B's weights are loaded into it, then
  its normal 3-phase schedule runs at the low learning rate. Model settings must match B's.
- **Choosing**: `jarvie/eval/compare_models.ipynb` has a *personal* table (your held-out recordings) and a
  *general* table (other people's held-out recordings, each speaker weighted equally). The personal model
  is the best on the first; the deployment model must beat M1 on the second. It refuses to score any test
  clip that is also in `~/jarvie-data/real/`.

## Running on RunPod

- **Pod:** a 100 GB network volume (mounted at `/workspace`), an A40 (any NVIDIA GPU with 24 GB+
  works; on 16 GB set `tts_batch_size=25`), the newest official RunPod PyTorch template (PyTorch
  >= 2.5, Python >= 3.11). Upload the notebook, `wakeword_pipeline.py`, `real_data.py` and `monitor.py` to
  `/workspace/wakeword/`.
- **Each session:** cell 1 (install) only after the pod was stopped; cell 2 (preflight) first after
  every kernel restart; cell 3 (settings); cell 4 (data, cached); then the variant cells.
- **Progress without the notebook:** `watch -n 60 python /workspace/wakeword/monitor.py`.
- **Billing:** GPU time and the volume's storage (~$7/month for 100 GB, charged even with no pod
  running) come out of prepaid credit. At $0 the pod stops and the volume can eventually be deleted;
  keep a buffer or enable Auto-Pay. Back up `models/`, `results.csv` and `noise/uploads/` before
  deleting a volume.

### Problems hit, and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Colab: an error every ~20 min asking for a new package | livekit imports most training packages mid-run; `onnx`/`onnxscript` and the `espeak-ng` program aren't in its extras | Install cell installs everything up front; preflight cell checks it |
| `CUDA out of memory ... Skipping 50 clips` on a 15 GB T4 | TTS batch of 50 | `tts_batch_size=25` on 16 GB cards; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Same error on the A40 (48 GB), in jarvie_sleep's synthesis right after jarvie_wake's training (42.6 GB still allocated) | The previous stage's objects still held GPU memory in the same process | `release_gpu_memory()` runs `gc.collect()` + `torch.cuda.empty_cache()` after every stage and prints GB before -> after; `tts_batch_size=32`. If memory stays high, run stages in subprocesses |
| Feature extraction ~1 clip/s, augmentation ~8 clips/s, GPU 0% | `os.cpu_count()` reports the host's 128 cores; the pod has a ~12-CPU quota; 128 threads oversubscribed it | `available_cpus()` reads the cgroup quota; `limit_threads()` caps numpy/torch/OpenMP; onnxruntime uses <= 8 threads. Benchmark (12-core Mac, livekit's feature models): 3.4 clips/s at 128 threads vs 71 at 8 |
| Notebook shows no output, cell `[*]` | Browser reconnected (refresh, back button, Wi-Fi); output isn't replayed | The run continues on the pod; check with `monitor.py` |
| Code changes have no effect | Kernel still holds the old module | Restart the kernel after uploading code |
| Uploaded notebook reverted | An open notebook autosaves over the uploaded file | Close the notebook before uploading; on "changed on disk", choose Revert |

## Evaluation (`jarvie/eval/`)

- Scores each clip by sliding the model's 16-embedding window (~1.96 s: each embedding covers
  0.76 s, one every 80 ms) over it; a clip's max score decides detection.
- The featurizer must match livekit's preprocessing exactly: float audio in [-1, 1] into
  livekit's bundled mel/embedding models. An earlier version used openWakeWord's featurizer
  (int16-scale audio), which shifted scores (normal-pace EER 29.4% vs 23.5%).
  `test_featurizer_matches_livekit_wakeword` pins this to livekit's `WakeWordModel`.
- Normal pace is the primary result; section 10 compares fast, slow, extreme (fast + slow) and
  composite, each against negatives of the same pace (`MATCH_NEGATIVE_PACE`); section 11 breaks
  negatives down by similarity tier.
- With minutes of negative audio, false accepts per hour can't be checked against a 0.1/h budget
  (one false accept = 1/H per hour), so the notebook uses the equal-error-rate (EER) threshold
  until there are >= 10 h of negatives.
- All reported numbers were re-derived with independent code and matched exactly.

## Results log

### 2026-10-12: first model (livekit quick config) on 57 real clips

Trained with livekit defaults except `n_samples=5000`, `steps=30000`; phrases "jarvie, wake up" /
"jarvie, I need some help" / "jarvie, wakie wakie". All clips `desk_loud`. Threshold 0.51 (EER).

| Set | Positives | Negatives | Recall | EER |
|---|---|---|---|---|
| normal (primary) | 17 | 17 | **76.5%** | **23.5%** |
| fast | 5 | 6 | 0% | 18.3% |
| slow | 6 | 6 | 16.7% | 33.3% |
| composite | 28 | 29 | 50.0% | 35.1% |

| Negatives (normal pace) | Clips firing | Worst score |
|---|---|---|
| not similar (6) | 0% | 0.07 |
| similar (8) | 37.5% | 0.88 |
| very similar (3, "jacky") | 33.3% | 0.80 |

Takeaways: everyday speech never triggers it; similar-sounding phrases do; fast/slow speech is
barely detected. Background noise doesn't explain the misses: the clips are clean (~30 dB SNR,
noise floor ~-55 dBFS) and SNR doesn't correlate with score (r = +0.07). Groups are small; treat
differences under ~20 points as noise.

### 2026-10-13: RunPod run A (in progress)

A40 pod, settings above. Speech synthesis ran at ~41 clips/s (vs ~24 on a Colab T4); 50,500
clips per model (positives, negatives, background; train + test). Augmentation and feature
extraction were throttled by the thread oversubscription above, fixed mid-run; augmentation was
kept and extraction restarted with the fix. A, B (and later C) comparison: pending.

### 2026-10-14: A and B vs the old model on the same 57 clips

`jarvie/eval/compare_models.ipynb`. Clips are now padded with 1 s of silence on each side before
scoring, as livekit's `predict_clip` does (training aligns each phrase to the end of its window;
without padding, phrases ending at the very end of a recording had no room to line up).

| Normal-pace EER / threshold | old | A | B |
|---|---|---|---|
| all clips | 23.5% / 0.66 | 35.3% / 0.78 | 32.4% / 0.66 |
| without positives 004-007 | 23.3% / 0.66 | 23.3% / 0.93 | 23.3% / 0.89 |

Recall fast / slow (all clips): old 40% / 17%, A 60% / 33%, B 60% / 33%.
"Similar" negatives firing: old 38%, A and B 63%.

Checks of the evaluation itself: file formats, featurizer parity with livekit, metric
calculations and recording volume (scores barely change at peak 0.9 or half volume) are all fine.
What looks wrong is in the inputs:
- Positives 004-007 score near 0 only on A/B: probably "jarvie, I need some help", which A/B
  weren't trained on (replaced by "jarvie, help me"). To be confirmed by listening.
- Speech runs to the very end of the 3 s recording in about a third of the clips, so some phrases
  are cut off. `record.py` now records 4 s after an audible beep.
- "Similar" negatives 010-013 and "very similar" 016 score 0.9+ on A/B, which forces their
  threshold to ~0.9. Next step: put their wording into `custom_negative_phrases` and retrain.
