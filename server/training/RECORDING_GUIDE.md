# Recording guide

Step-by-step: recording your own voice data for the three Jarvie wake-word models, then training and
choosing models. What to say is in `recording_prompts.txt`; the recorder is `record_session.py`.

| Model | Fires on | Used for | Prompt sections |
|---|---|---|---|
| `jarvie_wake` (stage 1) | "jarvie, wake up", "jarvie, help me", "jarvie, wakie wakie", "jarvie, are you up?" | starting the bot (the planned daemon) | `wake-*` |
| `jarvie_sleep` | "jarvie, go to sleep", "jarvie, sleep now" | putting it to sleep (planned; probably not needed, see below) | `sleep-*` |
| `jarvie_name` (stage 2) | "jarvie" alone | talking to the bot while it runs (planned; today bot.py matches the transcript) | `name-*` |

Negatives (`neg-*`) are things that must not trigger. "Jarvis" is always a negative.

**Priorities.** The name model is the one that needs the most data: it may gate audio to Deepgram
(see `STAGE2_GATE.md`), which needs high recall in every room condition and a false-fire rate
measured per hour of everyday audio. The sleep model is optional: "jarvie, go to sleep" is said while
the bot runs, so the transcript can catch it, and the sessions below no longer record it.

## 0. Once, before the first session

```bash
cd ~/jarvie-wakeword/server    # the wakeword-eval worktree (already on that branch)
git pull
uv run python training/record_session.py --list     # every section and how many lines it has
```
- macOS: Control Center -> Mic Mode -> **Standard** (Voice Isolation strips background noise).
- Use the microphone the detector will use, if you can.
- In another terminal tab: `caffeinate -d` (keeps the Mac awake).

## 1. How one take works

```
[3/16]  say:  hey jarvie
  Enter = record, s = skip, q = quit:        <- Enter, wait for the beep, speak (4 s)
  looks fine                                 <- or WARNING: clipped / very quiet / may be cut off / no speech
  Enter = keep, r = redo, p = play back, s = discard and skip:
```
Only text outside (parentheses) is said; the parentheses say *how*. For `name-*` say **only the name**,
nothing after it.

## 2. Training sessions

Each row is one sitting of about 10-15 minutes. Any order; spreading them over several days is better
(your voice and the room vary). Prefix every command with
`uv run python training/record_session.py`.

| # | Where / background | Sections and flags |
|---|---|---|
| 1 | desk, quiet | `wake-normal wake-areyouup wake-fast wake-areyouup-fast wake-slow wake-areyouup-slow --position desk --condition quiet` |
| 2 | desk, quiet | `neg-very-similar neg-jarvis neg-similar neg-similar-fast --position desk --condition quiet` |
| 3 | desk, quiet | `name-normal name-fast name-slow --position desk --condition quiet` |
| 4 | desk, quiet | `neg-name-very-similar neg-name-similar --position desk --condition quiet` |
| 5 | desk, quiet | `neg-everyday --position desk --condition quiet` (5 min talking/reading, one take) |
| 6 | across the room | `wake-normal wake-areyouup name-normal neg-very-similar neg-jarvis neg-name-very-similar --position far --condition quiet --shuffle` |
| 7 | desk, music on | `wake-normal wake-areyouup wake-fast name-normal neg-very-similar neg-jarvis --position desk --condition music --shuffle` |
| 8 | bed, quiet | `wake-normal wake-slow name-normal name-slow --position bed --condition quiet` |
| 9 | bed, quiet | `neg-similar neg-name-very-similar name-fast --position bed --condition quiet` |
| 10 | desk, fan/AC on | `wake-normal wake-areyouup name-normal neg-very-similar neg-name-very-similar --position desk --condition fan --repeats 2 --shuffle` |
| 11 | desk, TV/video on | `neg-everyday --position desk --condition talk` |
| 12 | desk, TV/video on | `name-normal name-fast neg-name-very-similar neg-name-similar --position desk --condition talk --shuffle` |
| 13 | across the room, music on | `name-normal name-slow neg-name-very-similar --position far --condition music --shuffle` |
| 14 | anywhere, normal dorm life | `neg-everyday --position desk --condition room --minutes 30` (passive; repeat on several days) |

Recorded so far: sessions 1, 2, 3 and 11.

**Session 14 is passive.** Start it and carry on: calls, videos, talking with your roommate, typing,
music. Don't say "jarvie" during it (if you do, delete that recording). Aim for 1-2 hours in total
across days, in 30-minute takes so a crash doesn't lose much. This is what the name model will
actually hear between turns, and what its false fires per hour are measured on.

**Sleep model (optional).** Only if you decide to keep `jarvie_sleep`: in bed, quiet,
`sleep-normal sleep-fast sleep-slow neg-sleep-similar --position bed --condition quiet --repeats 2`
(about 36 positives).

**Already recorded session 1 before "jarvie, are you up?" existed?** Record just the new phrase as a
catch-up, in the same place:
`wake-areyouup wake-areyouup-fast wake-areyouup-slow --position desk --condition quiet`. Nothing already
recorded needs redoing.

After sessions 1-14, roughly: ~240 stage-1 positives, ~150 name positives, ~320 negatives (~150 of
them for the name model) and 1-2 hours of everyday audio. For more name data, repeat 6, 10, 12 and 13
on other days; more clean desk takes (session 3) help least.

- Files go to `~/jarvie-data/real/v1/me/<wake|sleep|name>/` with a `manifest.csv` of what was said.
- `--position` / `--condition` must be lowercase letters or digits; copy them from the table so a
  typo doesn't create a new label.

### Stopping and resuming
- `q` stops the whole command and prints what to rerun, e.g.
  `Resume with: name-fast name-slow --start 3`. Run those sections with the same flags as before
  (`--position`, `--condition`, `--shuffle`, ...) plus that `--start`; it skips takes in the first
  section only. `--shuffle` gives the same order every time, so resuming skips exactly the takes
  already recorded.
- Numbering continues from the highest existing file per label, so stopping, resuming or recording
  over several days never overwrites anything.
- A bad take noticed later: delete the `.wav` and its line in that folder's `manifest.csv`, or keep it.

### Progress
```bash
uv run python -c "
import glob, pandas as pd
df = pd.concat(pd.read_csv(f) for f in glob.glob('$HOME/jarvie-data/real/v1/*/*/manifest.csv'))
print(df.groupby(['speaker', 'section']).size().to_string())
print(df.groupby(['position', 'condition']).size().to_string()); print('total takes:', len(df))"
```

## 3. Test sessions (a day with no training recordings)

These are what models are judged on; they are never uploaded for training.
```bash
uv run python training/record_session.py wake-normal wake-areyouup wake-fast wake-slow neg-very-similar neg-jarvis neg-similar --position desk --condition quiet --split test
uv run python training/record_session.py name-normal name-fast neg-name-very-similar neg-name-similar --position desk --condition quiet --split test
uv run python training/record_session.py wake-normal name-normal neg-jarvis neg-name-very-similar --position far --condition music --split test --shuffle
uv run python training/record_session.py name-normal name-fast neg-name-very-similar --position desk --condition talk --split test --shuffle
uv run python training/record_session.py neg-everyday --position desk --condition room --minutes 30 --split test
```
They go to `~/jarvie-data/audio/<wake|sleep|name>/`; the everyday recordings always go to `audio/wake/`,
and the Gate table finds them there for every `MODE`. Record the last one (passive, as in session 14)
on at least two days, for an hour or more in total: with only 10 minutes, a single false fire
already reads as 6 per hour. If you keep the sleep model, also record
`sleep-normal neg-sleep-similar --position bed --condition quiet --split test`.

## 4. Other people (for the general, deployable model)

Same sections, with `--speaker <short lowercase name>`; one session of about 10 minutes each:
```bash
uv run python training/record_session.py wake-normal name-normal neg-very-similar neg-jarvis neg-name-very-similar --speaker sam --position desk --condition quiet
```
Keep 1-2 people **test-only** (add `--split test`); their clips go to
`~/jarvie-data/audio-speakers/<name>/<mode>/`.

## 5. Train on RunPod

1. Start a pod with the **same network volume**, open JupyterLab.
2. Upload into `/workspace/wakeword/` (notebook tab closed): `wakeword_pipeline.py`, `real_data.py`,
   `monitor.py`, `train_wakeword_runpod.ipynb` (all from `server/training/`).
3. Upload the recordings: on the Mac `runpodctl send ~/jarvie-data/real/v1`; in a pod terminal
   `mkdir -p /workspace/wakeword/real/uploads && cd /workspace/wakeword/real/uploads && runpodctl receive <code>`.
4. Notebook: restart the kernel; cells 1-3; **the stage-2 cell** (adds `jarvie_name`); cell 4.
5. `pipe.prepare_real("v1")`: check the counts per speaker and label.
6. `pipe.run("B")`: M1. The first run after a phrase-list change regenerates that model's synthetic
   speech (~1-2 h each). The current models are from before "jarvie, are you up" was added, so the
   next run regenerates wake and sleep (sleep's negatives include the wake phrases), plus
   `jarvie_name` once. Runs after that reuse it.
7. Section 7b: M2 `pipe.run("B", real="v1")`, M3 `pipe.run("B", real="v1", synthetic=False)`,
   M4 (the fine-tune cell). Monitor with `watch -n 60 python /workspace/wakeword/monitor.py`.
   **No sleep recordings** (the sessions above don't make any): M3 trains on your recordings only,
   so run it as `pipe.run("B", real="v1", synthetic=False, models=["jarvie_wake", "jarvie_name"])`;
   `jarvie_sleep` would have no positives at all. M2 and M4 can keep `jarvie_sleep` (synthetic sleep
   speech, with your recordings as negatives); add the same `models=[...]` to skip it.
8. Download `/workspace/wakeword/models/` (`runpodctl send`), **stop the pod**. On the Mac, move the
   old models aside first, so new files with the same names (`jarvie_wake__B__base.onnx`) don't mix
   with them: `mkdir -p server/models/old && mv server/models/*.onnx server/models/results.csv
   server/models/old/` (the notebook only reads `server/models/` itself, not subfolders). Then copy
   the new `.onnx` files into `server/models/`.

## 6. Choose models (Mac)

`server/jarvie/eval/compare_models.ipynb`: set `MODE = "wake"`, Restart, Run All; then `"name"`
(and `"sleep"`, which needs the optional sleep test recording from section 3). The first cell prints `ready to evaluate` or what's still missing
(models in `server/models/`, test recordings); the run stops there until both exist.
- **Personal model**: the best row in the Personal table (lowest `eer_normal`, then fast/slow recall,
  then the lowest `fires_similar` / `fires_very-similar`). For the name model as a gate, what matters
  most is recall at a low threshold and false fires per hour of everyday audio: the Gate table
  (`gate_recall` within `MAX_FPPH`; see `STAGE2_GATE.md`).
- **Deployment model**: the best row in the General table (needs other people's test recordings);
  it must beat M1 (`B`) there.

Add your results to the results log in `README.md` (this folder).
