# Recording guide

Step-by-step: recording your own voice data for the three Jarvie wake-word models, then training and
choosing models. What to say is in `recording_prompts.txt`; the recorder is `record_session.py`.

| Model | Fires on | Used for | Prompt sections |
|---|---|---|---|
| `jarvie_wake` (stage 1) | "jarvie, wake up", "jarvie, help me", "jarvie, wakie wakie", "jarvie, are you up?" | starting the bot (the planned daemon) | `wake-*` |
| `jarvie_sleep` | "jarvie, go to sleep", "jarvie, sleep now" | putting it to sleep (planned) | `sleep-*` |
| `jarvie_name` (stage 2) | "jarvie" alone | talking to the bot while it runs (planned; today bot.py matches the transcript) | `name-*` |

Negatives (`neg-*`) are things that must not trigger. "Jarvis" is always a negative.

## 0. Once, before the first session

```bash
cd ~/pipecat-quickstart/server
git switch wakeword-eval && git pull
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
| 8 | bed, quiet | `wake-normal wake-slow sleep-normal sleep-fast sleep-slow --position bed --condition quiet` |
| 9 | bed, quiet | `neg-sleep-similar neg-similar name-normal --position bed --condition quiet` |
| 10 | desk, fan/AC on | `wake-normal wake-areyouup name-normal neg-very-similar neg-name-very-similar --position desk --condition fan --repeats 2 --shuffle` |
| 11 | desk, TV/video on | `neg-everyday --position desk --condition talk` |

**Already recorded session 1 before "jarvie, are you up?" existed?** Record just the new phrase as a
catch-up, in the same place:
`wake-areyouup wake-areyouup-fast wake-areyouup-slow --position desk --condition quiet`. Nothing already
recorded needs redoing.

After sessions 1-11, roughly: ~250 stage-1 positives, ~150 name positives, ~40 sleep positives,
~400 negatives and 10 minutes of everyday speech. For more, repeat 6, 7 and 10 on other days.

- Files go to `~/jarvie-data/real/v1/me/<wake|sleep|name>/` with a `manifest.csv` of what was said.
- `--position` / `--condition` must be lowercase letters or digits; copy them from the table so a
  typo doesn't create a new label.

### Stopping and resuming
- `q` stops and prints e.g. `Resume with --start 11`. Rerun **the same command** plus `--start 11`
  (it applies to the first section in the command).
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
uv run python training/record_session.py sleep-normal neg-sleep-similar --position bed --condition quiet --split test
```
They go to `~/jarvie-data/audio/<wake|sleep|name>/`.

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
6. `pipe.run("B")`: M1. Wake/sleep are cached; `jarvie_name` generates its synthetic speech once.
7. Section 7b: M2 `pipe.run("B", real="v1")`, M3 `pipe.run("B", real="v1", synthetic=False)`,
   M4 (the fine-tune cell). Monitor with `watch -n 60 python /workspace/wakeword/monitor.py`.
8. Download `/workspace/wakeword/models/` (`runpodctl send`), copy the `.onnx` files into
   `server/models/` on the Mac, **stop the pod**.

## 6. Choose models (Mac)

`server/jarvie/eval/compare_models.ipynb`: set `MODE = "wake"`, Restart, Run All; then `"name"`,
then `"sleep"`.
- **Personal model**: the best row in the Personal table (lowest `eer_normal`, then fast/slow recall,
  then the lowest `fires_similar` / `fires_very-similar`).
- **Deployment model**: the best row in the General table (needs other people's test recordings);
  it must beat M1 (`B`) there.

Add your results to the results log in `README.md` (this folder).
