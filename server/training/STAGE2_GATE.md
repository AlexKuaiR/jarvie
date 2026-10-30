# Stage-2 gate: the name model in front of Deepgram (design, not built)

Design notes for using the local `jarvie_name` model to decide when the bot streams audio to
Deepgram. Nothing here is implemented; it's recorded so the model work in this folder targets what
the gate needs. The stage-1 wake gate ("jarvie, wake up" starting the bot) is separate.

## Why

Today, while the bot runs, every sound in the room streams to Deepgram, and stage 2 ("jarvie"
alone) is matched in the transcript (`AwakeAtStartWakePhrase` in `server/jarvie/wake.py`,
`STAGE2_PHRASES` in `server/jarvie/wake_phrases.py`). Gating keeps a roommate's conversation from
going to the cloud between turns, and cuts STT cost if the bot stays up for long stretches.

## Architecture

```
mic ─► transport.input ─► [JarvieGate] ─► stt (Deepgram) ─► user_aggregator ─► llm ─► tts ─► out
                              │  ▲
                 CLOSED:      │  │ open / close signals
                 ring buffer  │  │  - wake strategy events (detected / timeout)
                 + NameDetector  │  - bot started / stopped speaking
                              ▼  │
                 score ≥ threshold → send buffer to Deepgram, then live audio
```

| State | Audio to Deepgram | Enter when | Leave when |
|---|---|---|---|
| OPEN | live | bot starts (it starts awake), a confirmed "jarvie", or the bot is speaking | `on_wake_phrase_timeout` (10 s of quiet) → CLOSED |
| CLOSED | none; KeepAlive pings only | the timeout | detector score ≥ threshold → CONFIRMING |
| CONFIRMING | the buffered ~1.5 s, then live | the detector fired | transcript has a "jarvie" spelling (`on_wake_phrase_detected`) → OPEN; none within ~3 s → CLOSED (false fire, logged) |

Pieces:
1. **`NameDetector` interface**: takes ~80 ms chunks of 16 kHz audio and returns a score.
   - `OnnxNameDetector` runs `jarvie_name.onnx`.
   - `SimulatedNameDetector` fires on a scripted schedule, so tests run without a mic.
2. **Ring buffer**: holds the last ~1.5 s while CLOSED.
3. **`JarvieGate`**: a Pipecat `FrameProcessor` that holds the state machine. It re-chunks transport
   frames into the model's 80 ms windows and listens to the wake strategy's events.
4. **Deepgram KeepAlive** while CLOSED.

### Who buffers, who confirms

- **The buffer is local, in the gate.** Deepgram receives no audio at all while CLOSED.
- **Deepgram only keeps the connection open.** The gate sends a KeepAlive message every few seconds
  (a "still here" ping with no audio) so Deepgram doesn't drop the idle connection after ~10 s.
  This only saves the reconnect delay on each activation.
- **Deepgram confirms the local model, but only after it fires.** The gate first sends the buffered
  1.5 s, which includes the spoken "jarvie" (the model fires after the word ends), then live audio.
  The existing transcript match decides whether a turn opens. The gate only decides whether
  Deepgram hears the audio.

## Activation speed

The gate shouldn't slow activation:
- **Running the model is cheap.** It's a small classifier run on every ~80 ms chunk: a few ms on a
  laptop, real time on a Pi 4/5. It runs alongside the audio stream.
- **Detection delay is about the same as today.** The model fires ~100–300 ms after the word ends
  (to be measured, see below). Today, matching the transcript also waits a few hundred ms for
  Deepgram's round trip.
- **Response time stays the same.** Deepgram works through the buffered audio faster than real time
  and catches up while you're still talking. When the bot answers is still decided by smart-turn
  and VAD's 1 s `stop_secs`.
- **The one slow case:** if the Deepgram connection were closed instead of kept open, every
  activation would pay for a new connection.

## Open questions

- **Server or edge.** The gate in the server pipeline keeps audio away from Deepgram. On Pipecat
  Cloud, though, the audio still leaves the room to reach the server. The end goal is the Pi daemon
  with three modes on one mic:
  - **OFF**: the stage-1 model listens for "jarvie, wake up".
  - **CLOSED**: the name model listens for "jarvie".
  - **OPEN**: audio streams to the bot.

  Keep the detector behind `NameDetector` so it can move from the server to the Pi.
- **What the gate blocks.** Silero VAD and smart-turn run in `user_aggregator`, after STT. If the
  gate drops audio frames, they hear nothing while CLOSED, then a 1.5 s burst. That's probably fine,
  but test it. The alternative is to mute only the audio going to Deepgram (Pipecat's
  `STTMuteFilter` may work this way; unchecked).
- **Recall is the risk.** "jarvie" alone is short and hard to catch without STT. A miss means you're
  ignored. Today Deepgram plus the "Jarvie" keyterm does this job.

## What this means for the name model (this folder)

The gate doesn't need a precise detector. It needs **high recall at a low threshold**, with a
tolerable false-fire rate: each false fire costs about 3 s of CONFIRMING, then the transcript check
rejects it. Equal-error rate (`eer_normal`) is the wrong balance for a gate. For `MODE = "name"`,
two numbers say whether the gate is feasible before anyone builds it:
- **Recall at a false-fire budget**: recall at ≤ N false fires per hour of `neg-everyday` audio.
  `compare_models.ipynb` reports this in its Gate table (`compare.gate_summary`, budget
  `MAX_FPPH`). A fire counts as one gate opening that holds for 3 s (`compare.gate_fires`), so a
  model stuck above its threshold counts once per 3 s instead of looking like a single fire.
- **How late it fires**: time from the end of the word to the detection. Not measured yet.

## Related decision: the sleep model

A separate `jarvie_sleep` model is probably unnecessary. "jarvie, go to sleep" is only said while
the bot runs, so Deepgram already hears it (or, with the gate, the name model opens STT because the
phrase starts with "jarvie"). A transcript match on sleep phrases, or an LLM tool call, can end the
session; that's bot work for another branch. If the sleep model is kept, `RECORDING_GUIDE.md` needs
fixes:
- Its "~40 sleep positives" is really ~18: the `sleep-*` prompt sections have 18 lines and
  session 8 is the only one recording them.
- All sleep clips are bed/quiet.
- No other speakers record sleep, so the General table has no sleep data.
