"""Jarvie's wake phrases, in one place.

Two stages:
    stage 1  a full phrase ("jarvie, wake up") starts the bot. Detected by the local wake-word
             model (planned daemon); the transcript check in jarvie/eval/gating.py confirms it.
    stage 2  while the bot is running, the name alone ("jarvie") opens a turn. Matched in the
             Deepgram transcript by bot.py.

"Jarvis" deliberately does NOT wake Jarvie. Deepgram doesn't know the word "jarvie" and may spell
it several ways, so both stages accept the spellings below, and bot.py asks Deepgram to favour
"Jarvie" (keyterm prompting) so it isn't transcribed as "Jarvis" in the first place.
"""

STAGE1_PHRASES = [
    "jarvie wake up",
    "jarvie help me",
    "jarvie wakie wakie",
    "jarvie are you up",
    "jarvie i need some help",
]

# How Deepgram may write "jarvie" in a transcript. "jarvis" is intentionally absent.
JARVIE_SPELLINGS = [
    "jarvie",
    "jarvy",
    "jarvey",
    "jarvee",
    "jarvi",
    "jervy",
    "jervie",
    "javi",
    "javy",
    "jarbi",
    "jarby",
]

# Stage 2: the name alone, in any of those spellings.
STAGE2_PHRASES = JARVIE_SPELLINGS

# Words Deepgram should listen for (Nova-3 keyterm prompting); improves "Jarvie" vs "Jarvis".
DEEPGRAM_KEYTERMS = ["Jarvie"]
