import pytest

from jarvie.wake import AwakeAtStartWakePhrase
from jarvie.wake_phrases import DEEPGRAM_KEYTERMS, JARVIE_SPELLINGS, STAGE1_PHRASES, STAGE2_PHRASES


def test_jarvis_does_not_wake_jarvie():
    assert "jarvis" not in JARVIE_SPELLINGS and "jarvis" not in STAGE2_PHRASES
    assert all(p.startswith("jarvie ") for p in STAGE1_PHRASES)
    assert DEEPGRAM_KEYTERMS == ["Jarvie"]


def stage2_matches(text: str) -> bool:
    # The patterns Pipecat's wake-phrase strategy compiles from bot.py's phrases, applied the way it
    # applies them (punctuation stripped). Calling its matcher directly would also start a pipeline
    # task, which needs a running pipeline.
    strategy = AwakeAtStartWakePhrase(phrases=STAGE2_PHRASES)
    cleaned = strategy._strip_punctuation(text)
    return any(p.search(cleaned) for p in strategy._patterns)


@pytest.mark.parametrize(
    "text", ["Jarvie", "Jarvie, what's on my calendar?", "hey jarvy", "Okay, Jarvee."]
)
def test_stage2_name_opens_a_turn(text):
    assert stage2_matches(text)


@pytest.mark.parametrize(
    "text", ["Jarvis", "Jarvis, what time is it?", "Harvey", "jar of jam", "Marvin"]
)
def test_stage2_ignores_other_words(text):
    assert not stage2_matches(text)
