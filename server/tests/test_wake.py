from pipecat.turns.user_start.wake_phrase_user_turn_start_strategy import (
    WakePhraseUserTurnStartStrategy,
)

from jarvie.wake import AwakeAtStartWakePhrase


def test_starts_awake():
    assert AwakeAtStartWakePhrase(phrases=["jarvis"]).state.value == "awake"


def test_pipecat_default_still_starts_idle():
    # Guards the private-state assumption in jarvie/wake.py: if Pipecat changes how
    # the strategy starts, this fails and the subclass needs a second look.
    assert WakePhraseUserTurnStartStrategy(phrases=["jarvis"]).state.value == "idle"
