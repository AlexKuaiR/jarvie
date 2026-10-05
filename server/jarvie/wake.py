from pipecat.turns.user_start.wake_phrase_user_turn_start_strategy import (
    WakePhraseUserTurnStartStrategy,
    _WakeState,
)


class AwakeAtStartWakePhrase(WakePhraseUserTurnStartStrategy):
    """Wake-phrase gate that starts awake, so you can answer the greeting directly.

    After `timeout` seconds of silence it falls back to requiring the wake phrase,
    exactly like the parent class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pipecat has no public "start awake" switch, so set its private state.
        # Re-check this on Pipecat upgrades; tests/test_wake.py will catch a rename.
        self._state = _WakeState.AWAKE
