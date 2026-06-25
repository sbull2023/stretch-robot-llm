"""Speech-to-text input.

The deployed system uses a microphone front-end (``mic_llm_v2``) on the
operator workstation. That module carries site-specific audio-device
configuration, so it ships separately; this wrapper imports it when
present and otherwise falls back to keyboard input, which keeps the
controller usable on machines with no microphone stack.
"""


def listen() -> str:
    """Return one transcribed utterance, or a typed line as a fallback."""
    try:
        from mic_llm_v2 import listen as _mic_listen
        return _mic_listen()
    except ImportError:
        print("[speech] Microphone front-end unavailable; type instead.")
        try:
            return input("You (typed) > ").strip()
        except EOFError:
            return ""
