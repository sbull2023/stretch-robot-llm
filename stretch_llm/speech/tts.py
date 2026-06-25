"""Text-to-speech output for the operator-facing reply channel.

The deployed system speaks every natural-language reply (the ``ANSWER:``
turns and per-command execution results) through the OpenAI TTS endpoint.
The function degrades gracefully: with no API key, no audio backend, or no
network, it prints the text and returns, so the control loop never blocks
on speech.
"""

import os
import tempfile

import requests

TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")


def speak_text(text: str) -> None:
    """Speak ``text`` aloud; fall back to stdout when speech is unavailable."""
    print(f"[speech] {text}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return  # text-only operation

    try:
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": TTS_MODEL, "voice": TTS_VOICE, "input": text},
            timeout=30,
        )
        if response.status_code != 200:
            print("[speech] TTS server response:", response.text)
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.content)
            path = f.name
        try:
            from playsound import playsound
            playsound(path)
        finally:
            os.remove(path)
    except Exception as e:  # noqa: BLE001 - speech must never crash the loop
        print(f"[speech] TTS error: {e}")
