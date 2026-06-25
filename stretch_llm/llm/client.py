"""HTTP clients for the two-tier LLM inference architecture.

Tier 1 (primary) is the fine-tuned Llama-3.1-8B served on the AI-Panther
HPC cluster and reached over the campus LAN, typically through an SSH
tunnel (see ``scripts/hpc/``). Tier 2 (monitor) is the fine-tuned
Gemma-3-4B served on the robot's onboard compute for execution
verification, visual-servoing decisions, and command-completion checks.

Configuration is environment-driven so the same code runs on the robot,
on a development laptop, and inside the HPC job:

    export OLLAMA_URL=http://127.0.0.1:11435      # tunnel to AI-Panther
    export OLLAMA_EDGE_URL=http://127.0.0.1:11434 # local edge instance
    export PRIMARY_MODEL=llama3.1:latest
    export MONITOR_MODEL=gemma3:4b
"""

import os
from typing import List, Optional

import requests

from .grammar import ALLOWED_COMMANDS
from .parser import extract_commands, extract_name
from .prompts import MONITOR_COMPLETION_PROMPT, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tier-1 endpoint. On the robot this points at the SSH tunnel to the HPC;
# port 11435 is the project convention because 11434 is taken by the edge
# instance on shared nodes.
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11435")

# Tier-2 endpoint on the robot itself.
OLLAMA_EDGE_URL = os.getenv("OLLAMA_EDGE_URL", "http://localhost:11434")

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama3.1:latest")
MONITOR_MODEL = os.getenv("MONITOR_MODEL", "gemma3:4b")

REQUEST_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _generate(base_url: str, model: str, prompt: str) -> str:
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def chat(messages: List[dict], model: Optional[str] = None,
         max_tokens: int = 500, temperature: float = 0.7,
         base_url: Optional[str] = None) -> str:
    """Chat-format completion against an Ollama endpoint.

    Defaults to the edge monitor model: this is the call the visual
    servoing loops make at the rate of the inner control loop, so the
    default must never silently route a camera frame off the robot.
    """
    if model is None:
        model = MONITOR_MODEL
    if base_url is None:
        base_url = OLLAMA_EDGE_URL if model == MONITOR_MODEL else OLLAMA_BASE_URL
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload,
                          timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:  # noqa: BLE001 - a robot loop must not crash here
        print("[llm] Ollama chat error:", e)
        return "ERROR"


# Backward-compatible alias for the v27 controller code.
tinyllama_chat = chat


# ---------------------------------------------------------------------------
# Tier 1: instruction grounding
# ---------------------------------------------------------------------------

def ask_llm(text: str) -> List[str]:
    """Ground one natural-language turn into a validated command list.

    Sends the deployed system prompt plus the user turn to the Tier-1
    primary model, extracts command-shaped tokens from the response,
    rejects anything outside the closed vocabulary, and falls back to a
    single ``stop`` if nothing valid survives. Natural-language replies
    (the ``ANSWER:`` prefix) pass through unchanged.
    """
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {text}\nAssistant:\n"
        raw = _generate(OLLAMA_BASE_URL, PRIMARY_MODEL, full_prompt)

        if raw.strip().lower().startswith("answer:"):
            return [raw]

        cmds = extract_commands(raw)
        validated = []
        for c in cmds:
            name = extract_name(c)
            if name in ALLOWED_COMMANDS:
                validated.append(c)
            else:
                print(f"[llm] Filtered invalid: '{c}' (name: '{name}')")

        if not validated:
            print("[llm] No valid commands -> safety stop")
            return ["stop"]
        return validated

    except Exception as e:  # noqa: BLE001
        print(f"[llm] Primary model error: {e}")
        return ["stop"]  # fail-safe


# ---------------------------------------------------------------------------
# Tier 2: command-completion checks
# ---------------------------------------------------------------------------

def ensure_command_completion(commands: List[str],
                              sensor_feedback: Optional[str] = None
                              ) -> List[str]:
    """Ask the edge monitor whether a command list has physically completed.

    The monitor must not invent new commands: anything it returns is
    re-validated against the closed vocabulary, and on any failure the
    original list is returned unchanged.
    """
    try:
        prompt = MONITOR_COMPLETION_PROMPT.format(
            commands=", ".join(commands),
            feedback=sensor_feedback or "No feedback yet.",
        )
        raw = _generate(OLLAMA_EDGE_URL, MONITOR_MODEL, prompt)
        continued = [c for c in extract_commands(raw)
                     if extract_name(c) in ALLOWED_COMMANDS]
        return continued or commands
    except Exception as e:  # noqa: BLE001
        print(f"[llm] Completion monitor error: {e}")
        return commands
