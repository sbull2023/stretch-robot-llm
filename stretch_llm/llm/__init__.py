"""Language-model side of the stack: grammar, prompts, parsers, clients.

This package supersedes the monolithic ``llm_v27.py`` module. The original
flat interface is re-exported here so legacy code that did
``from llm_v27 import *`` can switch to ``from stretch_llm.llm import *``
with no further edits.
"""

from .grammar import (ALLOWED_COMMANDS, COMMAND_SPECS, HIGH_LEVEL_BEHAVIOURS,
                      PERCEPTION_TOOLS, Mode, allowed_in_mode, clamp_argument,
                      spec_for)
from .parser import (extract_commands, extract_name, parse_cmd, parse_command,
                     validate_commands)
from .prompts import SYSTEM_PROMPT
from .client import (MONITOR_MODEL, OLLAMA_BASE_URL, OLLAMA_EDGE_URL,
                     PRIMARY_MODEL, ask_llm, chat, ensure_command_completion,
                     tinyllama_chat)

__all__ = [
    "ALLOWED_COMMANDS", "COMMAND_SPECS", "HIGH_LEVEL_BEHAVIOURS",
    "PERCEPTION_TOOLS", "Mode", "allowed_in_mode", "clamp_argument",
    "spec_for", "extract_commands", "extract_name", "parse_cmd",
    "parse_command", "validate_commands", "SYSTEM_PROMPT", "MONITOR_MODEL",
    "OLLAMA_BASE_URL", "OLLAMA_EDGE_URL", "PRIMARY_MODEL", "ask_llm", "chat",
    "ensure_command_completion", "tinyllama_chat",
]
