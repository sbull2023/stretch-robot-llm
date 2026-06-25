"""Parsers and validators for LLM command output.

Every token the model generates passes through this module before it can
reach an actuator. The pipeline is the one specified in Section 5.3 of the
paper: (i) extract command name and arguments by regex, (ii) reject any name
outside the closed vocabulary, (iii) clamp numeric arguments to per-command
safe ranges. If nothing survives, the caller falls back to an emergency
stop.

The v27 implementation used ``eval`` on raw model output to coerce
parameters; this module replaces that with :func:`ast.literal_eval` plus a
string fallback, so a malformed or adversarial argument can never execute
code on the robot.
"""

import ast
import re
from typing import List, Optional, Tuple

from .grammar import ALLOWED_COMMANDS, allowed_in_mode, clamp_argument, spec_for

# A command token: lowercase name, optional parenthesised argument list.
_CMD_PATTERN = re.compile(r"([a-z_]+(?:\([^\)]*\))?)")
_NAME_PATTERN = re.compile(r"([a-z_]+)")
_PARAM_PATTERN = re.compile(r"[a-z_]+\s*\((.*)\)")


def extract_name(cmd: str) -> str:
    """``base_forward(0.2)`` -> ``base_forward``; bare names pass through."""
    match = _NAME_PATTERN.match(cmd.strip())
    return match.group(1) if match else ""


def _coerce(token: str):
    """Coerce one raw argument token to a Python value, never via eval."""
    token = token.strip().strip("'\"")
    try:
        return ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return token  # symbolic argument such as a waypoint or object name


def parse_command(cmd: str) -> Tuple[str, Optional[object]]:
    """Parse one command string into ``(name, params)``.

    ``base_forward(0.2)``        -> ``('base_forward', 0.2)``
    ``grip_close()``             -> ``('grip_close', None)``
    ``nav_relative(forward,0.5)``-> ``('nav_relative', ('forward', 0.5))``
    """
    name = extract_name(cmd)
    params = None
    match = _PARAM_PATTERN.match(cmd.strip())
    if match:
        raw = match.group(1)
        if raw.strip():
            values = tuple(_coerce(p) for p in raw.split(","))
            params = values[0] if len(values) == 1 else values
    return name, params


# Backward-compatible alias for the controller-side helper from v27.
def parse_cmd(cmd: str) -> Tuple[Optional[str], Optional[object]]:
    name = extract_name(cmd)
    if not name:
        return None, None
    return parse_command(cmd)


def extract_commands(raw: str) -> List[str]:
    """Pull every command-shaped token out of a raw model response."""
    return [m.group(1).strip() for m in _CMD_PATTERN.finditer(raw)
            if m.group(1).strip()]


def validate_commands(cmds: List[str], mode: Optional[str] = None,
                      verbose: bool = True) -> List[str]:
    """Filter a candidate command list against the closed vocabulary.

    When ``mode`` is given, mode-incompatible commands are also rejected;
    this is the controller-routing layer of the three-layer mode-separation
    defence (the prompt rule and the hardware Trigger services are the other
    two).
    """
    validated = []
    for c in cmds:
        name = extract_name(c)
        if name not in ALLOWED_COMMANDS:
            if verbose:
                print(f"[parser] Filtered invalid command: '{c}' (name: '{name}')")
            continue
        if mode is not None and not allowed_in_mode(name, mode):
            if verbose:
                print(f"[parser] Rejected '{c}': not allowed in mode '{mode}'")
            continue
        validated.append(clamp_command(c))
    return validated


def clamp_command(cmd: str) -> str:
    """Clamp the numeric argument of a command to its safe range."""
    name, params = parse_command(cmd)
    spec = spec_for(name)
    if spec is None or spec.bounds is None or params is None:
        return cmd
    if isinstance(params, (int, float)):
        clamped = clamp_argument(name, float(params))
        if clamped != float(params):
            print(f"[parser] Clamped {name} argument {params} -> {clamped}")
        return f"{name}({clamped})"
    if isinstance(params, tuple) and len(params) == 2 and \
            isinstance(params[1], (int, float)):
        clamped = clamp_argument(name, float(params[1]))
        return f"{name}({params[0]},{clamped})"
    return cmd
