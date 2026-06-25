"""Constrained command grammar for the Stretch RE2 dual-mode controller.

This module is the single source of truth for the closed command
vocabulary C described in Section 5.3 of the paper ("Constrained
Command Grammar"). Every command carries structured metadata:

* ``mode``   -- which control mode may execute it
               (POSITION, NAVIGATION, BOTH, or ANY for mode-independent)
* ``level``  -- control-abstraction level
               (1 = primitive: one ROS action/service/publish;
                2 = high-level behaviour: a multi-step autonomy routine)
* ``params`` -- parameter signature for the regex parser
* ``bounds`` -- safe numeric ranges; the validator clamps arguments
               that fall outside these ranges before dispatch

The LLM never sees this module directly. It sees the system prompt
(:mod:`stretch_llm.llm.prompts`); the controller uses this module to
validate, clamp, and route everything the model emits.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class Mode(str, Enum):
    POSITION = "position"
    NAVIGATION = "navigation"
    BOTH = "both"          # legal in either mode (head, perception)
    ANY = "any"            # mode-independent (stop, mode switches)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    mode: Mode
    level: int                                   # 1 = primitive, 2 = behaviour
    params: Tuple[str, ...] = ()                 # e.g. ("float",), ("str", "float")
    bounds: Optional[Tuple[float, float]] = None # safe range for the numeric arg
    default: Optional[float] = None              # value when the user omits one
    description: str = ""
    in_vocabulary: bool = True                   # exposed to the LLM (member of C)


# ---------------------------------------------------------------------------
# The 47-command vocabulary (Table 2 of the paper)
# ---------------------------------------------------------------------------

COMMAND_SPECS = {spec.name: spec for spec in [
    # -- Base motion (position mode) ---------------------------------------
    CommandSpec("base_forward", Mode.POSITION, 1, ("float",), (0.0, 2.0), 0.2,
                "Translate the base forward by d metres."),
    CommandSpec("base_back",    Mode.POSITION, 1, ("float",), (0.0, 2.0), 0.2,
                "Translate the base backward by d metres."),
    CommandSpec("base_left",    Mode.POSITION, 1, ("float",), (0.0, 3.1416), 0.3,
                "Rotate the base left by r radians."),
    CommandSpec("base_right",   Mode.POSITION, 1, ("float",), (0.0, 3.1416), 0.3,
                "Rotate the base right by r radians."),

    # -- Navigation (navigation mode) --------------------------------------
    CommandSpec("nav_to",        Mode.NAVIGATION, 1, ("float", "float", "float"),
                description="Navigate to absolute SLAM-frame pose (x, y, yaw)."),
    CommandSpec("nav_to_named",  Mode.NAVIGATION, 1, ("str",),
                description="Navigate to a saved waypoint by name."),
    CommandSpec("nav_relative",  Mode.NAVIGATION, 1, ("str", "float"), (0.0, 5.0),
                description="Relative move resolved against the AMCL pose: "
                            "direction in {forward, back, left, right}."),

    # -- Lift ---------------------------------------------------------------
    CommandSpec("lift_up",   Mode.POSITION, 1, ("float",), (0.0, 1.1), 0.1,
                "Raise the lift by d metres."),
    CommandSpec("lift_down", Mode.POSITION, 1, ("float",), (0.0, 1.1), 0.1,
                "Lower the lift by d metres."),
    CommandSpec("lift_home", Mode.POSITION, 1,
                description="Send the lift to its home height."),

    # -- Wrist / arm extension ----------------------------------------------
    CommandSpec("wrist_out",  Mode.POSITION, 1, ("float",), (0.0, 0.52), 0.1,
                "Extend the telescoping arm by d metres."),
    CommandSpec("wrist_in",   Mode.POSITION, 1, ("float",), (0.0, 0.52), 0.1,
                "Retract the telescoping arm by d metres."),
    CommandSpec("wrist_home", Mode.POSITION, 1,
                description="Retract the arm fully."),
    CommandSpec("wrist_yaw_left",  Mode.POSITION, 1, ("float",), (0.0, 3.1416), 0.3,
                "Rotate the gripper yaw left by r radians."),
    CommandSpec("wrist_yaw_right", Mode.POSITION, 1, ("float",), (0.0, 3.1416), 0.3,
                "Rotate the gripper yaw right by r radians."),
    CommandSpec("wrist_yaw_home",  Mode.POSITION, 1,
                description="Centre the gripper yaw."),

    # -- Gripper --------------------------------------------------------------
    CommandSpec("grip_open",  Mode.POSITION, 1, description="Open the gripper."),
    CommandSpec("grip_close", Mode.POSITION, 1, description="Close the gripper."),
    CommandSpec("grip_home",  Mode.POSITION, 1, description="Home the gripper."),
    CommandSpec("grip_left",  Mode.POSITION, 1, ("float",), (0.0, 1.57), 0.2,
                "Rotate the gripper roll left."),
    CommandSpec("grip_right", Mode.POSITION, 1, ("float",), (0.0, 1.57), 0.2,
                "Rotate the gripper roll right."),

    # -- Head -----------------------------------------------------------------
    CommandSpec("head_left",  Mode.BOTH, 1, ("float",), (0.0, 3.1416), 0.3),
    CommandSpec("head_right", Mode.BOTH, 1, ("float",), (0.0, 3.1416), 0.3),
    CommandSpec("head_up",    Mode.BOTH, 1, ("float",), (0.0, 1.0), 0.2),
    CommandSpec("head_down",  Mode.BOTH, 1, ("float",), (0.0, 1.0), 0.2),
    CommandSpec("head_home",  Mode.BOTH, 1),

    # -- Directional camera views ---------------------------------------------
    CommandSpec("look_front",  Mode.BOTH, 1, description="Camera view, ahead."),
    CommandSpec("look_left",   Mode.BOTH, 1, description="Camera view, left."),
    CommandSpec("look_right",  Mode.BOTH, 1, description="Camera view, right."),
    CommandSpec("look_behind", Mode.BOTH, 1, description="Camera view, behind."),
    CommandSpec("look_up",     Mode.BOTH, 1, description="Camera view, upward."),
    CommandSpec("look_down",   Mode.BOTH, 1, description="Camera view, downward."),

    # -- Mode and safety --------------------------------------------------------
    CommandSpec("mode_position", Mode.ANY, 1, description="Switch to position mode."),
    CommandSpec("mode_nav",      Mode.ANY, 1, description="Switch to navigation mode."),
    CommandSpec("stop",          Mode.ANY, 1, description="Emergency stop."),
    CommandSpec("resume",        Mode.ANY, 1, description="Resume after a stop."),

    # -- Perception tools --------------------------------------------------------
    CommandSpec("get_state",              Mode.BOTH, 1),
    CommandSpec("get_camera_view",        Mode.BOTH, 1),
    CommandSpec("get_pointcloud_summary", Mode.BOTH, 1),
    CommandSpec("get_object_distance",    Mode.BOTH, 1, ("str",)),
    CommandSpec("get_slam_map",           Mode.BOTH, 1),

    # -- High-level behaviours (Level 2) -------------------------------------------
    CommandSpec("search_for_object", Mode.POSITION, 2, ("str",),
                description="Rotate-and-detect sweep, head fixed, up to 12 steps."),
    CommandSpec("approach_object",   Mode.POSITION, 2, ("str",),
                description="Thirds-of-image visual servoing toward a target."),
    CommandSpec("pick_object",       Mode.POSITION, 2, ("str",),
                description="Full grasp routine: yaw, look, lift, align, grasp."),
    # place_object is keyword-routed by the controller's run loop and never
    # emitted by the LLM, so it stays outside the 47-command vocabulary C
    # (it does not appear in Table 2 of the paper).
    CommandSpec("place_object",      Mode.POSITION, 2, ("str",),
                description="Visual placement on a named surface.",
                in_vocabulary=False),
    CommandSpec("analyze_three_camera_frames", Mode.POSITION, 2, ("str",),
                description="Three-frame capture, description, and distance."),
    CommandSpec("estimate_gripper_distance",   Mode.POSITION, 2, ("str",),
                description="Gripper-to-object distance from the camera."),
    CommandSpec("demo_sequence",     Mode.POSITION, 2,
                description="Fixed diagnostic gripper demonstration."),
]}

# Flat list, kept for backward compatibility with the v27 interface.
# Only vocabulary members (the 47 commands of C) are LLM-emittable.
ALLOWED_COMMANDS = sorted(n for n, s in COMMAND_SPECS.items() if s.in_vocabulary)

PERCEPTION_TOOLS = (
    "get_state", "get_camera_view", "get_pointcloud_summary",
    "get_object_distance", "get_slam_map",
    "look_front", "look_left", "look_right", "look_behind",
    "look_up", "look_down",
)

HIGH_LEVEL_BEHAVIOURS = tuple(
    s.name for s in COMMAND_SPECS.values() if s.level == 2 and s.in_vocabulary
)

NAVIGATION_ONLY = tuple(
    s.name for s in COMMAND_SPECS.values() if s.mode is Mode.NAVIGATION
)

POSITION_ONLY = tuple(
    s.name for s in COMMAND_SPECS.values() if s.mode is Mode.POSITION
)


def spec_for(name: str) -> Optional[CommandSpec]:
    """Return the :class:`CommandSpec` for a command name, or ``None``."""
    return COMMAND_SPECS.get(name)


def allowed_in_mode(name: str, mode: str) -> bool:
    """Mode-compatibility predicate ALLOWED(c, m) from the problem formulation."""
    spec = COMMAND_SPECS.get(name)
    if spec is None:
        return False
    if spec.mode in (Mode.BOTH, Mode.ANY):
        return True
    return spec.mode.value == mode


def clamp_argument(name: str, value: float) -> float:
    """Clamp a numeric argument to the per-command safe range."""
    spec = COMMAND_SPECS.get(name)
    if spec is None or spec.bounds is None:
        return value
    lo, hi = spec.bounds
    return max(lo, min(hi, value))
