#!/usr/bin/env python3
"""Synthetic counterpart dataset D_syn for the E1 comparison.

Generates a matched-scale dataset with the standard template-based
recipe the paper compares against: handcrafted paraphrase templates per
command, parameter values sampled from safe ranges, mode-aware filters.
Each example is a flat (instruction, command) pair wrapped in the same
chat format as the real-trajectory data, so the only difference between
FT-Real and FT-Synth is the data distribution itself.

Usage:
    python training/synthetic_baseline.py --n 870 \
        --out data/trajectories/processed_synth
"""

import argparse
import json
import random
from pathlib import Path

from stretch_llm.llm.grammar import COMMAND_SPECS, Mode
from stretch_llm.llm.prompts import SYSTEM_PROMPT

TEMPLATES = {
    "base_forward": ["move forward {v} meters", "go ahead {v} m",
                     "drive forward by {v} meters", "advance {v} meters"],
    "base_back": ["move back {v} meters", "back up {v} m", "reverse {v} meters"],
    "base_left": ["turn left {v} radians", "rotate left by {v}"],
    "base_right": ["turn right {v} radians", "rotate right by {v}"],
    "lift_up": ["raise the lift {v} meters", "lift up by {v} m",
                "move the lift higher by {v}"],
    "lift_down": ["lower the lift {v} meters", "bring the lift down {v} m"],
    "wrist_out": ["extend the arm {v} meters", "arm out {v} m",
                  "reach out by {v} meters"],
    "wrist_in": ["retract the arm {v} meters", "arm in {v} m",
                 "pull the arm back {v}"],
    "grip_open": ["open the gripper", "release the gripper", "open your hand"],
    "grip_close": ["close the gripper", "grip it", "close your hand"],
    "head_left": ["pan the head left {v}", "look a bit to the left"],
    "head_right": ["pan the head right {v}", "look a bit to the right"],
    "search_for_object": ["search for the {o}", "look around for a {o}",
                          "find the {o} in the room"],
    "approach_object": ["go to the {o}", "approach the {o}",
                        "get close to the {o}"],
    "pick_object": ["pick up the {o}", "grab the {o}", "grasp the {o}"],
    "get_camera_view": ["what do you see", "describe the camera view"],
    "get_pointcloud_summary": ["is the path clear", "check for obstacles"],
    "get_slam_map": ["show me the map", "where are you on the map"],
    "mode_nav": ["switch to navigation mode", "enter nav mode"],
    "mode_position": ["switch to position mode", "enter position mode"],
    "stop": ["stop", "halt", "freeze"],
}

OBJECTS = ["water_bottle", "coffee_mug", "tv_remote", "book", "tennis_ball",
           "game_controller", "stapler", "apple", "tissue_box", "screwdriver"]


def sample_pair(rng):
    name = rng.choice(list(TEMPLATES))
    spec = COMMAND_SPECS[name]
    template = rng.choice(TEMPLATES[name])
    obj = rng.choice(OBJECTS)
    if spec.bounds:
        lo, hi = spec.bounds
        v = round(rng.uniform(max(lo, 0.05), min(hi, 1.0)), 2)
    else:
        v = None
    instruction = template.format(v=v, o=obj.replace("_", " "))
    if "{o}" in template or "(o" in str(spec.params):
        command = f"{name}({obj})" if spec.params else name
    elif v is not None and spec.params:
        command = f"{name}({v})"
    else:
        command = name
    mode = "navigation" if spec.mode is Mode.NAVIGATION else "position"
    return instruction, command, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=870)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for i in range(args.n):
            instruction, command, mode = sample_pair(rng)
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": (f"Current robot mode: {mode}\n\n"
                                 f"User request: {instruction}")},
                    {"role": "assistant",
                     "content": ("<|verdict:success|> <|level:primitive|> "
                                 + command)},
                ],
                "verdict": "success",
                "trajectory_id": f"synth-{i:05d}",
                "weight": 1.0,
            }
            f.write(json.dumps(example) + "\n")
    print(f"Wrote {args.n} synthetic pairs to {out/'train.jsonl'}")


if __name__ == "__main__":
    main()
