#!/usr/bin/env python3
"""Entry point for the dual-mode controller on the robot.

Equivalent to ``rosrun``/``roslaunch`` execution, but usable from any
checkout: it puts the repository on sys.path and starts the node.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stretch_llm.controller.node import main  # noqa: E402

if __name__ == "__main__":
    main()
