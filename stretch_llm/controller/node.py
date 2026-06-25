#!/usr/bin/env python3
"""Dual-mode Stretch RE2 controller node.

Composition root for the deployed system. The controller is a single ROS
node assembled from six focused mixins:

* :class:`~stretch_llm.controller.core.CoreMixin` -- ROS interfaces and
  primitive motion (joint trajectories, velocity bursts, navigation goals).
* :class:`~stretch_llm.controller.monitor.MonitorMixin` -- Tier-2
  execution verification with the edge model.
* :class:`~stretch_llm.controller.perception.PerceptionMixin` -- the five
  perception tools plus distance estimation.
* :class:`~stretch_llm.controller.behaviors.BehaviorMixin` -- the Level-2
  closed-loop visual behaviours (search, approach, pick, place, demo).
* :class:`~stretch_llm.controller.execution.ExecutionMixin` -- the
  central command dispatch routine.
* :class:`~stretch_llm.controller.loop.ReasoningLoopMixin` -- the
  multi-turn observe--reason--act loop and the interactive ``run`` entry.

Run it directly, through ``scripts/run_controller.py``, or through the
launch file in ``launch/dual_mode_controller.launch``.
"""

import rospy

from .behaviors import BehaviorMixin
from .core import CoreMixin
from .execution import ExecutionMixin
from .loop import ReasoningLoopMixin
from .monitor import MonitorMixin
from .perception import PerceptionMixin


class DualModeStretchController(CoreMixin, MonitorMixin, PerceptionMixin,
                                BehaviorMixin, ExecutionMixin,
                                ReasoningLoopMixin):
    """LLM-driven dual-mode controller for the Hello Robot Stretch RE2."""


def main():
    try:
        controller = DualModeStretchController()
        controller.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
