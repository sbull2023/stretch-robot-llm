"""Tier-2 execution verification.

After every primitive command the controller captures the robot state
before and after execution and asks the edge monitor model for a verdict:
Verified Success, Verified Partial, or Verified Failed. A failed
verification triggers one automatic retry in the execution layer. The
final verdict of a trajectory is the label the fine-tuning recipe
conditions on (Section 4.6 of the paper).
"""

import base64
import math
import os
import re

import numpy as np
import requests
import yaml

import rospy
import actionlib
import tf
import cv2
from cv_bridge import CvBridge
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_srvs.srv import Trigger, TriggerRequest
from sensor_msgs.msg import JointState, Image, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import String
from nav_msgs.msg import Odometry, OccupancyGrid

from stretch_llm.llm import (ALLOWED_COMMANDS, MONITOR_MODEL, OLLAMA_BASE_URL,
                             PRIMARY_MODEL, SYSTEM_PROMPT, ask_llm,
                             ensure_command_completion, parse_cmd,
                             parse_command, tinyllama_chat)
from stretch_llm.speech.stt import listen
from stretch_llm.speech.tts import speak_text


class MonitorMixin:
    def verify_execution_with_llm(self, command, before_state, after_state):
        try:

            verification_prompt = f"""
    You are a robot execution verifier.

    The robot attempted to execute this command:

    {command}

    ROBOT STATE BEFORE:
    {before_state}

    ROBOT STATE AFTER:
    {after_state}

    Determine if the command completed successfully.

    Respond ONLY with:

    Verified SUCCESS: <short reason>
    Verified PARTIAL: <short reason>
    Verified FAILED: <short reason>
    """
    
            response = tinyllama_chat(
                messages=[
                    {"role": "user", "content": verification_prompt}
                ],
                model=MONITOR_MODEL
            )

            return response.strip()

        except Exception as e:
            return f"VERIFIER_ERROR: {e}"
