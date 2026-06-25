"""Multi-turn observe--reason--act (ORA) loop.

The interactive entry point of the controller. Each user instruction
starts a reasoning episode capped at T_max = 6 turns: the loop injects
the active control mode into the prompt, queries the Tier-1 model,
executes any requested perception tools, aggregates their observations
into a structured re-prompt, and repeats until the model commits to a
final action sequence or a natural-language answer (Algorithm 1 of the
paper).
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


class ReasoningLoopMixin:
    def run(self):
        print("Stretch Dual-Mode LLM Controller + Voice + Multi-turn Reasoning")
        print("→ Type your command normally")
        print("→ Or just press ENTER to speak into the microphone")
        print("Say 'stop', 'quit', or 'exit' to stop the program.\n")

        while not rospy.is_shutdown():
            try:
                user_input = input("You > ").strip()

                if user_input == "":
                    text = listen()          # voice input
                    if not text:
                        continue
                else:
                    text = user_input

                if text.lower() in ['q', 'quit', 'exit', 'stop']:
                    print("Exiting.")
                    break
                if not text:
                    continue

                print(f"\nYou: {text}")

                cmd_lower = text.lower().strip()
                if any(k in cmd_lower for k in ["place on", "put on", "set on", "drop on"]):
                    place_target = cmd_lower.split("on")[-1].strip().replace(" ", "_")

                    cmd = f"place_object({place_target})"

                    print("⚡ Direct command:", cmd)
                    print("-" * 80)
                    print(f"  Executing: {cmd}")

                    result = self.execute_cmd(cmd)
                    print(f"  → {result}")

                    speak_text(result)

                    continue

                # === Multi-turn Reasoning Loop ===
                max_turns = 6
                turn = 0
                observation = ""

                # Get current mode
                current_mode = self.current_mode if self.current_mode != "unknown" else "unknown (check with get_state)"

                # Inject mode into the prompt
                augmented_text = f"Current robot mode: {current_mode}\n\nUser request: {text}"

                while turn < max_turns:
                    commands = ask_llm(augmented_text) if turn == 0 else ask_llm(re_prompt)

                    raw_final = " ".join(commands).strip()

                    # LLM gives final answer
                    if raw_final.lower().startswith("answer:"):
                        answer = raw_final.split(":", 1)[1].strip()
                        print("LLM answer:", answer)
                        speak_text(answer)
                        break

                    # Check if any tools are requested
                    tool_requested = False
                    for c in commands:
                        c_lower = c.strip().lower()
                        if any(kw in c_lower for kw in [
                            "get_state", "get_camera_view", "get_pointcloud_summary",
                            "get_object_distance", "get_slam_map",
                            "look_front", "look_left", "look_right", "look_behind",
                            "look_up", "look_down"
                        ]):
                            tool_requested = True
                            break

                    if not tool_requested:
                        # No tools → treat as final action commands
                        final_commands = [c.strip() for c in commands if c.strip()]
                        if final_commands:
                            print("Final commands:", ", ".join(final_commands))
                            print("-" * 80)
                            for cmd in final_commands:
                                if cmd.split("(")[0] not in ALLOWED_COMMANDS:
                                    print(f"Skipping unknown command: {cmd}")
                                    continue

                                print(f"  Executing: {cmd}")

                                # Capture state BEFORE execution
                                before_state = self.get_robot_state_summary()

                                result = self.execute_cmd(cmd)
                                print(f"  → {result}")

                                speak_text(result)

                                rospy.sleep(0.5)

                                # Capture state AFTER execution
                                after_state = self.get_robot_state_summary()

                                # Skip verification for search_for_object
                                if cmd.split("(")[0] not in ["search_for_object"]:
                                    verification = self.verify_execution_with_llm(cmd, before_state, after_state)
                                    print(f"  LLM execution verification: {verification}")
                                else:
                                    verification = "SKIPPED"
                                    print(f"  Skipping LLM execution verification for: {cmd}")

                                print(f"  LLM execution verification: {verification}")

                                if verification.startswith("FAILED"):
                                    print("  Retrying command once...")
                                    retry_result = self.execute_cmd(cmd)
                                    print(f"  Retry result: {retry_result}")
                                    
                                # # === NEW: Call analyze_three_camera_frames if the command is explicitly requested ===
                                # if cmd.split("(")[0] == "analyze_three_camera_frames":
                                #     object_desc = cmd.split("(")[1].rstrip(")") if "(" in cmd else "main object"
                                #     object_desc = object_desc.strip().strip("'\"")
                                #     print(f"DEBUG - Running analyze_three_camera_frames for '{object_desc}'")
                                #     distance = self.analyze_three_camera_frames(object_desc)
                                #     distance_text = f"{distance:.2f} m" if distance else "unknown"
                                #     # speak_text(f"{description}. Estimated distance: {distance_text}")
                                #     self.captured_images.clear()  # reset for next command
                            print("-" * 80)
                        break

                    # ====================== FIXED TOOL EXECUTION ======================
                    print(f"→ Turn {turn+1}: Fetching sensor data...")

                    state_summary = ""
                    camera_desc = ""
                    pointcloud_desc = ""
                    object_dist_desc = ""
                    slam_map_desc = ""
                    look_desc = ""   # ← NEW

                    # Existing get_* handling (unchanged)
                    if any("get_state" in c.lower() for c in commands):
                        state_summary = self.get_robot_state_summary()
                    if any("get_camera_view" in c.lower() for c in commands):
                        camera_desc = self.get_camera_vision_description()
                    if any("get_pointcloud_summary" in c.lower() for c in commands):
                        pointcloud_desc = self.get_pointcloud_summary()
                    if any("get_object_distance" in c.lower() for c in commands):
                        for c in commands:
                            if "get_object_distance" in c.lower():
                                # Extract object name
                                object_desc = c.split("(", 1)[1].rstrip(")") if "(" in c else "main object in view"
                                # ← FIXED: properly unpack the tuple return value
                                result = self.get_object_distance(object_desc)
                                if isinstance(result, tuple) and len(result) > 0:
                                    object_dist_desc = result[0]          # textual description for LLM
                                else:
                                    object_dist_desc = str(result)
                                break
                    if any("get_slam_map" in c.lower() for c in commands):
                        slam_map_desc = self.get_slam_map_description()

                    # NEW: Execute ALL look_* commands so we actually get the camera descriptions
                    for c in commands:
                        c_clean = c.strip()
                        if c_clean.lower().startswith(("look_front", "look_left", "look_right",
                                                       "look_behind", "look_up", "look_down")):
                            result = self.execute_cmd(c_clean)
                            look_desc += result + "\n\n"

                    # Build observation for next LLM turn
                    observation = ""
                    if state_summary:
                        observation += state_summary + "\n\n"
                    if camera_desc:
                        observation += "Camera observation:\n" + camera_desc + "\n\n"
                    if pointcloud_desc:
                        observation += "Point cloud summary:\n" + pointcloud_desc + "\n\n"
                    if object_dist_desc:
                        observation += object_dist_desc + "\n\n"
                    if slam_map_desc:
                        observation += slam_map_desc + "\n\n"
                    if look_desc:
                        observation += "Directional camera views:\n" + look_desc

                    # # Build observation for next LLM turn
                    # observation = f"Current robot mode: {self.current_mode}\n\n"
                    # if state_summary:
                    #     observation += state_summary + "\n\n"

                    re_prompt = f"""You are in FOLLOW-UP reasoning for the original user request: "{text}"

                                CURRENT OBSERVATION:
                                {observation}

                                You now have the latest sensor data.
                                Based on this, decide what to do next.
                                If you have enough information, start with 'ANSWER:'.
                                If you need more information, output only the next tool(s) you need.

                                Your response:"""

                    turn += 1

                # End of multi-turn loop

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
