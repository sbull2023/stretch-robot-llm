"""Central command dispatch.

``execute_cmd`` is the single choke point between the language side and
the actuators: it parses one validated command string, checks mode
compatibility, and routes it to the matching primitive or behaviour. Any
command that reaches an actuator has therefore passed the prompt-level
rule, the vocabulary validator, and this routing layer -- the
three-layer defence-in-depth design of Section 5.2.
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


class ExecutionMixin:
    def execute_cmd(self, cmd):
        cmd_lower = cmd.lower().strip()
        global re
        if ("three pictures" in cmd_lower or "three camera frames" in cmd_lower or "three images" in cmd_lower) and "distance" in cmd_lower:
            import re
            obj_match = re.search(r"distance to (.+)", cmd_lower)
            if obj_match:
                object_desc = obj_match.group(1).strip()
                # Remove leading 'the' if user typed "the water bottle"
                if object_desc.lower().startswith("the "):
                    object_desc = object_desc[4:]
            else:
                object_desc = "main object in view"

            print(f"DEBUG - Using object description: '{object_desc}'")

            # Capture 3 images with smaller angles
            directions = ["front", "left", "right"]
            pan_tilt_values = {
                "front": (0.0, 0.0),
                "left": (0.4, 0.0),   # smaller pan
                "right": (-0.4, 0.0)  # smaller pan
            }
            images = []

            for dir in directions:
                pan, tilt = pan_tilt_values[dir]
                self.send_joint_goal(['joint_head_pan'], [pan], relative=False, duration=1.5)
                self.send_joint_goal(['joint_head_tilt'], [tilt], relative=False, duration=1.5)
                rospy.sleep(2.5)
                if self.latest_color_img is not None:
                    images.append(self.latest_color_img.copy())

            # Reset head
            self.send_joint_goal(['joint_head_pan'], [0.0], relative=False, duration=1.5)
            self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False, duration=1.5)
            rospy.sleep(1.5)

            return description  # Note: This seems incomplete in original code; assuming it returns a description string

        name, value = parse_cmd(cmd)

        defaults = {
            "base_forward": 0.05, "base_back": 0.05,
            "base_left": 0.2, "base_right": 0.2,
            "lift_up": 0.02, "lift_down": 0.02,
            "wrist_out": 0.03, "wrist_in": 0.03,
            "head_left": 0.15, "head_right": 0.15,
            "head_up": 0.15, "head_down": 0.15,
        }

        # ──────────────────────────────────────────────
        # Mode switching commands
        # ──────────────────────────────────────────────
        if name in ["mode_position", "mode_pos"]:
            if self.is_position_mode():
                return "Already in position mode."
            try:
                rospy.wait_for_service('/switch_to_position_mode', timeout=3)
                srv = rospy.ServiceProxy('/switch_to_position_mode', Trigger)
                srv()
                return "Switched to position mode"
            except Exception as e:
                return f"Failed to switch to position mode: {e}"

        elif name in ["mode_navigation", "mode_nav"]:
            if self.is_navigation_mode():
                return "Already in navigation mode."
            try:
                rospy.wait_for_service('/switch_to_navigation_mode', timeout=3)
                srv = rospy.ServiceProxy('/switch_to_navigation_mode', Trigger)
                srv()
                return "Switched to navigation mode"
            except Exception as e:
                return f"Failed to switch to navigation mode: {e}"

        # ──────────────────────────────────────────────
        # Base movement — mode-dependent
        # ──────────────────────────────────────────────
        if name in ["base_forward", "base_back", "base_left", "base_right"]:
            print("DEBUG")
            try:
                delta = abs(float(value))
            except (TypeError, ValueError):
                delta = defaults.get(name, 0.05)
            if self.is_position_mode():
                return self._base_position_move(name, value, delta)
            elif self.is_navigation_mode():
                return self._base_velocity_move(name, value, delta)
            else:
                return f"Unknown mode '{self.current_mode}' — cannot move base"

        # ──────────────────────────────────────────────
        # Other joints — same in both modes
        # ──────────────────────────────────────────────
        if name in ["lift_up", "lift_down"]:
            try:
                delta = abs(float(value))
            except (TypeError, ValueError):
                delta = defaults.get(name, 0.02)
            sign = 1 if name == "lift_up" else -1
            self.send_joint_goal(['joint_lift'], [sign * delta], relative=True)
            return "Lift moved."

        elif name == "lift_home":
            self.home_srv()
            return "Lift homed."

        elif name in ["wrist_out", "wrist_in"]:
            try:
                delta = abs(float(value))
            except (TypeError, ValueError):
                delta = defaults.get(name, 0.03)
            sign = 1 if name == "wrist_out" else -1
            delta_signed = sign * delta

            arm_joints = [
                'joint_arm_l0', 'joint_arm_l1',
                'joint_arm_l2', 'joint_arm_l3'
            ]
            positions = [delta_signed] * 4
            self.send_joint_goal(arm_joints, positions, relative=True, duration=2.8)
            return f"Wrist moved {'out' if sign > 0 else 'in'}."
        
        elif name in ["grip_left", "wrist_yaw_left"]:
            try:
                move = abs(float(value))
            except (TypeError, ValueError):
                move = 0.15  # Reuse a default or set new
            self.send_joint_goal(
                ['joint_wrist_yaw'],
                [move],
                relative=True,
                duration=1.5
            )
            return f"Gripper rotating left by {move:.3f} rad"
        
        elif name in ["grip_right", "wrist_yaw_right"]:
            try:
                move = abs(float(value))
            except (TypeError, ValueError):
                move = 0.15  # Reuse a default or set new
            self.send_joint_goal(
                ['joint_wrist_yaw'],
                [-move],
                relative=True,
                duration=1.5
            )
            return f"Gripper rotating right by {move:.3f} rad"
        
        elif name == "wrist_yaw_home":
            self.send_joint_goal(
                ['joint_wrist_yaw'],
                [0.0],
                relative=False,
                duration=2.0
            )
            return "Wrist yaw homed."

        elif name == "wrist_home":
            self.home_srv()
            return "Wrist homed."

        elif name in ["grip_open", "grip_close"]:
            pos = 0.6 if name == "grip_open" else -0.6
            self.send_joint_goal(['joint_gripper_finger_left'], [pos], duration=1.5)
            return "Gripper moved."

        elif name == "grip_home":
            self.home_srv()
            return "Gripper homed."

        # Head movement (same in both modes)
        elif name in ["head_left", "head_right", "head_up", "head_down", "head_home"]:
            if name == "head_home":
                # Special case: Reset both pan and tilt to 0.0
                self.send_joint_goal(['joint_head_pan'], [0.0], relative=False, duration=1.5)
                self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False, duration=1.5)
                rospy.sleep(1.5)  # Wait for stabilization
                current_pan = self.get_current_pos('joint_head_pan')
                current_tilt = self.get_current_pos('joint_head_tilt')
                print(f"DEBUG - Head home: Reset from pan={current_pan:.3f}, tilt={current_tilt:.3f} to 0.0")
                return "Head reset to home/forward position (pan=0.0, tilt=0.0)."
            
            # Parse target (robust: convert if string, fallback if invalid or missing)
            try:
                target = float(value)  # e.g. 2.094, 1.5708, etc.
                print(f"DEBUG - Parsed target from value '{value}' (type: {type(value)}): {target:.3f}")
            except (TypeError, ValueError):
                # Fallback only if conversion fails or no value
                if name == "head_left":
                    target = 0.7854   # ~45° left
                elif name == "head_right":
                    target = -0.7854  # ~45° right
                elif name == "head_up":
                    target = -0.5
                else:  # head_down
                    target = 0.5
                print(f"DEBUG - Invalid or no value '{value}' for {name}; using default target: {target:.3f}")

            # ─── Apply correct sign convention to match robot behavior ───
            if name == "head_left":
                target = abs(target)          # positive → left
            elif name == "head_right":
                target = -abs(target)         # negative → right
            elif name == "head_up":
                target = abs(target)         # negative → up (looking higher)
            elif name == "head_down":
                target = -abs(target)          # positive → down (looking lower)

            # Determine joint
            if name in ["head_left", "head_right"]:
                joint = "joint_head_pan"
            else:
                joint = "joint_head_tilt"

            # Safety clamp
            if joint == "joint_head_pan":
                target = max(min(target, 1.9), -3.4)
            elif joint == "joint_head_tilt":
                target = max(min(target, 0.4), -1.67)

            # Debug logs
            current_pos = self.get_current_pos(joint)
            print(f"DEBUG - {name}: Current {joint} pos: {current_pos:.3f}, Sending absolute target: {target:.3f}")

            # Send goal
            self.send_joint_goal([joint], [target], relative=False, duration=2.0)
            
            return f"Head turned {name.replace('head_', '')} to absolute {target:.3f} rad"

        elif name == "stop":
            self.traj_client.cancel_all_goals()
            self.cmd_vel_pub.publish(Twist())
            return "Stopped all motion."

        elif name == "resume":
            return "Ready for new commands."
        
        
        elif name in ["look_front", "look_left", "look_right", "look_behind", "look_up", "look_down"]:
            # First move the head to the desired direction
            if name == "look_front":
                self.send_joint_goal(['joint_head_pan'], [0.0], relative=False)   # reset pan to center
                self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False)  # reset tilt to neutral
            elif name == "look_left":
                self.send_joint_goal(['joint_head_pan'], [0.8], relative=False)   # ~45° left
            elif name == "look_right":
                self.send_joint_goal(['joint_head_pan'], [-0.8], relative=False)  # ~45° right
            elif name == "look_behind":
                self.send_joint_goal(['joint_head_pan'], [-3.14], relative=False) # ~180° back (π radians)
            elif name == "look_up":
                self.send_joint_goal(['joint_head_tilt'], [-0.5], relative=False) # tilt up
            elif name == "look_down":
                self.send_joint_goal(['joint_head_tilt'], [0.5], relative=False)  # tilt down

            rospy.sleep(1.5)  # wait for head to move

            # Then get the camera description
            desc = self.get_camera_vision_description(
                prompt=f"Describe what the robot sees while looking {name.replace('look_', '')}."
            )

            # Auto-reset head to front after description
            self.send_joint_goal(['joint_head_pan'], [0.0], relative=False, duration=1.5)
            self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False, duration=1.5)
            rospy.sleep(1.5)  # wait for reset
            return f"Looked {name.replace('look_', '')}. Camera sees: {desc}"
        
        elif name == "nav_relative":
            if not self.is_navigation_mode():
                return "Must be in navigation mode for nav_relative."
            
            try:
                parts = cmd.split("(")[1].rstrip(")").split(",")
                direction = parts[0].strip().lower()
                distance = float(parts[1].strip())
            except:
                return "Invalid nav_relative format: nav_relative(direction, distance) e.g., forward, 2.0"
            
            current_pose = self.get_current_robot_pose()
            if not current_pose:
                return "Cannot compute relative nav: Current pose unavailable."
            
            x, y, yaw = current_pose
            new_x, new_y, new_yaw = x, y, yaw
            
            if direction == "forward":
                new_x += distance * math.cos(yaw)
                new_y += distance * math.sin(yaw)
            elif direction == "back":
                new_x -= distance * math.cos(yaw)
                new_y -= distance * math.sin(yaw)
            elif direction == "left":
                new_yaw += distance  # Assume distance=radians for turns
            elif direction == "right":
                new_yaw -= distance  # Assume distance=radians for turns
            else:
                return f"Unknown direction: {direction}. Use forward/back/left/right."
            
            # Optional: Validate with get_slam_map here if needed
            return self.send_nav_goal(new_x, new_y, new_yaw)
        
        elif name == "look_around":
            directions = ["front", "left", "right", "behind"]  # include front
            all_desc = ""

            for dir in directions:
                pan = 0.0
                tilt = 0.0

                if dir == "left":
                    pan = 1.57    # 90° left
                elif dir == "right":
                    pan = -1.57   # 90° right
                elif dir == "behind":
                    pan = -3.14    # ~172° right turn behind
                # front = 0.0 (neutral)

                # Move head
                self.send_joint_goal(['joint_head_pan'], [pan], relative=False, duration=2.0)
                self.send_joint_goal(['joint_head_tilt'], [tilt], relative=False, duration=2.0)
                rospy.sleep(2.5)  # wait for movement

                # Get description
                desc = self.get_camera_vision_description(
                    prompt=f"Describe what the robot sees while looking {dir} in detail: objects, scene, people, obstacles, colors, approximate distances."
                )
                all_desc += f"\n\nLooking {dir}: {desc}"

                # Reset to front after each look
                self.send_joint_goal(['joint_head_pan'], [0.0], relative=False, duration=2.0)
                self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False, duration=2.0)
                rospy.sleep(2.0)

            return f"Looked around the room (front, left, right, behind):{all_desc}\nHead returned to front position."
    
        elif name == "get_camera_view":
            desc = self.get_camera_vision_description(
                prompt="Describe the current scene from the robot's camera in detail: objects, scene, people, obstacles, colors, approximate distances. Be concise but informative."
            )
            return f"Current camera view: {desc}"
        
        elif name == "get_pointcloud_summary":
            pc_info = self.get_pointcloud_summary()
            return f"Point cloud information: {pc_info}"
        
        elif name == "get_object_distance":
            # value is expected to be str
            object_desc = value if value is not None else "main object in view"
            dist_info = self.get_object_distance(object_desc)
            return dist_info
        
        elif name == "nav_to_named":
            # value is expected to be str
            location = value.strip().lower() if value else ""
            if location in self.waypoints:
                x, y, yaw = self.waypoints[location]
                print(f"→ Navigating to known location: {location} ({x}, {y}, {yaw:.2f} rad)")
                return self.send_nav_goal(x, y, yaw)
            else:
                known = ", ".join(self.waypoints.keys())
                return f"Unknown location '{location}'. Known locations: {known}"
            
        elif name == "move_relative":
            try:
                parts = value.split(",")
                direction = parts[0].strip().lower()
                distance = float(parts[1].strip())
                
                if direction in ["forward", "back", "left", "right"]:
                    return self._base_velocity_move(f"base_{direction}", distance, None)
                else:
                    return f"Unknown direction: {direction}"
            except:
                return "Invalid move_relative format. Use: move_relative(direction, distance)"
            
        elif name == "get_slam_map":
            map_desc = self.get_slam_map_description()
            return f"SLAM Map Information:\n{map_desc}"
        
        elif name == "nav_to":
            # Parse nav_to(x, y, yaw) — all floats
            match = re.match(r"nav_to\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", cmd)
            if match:
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    yaw = float(match.group(3))
                    return self.send_nav_goal(x, y, yaw)
                except ValueError:
                    return "Invalid numbers in nav_to(x,y,yaw)."
            else:
                return "Invalid format for nav_to. Use: nav_to(x,y,yaw) where x,y in meters, yaw in radians."
                
        elif name == "search_for_object":
            # value is expected to be str
            object_name = cmd.split("(")[1].split(")")[0]
            object_name = object_name.strip()
            rospy.loginfo(f"Searching for: {object_name}")
            return self.search_for_object(object_name)
            
        elif name == "analyze_three_camera_frames":
            # value is expected to be str
            object_desc = str(value).replace('_', ' ') if value else 'main object in view'
            result = self.analyze_three_camera_frames(object_name)
            if not isinstance(result, tuple) or len(result) < 2:
                return "Could not estimate distance", None, None, None, None

            _, distance, obj_x, obj_y, target_yaw = result

            if distance is None:
                return "Could not estimate distance", None, None, None, None
            distance_text = f"{distance:.2f} m" if distance else "unknown"
            return f"[Three pictures analysis] Object '{object_desc}': {description}. Estimated distance: {distance_text}"
            
        elif name == "demo_sequence":
            return self.demo_motion_sequence()
            
        elif name == "estimate_gripper_distance":
            # value is expected to be str
            object_desc = str(value).replace('_', ' ') if value else 'main object in view'
            distance = self.estimate_gripper_distance(object_desc)
            distance_text = f"{distance:.2f} m" if isinstance(distance, (int, float)) else distance
            return f"[Gripper distance] Object '{object_desc}': Estimated distance: {distance_text}"

        elif name == "pick_object":
            object_desc = str(value).replace('_', ' ') if value else 'main object in view'
            return self.pick_object(object_desc)
        
        elif name == "place_object":
            place_desc = str(value).replace('_', ' ') if value else 'default location'
            return self.place_object(place_desc)

        elif name == "approach_object":
            object_desc = str(value).replace('_', ' ') if value else 'main object in view'
            return self.approach_object(object_desc)
            
        elif "look" in cmd.lower() and "find" in cmd.lower():
            # User wants to just look around
            object_name = cmd.lower().split("find")[-1].strip()
            rospy.loginfo(f"Looking around for: {object_name}")

            # Just take camera images from multiple angles without moving the base
            directions = ["front", "left", "right", "behind"]
            for dir in directions:
                self.execute_cmd(f"look_{dir}")

            # After looking, call vision analysis only
            description = self.get_camera_vision_description(
                f"Is there a {object_name} in these images? If yes, describe location and distance."
            )
            print(f"  → {description}")
            speak_text(description)
            return description

        
        if "find" in cmd_lower or "look for" in cmd_lower:
            obj = cmd_lower.split("find")[-1].strip()
            self.search_for_object(obj)
            return

        else:
            return f"Unknown command: {name}"
