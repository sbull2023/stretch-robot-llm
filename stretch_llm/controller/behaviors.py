"""Level-2 closed-loop visual behaviours.

These are the high-level autonomy routines the LLM can invoke with one
command: object search (rotate-and-detect sweep), visual approach
(thirds-of-image servoing), grasp and placement routines, lift-height
correction, and the fixed diagnostic demo. Each routine expands
internally into Level-1 primitives interleaved with VLM decisions, as
described in Section 6 of the paper.
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


class BehaviorMixin:
    def pick_object(self, object_name: str):
        """Advanced pick sequence:
        - Approach object using visual servoing
        - head_right + head_down (fixed camera)
        - Rotate gripper left 180°
        - Vision height estimation + correction loop
        - Grasp and lift
        """
        rospy.loginfo(f"🚀 Starting advanced pick_object('{object_name}')")

        # 1. Ensure position mode
        if not self.is_position_mode():
            self.execute_cmd("mode_position")
            rospy.sleep(1.0)

        # 2. Move head to neutral before approach
        self.send_joint_goal(['joint_head_pan'], [0.0], relative=False)
        self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False)
        rospy.sleep(1.5)

        # 3. APPROACH OBJECT USING YOUR EXISTING LOGIC
        rospy.loginfo(f"🚀 Calling approach_object('{object_name}')")
        approach_result = self.approach_object(object_name)

        if approach_result is None or "timeout" in str(approach_result).lower():
            rospy.logerr("❌ Approach failed or timed out")
            return f"FAILED: could not approach {object_name}"

        rospy.loginfo(f"✅ Approach complete: {approach_result}")
        rospy.loginfo("🎯 Final centering before rotation")
        for center_step in range(5):  # small correction loop
            if self.latest_color_img is None:
                continue

            corrected_img = cv2.rotate(self.latest_color_img, cv2.ROTATE_90_CLOCKWISE)
            b64_img = self._img_to_base64(corrected_img)

            center_prompt = f"""
            You are now ONLY allowed to center the {object_name}.

            IMPORTANT RULES:
            - DO NOT move forward at all.
            - ONLY correct LEFT or RIGHT position.
            - If object is centered → return: Action: STOP
            - Only change direction if the object is clearly in a different third than the previous step. If uncertain, repeat previous action.

            Divide image:
            - LEFT OUTSIDE 30% → base_left(0.05)
            - CENTER 40% BAND → STOP
            - RIGHT OUTSIDE 30% → base_right(0.05)

            Respond EXACTLY:
            Observation: ...
            Reasoning: ...
            Action: base_left(...) OR base_right(...) OR STOP
            """

            output = tinyllama_chat(
                messages=[
                    {
                        "role": "user",
                        "content": center_prompt,
                        "images": [b64_img]
                    }
                ],
                model=MONITOR_MODEL
            )

            output = output.strip()
            print(f"[CENTER PHASE]\n{output}")

            action = ""
            for line in output.splitlines():
                if line.lower().startswith("action:"):
                    action = line.split(":", 1)[1].strip()

            if "stop" in action.lower():
                rospy.loginfo("✅ Object centered successfully")
                break

            if "left" in action.lower():
                self.execute_cmd("base_left(0.10)")
            elif "right" in action.lower():
                self.execute_cmd("base_right(0.10)")

            rospy.sleep(1.0)
        rospy.loginfo("🔄 Rotating 90° for grasp alignment")
        self.execute_cmd("base_left(1.57)")   # +90 degrees (CCW)
        rospy.sleep(2.5)

        # 2. FIXED SIDE-VIEW CAMERA (exactly as you requested)
        rospy.loginfo("Positioning camera: head_right(1.57079632679) + head_down(1.0472)")
        self.execute_cmd("head_right(1.57079632679)")
        rospy.sleep(2.0)
        self.execute_cmd("head_down(1.0472)")
        rospy.sleep(2.5)
        rospy.loginfo("✅ Camera locked at right 90° + down 60° for the entire grasping process")

        # 3. Grasp sequence (camera remains fixed)

        # Capture side-view image (you already positioned camera correctly above)
        if self.latest_color_img is None:
            rospy.logwarn("No image available for height estimation")
            return "FAILED: no image"

        img = cv2.rotate(self.latest_color_img, cv2.ROTATE_90_CLOCKWISE)
        b64_img = self._img_to_base64(img)

        height_prompt = f"""
        You are controlling a robot lift for grasping.

        The image shows:
        - the robot gripper (visible)
        - the {object_name}
        - a side view (camera is 90° right and tilted down)

        GOAL:
        Align the gripper vertically with the grasp point of the object.

        The gripper is visible in the image.

        Determine how high the lift should be so the gripper aligns with the object.

        IMPORTANT:
        - Base your answer ONLY on what you see in the image
        - Do NOT guess distances
        - Match the vertical position of the gripper to the object
        - The correct grasp point is slightly below the top of the object.

        Output:
        Lift: X.XX
        """

        output = tinyllama_chat(
            messages=[
                {
                    "role": "user",
                    "content": height_prompt,
                    "images": [b64_img]
                }
            ],
            model=MONITOR_MODEL
        )

        output = output.strip()
        print(f"[HEIGHT ESTIMATION]\n{output}")

        # Parse lift value
        lift_target = 0.3  # fallback
        for line in output.splitlines():
            if line.lower().startswith("lift:"):
                try:
                    lift_target = float(line.split(":")[1].strip())
                except:
                    pass

        # Clamp for safety
        lift_target = lift_target + 0.101
        lift_target = max(0.05, min(lift_target, 0.95))

        rospy.loginfo(f"🎯 LLM estimated lift height: {lift_target:.3f} m")

        # Move lift directly to target
        self.send_joint_goal(['joint_lift'], [lift_target], relative=False, duration=2.5)
        rospy.sleep(2.0)

        # Vision + LLM correction loop (camera is still in perfect side view)
        self.check_and_correct_lift_height(object_name, max_attempts=2)

        # Rotate gripper left 180°
        self.send_joint_goal(['joint_wrist_yaw'], [3.14], relative=False, duration=2.0)
        rospy.sleep(1.5)

        rospy.loginfo("🎯 Final gripper-to-object alignment (pre-grasp)")
        for step in range(3):
            if self.latest_color_img is None:
                continue

            img = cv2.rotate(self.latest_color_img, cv2.ROTATE_90_CLOCKWISE)
            b64_img = self._img_to_base64(img)

            align_prompt = f"""
            You are controlling a robot arm with:

            - grip_left / grip_right = rotate gripper orientation
            - extend = move forward/back
            - STOP = ready to grasp

            The camera is a fixed side view.

            GOAL:
            1. Align gripper orientation with the object first (if needed)
            2. Then extend toward the object
            3. Stop when gripper is at the object

            CRITICAL RULES:

            STEP 1 — ORIENTATION FIX:
            - If gripper fingers are NOT aligned with object shape → use grip_left(0.05) or grip_right(0.05)

            STEP 2 — POSITION:
            - If gripper is far from object → extend(0.05)
            - If gripper is at object → STOP

            STEP 3 — DO NOT rotate continuously
            - Only adjust orientation if clearly misaligned

            ACTIONS (ONLY THESE):
            - grip_left(0.05)
            - grip_right(0.05)
            - extend(0.05)
            - STOP

            Respond EXACTLY:
            Action: grip_left(0.05) OR grip_right(0.05) OR extend(0.05) OR STOP
            """

            output = tinyllama_chat(
                messages=[
                    {
                        "role": "user",
                        "content": align_prompt
                    }
                ],
                model=MONITOR_MODEL
            )

            print(f"[FINAL ALIGN]\n{output}")

            action = ""
            for line in output.splitlines():
                if line.lower().startswith("action:"):
                    action = line.split(":", 1)[1].strip()

            if "stop" in action.lower():
                rospy.loginfo("✅ Final alignment complete")
                break

            elif "grip_left" in action.lower():
                self.execute_cmd("grip_left(0.05)")

            elif "grip_right" in action.lower():
                self.execute_cmd("grip_right(0.05)")

            elif "extend" in action.lower():
                self.execute_cmd("extend(0.05)")

            rospy.sleep(0.8)

        # Open gripper
        self.execute_cmd("grip_open")
        rospy.sleep(0.8)

        # Extend arm until gripper touches object
        for step in range(6):
            if self.latest_color_img is None:
                continue

            img = cv2.rotate(self.latest_color_img, cv2.ROTATE_90_CLOCKWISE)
            b64_img = self._img_to_base64(img)

            extend_prompt = f"""
            You are controlling a robot arm extending toward a {object_name}.

            The camera is a fixed side view.

            GOAL:
            Extend the arm until the gripper is JUST BEHIND the object (ready to grasp).

            RULES:
            - If the gripper is still FAR from the object → extend forward
            - If the gripper is ALIGNED with object → stop
            - If the gripper is already past object → stop immediately
            - Do NOT overextend

            ACTIONS:
            - extend(0.05)
            - STOP

            Respond EXACTLY:
            Observation: ...
            Reasoning: ...
            Action: extend(0.05) OR STOP
            """

            output = tinyllama_chat(
                messages=[
                    {
                        "role": "user",
                        "content": extend_prompt,
                        "images": [b64_img]
                    }
                ],
                model=MONITOR_MODEL
            )

            print(f"[EXTEND]\n{output}")

            action = ""
            for line in output.splitlines():
                if line.lower().startswith("action:"):
                    action = line.split(":", 1)[1].strip()

            if "stop" in action.lower():
                rospy.loginfo("✅ Arm properly positioned for grasp")
                break

            if "extend" in action.lower():
                self.send_joint_goal(
                    ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3'],
                    [0.05] * 4,
                    relative=True,
                    duration=1.0
                )

            rospy.sleep(1.0)

        # Close gripper
        self.execute_cmd("grip_close")
        rospy.sleep(1.2)

        # Lift object up
        self.send_joint_goal(['joint_lift'], [0.3], relative=True)
        rospy.sleep(1.5)

        # Optional: retract arm slightly
        self.send_joint_goal(
            ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3'],
            [-0.15] * 4,
            relative=True,
            duration=2.0
        )

        # 4. Reset ONLY at the very end
        self.send_joint_goal(['joint_wrist_yaw'], [0.0], relative=False, duration=1.5)
        self.execute_cmd("head_home")
        rospy.sleep(1.5)

        speak_text(f"I successfully picked up the {object_name}!")
        return f"✅ Picked up '{object_name}' with vision-corrected height"
    
    def place_object(self, place_name: str):

        place_name = place_name.replace("_", " ").strip()
        rospy.loginfo(f"📦 Starting place sequence at '{place_name}'")

        # 2. Approach it using your existing pick-style approach
        self.approach_object(place_name)

        # 🔽 Match pick-style final positioning BEFORE placing
        rospy.loginfo("📐 Aligning for placement (match pick behavior)")

        # Rotate base 90° (same as pick)
        self.execute_cmd("base_left(1.57)")
        rospy.sleep(2.5)

        # Lock camera to side + down (same as pick)
        self.send_joint_goal(['joint_head_pan'], [-1.57], relative=False)
        self.send_joint_goal(['joint_head_tilt'], [-0.72], relative=False)
        rospy.sleep(2.5)

        # 🔍 Confirm surface below before placing
        confirm = self.get_camera_vision_description(
            f"Is there a {place_name} directly below the gripper? Reply YES or NO."
        )
        print("PLACE CONFIRM:", confirm)
        if "yes" not in confirm.lower():
            rospy.logwarn("⚠️ Surface not confirmed — nudging forward slightly")
            self.execute_cmd("base_forward(0.05)")
            rospy.sleep(1.5)

        # ──────────────────────────────────────────────
        # 3. FINAL PLACEMENT ALIGNMENT 
        # ──────────────────────────────────────────────
        rospy.loginfo("📐 Aligning gripper over placement surface")

        max_extend_steps = 8

        for i in range(max_extend_steps):

            if self.latest_color_img is None:
                self.execute_cmd("wrist_out(0.05)")
                rospy.sleep(1.5)
                continue

            corrected_img = cv2.rotate(self.latest_color_img, cv2.ROTATE_90_CLOCKWISE)
            b64_img = self._img_to_base64(corrected_img)

            prompt = f"""
            You are controlling the robot arm to place an object onto a {place_name}.

            The camera is looking DOWN at the table.

            GOAL:
            Move the gripper so it is clearly OVER the {place_name} surface.

            RULES:
            - If the gripper is NOT over the {place_name} → extend forward
            - If the gripper IS over the {place_name} → say DONE
            - Be strict: only say DONE when it is clearly above the surface

            Respond EXACTLY with one:
            extend(0.05)
            DONE
            """

            try:
                output = tinyllama_chat(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [b64_img]
                        }
                    ],
                    model=MONITOR_MODEL
                )

                decision = output.strip().lower()
                print(f"[PLACE ALIGN] {decision}")

                if "done" in decision:
                    rospy.loginfo("✅ Gripper is over surface")
                    break

                self.execute_cmd("wrist_out(0.05)")
                rospy.sleep(1.5)

            except Exception as e:
                print(f"Placement LLM error: {e}")
                self.execute_cmd("wrist_out(0.05)")
                rospy.sleep(1.5)


        # ──────────────────────────────────────────────
        # 4. ROTATE WRIST FOR DROP 
        # ──────────────────────────────────────────────
        rospy.loginfo("🔄 Rotating wrist for placement")

        self.send_joint_goal(
            ['joint_wrist_yaw'],
            [3.14],   # 180° rotate (adjust if needed)
            relative=True,
            duration=1.5
        )

        rospy.sleep(1.0)


        # ──────────────────────────────────────────────
        # 5. DROP OBJECT
        # ──────────────────────────────────────────────
        rospy.loginfo("📦 Releasing object")

        self.send_joint_goal(['joint_gripper_finger_left'], [0.6], duration=1.5)
        rospy.sleep(1.0)


        # ──────────────────────────────────────────────
        # 6. RETRACT (reverse of pick)
        # ──────────────────────────────────────────────
        self.send_joint_goal(['joint_lift'], [0.05], relative=True)
        rospy.sleep(1.0)

        self.send_joint_goal(
            ['joint_arm_l0','joint_arm_l1','joint_arm_l2','joint_arm_l3'],
            [-0.05,-0.05,-0.05,-0.05],
            relative=True,
            duration=2.5
        )

        self.send_joint_goal(['joint_wrist_yaw'], [0.0], relative=False)

        return f"Placed object at {place_name}"
    
    # def approach_object(self, object_name: str):
    #     """Fully autonomous visual approach.
    #     The decision LLM now receives the ACTUAL camera image after EVERY step."""
    #     rospy.loginfo(f"🚀 Starting visual approach to '{object_name}'")

    #     # speak_text(f"Starting to approach the {object_name.replace('_', ' ')}.")

    #     # 1. Setup
    #     if not self.is_position_mode():
    #         self.execute_cmd("mode_position")
    #         rospy.sleep(1.0)
    #     self.send_joint_goal(['joint_head_pan'], [0.0], relative=False)
    #     self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False)
    #     rospy.sleep(1.5)

    #     # 2. Search if needed
    #     search_result = self.search_for_object(object_name, attempts=8)
    #     if "not found" in search_result.lower():
    #         speak_text(f"Could not find the {object_name.replace('_', ' ')}.")
    #         return search_result

    #     # 3. Approach loop with REAL IMAGE sent to decision LLM every step
    #     history = []
    #     max_steps = 5

    #     for step in range(max_steps):
    #         rospy.loginfo(f"Approach step {step+1}/{max_steps}")

    #         # Capture the current live image
    #         if self.latest_color_img is None:
    #             rospy.logwarn("No camera image available")
    #             decision = "base_forward(0.15)"
    #         else:
    #             b64_img = self._img_to_base64(self.latest_color_img)  # reuse helper from your class

    #             # History for context
    #             history_str = "\n".join(history[-6:]) if history else "First step."

    #             # Prompt for the vision LLM (this is now the "small" decision LLM)
    #             decision_prompt = f"""
    #             You are guiding the robot to approach the {object_name}.
    #             Goal: keep the object centered and stop when it looks large or fills most of the frame (~0.30 m away).

    #             HISTORY OF PREVIOUS ACTIONS:
    #             {history_str}


    #             Decide the next action based on what you SEE in the image.
    #             Decide the NEXT movement.
    #             - Units: meters for forward/back, radians for left/right
    #             - Choose ANY distance or angle you think is best
    #             Respond with EXACTLY one line (no explanations):
    #             base_forward(0.2)
    #             base_back(0.1)
    #             base_left(0.2)
    #             base_right(0.2)
    #             ANSWER: close enough
    #             """

    #             try:
    #                 resp = client.chat.completions.create(
    #                     model="gpt-4o",                    # ← vision model, sees the real image
    #                     messages=[{
    #                         "role": "user",
    #                         "content": [
    #                             {"type": "text", "text": decision_prompt},
    #                             {
    #                                 "type": "image_url",
    #                                 "image_url": {
    #                                     "url": f"data:image/jpeg;base64,{b64_img}"
    #                                 }
    #                             }
    #                         ]
    #                     }],
    #                     temperature=0.0,
    #                     max_tokens=80
    #                 )
    #                 decision = resp.choices[0].message.content.strip()
    #             except Exception as e:
    #                 print(f"Decision LLM error: {e}")
    #                 decision = "ANSWER: close enough"

    #         print(f"LLM decision: {decision}")

    #         if decision.lower().startswith("answer:"):
    #             speak_text(f"I have successfully approached the {object_name.replace('_', ' ')}.")
    #             return f"✅ Approached '{object_name}'"

    #         # Execute
    #         if any(x in decision for x in ["base_forward", "base_back", "base_left", "base_right"]):
    #             result = self.execute_cmd(decision)
    #             print(f"  → {result}")
    #             speak_text(result)
    #             rospy.sleep(0.8)
    #         else:
    #             self.execute_cmd("base_forward(0.15)")
    #             rospy.sleep(0.8)

    #         # Record step in history (text only)
    #         history.append(f"Step {step+1}: {decision}")

    #     # Safety timeout
    #     speak_text(f"Approached as close as possible to the {object_name.replace('_', ' ')}.")
    #     return f"Approached '{object_name}' (max steps reached)"

    
    def approach_object(self, object_name: str):
        surface_keywords = ["desk", "table", "counter", "surface", "bench"]
        is_surface = any(k in object_name.lower() for k in surface_keywords)

        """Fully autonomous visual approach – FIXED IMAGE ROTATION + strict centering."""
        rospy.loginfo(f"🚀 Starting visual approach to '{object_name}'")

        # 1. Setup
        if not self.is_position_mode():
            self.execute_cmd("mode_position")
            rospy.sleep(1.0)
        self.send_joint_goal(['joint_head_pan'], [0.0], relative=False)
        self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False)
        rospy.sleep(1.5)

        # 2. Search
        search_result = self.search_for_object(object_name, attempts=8)
        if "not found" in search_result.lower():
            speak_text(f"Could not find the {object_name.replace('_', ' ')}.")
            return search_result

        # Force initial centering
        # rospy.loginfo("Forcing initial centering before loop...")
        # self.execute_cmd("base_left(0.30)")
        # rospy.sleep(2.2)
        # self.execute_cmd("base_forward(0.12)")
        # rospy.sleep(2.2)

        # 3. Approach loop
        history = []
        max_steps = 8
        prev_visible = True

        for step in range(max_steps):
            current_visible = True
            rospy.loginfo(f"\n=== Approach step {step+1}/{max_steps} ===")

            if self.latest_color_img is None:
                decision = "base_forward(0.20)"
                current_visible = False
            else:
                # ====================== FIX IMAGE ROTATION ======================
                # Rotate 90° clockwise to correct the 90° left (counter-clockwise) rotation
                corrected_img = cv2.rotate(self.latest_color_img, cv2.ROTATE_90_CLOCKWISE)

                # Save corrected debug image (so you can verify it's now upright)
                debug_path = f"/tmp/approach_step_{step:02d}_CORRECTED.jpg"
                cv2.imwrite(debug_path, corrected_img)

                b64_img = self._img_to_base64(corrected_img)
                history_str = "\n".join(history[-8:]) if history else "First step."

                if is_surface:
                    decision_prompt = f"""
                    You are controlling a robot approaching a SURFACE: {object_name}.

                    IMPORTANT DIFFERENCE FROM OBJECTS:
                    - Surfaces are LARGE and stay in view even when you are close
                    - You MUST STOP EARLY to avoid collision

                    RULES:
                    - If surface is far → base_forward(0.30)
                    - If surface is medium distance → base_forward(0.30)
                    - If edge of surface is in view → base_forward(0.10)
                    - ONLY say ANSWER: close enough when the gripper is visually ABOVE the surface OR the surface is directly under the gripper AND a small forward extension (≤0.05–0.10 m) would be enough to touch it
                    - If unsure, prefer base_forward(0.10)
                    - If not visible → lost

                    NEVER keep moving forward once the surface is large in frame.

                    Respond EXACTLY:

                    Observation: ...
                    Reasoning: ...
                    Action: base_forward(0.20) or base_forward(0.10) or ANSWER: close enough or lost

                    Previous steps:
                    {history_str}
                    """
                else:
                    decision_prompt = f"""
                    You are a visual servoing controller. Goal: center the {object_name} perfectly and drive forward until it fills ~30% of the frame (≈0.30 m away).

                    Divide the image into three EXACT vertical thirds:
                    - LEFT THIRD: 0–33% of image width
                    - CENTER THIRD: 33–66% of image width (wide safe zone)
                    - RIGHT THIRD: 66–100% of image width

                    STRICT RULES (follow exactly):
                    - If the object is ANYWHERE in the CENTER THIRD (even slightly off) → MUST go forward. NEVER turn.
                    - If the object is clearly in the LEFT THIRD → base_left(0.15)
                    - If the object is clearly in the RIGHT THIRD → base_right(0.15)
                    - If the object is NOT VISIBLE at all → output "lost"
                    - If the object is partially visible or unclear → treat it as centered and go forward

                    Look at the image very carefully. Do not guess "right" if it has moved left.

                    Respond in EXACTLY this 3-line format (nothing else):

                    Observation: <one short sentence describing exact position>
                    Reasoning: <why you chose this action>
                    Action: base_forward(0.20) or base_left(0.15) or base_right(0.15) or ANSWER: close enough or lost

                    Previous steps for context:
                    {history_str}
                    """

                try:
                    full_response = tinyllama_chat(
                        messages=[
                            {
                                "role": "user",
                                "content": decision_prompt,
                                "images": [b64_img]
                            }
                        ],
                        model=MONITOR_MODEL
                    )

                    print(f"\n🤖 LLM full reasoning:\n{full_response}\n")

                    observation_line = ""
                    action_line = ""
                    for line in full_response.splitlines():
                        line_clean = line.strip()
                        if line_clean.lower().startswith("observation:"):
                            observation_line = line_clean.lower()
                        elif line_clean.lower().startswith("action:"):
                            action_line = line_clean.split(":", 1)[1].strip()
                            
                    decision = action_line if action_line else "base_forward(0.20)"
                    decision = decision.strip()

                    print(f"RAW DECISION: '{decision}'")
                    # Determine if object is visible in CURRENT frame
                    current_visible = True
                    if any(x in observation_line for x in ["not visible", "can't see", "not in view", "no longer visible"]):
                        current_visible = False  
                        
                    # ===== TRANSITION: visible → not visible =====
                    if (not current_visible and step > 1) or decision.lower() == "lost":
                        rospy.loginfo("⚠️ Object disappeared → likely reached it")

                        # STOP immediately
                        self.execute_cmd("base_forward(0.0)")
                        rospy.sleep(0.5)

                        # Look down 30 degrees
                        self.send_joint_goal(['joint_head_tilt'], [-0.52], relative=False)
                        rospy.sleep(1.5)

                        confirm_prompt = f"""
                        You are verifying if the robot has reached a {object_name}.

                        If you see the surface (table, desk, or object_name) directly below or very close to the robot, respond:
                        YES

                        If not, respond:
                        NO
                        """
                        found = False
                        for _ in range(3):  # try multiple frames
                            rospy.sleep(0.5)

                            if self.latest_color_img is None:
                                continue

                            corrected_img = cv2.rotate(self.latest_color_img, cv2.ROTATE_90_CLOCKWISE)
                            b64_img = self._img_to_base64(corrected_img)

                            try:
                                answer = tinyllama_chat(
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": confirm_prompt,
                                            "images": [b64_img]
                                        }
                                    ],
                                    model=MONITOR_MODEL
                                ).strip().lower()

                                print(f"CONFIRM ANSWER: '{answer}'")  # DEBUG
                                if any(x in answer for x in ["yes", "visible", "see", "present"]):
                                    found = True
                                    break

                            except Exception as e:
                                print(f"Confirm LLM error: {e}")

                        if found:
                            rospy.loginfo("✅ Object confirmed below — reached target")
                            speak_text(f"I have reached the {object_name.replace('_', ' ')}.")
                            return f"✅ Reached '{object_name}'"

                        return f"⚠️ Lost '{object_name}' after approach"
                    
                except Exception as e:
                    print(f"Decision LLM error: {e}")
                    decision = "base_forward(0.20)"
                    current_visible = False

            if decision.lower().startswith("answer:"):
                speak_text(f"I have successfully approached the {object_name.replace('_', ' ')}.")
                return f"✅ Approached '{object_name}'"

            decision = decision.strip()

            if decision.startswith(("base_forward", "base_back", "base_left", "base_right")):
                result = self.execute_cmd(decision)
                print(f"  → Executing: {decision} | {result}")
            else:
                self.execute_cmd("base_forward(0.20)")
                print(f"  → Executing: base_forward(0.20) | fallback")

            rospy.sleep(2.2)

            prev_visible = current_visible
            history.append(f"Step {step+1}: {decision}")

        speak_text(f"Approached as close as possible to the {object_name.replace('_', ' ')}.")
        return f"Approached '{object_name}' (max steps reached)"

    def estimate_table_height(self):
        """
        Robust table height estimation using depth geometry (NOT LLM).
        """

        rospy.loginfo("📏 Estimating table height using depth geometry...")

        if self.latest_depth_img is None or self.latest_color_img is None:
            rospy.logwarn("No depth data → fallback 0.75m")
            return 0.75

        # Rotate to match your corrected frame
        corrected = cv2.rotate(self.latest_depth_img, cv2.ROTATE_90_CLOCKWISE)
        h, w = corrected.shape

        cx = w // 2

        # Sample lower half of image (table region candidates)
        sample_ys = list(range(int(h * 0.60), int(h * 0.85)))

        depths = []

        for y in sample_ys:
            try:
                d = corrected[y, cx]

                # mm → m conversion
                if d > 10:
                    d = d / 1000.0

                if 0.2 < d < 5.0:
                    depths.append(d)

            except:
                continue

        if len(depths) < 10:
            rospy.logwarn("Insufficient depth samples → fallback 0.75m")
            return 0.75

        # use LOWER percentile (table is closer than noise above it)
        depths.sort()
        table_depth = depths[int(len(depths) * 0.2)]  # robust low percentile

        rospy.loginfo(f"📏 Table depth estimate: {table_depth:.3f} m")

        return table_depth
    

    def check_and_correct_lift_height(self, object_name: str, max_attempts=2):
        """After initial lift move, use side-view camera (already positioned) to verify/correct table height"""
        rospy.loginfo("🔧 Running vision-based lift height correction...")

        for attempt in range(max_attempts):
            prompt = (
                f"You are looking at a side view of the robot arm and gripper approaching a table with a {object_name}. "
                "The camera is fixed at right 90° and tilted down. "
                "Is the gripper at the correct height to touch the table surface? "
                "If YES, reply ONLY with 'CORRECT'. "
                "If NO, reply with 'CORRECT_TO: X.XX' where X.XX is the exact lift joint position in meters (e.g. CORRECT_TO: 0.68)."
            )

            response = self.get_camera_vision_description(prompt=prompt)

            print(f"Height check attempt {attempt+1}: {response}")

            if "CORRECT" in response.upper():
                rospy.loginfo("✅ Lift height verified correct by vision.")
                return True

            # Try to extract correction
            match = re.search(r"CORRECT_TO:\s*(\d+\.?\d*)", response, re.IGNORECASE)
            if match:
                new_lift = float(match.group(1))
                new_lift = max(0.05, min(new_lift, 0.95))  # safety clamp
                rospy.loginfo(f"🔧 LLM correction: moving lift to {new_lift:.3f} m")
                self.send_joint_goal(['joint_lift'], [new_lift], relative=False, duration=1.8)
                rospy.sleep(1.5)
            else:
                rospy.logwarn("No clear correction from LLM — using last position.")

        rospy.loginfo("✅ Lift height correction finished.")
        return True

    def search_for_object(self, object_name, attempts=12):
        
        # Always ensure position mode
        if not self.is_position_mode():
            self.execute_cmd("mode_position")
            rospy.sleep(1.5)

        # Keep head fixed forward
        self.send_joint_goal(['joint_head_pan'], [0.0], relative=False)
        self.send_joint_goal(['joint_head_tilt'], [0.0], relative=False)
        rospy.sleep(1.5)
        
        for i in range(attempts):
            rospy.loginfo(f"Search attempt {i+1}/{attempts}")

            # Use position mode rotation
            self.send_joint_goal(['rotate_mobile_base'], [0.5])
            print("Executed position mode rotation.")

            rospy.sleep(1.5)

            # Capture image and analyze with improved prompt for yes/no
            description = self.get_camera_vision_description(
                f"Respond with 'YES' if there is a {object_name.replace('_', ' ')} in this image, otherwise 'NO'. "
                "If YES, add a brief description after 'YES:'. Do not add extra text."
            )

            print("Vision:", description)
            rospy.loginfo(f"Vision: {description}")

            desc_lower = description.lower()
            
            # Improved detection: Check for 'yes' at start
            if desc_lower.startswith('yes'):
                rospy.loginfo(f"{object_name} found!")
                # Extract description if present
                found_desc = description.split(':', 1)[1].strip() if ':' in description else ''
                # Optionally switch back if we switched modes
                return f"I found the {object_name.replace('_', ' ')}. {found_desc}"

        return "Object not found after all attempts."

    def demo_motion_sequence(self):
        rospy.loginfo("Starting demo motion sequence...")

        # Ensure position mode
        if not self.is_position_mode():
            self.execute_cmd("mode_position")
            rospy.sleep(1.5)

        # 1. Camera turn right 90 degrees
        self.send_joint_goal(
            ['joint_head_pan'],
            [-1.57],   # -90 deg
            relative=False,
            duration=2.0
        )

        rospy.sleep(1.0)

        self.send_joint_goal(
            ['joint_head_tilt'],
            [-0.785],   # -45 deg
            relative=True,
            duration=1.5
        )

        rospy.sleep(0.8)
        
        # 4. Rotate gripper left 180 degrees
        self.send_joint_goal(
            ['joint_wrist_yaw'],
            [3.14],   # 180 deg
            relative=True,
            duration=2.0
        )
        
        rospy.sleep(0.8)

        # 2. Lift arm up 15 cm
        self.send_joint_goal(
            ['joint_lift'],
            [0.15],    # 15 cm
            relative=True,
            duration=2.0
        )

        self.execute_cmd("wrist_in(1.0)")

        rospy.sleep(1.0)

        rospy.loginfo("Demo motion complete!")
        return "Completed demo motion sequence."
