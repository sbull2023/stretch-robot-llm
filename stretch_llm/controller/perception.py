"""Perception tools: camera, point cloud, SLAM map, distance estimates.

Each public method here backs one perception tool in the closed command
vocabulary. Every tool returns a structured textual observation that the
multi-turn loop injects into the next prompt, which anchors the model's
next reasoning step in the current physical state of the world.
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


class PerceptionMixin:
    def get_pointcloud_summary(self):
        try:
            pc_msg = rospy.wait_for_message('/camera/depth/color/points', PointCloud2, timeout=10.0)
            print("DEBUG - Point cloud received")

            # Read x, y, z points (skip NaN/invalid)
            points = list(point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=[]))

            if not points:
                return "No valid points in point cloud (scene may be empty or too far/close)."

            import numpy as np
            xyz = np.array(points)  # shape (N, 3)

            # Distance from camera (origin)
            distances = np.linalg.norm(xyz, axis=1)

            # Filter realistic indoor range
            valid_mask = (distances > 0.1) & (distances < 10.0)
            valid_dist = distances[valid_mask]

            if valid_dist.size == 0:
                return "No valid points in 0.1–10 m range. Scene may be empty, too far, or too close."

            min_d = valid_dist.min()
            mean_d = valid_dist.mean()
            max_d = valid_dist.max()

            # Center cone (simple: small y, positive z)
            center_mask = (np.abs(xyz[:,1]) < 0.5) & (xyz[:,2] > 0.1)
            center_dist = distances[center_mask & valid_mask]
            center_d = center_dist.mean() if center_dist.size > 0 else "N/A"

            desc = "Point cloud summary (meters, filtered 0.1–10 m):\n"
            desc += f"- Valid points: {valid_dist.size:,}\n"
            desc += f"- Closest point: {min_d:.2f}\n"
            desc += f"- Average distance: {mean_d:.2f}\n"
            desc += f"- Farthest in range: {max_d:.2f}\n"
            desc += f"- Straight ahead (center cone): {center_d if isinstance(center_d, str) else f'{center_d:.2f}'}\n"

            return desc

        except Exception as e:
            return f"Point cloud error: {str(e)} (topic may not be publishing or parsing failed)"
        
    
    def get_current_robot_pose(self):
        """
        Returns a tuple (x, y, yaw) in the map/odom frame.
        Priority:
            1. /amcl_pose (if available)
            2. /odom
        """
        try:
            # Try AMCL first
            pose_msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout=1.0)
            p = pose_msg.pose.pose.position
            q = pose_msg.pose.pose.orientation
            yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            return (p.x, p.y, yaw)
        except (rospy.ROSException, AttributeError):
            # Fallback to odometry
            try:
                odom = rospy.wait_for_message('/odom', Odometry, timeout=0.5)
                p = odom.pose.pose.position
                q = odom.pose.pose.orientation
                yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
                return (p.x, p.y, yaw)
            except Exception as e:
                rospy.logwarn(f"Cannot get robot pose: {e}")
                return (0.0, 0.0, 0.0)
        
    
    
    def get_object_distance(self, object_desc: str):
        print(f"DEBUG - Using three-frame distance estimation for: '{object_desc}'")

        # description, distance = self.analyze_three_camera_frames(object_desc)
        
        result = self.analyze_three_camera_frames(object_desc)
        if isinstance(result, tuple) and len(result) >= 5:
            _, distance, obj_x, obj_y, target_yaw = result[:5]
            print(obj_x, obj_y)
        else:
            distance = obj_x = obj_y = target_yaw = None

        if distance is None:
            return f"Could not estimate distance to '{object_desc}'."

        robot_x, robot_y, robot_yaw = self.get_current_robot_pose()

        obj_x = robot_x + distance * math.cos(robot_yaw)
        obj_y = robot_y + distance * math.sin(robot_yaw)
        print("DEBUG")

        return (
            f"Object '{object_desc}' is approximately {distance:.2f} meters away.\n"
            f"Estimated map coordinates: x={obj_x:.2f}, y={obj_y:.2f}"
        ), distance, obj_x, obj_y, robot_yaw
    

    def _img_to_base64(self, img):
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_camera_vision_description(self, prompt="Describe what the robot's camera is seeing in detail."):
        if self.latest_color_img is None:
            return "No recent camera image available."

        # Encode image to base64
        _, buffer = cv2.imencode(".jpg", self.latest_color_img)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        try:
            vision_response = client.chat.completions.create(
                model="gpt-4o",  # or "gpt-4o-mini" if you want cheaper/faster
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.2
            )

            description = vision_response.choices[0].message.content.strip()
            return description

        except Exception as e:
            print("Vision API error:", e)
            return "Failed to get camera description."

    def analyze_three_camera_frames(self, object_name='main object in view'):
        print(f"DEBUG - Running analyze_three_camera_frames for '{object_name}'")
        
        if self.latest_depth_img is None or self.latest_color_img is None:
            return "No depth or RGB image available.", None

        descriptions = []
        distances = []

        # Pan head for three views: left, center, right (adjust angles as needed)
        pan_angles = [-0.2, 0.0, 0.2]  # radians (left, center, right)
        images_base64 = []  # List to hold base64-encoded images

        for i, angle in enumerate(pan_angles):
            # Move head
            self.send_joint_goal(['joint_head_pan'], [angle], relative=False)
            rospy.sleep(2.5)  # Wait for movement and image stabilization

            # Capture the current color image
            if self.latest_color_img is not None:
                _, buffer = cv2.imencode(".jpg", self.latest_color_img)
                b64 = base64.b64encode(buffer).decode('utf-8')
                images_base64.append(b64)

        # Reset head to center
        self.send_joint_goal(['joint_head_pan'], [0.0], relative=False)
        rospy.sleep(1.5)

        if len(images_base64) < 3:
            return "Failed to capture three images.", None

        # Prepare image messages for the API
        image_messages = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in images_base64
        ]

        # Prompt for LLM: Estimate distance using the three images
        prompt_text = (
            f"Estimate the distance to the '{object_name}' from these three images (left, center, right views). "
            "Respond ONLY with a numeric distance in meters (e.g., '1.5'). Do not include any other text, descriptions, or units."
        )

        try:
            vision_response = tinyllama_chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                        "images": images_base64
                    }
                ],
                model=MONITOR_MODEL
            )

            text_response = vision_response.strip()

            # Extract numeric distance
            distance_match = re.search(r"\d+\.?\d*", text_response)
            distance = float(distance_match.group(0)) if distance_match else None
            
            if distance is not None:
                robot_x, robot_y, robot_yaw = self.get_current_robot_pose()

                print(robot_x)
                print(robot_y)
                print(robot_yaw)

                # Assume object is straight ahead of the robot (center view)
                bearing = 0.0  

                obj_x = robot_x + distance * math.cos(robot_yaw + bearing)
                obj_y = robot_y + distance * math.sin(robot_yaw + bearing)
                target_yaw = robot_yaw

                print(f"DEBUG - Estimated object position: x={obj_x:.2f}, y={obj_y:.2f}")
            else:
                obj_x, obj_y = None, None

            print(f"DEBUG - Estimated distance to '{object_name}': {distance} m" if distance else "Distance not estimated")

            # Optional: Generate a combined description if needed, but since prompt is for distance only, skip
            combined_desc = f"Analyzed three views for '{object_name}'."

            print(f"DEBUG - Estimated distance to '{object_name}': {distance} m" if distance else "Distance not estimated")

            

            return combined_desc, distance, obj_x, obj_y, target_yaw

        except Exception as e:
            print(f"Vision API error: {e}")
            return "Failed to analyze images.", None

        
            
    def estimate_distance_from_images(self, images, object_name):
        """
        Simple placeholder to estimate distance from images.
        Replace with real vision or depth calculation later.
        """
        if not images:
            return None

        # Example: pretend we measured distance
        print(f"Estimating distance to '{object_name}' from {len(images)} images...")
        estimated_distance = 0.75  # meters, placeholder value
        return estimated_distance

    def get_slam_map_description(self):
        """Get textual description of SLAM map based on waypoints and robot position"""
        print("DEBUG - get_slam_map_description started (text-only version)")

        try:
            # Get waypoints text
            waypoint_text = "WAYPOINTS (x, y, yaw in radians):\n"
            if hasattr(self, 'waypoints') and self.waypoints:
                for name, pos in self.waypoints.items():
                    x, y, yaw = pos
                    waypoint_text += f"• {name}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}\n"
            else:
                waypoint_text += "No waypoints loaded.\n"

            # Get robot pose text
            robot_pose = self.get_current_robot_pose()
            robot_text = "Robot current position: Unknown"
            if robot_pose:
                rx, ry, ryaw = robot_pose
                robot_text = f"Robot current position: x={rx:.2f}, y={ry:.2f}, yaw={ryaw:.2f} rad"

            # Send to GPT-4o for description based on coordinates only
            vision_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"{robot_text}\n\n"
                                f"Waypoint coordinates:\n{waypoint_text}\n"
                                "These are absolute coordinates based on the /map topic.\n"
                    }]
                }],
                max_tokens=700
            )

            return vision_response.choices[0].message.content.strip()

        except Exception as e:
            print(f"ERROR in get_slam_map_description: {str(e)}")
            return f"Failed to get SLAM map: {str(e)}"

    def estimate_gripper_distance(self, object_name='main object in view'):
        print(f"DEBUG - Estimating distance from gripper to '{object_name}'")

        if self.latest_color_img is None:
            return "No RGB image available.", None

        # Capture the current color image (after demo)
        _, buffer = cv2.imencode(".jpg", self.latest_color_img)
        b64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare image messages for the API (same style as analyze_three_camera_frames)
        image_messages = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]

        # Prompt for LLM: Estimate distance using the image
        prompt_text = (
            f"Estimate the distance from the robot's gripper to the '{object_name}' in this image. "
            "Respond ONLY with a numeric distance in meters (e.g., '0.45'). Do not include any other text, units, or explanations."
        )

        try:
            vision_response = tinyllama_chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                        "images": base64
                    }
                ],
                model=MONITOR_MODEL
            )

            text_response = vision_response.strip()

            # Extract numeric distance
            distance_match = re.search(r"\d+\.?\d*", text_response)
            distance = float(distance_match.group(0)) if distance_match else None

            print(f"DEBUG - Estimated distance to '{object_name}': {distance} m" if distance else "Distance not estimated")

            return f"Estimated distance to '{object_name}' using gripper view.", distance

        except Exception as e:
            print(f"Vision API error: {e}")
            return "Failed to estimate gripper distance.", None
