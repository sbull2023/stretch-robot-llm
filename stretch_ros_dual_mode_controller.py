#!/usr/bin/env python3
import rospy
import re
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger, TriggerRequest
from sensor_msgs.msg import JointState, Image, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import String
from llm_v13 import *
from nav_msgs.msg import Odometry, OccupancyGrid
import cv2
from cv_bridge import CvBridge
from mic_llm_v2 import listen
import base64
import os
import requests
import yaml
import numpy as np
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped   # optional but cleaner
import math

def speak_text(text):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No API key found!")
        return

    url = "https://api.openai.com/v1/audio/speech"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": text
    }

    try:
        response = requests.post(url, json=data, headers=headers)

        if response.status_code != 200:
            print("TTS server response:", response.text)
            return

        # temp audio file (auto deleted)
        filename = "speech_temp.mp3"

        with open(filename, "wb") as f:
            f.write(response.content)

        from playsound import playsound
        playsound(filename)

        # ✅ remove file immediately after speaking
        os.remove(filename)

    except Exception as e:
        print(f"TTS error: {e}")


def parse_cmd(cmd):
    match = re.match(r"([a-z_]+)(?:\(([-\d.]+)\))?", cmd)
    if not match:
        return None, None
    name = match.group(1)
    val = match.group(2)
    return name, float(val) if val else None


class DualModeStretchController:
    def __init__(self):
        rospy.init_node('dual_mode_stretch_controller', anonymous=True)

        # Action client for position-based joint control
        self.traj_client = actionlib.SimpleActionClient(
            '/stretch_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        rospy.loginfo("Waiting for trajectory action server...")
        self.traj_client.wait_for_server()
        rospy.loginfo("Trajectory action server connected.")

        # Velocity publisher for navigation mode base movement
        self.cmd_vel_pub = rospy.Publisher('/stretch/cmd_vel', Twist, queue_size=10)

        # Homing service
        self.home_srv = rospy.ServiceProxy('/home_the_robot', Trigger)

        # Mode tracking
        self.current_mode = "unknown"
        self.mode_sub = rospy.Subscriber('/mode', String, self._mode_callback)
        rospy.wait_for_message('/mode', String, timeout=8.0)

        # Joint states
        self.joint_states = None
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.wait_for_message('/joint_states', JointState, timeout=8.0)

        # Navigation action client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base action server.")

        self.bridge = CvBridge()
        self.latest_color_img = None

        self.waypoints = {}
        self.load_waypoints()

        # Subscribe to camera
        self.color_sub = rospy.Subscriber(
            '/camera/color/image_raw', 
            Image, 
            self.color_callback, 
            queue_size=1, 
            buff_size=2**24
        )

        # Depth image
        self.latest_depth_img = None
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_rect_raw',  # aligned depth is best for RGB correspondence
            Image,
            self.depth_callback,
            queue_size=1,
            buff_size=2**24
        )

    def _mode_callback(self, msg):
        new_mode = msg.data.strip().lower()
        if new_mode != self.current_mode:
            self.current_mode = new_mode
            rospy.loginfo(f"Current robot mode changed to: {self.current_mode}")
        else:
            self.current_mode = new_mode  # still update, just don't print

    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")  # float32 depth in meters
        except Exception as e:
            print("Depth callback error:", e)

    def color_callback(self, msg):
        try:
            self.latest_color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Optional: save latest for debugging
            # cv2.imwrite("/tmp/latest_robot_view.jpg", self.latest_color_img)
        except Exception as e:
            print("Camera callback error:", e)

    def is_position_mode(self):
        return "position" in self.current_mode

    def is_navigation_mode(self):
        return "navigation" in self.current_mode

    def joint_states_callback(self, data):
        self.joint_states = data
    
    def load_waypoints(self):
        """Load waypoints from config/waypoints.yaml"""
        try:
            config_path = os.path.expanduser("/home/gaobotics/stretch_llm_ros/config/waypoints.yaml")
            
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                self.waypoints = data.get('waypoints', {})
            
            print(f"✅ Loaded {len(self.waypoints)} waypoints from YAML:")
            for name, pos in self.waypoints.items():
                print(f"   • {name}: {pos}")
                
        except Exception as e:
            print(f"⚠️ Failed to load waypoints.yaml: {e}")
            self.waypoints = {}

    def get_current_pos(self, joint_name):
        if self.joint_states:
            try:
                idx = self.joint_states.name.index(joint_name)
                return self.joint_states.position[idx]
            except ValueError:
                rospy.logerr(f"Joint '{joint_name}' not found in joint_states")
        return 0.0

    def send_joint_goal(self, joint_names, positions, duration=2.5, relative=False):
        if relative:
            current = [self.get_current_pos(j) for j in joint_names]
            positions = [c + p for c, p in zip(current, positions)]

        goal = FollowJointTrajectoryGoal()
        traj = JointTrajectory()
        traj.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(duration)
        traj.points = [point]
        goal.trajectory = traj

        self.traj_client.send_goal(goal)
        self.traj_client.wait_for_result()
        return self.traj_client.get_result()

    def send_base_velocity(self, linear=0.0, angular=0.0, duration=1.0):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        rate = rospy.Rate(20)

        end_time = rospy.Time.now() + rospy.Duration(duration)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        self.cmd_vel_pub.publish(Twist())  # stop
    
    def send_nav_goal(self, x, y, yaw):
        if not self.is_navigation_mode():
            return "Must be in navigation mode to send nav goals. Switch using mode_nav."

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        self.move_base_client.send_goal(goal)
        rospy.loginfo(f"Sent nav goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

        # Wait for result (with timeout for safety)
        wait = self.move_base_client.wait_for_result(rospy.Duration(20.0))  # 60s timeout
        if not wait:
            self.move_base_client.cancel_goal()
            rospy.logerr("Navigation timed out or server unavailable.")
            return "Navigation failed: timed out or server unavailable."

        result = self.move_base_client.get_state()
        if result == actionlib.GoalStatus.SUCCEEDED:
            return "Reached goal successfully."
        else:
            return f"Navigation failed (status: {result}). Check obstacles or map."

    
    def get_robot_state_summary(self):
        summary = "=== CURRENT ROBOT STATE ===\n"
        summary += f"Mode: {self.current_mode}\n"

        if self.joint_states:
            summary += "Joint positions:\n"
            for name, pos in zip(self.joint_states.name, self.joint_states.position):
                summary += f"  {name}: {pos:.4f}\n"

        try:
            map_pose = self.get_current_robot_pose()  # already returns (x, y, yaw) or None
            if map_pose:
                x, y, yaw = map_pose
                summary += f"Global pose (map frame): x={x:.3f} m, y={y:.3f} m, yaw={yaw:.3f} rad\n"
            else:
                summary += "Global pose (map frame): unavailable (AMCL not localized?)\n"

            # Optional: still show odom for debugging drift
            odom = rospy.wait_for_message('/odom', Odometry, timeout=0.8)
            ox = odom.pose.pose.position.x
            oy = odom.pose.pose.position.y
            summary += f"Local odometry (for reference): x={ox:.3f} m, y={oy:.3f} m\n"
        except Exception as e:
            summary += f"Pose information unavailable: {str(e)}\n"

        return summary
    
    
    
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
        """Robust version: tries cached AMCL → wait_for_message → TF (base_link or base_footprint)"""
        rospy.loginfo("=== get_current_robot_pose called ===")

        # 1. Try cached subscriber (if you kept it)
        if hasattr(self, 'latest_amcl_pose') and self.latest_amcl_pose is not None:
            try:
                p = self.latest_amcl_pose.pose.pose.position
                q = self.latest_amcl_pose.pose.pose.orientation
                yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
                rospy.loginfo(f"✓ SUCCESS from cached /amcl_pose: x={p.x:.3f}, y={p.y:.3f}, yaw={yaw:.3f}")
                return (p.x, p.y, yaw)
            except Exception as e:
                rospy.logwarn(f"Cache parse failed: {e}")

        # 2. Direct wait_for_message (most reliable for one-shot calls)
        rospy.loginfo("Trying wait_for_message on /amcl_pose (5s timeout)...")
        try:
            pose_msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout=5.0)
            p = pose_msg.pose.pose.position
            q = pose_msg.pose.pose.orientation
            yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            rospy.loginfo(f"✓ SUCCESS from wait_for_message: x={p.x:.3f}, y={p.y:.3f}, yaw={yaw:.3f}")
            return (p.x, p.y, yaw)
        except rospy.ROSException as e:
            rospy.logwarn(f"wait_for_message timed out or failed: {e}")

        # 3. TF fallback — try both common Stretch frames
        rospy.loginfo("Falling back to TF lookup...")
        for frame in ["base_link", "base_footprint"]:
            try:
                self.tf_listener.waitForTransform("map", frame, rospy.Time(0), rospy.Duration(3.0))
                (trans, rot) = self.tf_listener.lookupTransform("map", frame, rospy.Time(0))
                yaw = np.arctan2(2*(rot[3]*rot[2] + rot[0]*rot[1]), 1 - 2*(rot[1]**2 + rot[2]**2))
                rospy.loginfo(f"✓ SUCCESS from TF map → {frame}: x={trans[0]:.3f}, y={trans[1]:.3f}, yaw={yaw:.3f}")
                return (trans[0], trans[1], yaw)
            except Exception as e:
                rospy.logwarn(f"TF map → {frame} failed: {type(e).__name__}: {e}")

        rospy.logerr("❌ ALL METHODS FAILED — returning None")
        return None
        
    
    
    def get_object_distance(self, object_desc: str):
        print(f"DEBUG - Starting object distance for: '{object_desc}'")

        if self.latest_color_img is None:
            return "No RGB image available."

        try:
            # Vision to locate object
            rgb_b64 = self._img_to_base64(self.latest_color_img)

            vision_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Focus on locating '{object_desc}' (person, desk, table, etc.) in this image. Return exact format:\n"
                                        "object_found: yes/no\n"
                                        "center_x: <int>\n"
                                        "center_y: <int>\n"
                                        "box_x1: <int> box_y1: <int> box_x2: <int> box_y2: <int>\n"
                                        "If not found: object_found: no and brief reason."
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_b64}"}}
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.2
            )

            vision_text = vision_response.choices[0].message.content.strip()
            print("DEBUG - Vision location:", vision_text)

            if "object_found: no" in vision_text.lower():
                return f"'{object_desc}' not visible in current view."

            cx, cy = None, None
            box = None

            cx_match = re.search(r"center_x:\s*(\d+)", vision_text, re.IGNORECASE)
            cy_match = re.search(r"center_y:\s*(\d+)", vision_text, re.IGNORECASE)
            if cx_match and cy_match:
                cx = int(cx_match.group(1))
                cy = int(cy_match.group(1))

            box_match = re.search(r"box_x1:\s*(\d+)\s*box_y1:\s*(\d+)\s*box_x2:\s*(\d+)\s*box_y2:\s*(\d+)", vision_text, re.IGNORECASE)
            if box_match:
                box = (int(box_match.group(1)), int(box_match.group(2)), int(box_match.group(3)), int(box_match.group(4)))

            if cx is None or cy is None:
                cx = self.latest_color_img.shape[1] // 2
                cy = self.latest_color_img.shape[0] // 2
                print("DEBUG - Fallback to center for object '{object_desc}'")

            # Step 2: Use point cloud for distance (better than depth image)
            pc_msg = rospy.wait_for_message('/camera/depth/color/points', PointCloud2, timeout=10.0)
            points = list(point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))

            if not points:
                return "No 3D points available."

            import numpy as np
            xyz = np.array(points)
            distances = np.linalg.norm(xyz, axis=1)

            # Filter
            valid_mask = (distances > 0.1) & (distances < 10.0)
            valid_xyz = xyz[valid_mask]
            valid_dist = distances[valid_mask]

            if valid_dist.size == 0:
                return f"'{object_desc}' detected at approx pixel ({cx}, {cy}), but no valid 3D points."

            # Find points near the pixel (simple: small cone or box projection)
            # Approximate: points with small pixel offset (requires projection)
            # For simplicity, use center cone
            center_mask = (np.abs(valid_xyz[:,1]) < 0.5) & (valid_xyz[:,2] > 0.1)
            region_dist = valid_dist[center_mask]

            if region_dist.size == 0:
                region_dist = valid_dist  # fallback to all valid

            distance = np.median(region_dist)
            return f"Object '{object_desc}' is approximately {distance:.2f} meters away (median 3D distance in region)."

        except Exception as e:
            return f"Object distance failed: {str(e)}"
    
    

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

    def execute_cmd(self, cmd):
        name, value = parse_cmd(cmd)

        defaults = {
            "base_forward": 0.05, "base_back": 0.05,
            "base_left": 0.2, "base_right": 0.2,
            "lift_up": 0.02, "lift_down": 0.02,
            "wrist_out": 0.03, "wrist_in": 0.03,
            "head_left": 0.15, "head_right": 0.15,
            "head_up": 0.15, "head_down": 0.15,
        }
        delta = abs(value) if value is not None else defaults.get(name, 0.05)

        # ──────────────────────────────────────────────
        # Mode switching commands
        # ──────────────────────────────────────────────
        if name in ["mode_position", "mode_pos"]:
            try:
                rospy.wait_for_service('/switch_to_position_mode', timeout=3)
                srv = rospy.ServiceProxy('/switch_to_position_mode', Trigger)
                srv()
                return "Switched to position mode"
            except Exception as e:
                return f"Failed to switch to position mode: {e}"

        elif name in ["mode_navigation", "mode_nav"]:
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
            sign = 1 if name == "lift_up" else -1
            self.send_joint_goal(['joint_lift'], [sign * delta], relative=True)
            return "Lift moved."

        elif name == "lift_home":
            self.home_srv()
            return "Lift homed."

        elif name in ["wrist_out", "wrist_in"]:
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
            move = delta
            self.send_joint_goal(
                ['joint_wrist_yaw'],
                [move],
                relative=True,
                duration=1.5
            )
            return f"Gripper rotating left by {move:.3f} rad"
        
        elif name in ["grip_right", "wrist_yaw_right"]:
            move = -delta
            self.send_joint_goal(
                ['joint_wrist_yaw'],
                [move],
                relative=True,
                duration=1.5
            )
            return f"Gripper rotating right by {abs(move):.3f} rad"
        
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
        elif name in ["head_left", "head_right", "head_up", "head_down"]:
            delta = abs(value) if value is not None else 0.15

            if name in ["head_left", "head_right"]:
                joint = "joint_head_pan"
                sign = 1 if name == "head_left" else -1
            else:  # head_up, head_down
                joint = "joint_head_tilt"
                sign = 1 if name == "head_up" else -1

            move = sign * delta
            self.send_joint_goal([joint], [move], relative=True)
            return f"Head {name.replace('head_', '')} by {move:.3f} rad"

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
            # Assume value is the object description
            object_desc = value if value is not None else "main object in view"
            dist_info = self.get_object_distance(object_desc)
            return dist_info
        
        elif name == "nav_to_named":
            location = value.strip().lower() if value else ""
            if location in self.waypoints:
                x, y, yaw = self.waypoints[location]
                print(f"→ Navigating to known location: {location} ({x}, {y}, {yaw:.2f} rad)")
                return self.send_nav_goal(x, y, yaw)
            else:
                known = ", ".join(self.waypoints.keys())
                return f"Unknown location '{location}'. Known locations: {known}"
            
        elif name == "move_relative":
            # Format: move_relative(forward, 2.0) or move_relative(left, 1.5)
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

        else:
            return f"Unknown command: {name}"
        
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
    

    def _base_position_move(self, name, value, delta):
        direction = 1 if name in ["base_forward", "base_left"] else -1
        if name in ["base_forward", "base_back"]:
            self.send_joint_goal(['translate_mobile_base'], [direction * delta])
            return f"Base {'forward' if direction > 0 else 'back'} {delta:.3f} m"
        else:
            self.send_joint_goal(['rotate_mobile_base'], [direction * delta])
            return f"Base {'left' if direction > 0 else 'right'} {delta:.3f} rad"

    def _base_velocity_move(self, name, value, delta):
        if name in ["base_forward", "base_back"]:
            speed = 0.15  # m/s
            linear = speed if name == "base_forward" else -speed
            duration = delta / speed if value is not None else 0.5
            self.send_base_velocity(linear=linear, duration=duration)
            return f"Base {'forward' if name == 'base_forward' else 'back'} ≈{delta:.3f} m"
        else:
            speed = 0.4   # rad/s
            angular = speed if name == "base_left" else -speed
            duration = delta / speed if value is not None else 0.8
            self.send_base_velocity(angular=angular, duration=duration)
            return f"Base {'left' if name == 'base_left' else 'right'} ≈{delta:.3f} rad"

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
                                print(f"  Executing: {cmd}")
                                result = self.execute_cmd(cmd)
                                print(f"  → {result}")
                                speak_text(result)
                                rospy.sleep(0.2)
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
                                object_desc = c.split("(", 1)[1].rstrip(")") if "(" in c else "main object in view"
                                object_dist_desc = self.get_object_distance(object_desc)
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


if __name__ == '__main__':
    try:
        controller = DualModeStretchController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
