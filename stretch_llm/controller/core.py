"""Core node state, ROS interfaces, and primitive motion.

This mixin owns the connection to the robot: action clients, publishers,
subscribers, mode tracking, joint-state bookkeeping, waypoint storage, and
the three motion primitives every behaviour builds on (joint-trajectory
goals, timed velocity bursts, and absolute navigation goals).
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


class CoreMixin:
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
        self.captured_images = []

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
        wait = self.move_base_client.wait_for_result(rospy.Duration(60.0))  # 60s timeout
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

        pose = self.get_current_robot_pose()
        x, y, yaw = pose
        summary += f"Robot pose (x, y, yaw): {x:.3f}, {y:.3f}, {yaw:.3f} rad\n"

        return summary

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
