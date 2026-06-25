# Command Reference

Auto-generated from `stretch_llm/llm/grammar.py` — edit the grammar, then
run `python scripts/gen_command_docs.py`.

| Command | Level | Mode | Params | Safe range | Description |
|---|---|---|---|---|---|
| `mode_nav` | 1 | any | — | — | Switch to navigation mode. |
| `mode_position` | 1 | any | — | — | Switch to position mode. |
| `resume` | 1 | any | — | — | Resume after a stop. |
| `stop` | 1 | any | — | — | Emergency stop. |
| `get_camera_view` | 1 | both | — | — | — |
| `get_object_distance` | 1 | both | str | — | — |
| `get_pointcloud_summary` | 1 | both | — | — | — |
| `get_slam_map` | 1 | both | — | — | — |
| `get_state` | 1 | both | — | — | — |
| `head_down` | 1 | both | float | [0.0, 1.0] | — |
| `head_home` | 1 | both | — | — | — |
| `head_left` | 1 | both | float | [0.0, 3.1416] | — |
| `head_right` | 1 | both | float | [0.0, 3.1416] | — |
| `head_up` | 1 | both | float | [0.0, 1.0] | — |
| `look_behind` | 1 | both | — | — | Camera view, behind. |
| `look_down` | 1 | both | — | — | Camera view, downward. |
| `look_front` | 1 | both | — | — | Camera view, ahead. |
| `look_left` | 1 | both | — | — | Camera view, left. |
| `look_right` | 1 | both | — | — | Camera view, right. |
| `look_up` | 1 | both | — | — | Camera view, upward. |
| `nav_relative` | 1 | navigation | str, float | [0.0, 5.0] | Relative move resolved against the AMCL pose: direction in {forward, back, left, right}. |
| `nav_to` | 1 | navigation | float, float, float | — | Navigate to absolute SLAM-frame pose (x, y, yaw). |
| `nav_to_named` | 1 | navigation | str | — | Navigate to a saved waypoint by name. |
| `base_back` | 1 | position | float | [0.0, 2.0] | Translate the base backward by d metres. |
| `base_forward` | 1 | position | float | [0.0, 2.0] | Translate the base forward by d metres. |
| `base_left` | 1 | position | float | [0.0, 3.1416] | Rotate the base left by r radians. |
| `base_right` | 1 | position | float | [0.0, 3.1416] | Rotate the base right by r radians. |
| `grip_close` | 1 | position | — | — | Close the gripper. |
| `grip_home` | 1 | position | — | — | Home the gripper. |
| `grip_left` | 1 | position | float | [0.0, 1.57] | Rotate the gripper roll left. |
| `grip_open` | 1 | position | — | — | Open the gripper. |
| `grip_right` | 1 | position | float | [0.0, 1.57] | Rotate the gripper roll right. |
| `lift_down` | 1 | position | float | [0.0, 1.1] | Lower the lift by d metres. |
| `lift_home` | 1 | position | — | — | Send the lift to its home height. |
| `lift_up` | 1 | position | float | [0.0, 1.1] | Raise the lift by d metres. |
| `wrist_home` | 1 | position | — | — | Retract the arm fully. |
| `wrist_in` | 1 | position | float | [0.0, 0.52] | Retract the telescoping arm by d metres. |
| `wrist_out` | 1 | position | float | [0.0, 0.52] | Extend the telescoping arm by d metres. |
| `wrist_yaw_home` | 1 | position | — | — | Centre the gripper yaw. |
| `wrist_yaw_left` | 1 | position | float | [0.0, 3.1416] | Rotate the gripper yaw left by r radians. |
| `wrist_yaw_right` | 1 | position | float | [0.0, 3.1416] | Rotate the gripper yaw right by r radians. |
| `analyze_three_camera_frames` | 2 | position | str | — | Three-frame capture, description, and distance. |
| `approach_object` | 2 | position | str | — | Thirds-of-image visual servoing toward a target. |
| `demo_sequence` | 2 | position | — | — | Fixed diagnostic gripper demonstration. |
| `estimate_gripper_distance` | 2 | position | str | — | Gripper-to-object distance from the camera. |
| `pick_object` | 2 | position | str | — | Full grasp routine: yaw, look, lift, align, grasp. |
| `search_for_object` | 2 | position | str | — | Rotate-and-detect sweep, head fixed, up to 12 steps. |

`place_object` exists as a dispatcher-only behaviour outside the vocabulary: the run loop routes placement keywords to it directly, so the LLM never emits it.
