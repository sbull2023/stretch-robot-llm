# Stretch Robot Dual-Mode Controller & LLM Integration

This is an extension of the original Stertch Robot work. This is the link to the original work: https://github.com/hello-robot/stretch_ros

This repository contains two core Python modules designed to control the **Stretch robot** using both direct commands and advanced LLM-driven reasoning. The system supports **manual position control** as well as **autonomous navigation**, while enforcing safety, sensor awareness, and precise command execution.

---

## 1. `stretch_ros_dual_mode_controller.py`

### Purpose
This file provides the **basic command interface** for controlling the Stretch robot. It allows sending robot commands through a simplified LLM-based system, validating instructions against a set of allowed commands. This module is ideal for straightforward movements and position control, such as moving the base, lifting the arm, or operating the gripper.

### Key Features
- **Allowed Commands:** Supports base movements (`base_forward`, `base_back`, `base_left`, `base_right`), lift operations, wrist and gripper control, head movements, safety commands, and basic perception tools.
- **Command Parsing:** Extracts command names from user inputs, even when numeric parameters or parentheses are used.
- **LLM Integration:** Uses an OpenAI model to parse natural language instructions and map them into validated robot commands.
- **Safety Fallback:** Returns a `stop` command if no valid commands are detected or if the LLM response cannot be parsed.
- **Simplicity:** Designed for basic use and initial prototyping, making it easy to integrate with ROS topics and robot action clients.

### Limitations
- Does **not enforce strict mode separation** between manual position control and navigation.
- Safety and sensor rules are minimal; the system may rely on LLM judgment without first checking actual sensors.
- Numeric parameters are not strictly preserved; defaults may be used if the user input is missing values.
- Limited support for complex navigation commands (`nav_to`, `nav_relative`) and multi-step reasoning.

---

## 2. `llm.py`

### Purpose
This file extends `stretch_ros_dual_mode_controller.py` with a **robust, safety-focused LLM interface**. It allows the Stretch robot to operate in two distinct modes — **position mode** and **navigation mode** — while strictly enforcing mode separation, safety rules, and sensor-first reasoning.

### Key Features
- **Strict Mode Separation**
  - **Position Mode (`mode_position`):** Enables manual base, lift, wrist, gripper, and head commands. Navigation commands are blocked in this mode.
  - **Navigation Mode (`mode_nav`):** Enables relative and absolute navigation commands (`nav_relative`, `nav_to`, `nav_to_named`). Position commands are blocked in this mode.
  - Automatic mode switching ensures that commands incompatible with the current mode are safely executed after switching.
  
- **Sensor-First Safety Rules**
  - Always checks sensors before acting on safety-sensitive or distance-related queries.
  - Uses tools like `get_camera_view`, `get_pointcloud_summary`, `get_slam_map`, and `get_object_distance` to validate commands and ensure safe operation.
  - Prevents unsafe moves by stopping when necessary.

- **Multi-Turn Reasoning**
  - Can query sensors, analyze results, and then decide on actions in subsequent steps.
  - Supports instructions like “check the room” or “is it safe to move forward?” with safe sensor-first workflows.

- **Command Accuracy**
  - Preserves numeric parameters exactly as provided by the user (e.g., distances, angles).
  - Validates all commands against the allowed command set.
  - Warns and filters out invalid commands.

- **Enhanced LLM Parsing**
  - Extracts multiple commands from natural language instructions.
  - Handles answers versus executable commands separately.
  - Includes detailed reasoning instructions in the system prompt for precise control.

### Advantages Over `stretch_ros_dual_mode_controller.py`
- Enforces safety and sensor-first rules rigorously.
- Supports both **position and navigation modes** with clear separation.
- Preserves numeric values exactly as given by the user.
- Supports multi-turn reasoning for complex instructions.
- Handles navigation in both relative and absolute terms while preventing unsafe mixing of commands.

---

## Combined Usage Overview
- **`stretch_ros_dual_mode_controller.py`** is ideal for **simple testing, basic robot movements, and initial LLM command parsing**.
- **`llm.py`** is designed for **full-featured deployment**, enabling safe autonomous operation, multi-step reasoning, and strict enforcement of operational modes.

Together, these modules provide a flexible foundation for controlling the Stretch robot using both natural language instructions and precise command execution, balancing **ease of use**, **safety**, and **advanced reasoning** capabilities.
