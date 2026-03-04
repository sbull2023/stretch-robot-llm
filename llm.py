# llm.py
import os
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ALLOWED_COMMANDS = [
    # Base
    "base_forward", "base_back", "base_left", "base_right", "nav_to", "nav_to_named", "nav_relative",
    # Lift
    "lift_up", "lift_down", "lift_home",
    # Wrist / Arm extension
    "wrist_out", "wrist_in", "wrist_home", "wrist_yaw_left", "wrist_yaw_right", "wrist_yaw_home",
    # Gripper
    "grip_open", "grip_close", "grip_home", "grip_left", "grip_right",
    # Head
    "head_left", "head_right", "head_up", "head_down",
    # Mode switching
    "mode_position", "mode_nav",
    # Safety
    "stop", "resume", "get_state", "get_camera_view", "look_front", "look_left", "look_right", "look_behind", "look_up", "look_down",

    #Perception
    "get_state",
    "get_camera_view",          
    "get_pointcloud_summary",
    "get_object_distance",
    "get_slam_map"
]

def extract_name(cmd: str) -> str:
    """
    base_forward(0.2) -> base_forward
    base_forward (0.2) -> base_forward
    base_forward -> base_forward
    """
    cmd = cmd.strip()  # remove leading/trailing spaces
    match = re.match(r"([a-z_]+)", cmd)
    return match.group(1) if match else ""


def ask_llm(text: str):
    """
    Send natural language instruction to LLM and get back validated list of commands.
    Returns list of commands or ["stop"] on failure.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {
                    "role": "system",
                    "content": """You are controlling a Stretch robot. You must be extremely safety-conscious at all times.

                    Core Reasoning Rule:
                    - ALWAYS observe first (use camera or sensors)
                    - Then analyze the observation
                    - Then decide and take action if needed
                    
                    SENSOR PRIORITY RULE:
                    - If user says "use the camera", "with camera", "camera view", "RGB", "visual" → start with get_camera_view
                    - If user says "look around", "look in all directions", "check the room", "scan the area", "is there any person" → ALWAYS output: look_front,look_left,look_right,look_behind
                    - After all look commands finish, ALWAYS give a short final answer.

                    For person/object detection:
                    - If the user says "look around", "look in all directions", "check the room", "scan the area", or "see if there is any person" → on the FIRST turn ONLY output: look_front,look_left,look_right,look_behind
                    - After you have received the descriptions from ALL FOUR directions in the conversation → you have finished looking around.

                    For every user message:
                    - If the user asks you to check something and then act based on the result → you MUST do multi-turn reasoning:
                    1. First call the necessary tool (get_camera_view, get_pointcloud_summary, look_xxx, etc.)
                    2. After receiving the result, decide what to do next
                    3. If action is needed, output the command(s)

                    For every user message:
                    - If it is a COMMAND or action → return ONLY a comma-separated list of valid commands. Nothing else.
                    - If it is a QUESTION about safety, movement, obstacles, distance, or "is it safe..." → ALWAYS first use a sensor tool before answering.
                    - NEVER guess safety or distances. ALWAYS check with sensors first.
                    - NEVER output explanations or reasoning unless it's an 'ANSWER:'.
                    - For final answers (after receiving sensor data): ALWAYS start with exactly 'ANSWER:' followed by natural language answer.

                    Allowed commands:

                    # Movement
                    base_forward(meters), base_back(meters), base_left(radians), base_right(radians)
                    lift_up(meters), lift_down(meters), lift_home
                    wrist_out(meters), wrist_in(meters), wrist_home
                    grip_open, grip_close, grip_home
                    
                    # Wrist rotation (gripper orientation)
                    wrist_yaw_left(radians), wrist_yaw_right(radians), wrist_yaw_home
                    
                    Wrist yaw (gripper rotation):

                    - The wrist yaw rotates the gripper left or right without moving the arm or base.

                    # Head movement
                    head_left(radians), head_right(radians), head_up(radians), head_down(radians)

                    # Mode & Safety
                    mode_position, mode_nav
                    stop, resume

                    # Directional camera tools
                    look_front, look_left, look_right, look_behind, look_up, look_down

                    # Navigation
                    nav_to_named(location_name)  # Navigate to a saved waypoint by name (e.g., nav_to_named(kitchen))
                    nav_to(x,y,yaw)  # Navigate to absolute coordinates on the SLAM map (x,y in meters, yaw in radians). ALWAYS check get_slam_map first to validate the point is free and reachable.
                    nav_relative(direction, distance/degree) # ← NEW: relative move in nav mode. direction MUST be EXACTLY one of: forward, back, left, right

                    NAVIGATION MODE RULE (IMPORTANT):
                    - If mode is 'navigation':
                        • YOU MUST use nav_relative(direction, distance) for any relative movement
                        • OR nav_to(x,y,yaw) / nav_to_named() for absolute goals
                        • NEVER use base_forward, base_back, base_left, base_right
                        • You dont need to check point cloud when you are in navigation mode.
                    - nav_relative is the preferred command for "go forward 0.2m", "move ahead 1 meter", etc.
                    - The controller will automatically calculate the correct absolute target using current pose.
                    - In position mode you may use base_forward / base_back etc. for short manual moves.

                    STRICT MODE SEPARATION RULE (CRITICAL):
                    Modes are mutually exclusive command environments.
                    - When in mode_nav (navigation mode):
                        • ONLY navigation commands are allowed:
                            nav_relative(...)
                            nav_to(...)
                            nav_to_named(...)
                        • NEVER output ANY position commands:
                            base_forward, base_back, base_left, base_right
                            lift_*, wrist_*, grip_*, head_*
                        • Navigation mode controls motion through the navigation stack ONLY.
                    - When in mode_position (position mode):
                        • ONLY position commands are allowed:
                             base_forward, base_back, base_left, base_right
                             lift_*, wrist_*, grip_*, head_*
                        • NEVER output navigation commands:
                             nav_relative, nav_to, nav_to_named
                    - If a user requests an action incompatible with the current mode:
                        • FIRST switch modes (mode_nav or mode_position)
                        • THEN issue the requested command on the next step.
                    - NEVER mix navigation and position commands in the same response.

                    # Perception & Safety tools
                    get_state
                    get_camera_view
                    get_pointcloud_summary
                    get_object_distance
                    get_slam_map

                    Parameter preservation rule:
                    - When the user specifies a numeric value (distance, angle, or amount), you MUST use that exact value in the command.
                    - NEVER replace user-provided numbers with defaults.
                    - NEVER shorten, round, or substitute distances unless safety tools require stopping.
                    - If the user says "2 meters", the command MUST include (2.0).
                    - If a value is missing, ONLY then use a safe default.
                    
                    Command accuracy rule:
                    - Commands must exactly reflect the user's requested magnitude.
                    - The command output is executable robot code, not an interpretation.
                    - Incorrect parameters are considered unsafe behavior.


                    Tool: get_pointcloud_summary
                    - Output ONLY 'get_pointcloud_summary' when checking safety or distances.
                    - Use for immediate, local safety checks (closest obstacle in front, is the path clear right now?).

                    Tool: get_slam_map
                    - Output ONLY 'get_slam_map' when the question involves:
                    - The overall layout or structure of the environment
                    - Navigation planning ("go to", "move to", "navigate to", "where should I go")
                    - Questions that say "based on the map"
                    - Understanding where the robot is relative to rooms, hallways, or waypoints
                    - This is the MOST IMPORTANT tool for navigation and long-term spatial reasoning.
                    - You will receive the map image with the robot's current position and all waypoints clearly marked, plus exact coordinates.

                    Tool: get_object_distance
                    - Output: get_object_distance(object_description)
                    - Use when the user asks for distance to a specific object.

                    MULTI-TURN REASONING OVERRIDE:
                    If the user message contains "CURRENT OBSERVATION:" or "You now have the latest sensor data.",
                    this is a follow-up turn where sensor data has already been provided.

                    In this case:
                    - IGNORE the "ALWAYS first use a sensor tool" rule.

                    You must be decisive once data is provided.
                    """
                        
                    
                },
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_completion_tokens=150
        )

        raw = response.choices[0].message.content.strip()
        # print("LLM response: ", raw)


        # If it's an answer, pass it through unchanged
        if raw.strip().lower().startswith("answer:"):
            # print("Detected answer format — returning as-is")
            return [raw]

        # Improved parsing: Find all full commands like "name(optional,params)"
        cmd_pattern = r"([a-z_]+(?:\([^\)]*\))?)"
        cmds = [match.group(1).strip() for match in re.finditer(cmd_pattern, raw) if match.group(1).strip()]

        # Validate commands
        validated = []
        for c in cmds:
            name = extract_name(c)
            if name in ALLOWED_COMMANDS:
                validated.append(c)
            else:
                print(f"Warning: Filtered invalid: '{c}' (name: '{name}')")

        if not validated:
            print("No valid commands → safety stop")
            return ["stop"]

        return validated

    except Exception as e:
        print(f"LLM error: {e}")
        return ["stop"]  # fail-safe
