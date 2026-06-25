# Trajectory Dataset Schema

Raw deployment logs are JSONL, one trajectory per line:

```json
{
  "session_id": "2025-11-04-A",
  "trajectory_id": "2025-11-04-A-003",
  "mode": "position",
  "instruction": "pick up the water bottle on the table",
  "turns": [
    {"role": "assistant", "commands": ["get_camera_view"], "level": 1},
    {"role": "observation", "text": "Camera observation: A water bottle ..."},
    {"role": "assistant", "commands": ["pick_object(water_bottle)"], "level": 2},
    {"role": "observation", "text": "Grasp verified: gripper width 41 mm"},
    {"role": "answer", "text": "I picked up the water bottle."}
  ],
  "verdict": "success",
  "truncated": false,
  "operator_intervention": false,
  "hardware_fault": false
}
```

Field notes:

* `verdict` ∈ {`success`, `partial`, `failed`} — the Tier-2 monitor's final
  outcome label. The two-class training formulation collapses `partial`
  into `failed`.
* `level` per assistant turn is optional; the builder infers it from the
  command list when absent (any high-level behaviour ⇒ Level 2).
* The three boolean flags drive curation: any true value drops the
  trajectory.

`training/build_dataset.py` consumes this schema and emits chat-format
JSONL with the level and verdict tokens applied, a deterministic
instruction-level ID/OOD split, and verdict-balanced sample weights.

The full dataset (~870 trajectories from ~640 sessions) is pending
institutional approval for public release; `data/trajectories/` holds a
two-trajectory sample that exercises the entire pipeline.
