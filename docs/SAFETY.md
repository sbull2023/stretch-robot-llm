# Safety Design

The system assumes the language model is fallible and builds three
independent layers between the model and the actuators.

## Layer 1 — prompt-level constraint

The system prompt carries the Strict Mode Separation Rule, the tool-first
safety rule (no answer to a safety question without a sensor check), and
the parameter-preservation rule. A fine-tuned model internalises these
from the trajectory data; the rule text remains in the prompt as the first
line of defence.

## Layer 2 — vocabulary validator and range clamps

Every model output passes through `stretch_llm/llm/parser.py`:

* command names outside the closed 47-command vocabulary are rejected;
* mode-incompatible commands are rejected against the live control mode;
* numeric arguments are clamped to per-command safe ranges from
  `grammar.py` (e.g. base translations capped at 2.0 m, lift travel at
  1.1 m);
* arguments are coerced with `ast.literal_eval`, never `eval`, so model
  output can never execute code on the robot;
* an empty post-filter command list resolves to an emergency `stop`.

## Layer 3 — hardware mode services

Position and navigation modes are mutually exclusive at the driver level:
the ROS Trigger services reconfigure the low-level controller, and the two
modes cannot be simultaneously active. A mode-violation error must
therefore defeat the fine-tuned model, the validator, and the hardware
driver at once to reach an actuator.

## Execution verification

After every primitive command the edge monitor compares the robot state
before and after execution and returns Verified Success / Partial /
Failed. One automatic retry follows a failed verdict; a second failure
surfaces to the reasoning loop as an observation the Tier-1 model can act
on.
