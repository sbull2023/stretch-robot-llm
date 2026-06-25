# System Architecture

## Two-tier inference

```
┌─────────────────────────────┐        campus LAN         ┌──────────────────────────┐
│   Stretch RE2 (on robot)    │  ←——— commands ————————   │  AI-Panther HPC cluster  │
│                             │  ———— observations ——→    │                          │
│  ┌───────────────────────┐  │      (HTTPS / SSH         │  ┌────────────────────┐  │
│  │ Dual-mode ROS node    │  │       tunnel, :11435)     │  │ Tier 1: Llama-3.1- │  │
│  │  core | execution     │  │                           │  │ 8B (LoRA, A100)    │  │
│  │  perception|behaviors │  │                           │  │ grounding, tools,  │  │
│  └──────────┬────────────┘  │                           │  │ map reasoning,     │  │
│             │ verify /      │                           │  │ NL answers         │  │
│             │ servo (8 Hz)  │                           │  └────────────────────┘  │
│  ┌──────────┴────────────┐  │                           └──────────────────────────┘
│  │ Tier 2: Gemma-3-4B    │  │
│  │ (edge, :11434)        │  │
│  └───────────────────────┘  │
│   RealSense D435i · NUC ·   │
│   lift · arm · gripper      │
└─────────────────────────────┘
```

Tier 1 operates at the rate of strategic decisions; Tier 2 operates at the
rate of the inner control loop and never leaves the robot. If the HPC link
drops, the controller can fall back to the edge model for top-level
reasoning at reduced grounding quality.

## The multi-turn observe–reason–act loop

Per user instruction (T_max = 6 turns):

1. Inject the active control mode into the prompt; query Tier 1.
2. A response with the `ANSWER:` prefix is a final natural-language reply —
   speak it and stop.
3. A response with perception-tool calls triggers tool execution; the
   observations form a structured re-prompt and the loop continues.
4. A response with action commands is the final sequence: each command
   passes the validator, executes with a pre/post state capture, and
   receives a Tier-2 verdict. A failed verdict allows one retry.

The implementation is `stretch_llm/controller/loop.py`; the formal
statement is Algorithm 1 of the paper.

## Module dependency direction

`stretch_llm.llm` has zero ROS dependencies and is importable anywhere
(dataset builds, evaluation, CI). `stretch_llm.controller` imports the llm
package, never the reverse. `training/` and `evaluation/` import the llm
package only, so the full fine-tuning pipeline runs on the HPC with no
robot present.
