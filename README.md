# Trajectory Fine-Tuning of Local Language Models for Mobile Manipulation

Reference implementation for the paper *"Trajectory Fine-Tuning of Local
Language Models for Mobile Manipulation: Sub-1,000 Real Robot Sessions Match
Frontier Models"* (Bullard, Huynh, Nguyen — Florida Institute of Technology).

The system controls a Hello Robot Stretch RE2 mobile manipulator through
natural language with two fine-tuned open-weight LLMs and no cloud
dependency on the reasoning path. A Llama-3.1-8B model on the AI-Panther
HPC cluster handles instruction grounding and multi-turn tool selection; a
Gemma-3-4B model on the robot's onboard compute handles execution
verification, visual-servoing decisions, and command-completion checks.
Both models are trained with **level-aware verdict-conditioned trajectory
fine-tuning**: the unit of training is the full observe–reason–act sequence
from real deployment logs, not a synthetic (instruction, command) pair.

## Repository layout

```
stretch-trajectory-llm/
├── stretch_llm/                  Python package (deployment stack)
│   ├── llm/                      Language-model side
│   │   ├── grammar.py            The 47-command vocabulary C, with mode,
│   │   │                         level, and safe-range metadata
│   │   ├── prompts.py            Deployed system prompt (byte-for-byte)
│   │   ├── parser.py             Regex parser, validator, range clamps
│   │   └── client.py             Two-tier Ollama clients (HPC + edge)
│   ├── controller/               ROS side (Noetic)
│   │   ├── node.py               Composition root: DualModeStretchController
│   │   ├── core.py               ROS interfaces, primitive motion
│   │   ├── monitor.py            Tier-2 execution verification
│   │   ├── perception.py         Camera, point cloud, SLAM map tools
│   │   ├── behaviors.py          Level-2 visual behaviours
│   │   ├── execution.py          Central command dispatch
│   │   └── loop.py               Multi-turn observe–reason–act loop
│   └── speech/                   TTS output and microphone input
├── training/                     Fine-tuning pipeline (Section 4)
│   ├── build_dataset.py          Curation, serialisation, token augmentation
│   ├── finetune_lora.py          LoRA recipe (r=16, α=32, q/k/v/o)
│   ├── verdict_inference.py      Forced-success logit-mask inference
│   └── synthetic_baseline.py     D_syn generator for the E1 comparison
├── evaluation/                   Metric suite and grounding eval (Section 7)
├── scripts/                      Robot entry point, HPC jobs, SSH tunnel
├── config/                       Model endpoints, waypoints
├── launch/                       ROS launch file
├── docs/                         Architecture, dataset schema, safety
├── tests/                        ROS-free unit tests (pytest)
└── data/                         Trajectory logs and maps (samples only)
```

## Quick start

### 1. Robot-free development

The language side has no ROS dependency, so the parser, grammar, dataset
builder, and metric suite run on any machine:

```bash
git clone https://github.com/<org>/stretch-trajectory-llm
cd stretch-trajectory-llm
pip install -e ".[dev]"
pytest tests/
```

### 2. Serve the Tier-1 model on the HPC

```bash
sbatch scripts/hpc/serve_ollama.sbatch        # on AI-Panther
./scripts/hpc/tunnel.sh <user> <compute_node> # on the robot's NUC
export OLLAMA_URL=http://127.0.0.1:11435
```

### 3. Run the controller on the Stretch

The robot needs ROS Noetic with `stretch_driver`, the navigation stack, and
AMCL active, plus a local Ollama instance with the monitor model:

```bash
ollama pull gemma3:4b
export OLLAMA_EDGE_URL=http://127.0.0.1:11434
python scripts/run_controller.py
```

Type an instruction, or press ENTER for voice input:

```
You > is it safe to move forward 1 meter? if yes, do it
```

The loop queries the Tier-1 model, executes the point-cloud safety check it
requests, injects the observation into a follow-up prompt, and dispatches
the verified action — the worked example of Figure 6 in the paper.

### 4. Reproduce the fine-tuning recipe

```bash
python training/build_dataset.py \
    --logs data/trajectories/raw_sessions.jsonl \
    --out  data/trajectories/processed
sbatch scripts/hpc/train_lora.sbatch
python evaluation/run_grounding_eval.py \
    --data data/trajectories/processed/test_ood.jsonl \
    --model llama3.1:latest
```

## The two control variables

**Level token.** Every assistant turn begins with `<|level:high|>` or
`<|level:primitive|>`. The token turns the implicit choice between one
high-level behaviour call (a parameterised pick, a target search) and an
explicit primitive sequence into a learned, observable variable. Level
prediction reaches 89.2% on the out-of-distribution split.

**Verdict token.** Every assistant turn carries `<|verdict:success|>` or
`<|verdict:failed|>` at training time, set to the trajectory's eventual
monitor-derived outcome. At inference, a logit mask forces the success
prefix, which elicits the model's success-conditional command distribution
(+12 points of command validity over unconditioned generation).

## Safety design

Mode separation is enforced at three independent layers — the prompt rule,
the vocabulary validator with per-command range clamps
(`stretch_llm/llm/parser.py`), and the hardware Trigger services — so a
mode-violation error must defeat all three to reach an actuator. A response
with no valid command resolves to an emergency stop. See `docs/SAFETY.md`.

## Citation

```bibtex
@article{huynh2026trajectory,
  title   = {Trajectory Fine-Tuning of Local Language Models for Mobile
             Manipulation: Sub-1,000 Real Robot Sessions Match Frontier Models},
  author  = {Bullard, Samantha and Huynh, Truong Nhut and Nguyen, Kim-Doang},
  journal = {Robotics and Autonomous Systems (under review)},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
