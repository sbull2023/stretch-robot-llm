#!/bin/bash
#SBATCH --job-name=traj-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out

# LoRA fine-tuning job on AI-Panther (4x A100 80GB, ~18 h per model).

source $HOME/venvs/trajllm/bin/activate
cd $SLURM_SUBMIT_DIR

torchrun --nproc_per_node=4 training/finetune_lora.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data data/trajectories/processed \
    --out runs/llama31-8b-traj-lora
