#!/bin/bash
#SBATCH --job-name=ollama-tier1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ollama_%j.out

# Serve the Tier-1 primary model on an A100 partition of AI-Panther.
# Port 11435 is the project convention: 11434 is frequently taken by
# other Ollama instances on shared nodes.

export OLLAMA_HOST=0.0.0.0:11435
export OLLAMA_MODELS=$HOME/.ollama/models

module load cuda 2>/dev/null || true

ollama serve &
SERVER_PID=$!
sleep 10

# Warm load so the first robot query avoids the cold-start penalty.
OLLAMA_HOST=127.0.0.1:11435 ollama run llama3.1:latest "ready" >/dev/null

echo "Tier-1 endpoint live on $(hostname):11435"
wait $SERVER_PID
