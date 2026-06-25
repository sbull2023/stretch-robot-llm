#!/bin/bash
# Robot-to-HPC SSH tunnel for the Tier-1 inference endpoint.
#
# Run on the robot's NUC. Forwards local port 11435 to the compute node
# that holds the Ollama job, through the cluster login node. Find the
# node name in the SLURM job output (e.g. gpu-node-07).
#
# Usage: ./tunnel.sh <hpc_user> <compute_node>

set -euo pipefail
HPC_USER=${1:?usage: tunnel.sh <hpc_user> <compute_node>}
NODE=${2:?usage: tunnel.sh <hpc_user> <compute_node>}
LOGIN_HOST=${LOGIN_HOST:-ai-panther.fit.edu}

echo "Forward 127.0.0.1:11435 -> ${NODE}:11435 via ${LOGIN_HOST}"
exec ssh -N -L 11435:${NODE}:11435 ${HPC_USER}@${LOGIN_HOST}
