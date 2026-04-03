#!/bin/bash
#SBATCH --job-name=transdiff_sensor
#SBATCH --partition=vinai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=sdc2-hpc-dgx-a100-020
#SBATCH --time=01:00:00
#SBATCH --output=/home/vinfast_tamp/slurm_%x_%j.out
#SBATCH --error=/home/vinfast_tamp/slurm_%x_%j.err

set -o pipefail

DDPM_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp/data/DPJI/transdiffuser/DDPM"
ENV_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp/envs/l4_robotaxi"
USER_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp"

for i in {1..20}; do
    ls "$USER_ROOT/data" > /dev/null 2>&1
    ls "$USER_ROOT/envs" > /dev/null 2>&1
    if ls "$DDPM_ROOT" > /dev/null 2>&1 && ls "$ENV_ROOT/bin" > /dev/null 2>&1; then
        echo "Lustre mounted."
        break
    fi
    echo "Waiting... $i/20"
    sleep 3
done

ls "$DDPM_ROOT"    > /dev/null 2>&1 || { echo "ERROR: DDPM_ROOT not accessible"; exit 1; }
ls "$ENV_ROOT/bin" > /dev/null 2>&1 || { echo "ERROR: ENV_ROOT not accessible"; exit 1; }

mkdir -p "$DDPM_ROOT/logs"

source /sw/software/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_ROOT"

export PROJECT_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp/data/DPJI/transdiffuser"
export DDPM_ROOT="$PROJECT_ROOT/DDPM"
export NAVSIM_ROOT="$DDPM_ROOT/datasets/navsim"
export NAVSIM_UTILIZE="$NAVSIM_ROOT/navsim_utilize"
export PYTHONPATH="$DDPM_ROOT:$NAVSIM_ROOT:$NAVSIM_UTILIZE:$PYTHONPATH"
export OPENSCENE_DATA_ROOT="$DDPM_ROOT/datasets/navsim/download"
export NUPLAN_MAPS_ROOT="$DDPM_ROOT/datasets/navsim/download/maps"
export NAVSIM_DEVKIT_ROOT="$DDPM_ROOT/datasets/navsim"

echo "================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "Start  : $(date)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "================================================"

cd "$DDPM_ROOT" || { echo "ERROR: cannot cd"; exit 1; }

python inference.py \
    --checkpoint outputs/run_20260323_165439_job179646/checkpoints/best.pt \
    --num_samples 12 \
    --num_inference_steps 20 \
    --hidden_size 768 \
    --depth 12 \
    --num_heads 12 \
    --max_agents 8

EXIT_CODE=$?
echo "================================================"
echo "Finished : $(date)"
echo "Exit     : $EXIT_CODE"
echo "================================================"
exit $EXIT_CODE