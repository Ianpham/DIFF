#!/bin/bash
#SBATCH --job-name=transdiff_4gpu
#SBATCH --partition=vinai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# ============================================================
# SLURM Job: 4× A100 DDP Training — TransDiffuser
# ============================================================

PROJECT_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp/data/DPJI/transdiffuser"
DDPM_ROOT="$PROJECT_ROOT/DDPM"

# Create directories
mkdir -p "$DDPM_ROOT/logs"
mkdir -p "$DDPM_ROOT/outputs"

# --- Load environment ---
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/vinfast/users/vinfast_tamp/envs/l4_robotaxi

# --- A100 + NCCL optimizations ---
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# --- DDP environment ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# --- Print job info ---
echo "================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Node         : $SLURM_NODELIST"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "Num GPUs     : 4"
echo "Master       : $MASTER_ADDR:$MASTER_PORT"
echo "Start Time   : $(date)"
echo "Python       : $(which python) ($(python --version 2>&1))"
echo "PyTorch      : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA (torch) : $(python -c 'import torch; print(torch.version.cuda)')"
echo "Project      : $DDPM_ROOT"
echo "================================================"
nvidia-smi
echo "================================================"

cd "$DDPM_ROOT"

# Launch with torchrun
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config config_4xA100.py \
    2>&1 | tee "logs/train_4gpu_${SLURM_JOB_ID}.log"

echo "================================================"
echo "Job finished : $(date)"
echo "================================================"