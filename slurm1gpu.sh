#!/bin/bash
#SBATCH --job-name=transdiff_1gpu
#SBATCH --partition=vinai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# ============================================================
# SLURM Job: 1× A100 Training — TransDiffuser
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

# --- A100 optimizations ---
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8

# --- Print job info ---
echo "================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Node         : $SLURM_NODELIST"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "Start Time   : $(date)"
echo "GPU          : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Driver       : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "Python       : $(which python) ($(python --version 2>&1))"
echo "PyTorch      : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA (torch) : $(python -c 'import torch; print(torch.version.cuda)')"
echo "Project      : $DDPM_ROOT"
echo "================================================"

cd "$DDPM_ROOT"

# Run training — all outputs go to $DDPM_ROOT/outputs/
python train.py \
    --dataset basic \
    --data_split mini \
    --mode efficient \
    --batch_size 8 \
    --hidden_size 768 \
    --depth 12 \
    --num_heads 12 \
    --max_agents 8 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --num_workers 8 \
    --log_interval 10 \
    --save_every 5 \
    2>&1 | tee "logs/train_1gpu_${SLURM_JOB_ID}.log"

echo "================================================"
echo "Job finished : $(date)"
echo "================================================"