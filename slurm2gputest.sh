#!/bin/bash
#SBATCH --job-name=transdiff_2gpu
#SBATCH --partition=vinai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodelist=sdc2-hpc-dgx-a100-020
#SBATCH --time=24:00:00
#SBATCH --output=/home/vinfast_tamp/slurm_%x_%j.out
#SBATCH --error=/home/vinfast_tamp/slurm_%x_%j.err

set -o pipefail

DDPM_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp/data/DPJI/transdiffuser/DDPM"
ENV_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp/envs/l4_robotaxi"
USER_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp"

# ---- Trigger autofs for ALL subdirs, then wait ----
for i in {1..20}; do
    ls "$USER_ROOT/data" > /dev/null 2>&1
    ls "$USER_ROOT/envs" > /dev/null 2>&1
    if ls "$DDPM_ROOT" > /dev/null 2>&1 && ls "$ENV_ROOT/bin" > /dev/null 2>&1; then
        echo "Lustre mounted and env accessible."
        break
    fi
    echo "Waiting for Lustre mount... attempt $i/20"
    sleep 3
done

# Hard verify
ls "$DDPM_ROOT" > /dev/null 2>&1    || { echo "ERROR: DDPM_ROOT not accessible"; exit 1; }
ls "$ENV_ROOT/bin" > /dev/null 2>&1 || { echo "ERROR: ENV_ROOT not accessible"; exit 1; }

mkdir -p "$DDPM_ROOT/logs"
mkdir -p "$DDPM_ROOT/outputs"

# ---- Conda ----
source /sw/software/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_ROOT"

# ---- Verify torchrun ----
TORCHRUN="$ENV_ROOT/bin/torchrun"
if [[ ! -f "$TORCHRUN" ]]; then
    echo "ERROR: torchrun not found at $TORCHRUN"
    exit 1
fi

# ---- PYTHONPATH ----
export PROJECT_ROOT="/lustre/scratch/client/vinfast/users/vinfast_tamp/data/DPJI/transdiffuser"
export DDPM_ROOT="$PROJECT_ROOT/DDPM"
export NAVSIM_ROOT="$DDPM_ROOT/datasets/navsim"
export NAVSIM_UTILIZE="$NAVSIM_ROOT/navsim_utilize"
export PYTHONPATH="$DDPM_ROOT:$NAVSIM_ROOT:$NAVSIM_UTILIZE:$PYTHONPATH"

# ---- Unbuffered output — see logs immediately ----
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1  # print stack trace on segfault

# ---- A100 + NCCL ----
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN          # was INFO — reduces NCCL spam in logs
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# ---- DDP ----
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# ============================================================
# Training configuration
# ============================================================
# Change these to experiment — no need to edit the launch cmd below.
#
# encoder_level:
#   0 = scene features only (no history, no encoder-level decorr)
#   1 = + history temporal encoding  + decorr_temporal
#   2 = + interaction features       + decorr_interaction + block decorr
#   3 = + scene-level agent features + decorr_scene (full)
#
# Recommended starting point: level 1, cosine schedule, 10 diff steps
# Paper config: lr=1e-4, batch=256 (global), 120 epochs, β=0.02, T=10

ENCODER_LEVEL=1
DIFFUSION_STEPS=10
NOISE_SCHEDULE="cosine"
DECORR_WEIGHT=0.02
SCHEDULER="onecycle"
NUM_EPOCHS=120
LEARNING_RATE=1.5e-4
BATCH_SIZE=8
EVAL_EVERY=10

# Action-space: converts GT waypoints to deltas (paper Eq. 1)
# Set to "" to disable
ACTION_SPACE_FLAG="--use_action_space"

# ============================================================

# ---- Info ----
echo "================================================"
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURM_NODELIST"
echo "Num GPUs        : 2"
echo "Master          : $MASTER_ADDR:$MASTER_PORT"
echo "Start Time      : $(date)"
echo "Python          : $(which python) ($(python --version 2>&1))"
echo "PyTorch         : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA (torch)    : $(python -c 'import torch; print(torch.version.cuda)')"
echo "torchrun        : $TORCHRUN"
echo "Working dir     : $DDPM_ROOT"
echo "── Training config ──"
echo "encoder_level   : $ENCODER_LEVEL"
echo "diffusion_steps : $DIFFUSION_STEPS"
echo "noise_schedule  : $NOISE_SCHEDULE"
echo "decorr_weight   : $DECORR_WEIGHT"
echo "scheduler       : $SCHEDULER"
echo "num_epochs      : $NUM_EPOCHS"
echo "learning_rate   : $LEARNING_RATE"
echo "batch_size      : $BATCH_SIZE (per GPU) × 2 GPUs = $(( BATCH_SIZE * 2 )) effective"
echo "action_space    : $ACTION_SPACE_FLAG"
echo "eval_every      : $EVAL_EVERY"
echo "================================================"
nvidia-smi
echo "================================================"

cd "$DDPM_ROOT" || { echo "ERROR: cannot cd to $DDPM_ROOT"; exit 1; }

# ---- Resume flag ----
RESUME_FLAG=""
if [[ -n "$RESUME" ]]; then
    echo "Resuming from: $RESUME"
    RESUME_FLAG="--resume $RESUME"
fi

# ---- Log file with timestamp ----
LOG_FILE="logs/train_2gpu_${SLURM_JOB_ID}.log"
echo "Live log: $LOG_FILE"

# ---- Launch ----
$TORCHRUN \
    --nproc_per_node=2 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    trainslurm.py \
    --dataset basic \
    --data_split mini \
    --mode efficient \
    --hidden_size 768 \
    --depth 4 \
    --num_heads 4 \
    --max_agents 1 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --num_workers 4 \
    --log_interval 1 \
    --save_every 5 \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --encoder_level $ENCODER_LEVEL \
    --diffusion_steps $DIFFUSION_STEPS \
    --noise_schedule $NOISE_SCHEDULE \
    --decorr_weight $DECORR_WEIGHT \
    --scheduler $SCHEDULER \
    --eval_every $EVAL_EVERY \
    --eval_batches 20 \
    --eval_steps 10 \
    $ACTION_SPACE_FLAG \
    $RESUME_FLAG \
    2>&1 | stdbuf -oL -eL tee "$LOG_FILE"
# stdbuf -oL -eL forces line-buffered output so tee writes immediately

EXIT_CODE=${PIPESTATUS[0]}
echo "================================================"
echo "Job finished : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "================================================"
exit $EXIT_CODE