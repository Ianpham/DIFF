#!/bin/bash
# Phase 2: Frozen Gaussians + TransDiffuser planning
# Load Phase 1 occ checkpoint, freeze Gaussian branch, train decoder
# Expected: 120 epochs

# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# torchrun \
#     --nproc_per_node=4 \
#     --master_port=29501 \
#     train.py \
#     --phase 2 \
#     --config configs/base.py \
#     --occ-ckpt work_dirs/phase1_occ/best.pth \
#     --work-dir work_dirs/phase2_plan

#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python train.py \
    --phase 2 \
    --config configs/base.py \
    --occ-ckpt work_dirs/phase1_occ/best.pth \
    --work-dir work_dirs/phase2_plan