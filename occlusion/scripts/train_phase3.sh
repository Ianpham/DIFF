#!/bin/bash
# Phase 3: Joint fine-tuning
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# torchrun \
#     --nproc_per_node=8 \
#     --master_port=29502 \
#     train.py \
#     --phase 3 \
#     --config configs/base.py \
#     --ckpt work_dirs/phase2_plan/best.pth \
#     --work-dir work_dirs/phase3_joint

#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python train.py \
    --phase 3 \
    --config configs/base.py \
    --ckpt work_dirs/phase2_plan/best.pth \
    --work-dir work_dirs/phase3_joint