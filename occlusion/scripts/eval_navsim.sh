#!/bin/bash
# Evaluate on NavSim navtest
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python evaluate.py \
    --config configs/base.py \
    --ckpt work_dirs/phase3_joint/best.pth \
    --split test \
    --output-dir eval_results/ \
    --batch-size 4
