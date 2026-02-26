#!/bin/bash
# Phase 1: Train GaussianFormer3D on OpenScene occupancy labels
# Expected: 24 epochs, ~2-3 weeks on 8×A40

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train.py \
    --phase 1 \
    --config configs/base.py \
    --work-dir work_dirs/phase1_occ
