# GaussianFormer3D × TransDiffuser — NavSim Integration

Occlusion-aware trajectory planning via 3D Gaussian scene representation.

## Architecture (Strategy C: Hybrid)

```
8 Cameras + 5 LiDARs (OpenScene)
     │                          │
TransFuser (frozen)      GaussianFormer3D
  F_bev, F_img, F_LiDAR    V2G → 3D DFA → Refinement → 25,600 Gaussians
     │                          │ pool → 256 tokens
     └──────────┬───────────────┘
         Denoising Decoder (DDPM, 10 steps)
         Cross-attn: [F_bev, F_img, F_LiDAR, F_gauss, motion]
                │
         30 candidates → rejection sampling → PDMS eval
```

## Three Training Phases

| Phase | What | Loss | Duration |
|-------|------|------|----------|
| 1 | GaussianFormer3D on OpenScene occ labels | CE + Lovász | 24 epochs |
| 2 | Freeze Gaussians, train decoder + planning | L_diff + β·L_rep | 120 epochs |
| 3 | Joint fine-tuning (unfreeze all) | All losses | 60 epochs |

## Quick Start

```bash
pip install -r requirements.txt
# Phase 1: Occupancy training
bash scripts/train_phase1.sh
# Phase 2: Planning (needs Phase 1 checkpoint)
bash scripts/train_phase2.sh
# Phase 3: Joint (needs Phase 2 checkpoint)
bash scripts/train_phase3.sh
# Evaluate on navtest
bash scripts/eval_navsim.sh
```

## Key Dependencies

- PyTorch >= 2.0, mmdet3d, spconv
- DFA3D CUDA ops (github.com/IDEA-Research/3D-deformable-attention)
- NavSim toolkit (github.com/autonomousvision/navsim, branch v1.1)
- OpenScene data + occupancy labels (github.com/OpenDriveLab/OpenScene)

## What to Replace for Production

The codebase uses **placeholder backbones** so you can run, test, and develop the pipeline
without the full dependency chain. Before real training, replace these:

| Placeholder | Replace With | Source |
|-------------|-------------|--------|
| `TransFuserBackbone` | Actual TransFuser (ResNet-34 image + LiDAR BEV, multi-stage Transformer fusion) | NavSim repo, `navsim/agents/transfuser/` |
| `ImageBackbone` | ResNet101-DCN + FPN from mmdet3d | `mmdet3d.models.backbones.ResNet` + `mmdet3d.models.necks.FPN` |
| `SparseConv3DBlock` | Real sparse convolution via spconv | `spconv.pytorch.SparseConvTensor` + `spconv.pytorch.SubMConv3d` |
| `DeformableAttention3D` | DFA3D CUDA kernel (two-stage 3D sampling) | github.com/IDEA-Research/3D-deformable-attention |
| `GaussianSplattingDecoder` | CUDA splatting kernel from GaussianFormer | github.com/huang-yh/GaussianFormer, `model/encoder/gaussian_encoder.py` |
| `_rejection_sampling` | Kinematic feasibility check + PDMS-based scoring | NavSim's trajectory evaluation utilities |
| `compute_metrics_placeholder` | Official NavSim PDMS evaluation | `navsim.evaluate.evaluate_predictions()` |

## Project Structure

```
gaussianformer3d_navsim/
├── configs/
│   └── base.py                  # All hyperparameters and paths
├── models/
│   ├── encoders/
│   │   ├── gaussian_lifter.py   # V2G: LiDAR voxels → initial Gaussians
│   │   ├── gaussian_encoder.py  # Iterative refinement (sparse conv + 3D DFA)
│   │   └── deformable_attn_3d.py # 3D deformable attention module
│   ├── decoders/
│   │   ├── gaussian_splatting.py # Gaussian → voxel grid for occ prediction
│   │   └── diffusion_decoder.py  # DDPM trajectory decoder + motion encoder
│   ├── losses/
│   │   └── occ_loss.py          # CE + Lovász + diffusion + decorrelation
│   └── gaussian_transdiffuser.py # Full pipeline (Strategy C)
├── data/
│   └── openscene_dataset.py     # OpenScene loader with occ labels
├── utils/
│   ├── gaussian_utils.py        # Quaternions, covariance, FPS, property parsing
│   └── pooling.py               # FPS / learned pooling (25,600 → 256 tokens)
├── scripts/
│   ├── train_phase1.sh          # torchrun launch for Phase 1
│   ├── train_phase2.sh          # torchrun launch for Phase 2
│   ├── train_phase3.sh          # torchrun launch for Phase 3
│   └── eval_navsim.sh           # Evaluation on navtest
├── train.py                     # Main training loop (phase-aware)
├── evaluate.py                  # Inference + PDMS evaluation
├── requirements.txt
└── README.md
```

## Data Setup

### 1. OpenScene (required)

```bash
# Download OpenScene sensor data + occupancy labels
# See: https://github.com/OpenDriveLab/OpenScene
# Place at /data/openscene/ (or update configs/base.py)
```

OpenScene provides:
- **Sensor data**: 8 cameras + 5 merged LiDARs at 2Hz, compact nuPlan redistribution
- **Occupancy labels**: 200×200×16 voxel grid (0.4m resolution), 20-second LiDAR accumulation
- **Occupancy flow**: per-voxel motion direction + velocity (bonus for dynamic context)

### 2. NavSim v1 (required for planning)

```bash
# Clone NavSim and checkout v1.1 branch
git clone https://github.com/autonomousvision/navsim
cd navsim && git checkout v1.1
pip install -e .
```

NavSim provides:
- **nav-train**: 1,192 scenarios for training
- **nav-test**: 136 scenarios for PDMS evaluation
- **TransFuser checkpoint**: pre-trained backbone

### 3. DFA3D CUDA ops (required for 3D attention)

```bash
git clone https://github.com/IDEA-Research/3D-deformable-attention
cd 3D-deformable-attention && python setup.py install
```

## Training Details

### Phase 1: Occupancy Pre-training
- **Goal**: Train GaussianFormer3D to predict occupancy from cameras + LiDAR
- **Supervision**: OpenScene occupancy labels (CE + Lovász-Softmax)
- **Output**: Checkpoint with trained Gaussian lifter + encoder + splatting decoder
- **Validation**: IoU / mIoU on OpenScene occupancy validation split
- **Config**: 24 epochs, lr=2e-4, batch=8/GPU × 8 GPUs, cosine schedule

### Phase 2: Planning with Frozen Gaussians
- **Goal**: Validate that Gaussian features help trajectory prediction
- **Setup**: Freeze GaussianFormer3D, add cross-attention in diffusion decoder
- **Supervision**: Diffusion noise prediction (MSE) + decorrelation regularization
- **Output**: If PDMS improves → Gaussian context is valuable → proceed to Phase 3
- **Config**: 120 epochs, lr=1e-4, batch=64/GPU × 4 GPUs, OneCycle schedule

### Phase 3: Joint Fine-tuning
- **Goal**: End-to-end optimization of scene understanding + planning
- **Setup**: Unfreeze GaussianFormer3D, train all components jointly
- **Supervision**: All losses (occ CE + Lovász + diffusion + decorrelation)
- **Key**: Lower LR for Gaussian branch (0.1× of decoder LR)
- **Config**: 60 epochs, lr=2e-5, batch=8/GPU × 8 GPUs, cosine schedule

## Expected Results

Based on TransDiffuser's PDMS 94.85 baseline, Gaussian context should primarily improve:

| Sub-metric | Baseline | Expected Impact |
|-----------|----------|----------------|
| NC (No Collision) | 99.4 | ↑ Occupancy completion reveals occluded agents |
| TTC (Time to Collision) | 97.8 | ↑ Earlier warning of hidden hazards |
| DAC (Drivable Area) | 96.5 | → Modest: Gaussian semantics encode road boundaries |
| EP (Ego Progress) | 94.1 | → Indirect: better scene understanding enables confident progress |
| C (Comfort) | 99.4 | → Indirect: better anticipation leads to smoother maneuvers |

**Critical milestone**: Phase 2 completion. If PDMS improves with frozen Gaussians,
the thesis is validated. If not, investigate Gaussian feature quality before Phase 3.

## References

- **GaussianFormer3D**: Zhao et al., "Multi-Modal Gaussian-based Semantic Occupancy Prediction with 3D Deformable Attention", arXiv:2505.10685
- **GaussianFormer**: Huang et al., "Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction", ECCV 2024
- **TransDiffuser**: Jiang et al., "End-to-end Trajectory Generation with Decorrelated Multi-modal Representation", arXiv:2505.09315
- **NavSim**: Dauner et al., "Data-Driven Non-Reactive Autonomous Vehicle Simulation", NeurIPS 2024
- **OpenScene**: Li et al., "3D Occupancy Prediction Benchmark in Autonomous Driving"
- **DFA3D**: Yin et al., "3D Deformable Attention for 3D Feature Learning", ICCV 2023
