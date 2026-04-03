"""
configs/moe_base.py

Master configuration file for MoETransDiffuser.

Every hyperparameter for every component is defined here.
Values are set as module-level variables so the config loader
(importlib.util.module_from_spec) can read them as attributes.


How to read this file:
──────────────────────────────────────────────────────────────
  Each section corresponds to one module in the codebase.
  The section header names the file it configures.
  Every non-obvious value has a one-line rationale comment.

  To create an experiment variant, copy this file and override
  only the values you want to change — everything else inherits
  from the base.


Recommended experiment variants:
──────────────────────────────────────────────────────────────
  moe_small.py      — half embed_dim, 2 blocks, 2 experts; for ablations
  moe_large.py      — 8 blocks, 8 experts; full-scale
  moe_no_stopgrad.py — stopgrad_relax_step=0 (always relaxed, ablation)
  moe_no_decorr.py  — lambda_A=B=C=0 (no decorrelation, ablation)
"""


# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

data_root   = "/data/openscene"
work_dir    = "work_dirs/moe_base"

# Checkpoints for stage transitions
# dense_ckpt: GaussianTransDiffuser Phase 3 checkpoint used for MoE upcycling
# Set to None if training from scratch (much slower to specialise)
dense_ckpt  = "work_dirs/phase3_joint/best.pth"

# Phase 2 checkpoint (MoE router warmup + planning)
# Used as starting point for Phase 3 joint fine-tuning
phase2_ckpt = None   # set after Phase 2 completes


# ═══════════════════════════════════════════════════════════════
# Data / dataloader
# ═══════════════════════════════════════════════════════════════

batch_size      = 8        # per-GPU; total effective batch = batch_size × n_gpus × accum_steps
num_workers     = 4
train_split     = "train"
val_split       = "val"
test_split      = "test"

# OpenScene occupancy grid geometry (must match openscene_dataset.py)
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ_size          = [200, 200, 16]   # X × Y × Z voxels at 0.4m resolution


# ═══════════════════════════════════════════════════════════════
# GaussianFormer3D backbone   (models/encoders/)
# ═══════════════════════════════════════════════════════════════

num_gaussians      = 25600   # number of 3D Gaussians per scene (from GaussianFormer)
embed_dims         = 128     # Gaussian feature dimension
num_classes        = 17      # OpenScene 17-class taxonomy
num_encoder_blocks = 6       # iterative refinement blocks (sparse conv + 3D DFA each)
num_gaussian_tokens = 256    # FPS-pooled tokens fed into MoE Group A
pooling_method     = "fps"   # "fps" | "learned"

# Image backbone
img_backbone_type  = "resnet101_dcn_fpn"   # placeholder; set to actual mmdet3d name
img_embed_dim      = 128

# TransFuser (frozen; provides BEV, image, and LiDAR feature tokens)
transfuser_dim     = 512     # TransFuser output feature dimension
# Projection: transfuser_dim → decoder_embed_dim handled in MoETransDiffuser
# via proj_bev / proj_img / proj_lidar linear layers


# ═══════════════════════════════════════════════════════════════
# Decoder / diffusion trajectory model   (models/decoders/)
# ═══════════════════════════════════════════════════════════════

decoder_embed_dim   = 256    # shared feature dimension for all decoder layers
decoder_num_heads   = 8      # attention heads in diffusion decoder
decoder_num_layers  = 6      # DenosingDecoderLayer stack depth
decoder_ff_dim      = 1024   # FFN hidden dim inside each decoder layer
trajectory_length   = 8      # number of future waypoints to predict
action_dim          = 2      # (x, y) displacement per waypoint
num_diffusion_steps = 10     # DDPM denoising steps at inference (training uses full schedule)
num_candidates      = 30     # K trajectory candidates sampled per scene
ego_status_dim      = 7      # speed, accel, yaw_rate, steering_angle, (+ 3 padding)


# ═══════════════════════════════════════════════════════════════
# MoE token counts   (models/moe/token_router.py — MoEConfig)
# ═══════════════════════════════════════════════════════════════
#
# These define the per-group token budgets fed into the MoE backbone.
# Must be consistent with MoEBlockConfig defaults in moe_block.py.

# Group A (sensory): pooled Gaussian + lidar tokens
num_tokens_A            = 64

# Group B (interaction): per-agent tokens (ego + surrounding agents)
num_tokens_B            = 64

# Group C (map/context): BEV structural + image context tokens
num_tokens_C            = 128


# ═══════════════════════════════════════════════════════════════
# MoE expert configuration   (models/moe/expert_ffn.py)
# ═══════════════════════════════════════════════════════════════

num_experts_A   = 4    # spatial experts: near/far, dynamic/static sensory regions
num_experts_B   = 4    # behavioral experts: yielding / crossing / following / stopped
                       # Expert 0 is the shared anchor (gets ≥ routing_floor_base probability)
num_experts_C   = 6    # structural experts: one per TTYPE (phantom/vectormap/traffic_light/
                       #                    intersection/bev_struct/unknown)
                       # Must be ≥ NUM_C_TYPES (6) so each type can map to a dedicated expert

top_k_A         = 2    # top-2 expert routing for Group A
top_k_B         = 2    # top-2 for Group B (soft combine, not hard select)
top_k_C         = 1    # top-1 for Group C (near-deterministic structural routing)

expert_ff_mult  = 4    # FFN hidden dim = embed_dim × expert_ff_mult (SwiGLU gate + value)
expert_dropout  = 0.0  # dropout inside expert FFN (set >0 only if overfitting)

# Upcycling stability: output_scale for expert out_proj (set low for first N steps)
# The GatedExpertFFN.output_scale is a learnable scalar initialised to this value.
# 0.1 means expert outputs are initially small residuals on top of the dense model.
expert_output_scale_init = 0.1   # ramp toward 1.0 as training stabilises

# [FIX-5] Per-group capacity factors — explicit, no derived *1.33.
# Group C gets higher capacity because structural token distribution is heavy-tailed.
capacity_factor_A       = 1.5
capacity_factor_B       = 1.5
capacity_factor_C       = 2.0    # plan 1.2: C has heavy-tailed distribution

# capacity_penalty_coeff removed — replaced by bias_penalty in router

# Shared expert floor (Group B Expert 0 always gets ≥ routing_floor_base probability)
routing_floor_base      = 0.05   # lowered from 0.12; bias_penalty handles load balancing


# ═══════════════════════════════════════════════════════════════
# MoE routing   (models/moe/token_router.py)
# ═══════════════════════════════════════════════════════════════

moe_embed_dim       = decoder_embed_dim   # 256 — same as decoder (no projection needed)
moe_num_blocks      = 4    # stacked MoEBlocks; 4 gives good depth:compute tradeoff
                            # (ablation: try 2 and 6 to measure sensitivity)

# Group C structural router temperature
# Low temperature → near-deterministic routing by token type
# High temperature → softer, more data-driven routing
struct_router_temp  = 0.1   # very low → quasi-lookup-table behaviour

# Group B gate
gate_num_heads      = 4    # cross-attention heads in Group B gate MLP

# Group identity embedding orthogonality regularisation
ortho_reg_weight    = 1e-3  # penalises Gram-matrix off-diagonal elements
                             # keeps A/B/C identity vectors orthonormal throughout training

# Stop-gradient on Group C → Group B gate path
stopgrad_C_to_B     = True  # active throughout training; relaxed at stopgrad_relax_step
stopgrad_relax_step = 15_000  # global step at which stop-grad is relaxed (None = never)

# Attention heads for shared self-attention layer
num_attn_heads      = 8
attn_dropout        = 0.0

# Input dimensions for tokeniser projections (set to embed_dim when upstream
# features are already projected; change if upstream dims differ)
dim_A_in            = decoder_embed_dim   # 256
dim_B_in            = decoder_embed_dim   # 256
dim_C_in            = decoder_embed_dim   # 256


# ═══════════════════════════════════════════════════════════════
# Directed cross-attention   (models/moe/direct_attention.py)
# ═══════════════════════════════════════════════════════════════
# Three separate cross-attention modules: A→C, A→B, C→B
# Each has its own Q/K/V projections (no weight sharing).
# Half the heads of shared self-attention (plan 3.2).

directed_cross_attn_num_heads = max(1, num_attn_heads // 2)   # 4


# ═══════════════════════════════════════════════════════════════
# Warmup cross-attention layer   (models/moe/warmup_attention.py)
# ═══════════════════════════════════════════════════════════════
# Lightweight cross-attention layer BEFORE the main block stack (plan 3.3).
# Gives Group B's gate a cross-modal representation before any routing.
# Estimated cost: ~3-5% FLOPs.
# Active — runs once before the block loop in StackedMoEBlocks.


# ═══════════════════════════════════════════════════════════════
# Group B internal pipeline   (models/moe/groupb_pipeline.py)
# ═══════════════════════════════════════════════════════════════
# Stages 1-3 enriching Group B tokens before the gate fires (plan 2.3-2.6):
#   Stage 1: Ego-centric cross-attention
#   Stage 2: Ego-proximity-filtered agent-agent attention
#   Stage 3: Map context re-weighting scalar gate

num_attn_heads_pipeline = 4      # heads for Stage 1 & 2 attention
history_len             = 15     # LSTM history steps
use_history_encoder     = True   # LSTM history encoder before pipeline


# ═══════════════════════════════════════════════════════════════
# Intention heads   (models/moe/intention_heads.py)
# ═══════════════════════════════════════════════════════════════
# Vehicle 6-class + pedestrian 2-class MLP heads (plan 2.7).
# Logits feed into gate query formation.
# Loss: L_intention = CE against GT intention labels from future trajectories.

lambda_intention    = 0.1    # weight for L_intention in total auxiliary loss


# ═══════════════════════════════════════════════════════════════
# DyDiT skip thresholds   (models/moe/token_router.py — DyDiTSkipScheduler)
# ═══════════════════════════════════════════════════════════════
#
# Opposite schedules for B and C (from design conversation):
#   Group C skips at LOW  t (mode committed, map context not needed)
#   Group B skips at HIGH t (fine agent dynamics not relevant yet)
#   Group A caches every cache_interval_A DDPM steps (sensory is timestep-stable)

T_max             = 1000   # DDPM max timestep (matches num_diffusion_steps * 100)
t_skip_C          = 0.2    # t/T_max < 0.2 → skip Group C (cache and reuse)
t_skip_B          = 0.7    # t/T_max > 0.7 → skip Group B (identity pass-through)
cache_interval_A  = 5      # reuse Group A expert output every 5 DDPM steps


# ═══════════════════════════════════════════════════════════════
# Decorrelation loss   (models/moe/decorrelation_loss.py — DecorrConfig)
# ═══════════════════════════════════════════════════════════════
#
# Staged introduction (from design conversation):
#   Phase 0: steps 0     → step_start_A   — no decorrelation
#   Phase 1: steps N_a   → step_start_B   — Group A only (warmup ramp)
#   Phase 2: steps N_b   → step_start_C   — + Group B (timestep-gated, buffer)
#   Phase 3: steps N_c   → ∞              — + Group C (structural MMD)

step_start_A    = 2_000    # N_a: Group A decorrelation turns on (within Phase 2 Stage B)
step_start_B    = 6_000    # N_b: Group B decorrelation turns on
step_start_C    = 10_000   # N_c: Group C decorrelation turns on

warmup_steps_A  = 2_000    # linear ramp from 0 → lambda_A over these steps
warmup_steps_B  = 3_000    # longer warmup for Group B (more fragile)
warmup_steps_C  = 2_000

# Loss weights (relative to diffusion loss ≈ 1.0)
# Small values — decorrelation is a regulariser, not the primary objective
lambda_A        = 0.02
lambda_B        = 0.03    # slightly higher: Group B diversity is the most important
lambda_C        = 0.02

# Group B timestep gate: only fire decorrelation at low-t (behaviorally meaningful)
t_decorr_B_threshold = 0.3    # t/T_max < 0.3 → decorrelation fires

# Group B anchor buffer (decouples decorrelation from sharp-gate token sparsity)
buffer_size           = 512   # max stored activations per expert (FIFO)
min_tokens_to_update  = 8     # expert must have ≥ this many tokens to update buffer
sigma_C               = 1.0   # RBF kernel bandwidth for Group C MMD
sigma_B               = 0.5   # tighter bandwidth for Group B (velocity/heading scale)
min_expert_tokens     = 4     # skip decorrelation for experts with < this many tokens

# MMD margin: push experts until MMD^2 ≥ margin before loss reaches 0
mmd_margin      = 0.1


# ═══════════════════════════════════════════════════════════════
# Training stage thresholds   (train_moe.py — MoETrainConfig)
# ═══════════════════════════════════════════════════════════════

# Phase 2 Stage A: experts frozen, only router + diffusion decoder train
# No decorrelation (frozen activations → spurious gradients)
# Monitor KL(routing || uniform) — must rise from near-zero during this phase
router_warmup_steps = 3_000

# Total steps per phase
total_steps_phase2  = 20_000
total_steps_phase3  = 15_000


# ═══════════════════════════════════════════════════════════════
# Learning rates   (StackedMoEBlocks.get_param_groups, train_moe.py)
# ═══════════════════════════════════════════════════════════════
#
# Per-component LR multipliers (applied relative to base_lr).
# Must match get_param_groups() in moe_block.py StackedMoEBlocks.

base_lr                 = 1e-4

# Shared self-attention (protect cross-modal integration)
lr_attn_mult            = 0.1

# Expert pools
lr_expert_A_mult        = 0.3
lr_expert_C_mult        = 0.3
lr_expert_B_specialist_mult = 0.5
lr_expert_B_shared_mult = 0.05   # anchor — moves very slowly

# Routers (fresh heads, need to move)
lr_router_mult          = 1.0

# Tokenizer / poolers
lr_tokenizer_mult       = 1.0

# LayerNorms
lr_layernorm_mult       = 1.0

# Warmup cross-attention layer (plan 3.3)
lr_warmup_cross_attn_mult = 0.3

# Directed cross-attention (A→C, A→B, C→B)
lr_directed_cross_attn_mult = 0.3

# Group B internal pipeline (stages 1-3 attention)
lr_group_b_pipeline_mult = 0.3

# History encoder (LSTM, if enabled)
lr_history_encoder_mult  = 1.0

# Intention heads
lr_intention_heads_mult  = 1.0

# Gaussian branch (frozen in Phase 2, low LR in Phase 3)
lr_gaussian_branch_mult = 0.1   # in Phase 3 joint fine-tuning

# Router-warmup LR (Phase 2 Stage A): higher because router is freshly initialised
# and experts are frozen, so aggressive routing LR won't destabilise experts
lr_router_warmup_mult   = 3.0   # applied on top of lr_router_mult during Stage A

# LR schedule
lr_warmup_steps         = 500   # linear warmup steps at start of each phase
eta_min_lr              = 1e-6  # cosine decay floor
max_grad_norm           = 1.0   # gradient clipping norm

weight_decay            = 1e-4


# ═══════════════════════════════════════════════════════════════
# State machine / recovery interventions   (train_moe.py)
# ═══════════════════════════════════════════════════════════════

# Duration of each recovery state (steps)
gate_freeze_steps       = 500    # Router Collapse recovery
expert_noise_steps      = 300    # Expert Degeneration recovery
grad_block_cb_steps     = 400    # Cross-Group Leakage recovery
focal_oversample_steps  = 1_000  # Expert Starvation recovery

# Minimum steps between intervention triggers (cooldown prevents re-entry thrashing)
intervention_cooldown   = 2_000

# Noise injection for Router Collapse recovery (sigma anneals from this to 0)
recovery_noise_sigma    = 2.0

# Partial reinit scale for Expert Degeneration recovery
reinit_noise_scale      = 0.01   # σ for additive noise on expert FFN weights


# ═══════════════════════════════════════════════════════════════
# Monitoring thresholds   (models/moe/probe_evaluator.py — ProbeConfig)
# ═══════════════════════════════════════════════════════════════

# Level 1: Routing Health
cosine_similarity_warn_threshold = 0.7    # pair similarity > this → degeneration warning
t_split_norm                     = 0.5   # high-t / low-t split for similarity heatmap
entropy_collapse_threshold       = 0.3   # entropy < this → gate collapsed (log2(4) ≈ 2 healthy)

# Level 2: Semantic Probes
probe_lr            = 1e-3
probe_epochs        = 20
probe_hidden        = 0      # 0 = linear probe (preferred; more interpretable)
cross_group_leak_threshold = 0.60   # cross-group accuracy > this → leakage

# Level 3: Counterfactual
ablation_min_relative_spike = 0.05   # ≥5% loss increase → expert is doing unique work
gate_sensitivity_min_jsd    = 0.05   # JSD < this → gate not using map context

# Level 4: Mode Coverage
num_trajectory_samples  = 10   # K samples per scene for mode entropy
num_maneuver_clusters   = 5    # k-means clusters in final-waypoint space
ambiguous_entropy_min   = 1.0  # bits; ambiguous scenes should exceed this

# Level 5: Failure Mode Classifier
router_collapse_entropy_var_max    = 0.1    # variance of entropy across scene types
starvation_gradient_fraction       = 0.1   # min expert grad < fraction × mean → starvation


# ═══════════════════════════════════════════════════════════════
# Monitoring / logging intervals   (train_moe.py)
# ═══════════════════════════════════════════════════════════════

log_interval         = 50      # steps between console log lines
sim_check_interval   = 100     # steps between activation similarity matrix updates
diag_interval        = 1_000   # steps between async diagnostics snapshots
val_interval         = 2_000   # steps between validation runs
micro_ckpt_interval  = 500     # steps between micro-checkpoints
probe_eval_interval  = 5_000   # steps between full 5-level probe evaluation runs


# ═══════════════════════════════════════════════════════════════
# Scene type labels   (for probe evaluator and routing entropy breakdown)
# ═══════════════════════════════════════════════════════════════
#
# Expected as integer field "scene_type" in each batch dict.
# 0=pedestrian  1=intersection  2=highway  3=occluded
# Assign these labels in the dataset or via a scene classifier.

num_scene_types   = 4
scene_type_names  = ["pedestrian", "intersection", "highway", "occluded"]


# ═══════════════════════════════════════════════════════════════
# Loss weights   (models/losses/occ_loss.py — CombinedLoss)
# ═══════════════════════════════════════════════════════════════

loss_weights_phase2 = dict(
    diffusion      = 1.0,    # primary DDPM noise prediction loss
    moe_bias       = 0.01,   # bias penalty (replaces capacity_penalty_coeff)
    moe_ortho      = 1e-3,   # orthogonality regularisation
    moe_intention  = 0.1,    # intention head CE loss (lambda_intention)
    # decorrelation is handled inside ThreeGroupDecorrLoss with its own schedule
    # and is added separately from _build_moe_context → forward_phase2
)

loss_weights_phase3 = dict(
    occ_ce         = 1.0,    # occupancy cross-entropy
    occ_lovasz     = 1.0,    # Lovász-Softmax for boundary sharpness
    diffusion      = 1.0,    # DDPM noise prediction
    moe_bias       = 0.01,
    moe_ortho      = 1e-3,
    moe_intention  = 0.1,
)


# ═══════════════════════════════════════════════════════════════
# Phase-specific overrides (accessed in train_moe.py as cfg.phaseN)
# ═══════════════════════════════════════════════════════════════
#
# These dicts collect per-phase settings in one place so the training loop
# can call cfg.phase2["lr"] without conditional logic throughout.

phase2 = dict(
    total_steps          = total_steps_phase2,
    lr                   = base_lr,
    batch_size           = batch_size,
    router_warmup_steps  = router_warmup_steps,
    step_start_A         = step_start_A,
    step_start_B         = step_start_B,
    step_start_C         = step_start_C,
    load_occ             = False,  # no occ labels needed in Phase 2
    load_planning        = True,
    freeze_gaussian      = True,   # Gaussian branch frozen in Phase 2
    loss_weights         = loss_weights_phase2,
)

phase3 = dict(
    total_steps          = total_steps_phase3,
    lr                   = base_lr * 0.2,   # lower LR for joint fine-tuning
    batch_size           = batch_size,
    router_warmup_steps  = 0,    # no warmup needed; router already trained
    step_start_A         = 0,    # decorrelation already scheduled from Phase 2
    step_start_B         = 0,    # (global step offset handled in train_moe.py)
    step_start_C         = 0,
    load_occ             = True,
    load_planning        = True,
    freeze_gaussian      = False,  # everything trains in Phase 3
    loss_weights         = loss_weights_phase3,
)


# ═══════════════════════════════════════════════════════════════
# Derived / computed values (do not override — computed from above)
# ═══════════════════════════════════════════════════════════════

# Total MoE token count
num_tokens_total = num_tokens_A + num_tokens_B + num_tokens_C   # 64 + 64 + 128 = 256

# Expert FFN hidden dim (for parameter count estimation)
expert_hidden_dim = decoder_embed_dim * expert_ff_mult            # 1024

# Approximate parameter counts per module (for sanity checking)
_expert_params_per_ffn = (
    decoder_embed_dim * expert_hidden_dim * 3   # gate + value + out
)   # ≈ 786K per expert

_total_expert_params = _expert_params_per_ffn * (
    num_experts_A + num_experts_B + num_experts_C
) * moe_num_blocks   # ≈ 786K × 14 × 4 ≈ 44M


# ═══════════════════════════════════════════════════════════════
# Sanity assertions (run when config is imported)
# ═══════════════════════════════════════════════════════════════

assert num_experts_C >= 6, \
    f"num_experts_C ({num_experts_C}) must be >= NUM_C_TYPES (6)"

assert t_skip_C < t_skip_B, \
    f"t_skip_C ({t_skip_C}) must be < t_skip_B ({t_skip_B}) for opposite schedules"

assert t_decorr_B_threshold <= t_skip_B, \
    "Group B decorrelation should only fire in the low-t regime where Group B is active"

assert step_start_A < step_start_B < step_start_C, \
    "Decorrelation stages must be introduced in order: A < B < C"

assert lambda_A > 0 and lambda_B > 0 and lambda_C > 0, \
    "Decorrelation weights must be positive"

assert 0.0 < routing_floor_base < 0.5, \
    "Shared expert floor must be in (0, 0.5)"

assert stopgrad_relax_step is None or stopgrad_relax_step > router_warmup_steps, \
    "Stop-gradient should not relax during router warmup"

assert capacity_factor_A > 1.0, "capacity_factor_A should be > 1.0"
assert capacity_factor_B > 1.0, "capacity_factor_B should be > 1.0"
assert capacity_factor_C > capacity_factor_A, \
    "capacity_factor_C should be higher (heavy-tailed distribution)"

assert lambda_intention > 0.0, "lambda_intention must be positive"

assert history_len > 0, "history_len must be positive"


# ═══════════════════════════════════════════════════════════════
# Quick summary (printed on import if run directly)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("  MoE Base Config Summary")
    print("=" * 64)
    print(f"  embed_dim          : {decoder_embed_dim}")
    print(f"  moe_num_blocks     : {moe_num_blocks}")
    print(f"  experts A/B/C      : {num_experts_A}/{num_experts_B}/{num_experts_C}")
    print(f"  tokens  A/B/C      : {num_tokens_A}/{num_tokens_B}/{num_tokens_C} → total {num_tokens_total}")
    print(f"  top-k   A/B/C      : {top_k_A}/{top_k_B}/{top_k_C}")
    print(f"  capacity A/B/C     : {capacity_factor_A}/{capacity_factor_B}/{capacity_factor_C}")
    print(f"  routing_floor_base : {routing_floor_base}")
    print(f"  t_skip  C/B        : {t_skip_C}/{t_skip_B}  (opposite schedules)")
    print(f"  decorr start A/B/C : {step_start_A}/{step_start_B}/{step_start_C}")
    print(f"  λ_A / λ_B / λ_C   : {lambda_A}/{lambda_B}/{lambda_C}")
    print(f"  λ_intention        : {lambda_intention}")
    print(f"  history_len        : {history_len}")
    print(f"  use_history_encoder: {use_history_encoder}")
    print(f"  stopgrad_relax_step: {stopgrad_relax_step}")
    print(f"  router_warmup_steps: {router_warmup_steps}")
    print(f"  base_lr            : {base_lr}")
    print(f"  expert params est. : ~{_total_expert_params/1e6:.1f}M")
    print(f"  data_root          : {data_root}")
    print(f"  work_dir           : {work_dir}")
    print("=" * 64)
    print("All sanity assertions passed.")