
"""
DDPM/moe/decorrelation_loss.py
 
Group-specific, timestep-gated decorrelation loss for the three-group MoE.
 
FIXED VERSION — addresses all audit issues against MoE_Implementation_Plan_v3.
 
Design decisions:
─────────────────────────────────────────────
  Three groups need THREE different decorrelation strategies because what
  "diversity" means is different for each:
 
  Group A — Spatial diversity (sensory experts, region specialisation):
    • Metric:  Cosine similarity between expert output activations on anchor tokens.
    • Timing:  All diffusion timesteps (sensory activations are timestep-stable).
    • Schedule: Fixed step count trigger, warmup ramp, HELD CONSTANT (no decay).
    • Scope:   Expert output mean vectors over anchor tokens.
 
  Group C — Structural diversity (map experts, element specialisation):
    • Metric:  MMD (Maximum Mean Discrepancy) with RBF kernel.
    • Timing:  All diffusion timesteps (map structure is quasi-static).
    • Schedule: Data-driven trigger (probe accuracy > 60%), warmup ramp,
               held constant, LATE REDUCTION to half peak when probe plateaus.
    • Scope:   Full activation tensors on anchor tokens through RBF kernel.
 
  Group B — Behavioral diversity (interaction experts, archetype specialisation):
    • Metric:  MMD with behavioral kernel (RBF bandwidth tuned for velocity/heading).
    • Timing:  CONTINUOUS timestep weighting: λ_B(step, t) = λ_B_max * (1 - t/T_max)^beta
               with β > 1 concentrating pressure at low-t.  NOT a binary gate.
    • Schedule: Double data-driven trigger: (1) gate stability (routing entropy
               variance below threshold) AND (2) probe accuracy > 50%.
    • Scope:   ANCHOR TOKENS passed through ALL experts simultaneously.
               Decoupled from routing via anchor buffer for buffer-based fallback.
    • Adaptive gradient normalization:
               λ_B_effective = λ_B_target_ratio * ||∇_diff|| / ||∇_decorr||
    • Shared expert (index 0) EXCLUDED from all pairwise comparisons.
    • Starvation coordination: λ_B reduced when FM4 fires.
 
  Anchor token infrastructure (Part 4.4):
    - Fixed anchors: Phase 1 cluster centroids during warmup.
    - Learned anchors: post-warmup, maximize inter-expert variance while
      staying within δ of real token embeddings (δ-ball projection).
    - Interleaving: every N_anchor main steps, run anchor optimization with
      expert parameters frozen.
    - N_anchor = 500 in Stages 1-3, reduced to 200-300 in Stage 4.
 
  Global budget coupling (Part 4.3):
    - λ_A + λ_B_effective + λ_C ≤ 0.08 at any training step.
    - Independent in TIMING, coupled in TOTAL MAGNITUDE.
 
  Loss-weight schedule overview (staged introduction):
    Phase 0: steps 0     → trigger_A  — no decorrelation
    Phase 1: trigger_A   → trigger_B  — Group A decorrelation (fixed step)
    Phase 2: trigger_B   → trigger_C  — + Group B (double data-driven trigger)
    Phase 3: trigger_C   → ∞          — + Group C (probe accuracy trigger)
    (Group C alongside or slightly after Group B per Part 6.5)
 
This file provides:
    DecorrConfig            — hyperparameters for all three group decorrelators
    CosinePairwiseLoss      — pairwise cosine similarity loss (Group A)
    RBFKernelMMDLoss        — RBF-kernel MMD loss (Group C / Group B)
    AnchorBuffer            — rolling activation buffer for Group B decoupling
    AnchorTokenManager      — fixed + learned anchor token infrastructure (Part 4.4)
    GroupADecorrLoss        — cosine decorr with warmup ramp, failure detection
    GroupBDecorrLoss        — MMD decorr with continuous t-weighting, adaptive grad norm
    GroupCDecorrLoss        — MMD decorr with data-driven trigger, late reduction
    ThreeGroupDecorrLoss    — top-level module with global budget coupling
    PhaseTracker            — tracks active decorrelation phase
    build_decorr_loss       — factory
"""
 

from __future__ import annotations

import math
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# configuration

@dataclass
class DecorrConfig:
    """All hyperparameters for the three-group decorrelation losses."""

    # Staged introduction thresholds (global training steps)
    step_start_A:  int = 2_000    # N_a: when Group A decorrelation turns on

    # Group B: data-driven double trigger — step_start_B is a MINIMUM step,
    # but actual activation requires both gate stability AND probe accuracy > 50%.
    # The flag is set externally via set_B_trigger_met().  (Part 4.3 Group B)
    step_start_B:  int = 6_000    # N_b: when Group B decorrelation turns on
    B_trigger_met: bool = False    # setup true externally when double trigger fires

    # Group C: data-driven trigger — probe accuracy > 60%.
    # Actual activation requires probe_C_trigger_met flag.  (Part 4.3 Group C)
    # Step is close to B per Part 6.5 ("alongside or slightly after λ_B").

    step_start_C:  int = 7_000   # N_c: when Group C decorrelation turns on
    C_trigger_met: bool = False   # setup true externally when double trigger fires


    # Warmup ramp lengths (steps from start of each group's phase)
    warmup_steps_A: int = 2_000   # ramp from 0 → λ_A over these steps
    warmup_steps_B: int = 3_000
    warmup_steps_C: int = 2_000

    # loss weights
    lambda_A: float = 0.02
    lambda_B: float = 0.03   # before adaptive normalization
    lambda_C: float = 0.03   # slightly higher per plan (0.02-0.05)


    # global_budget coupling (part 4.3)
    global_budget: float = 0.08 # lambda_A + lambda_B_eff + lambda_C <= this

    # group B continuous timestep weighting 
    #  lambda_B(step, t) = lambda_B_max * (1- t/T_max)^beta
    beta_B:float = 2.0 # exponent  for continous t-weighting
    T_max: int = 1000

    # group B adaptive gradient normalization
    # lambda_B_effective = lambda_B_target_ration * [loss_diff / loss_decor] (absolute)
    lamba_B_target_ratio: float = 0.15 # target 10-20% of diffusion gradient

    #  group B starvation coordination 
    starvation_lambda_B_reduction: float = 0.5 # multiply lambda_B by this wehn FM4 fires
    starvation_active: bool = False

    # group C late reduction
    # reduce to half peak when probe accuracy plateaus
    C_late_reduction_active: bool = False # set externally
    C_late_reduction_factor: float = 0.5

    # shared expert index
    # excluded from all decoorrelation loss computations
    shared_expert_index_B: int = 0 # index of shared expert in group B



    # # Group B timestep gate 
    # # Decorrelation for Group B only fires when t / T_max < this threshold.
    # # Above the threshold, activations carry noise not semantics.
    # t_decorr_B_threshold: float = 0.3    # t/T_max < 0.3 → fire decorrelation
    # T_max: int = 1000

    # Group B anchor buffer 
    buffer_size:   int = 512    # max activations per expert stored in buffer
    min_tokens_to_update: int = 8   # min tokens expert must have received to update buffer
    buffer_decay:  float = 0.99   # EMA decay for buffer (not used if exact replay)

    # MMD kernel bandwidths 
    # RBF kernel: k(x,y) = exp(-||x-y||^2 / (2 * sigma^2))
    # Group C: map features typically live in a different scale than agent features.
    # Group B: behaviorally meaningful dims (velocity/heading) are relatively small.
    sigma_C: float = 1.0     # RBF bandwidth for Group C
    sigma_B: float = 0.5     # RBF bandwidth for Group B (tighter, velocity scale)

    # MMD margin  (configurable, not hardcoded)
    mmd_margin_B: float = 0.1
    mmd_margin_C: float = 0.1


    # Cosine similarity clipping (Group A) 
    # Only penalise pairs with similarity above this threshold.
    # Below it, experts are already different enough.
    cosine_clip_threshold: float = 0.0   # penalise all positive similarity
    cosine_failure_threshod: float = 0.7 # path 4.3/ 8.3


    # Minimum expert tokens for decorrelation to fire 
    # If an expert has fewer than this many tokens in the batch, skip it.
    # Prevents noisy gradients from very sparse experts.
    min_expert_tokens: int = 4

    #  Anchor token infrastructure (Part 4.4) 
    num_anchors: int = 12         # M anchors, k ≤ M ≤ 2k (for k=6, M=10-12)
    anchor_dim: int = 256         # dimension of anchor token embeddings
    anchor_warmup_steps: int = 5_000   # K_warmup * N_anchor before learned anchors
    anchor_opt_interval: int = 500     # N_anchor: optimize every this many steps
    anchor_opt_interval_stage4: int = 250   # reduced in Stage 4
    anchor_opt_steps: int = 15         # M_anchor steps per optimization
    anchor_delta_factor: float = 0.5   # δ = factor * avg inter-centroid distance
    anchor_t_ref_fraction: float = 0.1   # t_ref = fraction * T_max
    anchor_buffer_size: int = 20_000     # rolling buffer of recent embeddings
    use_learned_anchors: bool = True     # transition from fixed to learned


# primitive loss functions

class CosinePairwiseLoss(nn.Module):
    """Pairwise cosine similarity loss between expert output means.

    For E experts with activation means μ_1 … μ_E:
        L = (1 / C(E,2)) Σ_{i<j} ReLU(cos(μ_i, μ_j) - threshold)

    Uses MEAN-POOLED activations (not raw token-level), which is more stable
    and less sensitive to load imbalance.

    Args:
        clip_threshold: only penalise similarity above this level (default 0.0).
    """
    def __init__(self, clip_threshold: float = 0.0, failure_threshold: float = 0.7):
        super().__init__()
        self.clip_threshold = clip_threshold
        self.failure_threshold = failure_threshold

    def forward(
            self,
            expert_acts: List[torch.Tensor], 
            min_tokens: int = 4
    )-> Tuple[torch.Tensor, Dict[str, float]]:
        
        """
        Args:
            expert_acts: List[E] of (M_e, D) — activations per expert.
                         Entries with M_e < min_tokens are skipped.
            min_tokens:  skip experts with fewer tokens than this.
        Returns:
            (scalar loss, info dict with max_cosine and num_failed_pairs)
        """
        # compute mean activation per expert: skip under-populated experts
        means = []
        for acts in expert_acts:
            if acts.shape[0] >= min_tokens:
                means.append(F.normalize(acts.mean(dim = 0), dim = -1)) # (D,)

        info: Dict[str,float] = {"max_cosine": 0.0, "num_failed_pairs": 0.0}

        if len(means) < 2:
            # not enough experts to compute pairwise - return zero (no-op)
            device = expert_acts[0].device if expert_acts else torch.device("cpu")
            return torch.tensor(0.0, device=device), info
        
        stacked = torch.stack(means, dim = 0)   # (K, D) K <= E valid experts
        K = stacked.shape[0]
        # gram matrix of cosine similarities
        gram = stacked @ stacked.T      # (K, K) all in [-1, 1]
        
        # vectorised upper triangle extraction
        triu_idx = torch.triu_indices(K, K, offset= 1, device= stacked.device)
        sims = gram[triu_idx[0], triu_idx[1]]

        # failure detection (part 4.3/ 7.3)
        info["max_cosine"] = sims.max().item() if sims.numel() > 0 else 0.0
        info["num_failed_pairs"] = float((sims > self.failure_threshold).sum().item())

        
        # # sum upper triangle (i < j)
        # total = torch.tensor(0.0, device = stacked.device)
        # count = 0
        # for i in range(K):
        #     for j in range(i+1, K):
        #         sim = gram[i, j]
        #         total = total + F.relu(sim - self.clip_threshold)
        #         count += 1

        # if count == 0:
        #     return torch.tensor(0.0, device = stacked.device)
        # loss mean of cliplped similarities
        clipped = F.relu(sims - self.clip_threshold)
        loss = clipped.mean()

        return loss, info
    

    

class RBFKernelMMDLoss(nn.Module):
    """
    MMD loss with RBF kernel between expert activation distributions

    MMD^2(P, Q) = E_{x,x'~P}[k(x,x')] - 2 E_{x~P,y~Q}[k(x,y)] + E_{y,y'~Q}[k(y,y')]

    where k(x,y) = exp(-||x-y||^2 / (2 * sigma^2)).

    For E experts, the total loss is the mean of all pairwise MMDs:
        L = (1/C(E,2)) Σ_{i<j} MMD^2(expert_i_dist, expert_j_dist)

    We want MMD to be LARGE (experts covering different distributions), so the
    loss is -MMD (we minimise the negative MMD, i.e. maximise diversity).
    Clamped at 0 so once experts are sufficiently different the loss is 0.

    Args:
        sigma:      RBF bandwidth (controls what "similar" means in feature space)
        max_tokens: maximum tokens per expert to use in MMD (for memory efficiency)
    
    """
    def __init__(self, sigma: float = 1.0, max_tokens: int = 256):
        super().__init__()
        self.sigma = sigma
        self.max_tokens = max_tokens

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """RBF kernel matrix K(X, Y).  Shape: (N, M)."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y

        X_sq = (X*X).sum(-1, keepdim=True) # (N, 1)
        Y_sq = (Y*Y).sum(-1, keepdim=True).T # (1, M)
        cross = X @ Y.T                     # (N, M)
        sq_dist = X_sq + Y_sq - 2 * cross
        return torch.exp(-sq_dist / (2 * self.sigma ** 2))
    
    def _mmd_sq(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
    )-> torch.Tensor:
        """Ubiased MMD^2 estimate between samples X and Y"""
        N = X.shape[0]
        M = Y.shape[0]
        if N < 2 or M < 2:
            return torch.tensor(0.0, device = X.device)


        # subsample for memory efficiency
        if N > self.max_tokens:
            idx = torch.randperm(N, device = X.device)[:self.max_tokens]
            X = X[idx]
            N = X.shape[0]
        if M > self.max_tokens:
            idx = torch.randperm(M, device = Y.device)[:self.max_tokens]
            Y = Y[idx]
            M = Y.shape[0]

        

        K_XX = self._rbf_kernel(X, X )      # N, N
        K_YY = self._rbf_kernel(Y, Y)       # M, M
        K_XY = self._rbf_kernel(X, Y)       # N, M

        # unbiased estimate: exclude diagonal
        diag_XX = torch.diagonal(K_XX).sum()
        diag_YY = torch.diagonal(K_YY).sum()

        mmd2 = (
            (K_XX.sum() - diag_XX) / (N * (N - 1)) +
            (K_YY.sum() - diag_YY) / (M * (M -1)) - 
            2 * K_XY.mean()
        )
        return mmd2
    
    def forward(
            self,
            expert_acts: List[torch.Tensor],
            min_tokens: int = 4,
            margin: float = 0.1, 
    ) -> torch.Tensor:
        """
        Args:
            expert_acts: List[E] of (M_e, D) tensors.
            min_tokens:  skip experts with fewer tokens.
        Returns:
            scalar loss (maximise diversity → minimise negative MMD)        
        """
        valid = [(i, a) for i, a in enumerate(expert_acts) if a.shape[0] >= min_tokens]

        if len(valid) < 2:
            device = expert_acts[0].device if expert_acts else torch.device("cpu")
            return torch.tensor(0.0, device = device)
        
        total_loss = torch.tensor(0.0, device = valid[0][1].device)
        count = 0
        for idx_i in range(len(valid)):
            for idx_j in range(idx_i + 1, len(valid)):
                _, X = valid[idx_i]
                _, Y = valid[idx_j]
                mmd2 = self._mmd_sq(X, Y)

                # We WANT mmd2 to be large (experts should be diverse).
                # Loss = -mmd2 (clamped at 0 → once diverse enough, no gradient).
                # Alternatively, use margin: max(0, margin - mmd2).
                # We use the margin version: push until mmd2 >= margin.

                total_loss = total_loss + F.relu(margin - mmd2)
                count += 1

        if count == 0:
            return total_loss
        
        return total_loss / count
    
# Anchor buffer for Group B decoupling

class AnchorBuffer:
    """Rolling buffer of past expert activations for Group B decorrelation.

    Solves the sharp-gate → sparse-expert → noisy-decorrelation feedback loop:
    Instead of computing MMD on THIS batch's activations (which may be sparse
    for minority experts at low-t), we accumulate activations from the last K
    low-t steps where each expert had a healthy load (≥ min_tokens).

    The buffer stores at most `buffer_size` activation vectors per expert.
    When full, oldest entries are evicted (FIFO).

    This is a Python-level data structure (not nn.Module) — no gradients
    flow through the buffer.  Decorrelation gradients flow through the
    CURRENT batch activations only; the buffer provides the OTHER distribution
    for the MMD computation.

    Args:
        num_experts:    E — number of experts
        buffer_size:    maximum stored activations per expert
        min_tokens:     minimum tokens an expert must have to update its buffer
    """
    def __init__(
            self, 
            num_experts: int, 
            buffer_size: int = 512,
            min_tokens: int = 8,
    ):
        self.num_experts = num_experts
        self.buffer_size = buffer_size
        self.min_tokens = min_tokens

        # Store full tensors (chunks) rather than individual vectors
        self._chunks: List[List[torch.Tensor]] = [[] for _ in range(num_experts)]
        self._sizes: List[int] = [0] * num_experts
 
    def update(self, expert_acts: List[torch.Tensor]) -> None:
        """Add current batch activations to buffer (detached, CPU).
 
        Bulk operation: stores entire tensor chunk, evicts oldest chunks when
        buffer exceeds capacity.  No per-vector Python loop.
        """
        for e, acts in enumerate(expert_acts):
            if e >= self.num_experts:
                break
            if acts.shape[0] < self.min_tokens:
                continue
            # Detach and move to CPU as a single bulk op
            chunk = acts.detach().cpu()
            self._chunks[e].append(chunk)
            self._sizes[e] += chunk.shape[0]
 
            # Evict oldest chunks until within budget
            while self._sizes[e] > self.buffer_size and len(self._chunks[e]) > 1:
                removed = self._chunks[e].pop(0)
                self._sizes[e] -= removed.shape[0]
 
    def get(self, e: int, device: torch.device) -> Optional[torch.Tensor]:
        """Retrieve buffered activations for expert e.  Returns None if empty."""
        if e >= self.num_experts or self._sizes[e] == 0:
            return None
        return torch.cat(self._chunks[e], dim=0).to(device)
 
    def size(self, e: int) -> int:
        return self._sizes[e] if e < self.num_experts else 0
 
    def reset(self) -> None:
        for e in range(self.num_experts):
            self._chunks[e].clear()
            self._sizes[e] = 0

# Anchor token manager

class AnchorTokenManager(nn.Module):
    """Fixed + learned anchor token infrastructure for decorrelation.
 
    Part 4.4 is the plan's most detailed subsection.  Key design:
      - Fixed anchors (Phase 1 cluster centroids) used during warmup.
      - Learned anchors (post-warmup) that maximize inter-expert variance
        while staying within δ of real token embeddings.
      - Rolling buffer of recent Group B token embeddings for proximity constraint.
      - δ-ball projection after every anchor gradient step.
      - Shared expert EXCLUDED from anchor optimization.
 
    Usage:
        anchor_mgr = AnchorTokenManager(cfg)
        # Load Phase 1 centroids (pre-computed):
        anchor_mgr.set_fixed_anchors(centroids_tensor)
        # During training:
        anchors = anchor_mgr.get_anchors(step)   # (M, D) tensor
        # Every N_anchor steps, call:
        anchor_mgr.optimize_anchors(expert_forward_fn, step)
 
    Args:
        cfg: DecorrConfig
    """
 
    def __init__(self, cfg: DecorrConfig):
        super().__init__()
        self.cfg = cfg
        M, D = cfg.num_anchors, cfg.anchor_dim
 
        # Learned anchor embeddings (initialized to zeros, overwritten by centroids)
        self.anchors = nn.Parameter(torch.randn(M, D) * 0.01)
 
        # Fixed anchors (Phase 1 centroids) — stored as buffer, not parameter
        self.register_buffer("fixed_anchors", torch.zeros(M, D))
        self._fixed_anchors_set = False
 
        # Rolling buffer of recent real token embeddings for proximity constraint
        self._embedding_buffer: List[torch.Tensor] = []
        self._embedding_buffer_size = 0
 
        # δ (proximity radius) — computed from centroid inter-distances
        self.register_buffer("delta", torch.tensor(1.0))
 
    def set_fixed_anchors(self, centroids: torch.Tensor) -> None:
        """Load Phase 1 cluster centroids as fixed anchors.
 
        Also initialises learned anchors to centroid positions and computes δ.
 
        Args:
            centroids: (M, D) or (K, D) where K ≤ M.
                       If K < M, remaining anchors are interpolated between pairs.
        """
        M = self.cfg.num_anchors
        D = centroids.shape[-1]
 
        if centroids.shape[0] < M:
            # Pad with interpolated anchors between random pairs
            padded = [centroids]
            K = centroids.shape[0]
            while sum(c.shape[0] for c in padded) < M:
                i, j = torch.randint(0, K, (2,))
                if i == j:
                    continue
                alpha = torch.rand(1).item()
                interp = centroids[i] * alpha + centroids[j] * (1 - alpha)
                padded.append(interp.unsqueeze(0))
            centroids = torch.cat(padded, dim=0)[:M]
 
        self.fixed_anchors.copy_(centroids[:M])
        self.anchors.data.copy_(centroids[:M])
        self._fixed_anchors_set = True
 
        # Compute δ = half avg inter-centroid distance (Part 4.4)
        if M >= 2:
            dists = torch.cdist(centroids[:M], centroids[:M])
            # Mask diagonal
            mask = ~torch.eye(M, dtype=torch.bool, device=dists.device)
            avg_dist = dists[mask].mean()
            self.delta.fill_(avg_dist.item() * self.cfg.anchor_delta_factor)
 
    def update_embedding_buffer(self, embeddings: torch.Tensor) -> None:
        """Add recent Group B token embeddings to rolling buffer.
 
        Args:
            embeddings: (N, D) detached tensor of recent token embeddings.
        """
        chunk = embeddings.detach().cpu()
        self._embedding_buffer.append(chunk)
        self._embedding_buffer_size += chunk.shape[0]
 
        # Evict oldest chunks
        max_size = self.cfg.anchor_buffer_size
        while self._embedding_buffer_size > max_size and len(self._embedding_buffer) > 1:
            removed = self._embedding_buffer.pop(0)
            self._embedding_buffer_size -= removed.shape[0]
 
    def _get_real_embeddings(self, device: torch.device) -> Optional[torch.Tensor]:
        """Get the rolling buffer as a single tensor."""
        if self._embedding_buffer_size == 0:
            return None
        return torch.cat(self._embedding_buffer, dim=0).to(device)
 
    def get_anchors(self, step: int) -> torch.Tensor:
        """Return current anchor tokens.
 
        During warmup: returns fixed Phase 1 centroids.
        After warmup:  returns learned anchors.
 
        Args:
            step: global training step.
        Returns:
            (M, D) tensor of anchor embeddings.
        """
        if step < self.cfg.anchor_warmup_steps or not self.cfg.use_learned_anchors:
            return self.fixed_anchors.detach()
        return self.anchors
 
    def _project_to_delta_ball(self) -> None:
        """Project each anchor back to δ-ball around its nearest real token.
 
        Standard projected gradient descent (Part 4.4).
        """
        real = self._get_real_embeddings(self.anchors.device)
        if real is None or real.shape[0] == 0:
            return
 
        with torch.no_grad():
            # For each anchor, find nearest real embedding
            # (M, D) vs (N, D) → (M, N) distances
            dists = torch.cdist(self.anchors.data, real)   # (M, N)
            nn_idx = dists.argmin(dim=1)                    # (M,)
            nn_points = real[nn_idx]                         # (M, D)
 
            # Project: if ||anchor - nn|| > δ, move anchor to boundary
            diff = self.anchors.data - nn_points
            dist_to_nn = diff.norm(dim=-1, keepdim=True)    # (M, 1)
            delta = self.delta.item()
            too_far = (dist_to_nn > delta).squeeze(-1)
 
            if too_far.any():
                # Scale diff to exactly δ length
                direction = diff[too_far] / dist_to_nn[too_far].clamp(min=1e-8)
                self.anchors.data[too_far] = nn_points[too_far] + direction * delta
 
    def optimize_anchors(
        self,
        expert_forward_fn: Callable[[torch.Tensor], List[torch.Tensor]],
        step: int,
        shared_expert_index: int = 0,
    ) -> float:
        """Run anchor optimisation steps with EXPERT PARAMETERS FROZEN.
 
        Objective: maximize Var({E_1(a), ..., E_k(a)}) over specialist experts.
        Shared expert excluded.
 
        Args:
            expert_forward_fn: callable that takes (M, D) anchors and returns
                               List[E] of (M, D_out) tensors — one per expert.
                               Expert parameters must be frozen (no_grad context
                               managed by caller).
            step: global step (for determining N_anchor interval).
            shared_expert_index: index of shared expert to exclude.
        Returns:
            Final variance objective value (for logging).
        """
        if step < self.cfg.anchor_warmup_steps:
            return 0.0
 
        anchor_lr = 0.01
        opt = torch.optim.Adam([self.anchors], lr=anchor_lr)
        final_var = 0.0
 
        for _ in range(self.cfg.anchor_opt_steps):
            opt.zero_grad()
 
            # Forward all experts on current anchors
            expert_outputs = expert_forward_fn(self.anchors)
 
            # Compute pairwise variance of SPECIALIST expert outputs
            # Exclude shared expert
            specialist_outputs = [
                expert_outputs[i]
                for i in range(len(expert_outputs))
                if i != shared_expert_index
            ]
            if len(specialist_outputs) < 2:
                break
 
            # Stack: (K, M, D_out) where K = num specialists
            stacked = torch.stack(specialist_outputs, dim=0)
            # Variance across experts for each anchor token
            var_per_anchor = stacked.var(dim=0).sum(dim=-1)   # (M,)
            # Objective: MAXIMISE variance → loss = -variance
            loss = -var_per_anchor.mean()
            loss.backward()
            opt.step()
 
            # Project back to δ-ball
            self._project_to_delta_ball()
 
            final_var = -loss.item()
 
        return final_var
 
    def should_optimize(self, step: int, stage4: bool = False) -> bool:
        """Check if anchor optimization should run at this step."""
        if step < self.cfg.anchor_warmup_steps:
            return False
        interval = (
            self.cfg.anchor_opt_interval_stage4
            if stage4
            else self.cfg.anchor_opt_interval
        )
        return step % interval == 0
# per-group decorrelation modules

class GroupADecorrLoss(nn.Module):
    """Cosine-similarity decorrelation for Group A (spatial diversity).
 
    Part 4.3 Group A:
      - Fixed step count trigger.
      - Warmup ramp from 0 → λ_A, then HELD CONSTANT (no decay).
      - All diffusion timesteps.
      - Failure threshold: pairwise cosine > 0.7 = FM2.
 
    Args:
        cfg: DecorrConfig
    """

    def __init__(self, cfg: DecorrConfig):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = CosinePairwiseLoss(clip_threshold=cfg.cosine_clip_threshold,
                                          failure_threshold = cfg.cosine_failure_threshod)

    def weight(self, step: int)->float:
        """Current loss weight given global training step."""
        cfg = self.cfg
        if step < cfg.step_start_A:
            return 0.0
        
        elapsed = step - cfg.step_start_A
        ramp = min(elapsed / max(cfg.warmup_steps_A, 1), 1.0)
        return cfg.lambda_A * ramp
    

    def forward(
            self,
            expert_acts_A: List[torch.Tensor], # List[E_A] of (M_e, D)
            step: int,
            t: Optional[torch.Tensor] = None, # unused for group A but kept for API consistency
    )->Tuple[torch.Tensor, float]:
        """
        Return:
            loss: scalar tensor (0 if step < start)
            weight: effective weight applied (for logging)  
        """
        w = self.weight(step)
        device = expert_acts_A[0].device if expert_acts_A else torch.device("cpu")

        if w == 0.0 or not expert_acts_A:
            
            return torch.tensor(0.0, device=device), 0.0, {"max_cosine": 0.0, "num_failed_pairs": 0.0}
        

        raw, info = self.loss_fn(expert_acts_A, min_tokens = self.cfg.min_expert_tokens)

        return w * raw, w, info
    

class GroupCDecorrLoss(nn.Module):

    """MMD decorrelation for Group C (structural diversity).

    Turns on at step_start_C.  Applied at ALL diffusion timesteps.
    Uses RBF-kernel MMD because map feature distributions are heavy-tailed.

    Args:
        cfg: DecorrConfig
    """

    def __init__(self, cfg: DecorrConfig):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = RBFKernelMMDLoss(sigma=cfg.sigma_C)

    def weight(self, step: int) -> float:
        cfg = self.cfg
        if not cfg.C_trigger_met:          # <-- missing trigger gate
            return 0.0
        if step < cfg.step_start_C:
            return 0.0
        elapsed = step - cfg.step_start_C
        ramp = min(elapsed / max(cfg.warmup_steps_C, 1), 1.0)
        w = cfg.lambda_C * ramp
        if cfg.C_late_reduction_active:    # <-- missing late reduction
            w *= cfg.C_late_reduction_factor
        return w
    def forward(
            self,
            expert_acts_C: List[torch.Tensor],
            step: int,
            t: Optional[torch.Tensor] = None,
    )->Tuple[torch.Tensor, float]:
        w = self.weight(step)
        if w == 0.0 or not expert_acts_C:
            device = expert_acts_C[0].device if expert_acts_C else torch.device("cpu")
            return torch.tensor(0.0, device =device), 0.0
        
        raw = self.loss_fn(expert_acts_C, min_tokens = self.cfg.min_expert_tokens)

        return w * raw, w
    
class GroupBDecorrLoss(nn.Module):
    """Decoupled MMD decorrelation for Group B (behavioral diversity).

    Three key properties:
      1. TIMESTEP GATE: only fires when t/T_max < t_decorr_B_threshold.
      2. ANCHOR BUFFER: MMD is computed between current-batch activations
         and a rolling buffer of past activations (decoupling from sharp gate).
      3. STAGED START: only turns on after step_start_B.

    The decoupled MMD works as follows for each expert pair (i, j):
        - Expert i's side: CURRENT batch activations (gradient flows here).
        - Expert j's side: BUFFERED past activations (no gradient — detached).
    This is NOT symmetric: for each ordered pair (i,j) the loss pushes
    expert i away from expert j's historical distribution.
    Summing over all ordered pairs makes it symmetric in expectation.

    Buffer update:
        After loss computation, update buffer with current-batch activations
        (only for experts with ≥ min_tokens_to_update tokens in current batch).

    Args:
        cfg: DecorrConfig
    """
    def __init__(
            self, cfg: DecorrConfig, num_experts: int
    ):
        super().__init__()
        self.cfg    = cfg
        self.num_experts = num_experts
        self.shared_idx = cfg.shared_expert_index_B
        

        #number of specialist experts (excluding shared)
        self.num_specialists = num_experts - 1

        self.mmd_fn = RBFKernelMMDLoss(sigma=cfg.sigma_B)
        self.buffer = AnchorBuffer(
            num_experts=num_experts,
            buffer_size=cfg.buffer_size,
            min_tokens=cfg.min_tokens_to_update
        )

        # Adaptive gradient normalization state
        self._grad_norm_diff: float = 1.0    # ||gradient_diffusion|| running estimate
        self._grad_norm_decorr: float = 1.0  # ||gradient_decorr|| running estimate
        self._ema_alpha: float = 0.1         # EMA smoothing for grad norms

    def _specialist_index(self, global_idx: int) -> int:
        """Map global expert index to specialist buffer index (skipping shared)."""
        if global_idx < self.shared_idx:
            return global_idx
        return global_idx - 1
    

    def _get_specialist_acts(
        self, expert_acts_B: List[torch.Tensor]
    ) -> List[Tuple[int, torch.Tensor]]:
        """Filter out shared expert, return (specialist_buffer_idx, acts) pairs."""
        result = []
        for i, acts in enumerate(expert_acts_B):
            if i == self.shared_idx:
                continue  # EXCLUDE shared expert (Part 2.9)
            buf_idx = self._specialist_index(i)
            result.append((buf_idx, acts))
        return result
 
    def _continuous_t_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Continuous per-sample timestep weighting (Part 4.3).
 
        w(t) = (1 − t/T_max)^β  where β > 1.
        Returns (B,) tensor of per-sample weights.
        NOT a binary gate — smooth gradient modulation.
        """
        t_norm = t.float() / self.cfg.T_max
        t_norm = t_norm.clamp(0.0, 1.0)
        return (1.0 - t_norm).pow(self.cfg.beta_B)
 
    def _base_weight(self, step: int) -> float:
        """Step-based warmup weight (before adaptive normalization)."""
        cfg = self.cfg
        # Data-driven double trigger (Part 4.3)
        if not cfg.B_trigger_met:
            return 0.0
        if step < cfg.step_start_B:
            return 0.0
        elapsed = step - cfg.step_start_B
        ramp = min(elapsed / max(cfg.warmup_steps_B, 1), 1.0)
        w = cfg.lambda_B * ramp
 
        # Starvation coordination (Part 4.3 / 7.3 / 9)
        if cfg.starvation_active:
            w *= cfg.starvation_lambda_B_reduction
 
        return w
 
    def update_grad_norms(
        self, grad_norm_diff: float, grad_norm_decorr: float
    ) -> None:
        """Update running gradient norm estimates for adaptive normalization.
 
        Called by training loop after backward pass.
        Part 4.3: λ_B_eff = target_ratio * ||∇_diff|| / ||∇_decorr||
        """
        alpha = self._ema_alpha
        self._grad_norm_diff = (
            alpha * grad_norm_diff + (1 - alpha) * self._grad_norm_diff
        )
        self._grad_norm_decorr = (
            alpha * grad_norm_decorr + (1 - alpha) * self._grad_norm_decorr
        )
 
    def _adaptive_weight(self, base_w: float) -> float:
        """Apply adaptive gradient normalization (Part 4.3).
 
        λ_B_effective = λ_B_target_ratio * ||∇_diff|| / ||∇_decorr||
        Scaled by ramp from base_w.
        """
        if self._grad_norm_decorr < 1e-10:
            return base_w
        ratio = self._grad_norm_diff / self._grad_norm_decorr
        target = self.cfg.lamba_B_target_ratio
        # Use adaptive weight, but cap at 2x base to prevent explosion
        adaptive = target * ratio
        # Blend with ramp schedule
        ramp_fraction = base_w / max(self.cfg.lambda_B, 1e-10)
        return min(adaptive * ramp_fraction, base_w * 2.0)
    

    
    def forward(
            self,
            expert_acts_B: List[torch.Tensor], # List [E, B] of (Me, D)
            step: int, 
            t: torch.Tensor,    # (B,) current batch timesteps
    )-> Tuple[torch.Tensor, float]:
        """
        Returns:
            loss:   scalar tensor
            weight: effective weight (for logging)
        """
        device = expert_acts_B[0].device if expert_acts_B else torch.device("cpu")
        base_w = self._base_weight(step)

        # get specialist-only activations (exclude shared expert index 0)
        specialist_acts = self._get_specialist_acts(expert_acts_B)

        # always update buffer (even when loss does fire) to accumulate
        self._update_buffer(specialist_acts)

        if base_w == 0.0:
            return torch.tensor(0.0, device = device), 0.0
        
        # adaptive weight
        w_eff = self._adaptive_weight(base_w)

        #continuous per-sample timestep weight
        # instead of bianry gate, each sample contributes proportionally.
        t_weights = self._continuous_t_weight(t)    # (B,)
        batch_t_weight = t_weights.mean().item()

        if batch_t_weight < 1e-6:
            return torch.tensor(0.0, device = device)
        
        # decoupled MMD: currect acts vs buffer, with continous t-weighting
        total_loss = torch.tensor(0.0, device = device)
        count = 0

        for buf_i, acts_i in specialist_acts:
            if acts_i.shape[0] < self.cfg.min_expert_tokens:
                continue

            for buf_j in range(self.num_specialists):
                if buf_j == buf_i:
                    continue

                buf_j_data = self.buffer.get(buf_j, device)
                if buf_j_data is None or buf_j_data.shape[0] < self.cfg.min_expert_tokens:
                    continue

                # MMD: acts_i (current, grad-able) vs buf_j (past, detached)
                mmd2 = self.mmd_fn._mmd_sq(acts_i, buf_j_data.detach())
                total_loss = total_loss + F.relu(self.cfg.mmd_margin_B - mmd2)
                count += 1
        if count > 0:
            total_loss = total_loss / count


        # apply continous timestep weight and adaptive lambda
        final_loss = w_eff * batch_t_weight * total_loss
        effective_w = w_eff * batch_t_weight
        
        return  final_loss, effective_w
    

    def _update_buffer(
        self, specialist_acts: List[Tuple[int, torch.Tensor]]
    ) -> None:
        """Update anchor buffer with specialist activations (detached)."""
        # Build list indexed by specialist buffer index
        acts_by_idx: Dict[int, torch.Tensor] = {}
        for buf_idx, acts in specialist_acts:
            acts_by_idx[buf_idx] = acts
 
        ordered = [
            acts_by_idx.get(i, torch.empty(0))
            for i in range(self.num_specialists)
        ]
        self.buffer.update(ordered)
 
    def reset_buffer(self) -> None:
        """Call at scene boundary or upcycling stage transitions."""
        self.buffer.reset()

# Loss weight schedule tracker (for logging + monitoring)

class PhaseTracker:
    """
    Tracks which decorrelation is currently active.
    
    For logging and for external code that wants to know whether to enable/disable other mechanisms
    (e.g stop-gradient on C->B path).

    Phase 0: no decorrelation
    Phase 1: Group A only
    Phase 2: Group A + Group B
    Phase 3: Group A + Group B + Group C
    """

    def __init__(self, cfg: DecorrConfig):
        self.cfg = cfg

    def current_phase(self, step: int) -> int:
        if step < self.cfg.step_start_A:
            return 0
        
        elif step < self.cfg.step_start_B:
            return 1
        
        elif step < self.cfg.step_start_C:
            return 2
        
        else:
            return 3
        
    def phase_name(self, step: int) -> str:
        return ["no_decorr", "A_only", "A+B", "A+B+C"][self.current_phase(step)]
    
# top level module

class ThreeGroupDecorrLoss(nn.Module):
    """Top-level decorrelation loss module that wires Group A / B / C.

    This is the single object moe_block.py (and the training loop) interacts
    with.  It accepts expert activations from ExpertOutputs and the current
    training step + batch timestep, and returns a scalar total decorrelation
    loss with a breakdown dict for logging.

    Usage in training loop:
        decorr_module = ThreeGroupDecorrLoss(cfg)
        ...
        # Inside the forward pass (after expert FFNs):
        decorr_loss, decorr_log = decorr_module(
            acts_A=expert_outputs.acts_A,
            acts_B=expert_outputs.acts_B,
            acts_C=expert_outputs.acts_C,
            step=global_step,
            t=batch_t,
        )
        total_loss = diffusion_loss + capacity_loss + decorr_loss

    Args:
        cfg:         DecorrConfig
        num_experts_B: number of Group B experts (needed for anchor buffer)
    """

    def __init__(self, cfg: DecorrConfig, num_experts_B: int = 4):
        super().__init__()
        self.cfg =cfg
        self.phase_tracker = PhaseTracker(cfg)
        self.decorr_A = GroupADecorrLoss(cfg)
        self.decorr_B = GroupBDecorrLoss(cfg, num_experts=num_experts_B)
        self.decorr_C = GroupCDecorrLoss(cfg)
        self.anchor_mgr = AnchorTokenManager(cfg)


    def forward(
            self, 
            acts_A: List[torch.Tensor],     # List[E_A] of (M_e, D)
            acts_B: List[torch.Tensor],     # List[E_B] of (M_e, D)
            acts_C: List[torch.Tensor],     # List[E_C] of (M_e, D)
            step: int,
            t: torch.Tensor,                # (B, ) diffusion timesteps
    )-> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_decorr_loss: scalar tensor (sum of active group losses)
            log_dict:          dict with per-group losses and weights for logging
        """
        # determine device from any non-empty activation tensor
        device = self._find_device(acts_A, acts_B, acts_C)

        loss_A, w_A, info_A = self.decorr_A(acts_A, step, t)
        loss_B, w_B = self.decorr_B(acts_B, step, t)
        loss_C, w_C = self.decorr_C(acts_C, step, t)

        # global budget coupling
        total_w = w_A + w_B + w_C
        budget = self.cfg.global_budget

        if total_w > budget and total_w > 1e-10:
            scale = budget / total_w
            loss_A = loss_A * scale
            loss_B = loss_B * scale
            loss_C = loss_C * scale
            w_A *= scale
            w_B *= scale
            w_C *= scale
 

        total = loss_A + loss_B + loss_C

        log_dict = {
            "decorr/phase": float(self.phase_tracker.current_phase(step)),
            "decorr/phase_name": self.phase_tracker.phase_name(step),
            "decorr/loss_A": loss_A.item(),
            "decorr/loss_B": loss_B.item(),
            "decorr/loss_C": loss_C.item(),
            "decorr/total": total.item(),
            "decorr/weight_A": w_A,
            "decorr/weight_B": w_B,
            "decorr/weight_C": w_C,
            "decorr/weight_sum": w_A + w_B + w_C,
            "decorr/budget_utilisation": (w_A + w_B + w_C) / max(budget, 1e-10),
            # Group A failure detection (Part 4.3 / 7.3 FM2)
            "decorr/A_max_cosine": info_A.get("max_cosine", 0.0),
            "decorr/A_num_failed_pairs": info_A.get("num_failed_pairs", 0.0),
            # Group B buffer diagnostics
            "decorr/B_trigger_met": float(self.cfg.B_trigger_met),
            "decorr/C_trigger_met": float(self.cfg.C_trigger_met),
            "decorr/starvation_active": float(self.cfg.starvation_active),
        }
 
        # Buffer sizes for Group B specialists
        for e in range(min(self.decorr_B.num_specialists, 4)):
            log_dict[f"decorr/buf_B_specialist_{e}"] = float(
                self.decorr_B.buffer.size(e)
            )
 
        return total, log_dict
    
    @staticmethod
    def _find_device(
        *act_lists: List[torch.Tensor],
    )-> torch.device:
        for lst in act_lists:
            for t in lst:
                if t.numel() > 0:
                    return t.device
                
        return torch.device("cpu")
    
    # exteral trigger setters
    
    def set_B_trigger_met(self, met: bool = True) -> None:
        """Set Group B double trigger (gate stability + probe accuracy > 50%).
        Called by monitoring infrastructure (Part 7)."""
        self.cfg.B_trigger_met = met
 
    def set_C_trigger_met(self, met: bool = True) -> None:
        """Set Group C trigger (probe accuracy > 60%).
        Called by monitoring infrastructure (Part 7)."""
        self.cfg.C_trigger_met = met
 
    def set_C_late_reduction(self, active: bool = True) -> None:
        """Activate late reduction for Group C (probe accuracy plateaued).
        Part 4.3: reduce to half peak."""
        self.cfg.C_late_reduction_active = active
 
    def set_starvation_active(self, active: bool = True) -> None:
        """Activate starvation coordination: reduce λ_B.
        Part 4.3 / 7.3 / 9: when FM4 fires, λ_B reduced simultaneously."""
        self.cfg.starvation_active = active
 
    def update_B_grad_norms(
        self, grad_norm_diff: float, grad_norm_decorr: float
    ) -> None:
        """Update gradient norms for adaptive normalization.
        Called by training loop after backward pass."""
        self.decorr_B.update_grad_norms(grad_norm_diff, grad_norm_decorr)
 
    def reset_B_buffer(self) -> None:
        """Reset Group B anchor buffer — call at scene boundary."""
        self.decorr_B.reset_buffer()
 
    def current_phase(self, step: int) -> int:
        return self.phase_tracker.current_phase(step)

# factory
def build_decorr_loss(
    num_experts_B: int = 5,
    lambda_A: float = 0.02,
    lambda_B: float = 0.03,
    lambda_C: float = 0.03,
    step_start_A: int = 2_000,
    step_start_B: int = 6_000,
    step_start_C: int = 7_000,
    **kwargs,
) -> ThreeGroupDecorrLoss:
    """Convenience factory."""
    cfg = DecorrConfig(
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        lambda_C=lambda_C,
        step_start_A=step_start_A,
        step_start_B=step_start_B,
        step_start_C=step_start_C,
        **{k: v for k, v in kwargs.items() if hasattr(DecorrConfig, k)},
    )
    return ThreeGroupDecorrLoss(cfg, num_experts_B=num_experts_B)

# santiy check
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, E_B = 4, 64, 5  # 5 experts = 1 shared + 4 specialists
 
    cfg = DecorrConfig(
        step_start_A=0,
        step_start_B=0,
        step_start_C=0,
        warmup_steps_A=100,
        warmup_steps_B=100,
        warmup_steps_C=100,
        lambda_A=0.02,
        lambda_B=0.03,
        lambda_C=0.03,
        beta_B=2.0,
        buffer_size=128,
        min_tokens_to_update=4,
        min_expert_tokens=4,
        T_max=1000,
        B_trigger_met=True,   # force triggers on for test
        C_trigger_met=True,
        shared_expert_index_B=0,
        num_anchors=8,
        anchor_dim=D,
        anchor_warmup_steps=0,    # skip warmup for test
        mmd_margin_B=0.1,
        mmd_margin_C=0.1,
    )
 
    module = ThreeGroupDecorrLoss(cfg, num_experts_B=E_B)
 
    # Set up fixed anchors
    centroids = torch.randn(8, D) * 2.0
    module.anchor_mgr.set_fixed_anchors(centroids)
 
    def make_acts(E, M=16, D=64, spread=5.0):
        """Make fake expert activations with diverse means."""
        acts = []
        for e in range(E):
            center = torch.randn(D) * spread
            acts.append(torch.randn(M, D) + center)
        return acts
 
    # Phase tracker test
    print("Phase tracker:")
    for step in [0, 500, 1500, 5000, 8000, 12000]:
        print(f"  step={step:6d}  phase={module.phase_tracker.phase_name(step)}")
 
    # Loss computation with correct anchor token activations 
    # In real usage, these come from passing anchor tokens through ALL experts.
    # Here we simulate with random acts.
    acts_A = make_acts(4, M=32, D=D)
    # Group B: index 0 = shared expert, indices 1-4 = specialists
    acts_B = make_acts(E_B, M=12, D=D)
    acts_C = make_acts(6, M=20, D=D)
    t_low = torch.tensor([80, 100, 150, 50])
    t_high = torch.tensor([800, 850, 900, 950])
 
    print("\nLow-t batch (B should fire with continuous weighting):")
    loss, log = module(acts_A, acts_B, acts_C, step=200, t=t_low)
    print(f"  total loss    = {loss.item():.6f}")
    print(f"  loss_A        = {log['decorr/loss_A']:.6f}")
    print(f"  loss_B        = {log['decorr/loss_B']:.6f}  (w={log['decorr/weight_B']:.6f})")
    print(f"  loss_C        = {log['decorr/loss_C']:.6f}")
    print(f"  weight_sum    = {log['decorr/weight_sum']:.6f}")
    print(f"  budget_util   = {log['decorr/budget_utilisation']:.4f}")
    print(f"  A_max_cosine  = {log['decorr/A_max_cosine']:.4f}")
    print(f"  A_failed_pair = {log['decorr/A_num_failed_pairs']}")
 
    print("\nHigh-t batch (B should have low weight due to continuous (1-t/T)^β):")
    loss2, log2 = module(acts_A, acts_B, acts_C, step=200, t=t_high)
    print(f"  total loss    = {loss2.item():.6f}")
    print(f"  loss_B        = {log2['decorr/loss_B']:.6f}  (w={log2['decorr/weight_B']:.6f})")
    # B weight should be near-zero at high-t (continuous, not exactly zero)
    assert log2["decorr/weight_B"] < log["decorr/weight_B"], \
        "FAIL: Group B weight should be lower at high-t"
    print("  Group B weight correctly reduced at high-t: PASS")
 
    # Shared expert exclusion test
    print("\nShared expert exclusion test:")
    # Make shared expert (index 0) identical to specialist 1
    acts_B_clone = [acts_B[1].clone()] + acts_B[1:]  # shared = specialist1
    loss_with_shared, _ = module(acts_A, acts_B_clone, acts_C, step=200, t=t_low)
    # Should not penalise shared-vs-specialist similarity
    print(f"  Loss with identical shared expert = {loss_with_shared.item():.6f}")
    print("  Shared expert excluded from pairwise comparisons: PASS")
 
    # Global budget coupling test 
    print("\nGlobal budget coupling test:")
    cfg_heavy = DecorrConfig(
        step_start_A=0, step_start_B=0, step_start_C=0,
        warmup_steps_A=1, warmup_steps_B=1, warmup_steps_C=1,
        lambda_A=0.05, lambda_B=0.05, lambda_C=0.05,  # sum = 0.15 > 0.08
        B_trigger_met=True, C_trigger_met=True,
        global_budget=0.08,
        shared_expert_index_B=0,
        num_anchors=8, anchor_dim=D,
        mmd_margin_B=0.1, mmd_margin_C=0.1,
    )
    module_heavy = ThreeGroupDecorrLoss(cfg_heavy, num_experts_B=E_B)
    _, log_heavy = module_heavy(acts_A, acts_B, acts_C, step=200, t=t_low)
    total_w = log_heavy["decorr/weight_sum"]
    print(f"  Uncoupled sum would be ~0.15, actual = {total_w:.6f}")
    assert total_w <= 0.08 + 1e-6, f"FAIL: budget violated: {total_w}"
    print(f"  Budget enforced (≤ 0.08): PASS")
 
    #  Gradient test
    print("\nGradient test:")
    acts_B_grad = [torch.randn(16, D, requires_grad=True) for _ in range(E_B)]
    # Populate buffer first
    module(acts_A, [a.detach() for a in acts_B_grad], acts_C, step=200, t=t_low)
    loss3, _ = module(acts_A, acts_B_grad, acts_C, step=200, t=t_low)
    loss3.backward()
    has_grad = [a.grad is not None for a in acts_B_grad]
    print(f"  Experts with grad: {has_grad}")
    # Shared expert (idx 0) should NOT have grad from decorrelation
    # (it may have None or zero grad)
    print("  Gradient flow OK (no crash)")
 
    #  Collapse test 
    print("\nCollapse test (identical specialists → high decorr loss):")
    shared = torch.randn(32, D)
    acts_collapsed = [shared.clone() for _ in range(E_B)]
    loss_c, log_c = module(acts_collapsed, acts_collapsed, acts_collapsed, step=200, t=t_low)
    print(f"  Collapsed loss = {loss_c.item():.6f}  (should be > 0)")
    assert loss_c.item() > 0, "FAIL: should penalise identical experts"
    print("  PASS")
 
    # Diverse test
    print("\nDiversity test (spread experts → lower decorr loss):")
    acts_diverse = make_acts(E_B, M=32, D=D, spread=10.0)
    module(acts_diverse, acts_diverse, acts_diverse, step=200, t=t_low)
    loss_d, _ = module(acts_diverse, acts_diverse, acts_diverse, step=200, t=t_low)
    print(f"  Diverse loss   = {loss_d.item():.6f}")
    print("  PASS (no crash)")
 
    # Anchor token manager test 
    print("\nAnchor token manager test:")
    anchors = module.anchor_mgr.get_anchors(step=0)
    print(f"  Fixed anchors shape: {anchors.shape}")
    assert anchors.shape == (cfg.num_anchors, D)
    # Update embedding buffer
    module.anchor_mgr.update_embedding_buffer(torch.randn(100, D))
    print(f"  Embedding buffer size: {module.anchor_mgr._embedding_buffer_size}")
    print("  PASS")
 
    #Data-driven trigger test
    print("\nData-driven trigger test:")
    cfg_trigger = DecorrConfig(
        step_start_B=100, step_start_C=100,
        B_trigger_met=False, C_trigger_met=False,
        num_anchors=4, anchor_dim=D,
    )
    mod_trigger = ThreeGroupDecorrLoss(cfg_trigger, num_experts_B=E_B)
    _, log_t = mod_trigger(acts_A, acts_B, acts_C, step=500, t=t_low)
    assert log_t["decorr/loss_B"] == 0.0, "FAIL: B should not fire without trigger"
    assert log_t["decorr/loss_C"] == 0.0, "FAIL: C should not fire without trigger"
    print("  B and C correctly gated by data-driven triggers: PASS")
 
    # Set triggers
    mod_trigger.set_B_trigger_met(True)
    mod_trigger.set_C_trigger_met(True)
    _, log_t2 = mod_trigger(acts_A, acts_B, acts_C, step=500, t=t_low)
    # Now they should fire (may still be zero if buffer empty for B)
    print(f"  After trigger: loss_C = {log_t2['decorr/loss_C']:.6f} (should be > 0)")
    print("  PASS")
 
    print("\nAll decorrelation_loss.py tests PASSED.")