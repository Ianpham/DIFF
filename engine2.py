import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType
from encode.raw_router import ModalityEncoder, ModalityGateInfo, ModalityEncoderAdapter
from encode.modality_encoder import (
    LidarEmbedding, ResNetImageEncoder, BEVEncoder, MultiCameraEncoder,
    AgentEncoder, IntersectionEncoder, GoalIntentEncoder,
    PedestrianEncoder, BehaviorEncoder, OcclusionEncoder, TrafficControlEncoder,
    HistoryEncoder, FutureTimeEmbedder, TrajectoryEmbedder, TimestepEmbedder,
)
from encode.requirements import EncoderRequirements, StandardRequirements

from datasets.navsim.navsim_utilize.data import (
    NavsimDataset, EnhancedNavsimDataset, PhaseNavsimDataset
)
from model.dydittraj import DiTBlock, DiffRate, DynaLinear
from model.MMRD import MultiModalDecorrelation

from adapters import EncoderAdapter
from diffusion import create_diffusion


# ============================================================
# EvalMetrics
# ============================================================

@dataclass
class EvalMetrics:
    """
    All metrics from one eval pass.
    ADE / FDE  : lower is better (metres).
    PDMS + components : higher is better (0–1).
    Diagnostics: silent-bug catchers.
    """
    ade:              float = 0.0
    fde:              float = 0.0
    pdms:             float = 0.0
    goal_score:       float = 0.0
    collision_score:  float = 0.0
    smoothness_score: float = 0.0
    kinematic_score:  float = 0.0
    pred_xy_std:      float = 0.0   # near 0  → mode collapse
    gt_xy_mean:       float = 0.0   # > 500   → coords in pixels not metres
    gt_xy_std:        float = 0.0
    n_batches_evaluated: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "eval/ADE":              self.ade,
            "eval/FDE":              self.fde,
            "eval/PDMS":             self.pdms,
            "eval/goal_score":       self.goal_score,
            "eval/collision_score":  self.collision_score,
            "eval/smoothness_score": self.smoothness_score,
            "eval/kinematic_score":  self.kinematic_score,
            "eval/pred_xy_std":      self.pred_xy_std,
            "eval/gt_xy_mean":       self.gt_xy_mean,
            "eval/gt_xy_std":        self.gt_xy_std,
        }

    def log(self, logger: logging.Logger, epoch: int):
        logger.info("=" * 70)
        logger.info(f"  EVAL — Epoch {epoch}  ({self.n_batches_evaluated} batches)")
        logger.info("=" * 70)
        logger.info(f"  ADE            : {self.ade:.4f} m   (↓ lower = better)")
        logger.info(f"  FDE            : {self.fde:.4f} m   (↓ lower = better)")
        logger.info(f"  PDMS           : {self.pdms:.4f}    (↑ higher = better)")
        logger.info(f"  goal           : {self.goal_score:.4f}")
        logger.info(f"  collision      : {self.collision_score:.4f}")
        logger.info(f"  smoothness     : {self.smoothness_score:.4f}")
        logger.info(f"  kinematic      : {self.kinematic_score:.4f}")
        collapse_flag = "⚠ possible mode collapse" if self.pred_xy_std < 0.01 else "✓"
        logger.info(f"  pred_xy_std    : {self.pred_xy_std:.4f}  {collapse_flag}")
        coord_flag = "⚠ coords may be pixels not metres" if self.gt_xy_mean > 500 else "✓"
        logger.info(
            f"  gt_xy          : mean={self.gt_xy_mean:.3f}  "
            f"std={self.gt_xy_std:.3f}  {coord_flag}"
        )
        logger.info("=" * 70)


# ============================================================
# Action-space conversion helpers  (paper Eq. 1)
# ============================================================

def waypoints_to_actions(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Convert absolute waypoints to action-space (deltas).
    waypoints: [B, 1, T, C]  (C >= 2, first two channels are x, y)
    returns:   [B, 1, T, C]  where t=0 is s1, t>0 is s_t - s_{t-1}
    Paper Eq. 1:  x_hat_1 = s_1,  x_hat_tau = s_tau - s_{tau-1}
    """
    actions = waypoints.clone()
    actions[:, :, 1:, :] = waypoints[:, :, 1:, :] - waypoints[:, :, :-1, :]
    # actions[:, :, 0, :] stays as s_1 (first waypoint, already relative to ego)
    return actions


def actions_to_waypoints(actions: torch.Tensor) -> torch.Tensor:
    """
    Convert action-space back to absolute waypoints via cumulative sum.
    actions:   [B, 1, T, C]
    returns:   [B, 1, T, C]
    """
    return torch.cumsum(actions, dim=2)


# ============================================================
# TransDiffuserIntegrated
# ============================================================

class TransDiffuserIntegrated(nn.Module):
    """
    Complete TransDiffuser model integrating:
    1. Multi-modal encoder (from DiT)
    2. Decorrelation mechanism - MMRD (paper Algorithm 2)
    3. Diffusion-based trajectory decoder (DiT)
    4. Contract dataset and encoder, compiled with encoder to decorrelation
    """

    def __init__(
        self,
        adapter: EncoderAdapter,
        # DiT parameters
        input_size=64,
        patch_size=5,
        traj_channels=7,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        # decorrelation parameters — applied on fused features (paper Sec 3.4)
        decorr_weights=0.02,
        decorr_similiary='cosine',
        # agent parameters
        max_agents=32,
        future_horizon=8,
        history_length=4,
        # other
        max_timesteps=10,          # paper uses T=10
        trajectory_dropout_prob=0.1,
        learn_sigma=False,         # paper does not learn sigma
        use_modality_specific=True,
        parallel=True,
        # encoder parameters
        use_improved_encoder=True,
        use_modality_gating=True,
        gate_type='soft',
        output_tokens_per_modality=16,
        # agent encoder
        use_agent_encoder=True,
        # action-space mode (paper Eq. 1)
        use_action_space=True,
        # temporal downsample (disabled by default — not in paper)
        use_temporal_downsample=False,
    ):
        super().__init__()

        self.adapter               = adapter
        self.contract              = adapter.contract
        self.hidden_size           = hidden_size
        self.traj_channels         = traj_channels
        self.max_agents            = max_agents
        self.future_horizon        = future_horizon
        self.max_timesteps         = max_timesteps
        self.learn_sigma           = learn_sigma
        self.use_improved_encoder  = use_improved_encoder
        self.use_agent_encoder     = use_agent_encoder
        self.use_action_space      = use_action_space
        self.use_temporal_downsample = use_temporal_downsample
        self.decorr_weight         = decorr_weights

        # dynamic modality configuration
        self.modality_config = self._build_modality_config()
        print("\nDetected modality configuration:")
        for name, info in self.modality_config.items():
            print(f"  {name}: {info['channels']} channels, shape={info['shape']}")

        self.modality_embedders = self._wrap_adapter_encoders()

        if use_improved_encoder:
            self.modality_adapter = ModalityEncoderAdapter(
                modality_embedders       = self.modality_embedders,
                modality_config          = {k: v['channels'] for k, v in self.modality_config.items()},
                hidden_size              = hidden_size,
                num_heads                = num_heads,
                dropout                  = 0.1,
                parallel                 = parallel,
                use_gating               = use_modality_gating,
                gate_type                = gate_type,
                output_tokens_per_modality = output_tokens_per_modality,
            )
            self.context_token_fixed = self.modality_adapter.get_output_token_count()
        else:
            self.context_token_fixed = None

        if use_agent_encoder:
            self.agent_encoder      = AgentEncoder(hidden_size=hidden_size, input_dim=self.traj_channels)
            self.agents_per_sample  = 10
            self.agent_token_count  = self.agents_per_sample * 3
        else:
            self.agent_token_count  = 0

        self.history_encoder_temporal = HistoryEncoder(
            input_size     = traj_channels,
            hidden_size    = hidden_size,
            history_length = history_length,
            num_layers     = 2,
            batch_first    = True,
        )

        # ── Decorrelation on FUSED multi-modal features (paper Algorithm 2) ──
        # This replaces the old agent-level decorrelation.
        # Applied on the concatenated conditional representation before DiT blocks.
        self.decorrelation = MultiModalDecorrelation(
            decorr_weight   = decorr_weights,
            similarity_type = decorr_similiary,
        )

        self.trajectory_embed    = TrajectoryEmbedder(traj_channels, hidden_size, trajectory_dropout_prob)
        self.t_embedder          = TimestepEmbedder(hidden_size)
        self.future_time_embedder= FutureTimeEmbedder(hidden_size, future_horizon)

        self.num_patches  = (input_size // patch_size) ** 2
        self.total_tokens = self._calculate_total_tokens()
        self.pos_embed    = nn.Parameter(
            torch.zeros(1, self.total_tokens, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.output_norm       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        output_channels        = traj_channels * 2 if learn_sigma else traj_channels
        self.output_projection = nn.Linear(hidden_size, output_channels)
        self.output_adaLN      = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        self.initialize_weights()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_modality_config(self) -> Dict[str, Dict[str, Any]]:
        modality_config = {}
        if self.contract.has(FeatureType.LIDAR_BEV):
            spec = self.contract.get_spec(FeatureType.LIDAR_BEV)
            modality_config['lidar'] = {
                'channels':    spec.shape[0] if len(spec.shape) > 0 else 2,
                'shape':       spec.shape,
                'encoder_name':'lidar',
            }
        if self.contract.has(FeatureType.BEV_LABELS):
            modality_config['BEV'] = {
                'channels':    self.contract.bev_channels,
                'shape':       (self.contract.bev_channels, 200, 200),
                'encoder_name':'bev',
            }
        if self.contract.has(FeatureType.CAMERA_IMAGES):
            modality_config['img'] = {
                'channels':    3,
                'shape':       (3, 900, 1600),
                'encoder_name':'camera',
            }
        return modality_config

    def _wrap_adapter_encoders(self) -> nn.ModuleDict:
        wrapped = nn.ModuleDict()
        for modality_name, modality_info in self.modality_config.items():
            channels = modality_info['channels']
            if modality_name == 'lidar':
                wrapped[modality_name] = LidarEmbedding(self.hidden_size)
            elif modality_name == 'BEV':
                wrapped[modality_name] = BEVEncoder(channels, self.hidden_size, patch_size=4)
            elif modality_name == 'img':
                camera_cfg   = self.adapter.config.encoders.get('camera', None)
                camera_names = (
                    camera_cfg.build_params.get('camera_names', ['cam_f0', 'cam_l0', 'cam_r0'])
                    if camera_cfg is not None and hasattr(camera_cfg, 'build_params')
                    else ['cam_f0', 'cam_l0', 'cam_r0']
                )
                wrapped[modality_name] = MultiCameraEncoder(
                    hidden_size    = self.hidden_size,
                    use_pretrained = True,
                    camera_names   = camera_names,
                )
        return wrapped

    def _calculate_total_tokens(self) -> int:
        context_tokens = 0
        for modality_name in self.modality_config:
            if modality_name == 'lidar':
                context_tokens += self.num_patches + 1
            elif modality_name == 'BEV':
                context_tokens += self.num_patches + 1
            elif modality_name == 'img':
                num_cams        = len(self.modality_embedders['img'].camera_names)
                context_tokens += (self.num_patches + 1) * num_cams
            else:
                context_tokens += 1

        if self.use_agent_encoder:
            context_tokens += self.max_agents * (1 + 1 + self.future_horizon)

        trajectory_tokens = self.max_agents * self.future_horizon
        total = context_tokens + trajectory_tokens
        return int(total * 1.5)   # safety buffer

    def initialize_weights(self):
        from model.dydittraj import get_2d_sincos_pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token   = True,
            extra_tokens= self.total_tokens - self.num_patches,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.constant_(self.output_adaLN[-1].weight, 0)
        nn.init.constant_(self.output_adaLN[-1].bias,   0)
        nn.init.constant_(self.output_projection.weight, 0)
        nn.init.constant_(self.output_projection.bias,   0)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_modalities(
        self, adapted_batch, return_gate_info=False
    ) -> Tuple[torch.Tensor, Optional[ModalityGateInfo]]:
        if self.use_improved_encoder:
            return self.modality_adapter(adapted_batch, return_gate_info=return_gate_info)

        # fallback: raw concat with token cap
        MAX_TOKENS = 64
        context = {
            info['encoder_name']: adapted_batch[info['encoder_name']]
            for info in self.modality_config.values()
            if info['encoder_name'] in adapted_batch
        }
        feats = []
        for name, modality_info in self.modality_config.items():
            key = modality_info['encoder_name']
            if key not in context:
                continue
            f = self.modality_embedders[name](context[key])   # (B, T, D)
            if f.shape[1] > MAX_TOKENS:
                f = F.adaptive_avg_pool1d(f.permute(0, 2, 1), MAX_TOKENS).permute(0, 2, 1)
            feats.append(f)

        if feats:
            return torch.cat(feats, dim=1), None

        B      = next(v for v in adapted_batch.values() if isinstance(v, torch.Tensor)).shape[0]
        device = next(v for v in adapted_batch.values() if isinstance(v, torch.Tensor)).device
        dtype  = next(v for v in adapted_batch.values() if isinstance(v, torch.Tensor)).dtype
        return torch.zeros(B, 1, self.hidden_size, device=device, dtype=dtype), None

    # def encode_history_temporal(self, history: torch.Tensor) -> torch.Tensor:
    #     """[B, N, T, C] → [B, N, D]"""
    #     B, N, T, C   = history.shape
    #     flat         = history.reshape(B * N, T, C)
    #     _, (h_n, _)  = self.history_encoder_temporal(flat)
    #     return h_n[-1].reshape(B, N, -1)
    def encode_history_temporal(self, history: torch.Tensor) -> torch.Tensor:
        """[B, N, T, C] → [B, N, D]"""
        return self.history_encoder_temporal(history) 

    # ------------------------------------------------------------------
    # Per-level decorrelation  (paper Algorithm 2, applied cumulatively)
    # ------------------------------------------------------------------
    #
    # Each encoder level computes its own decorrelation on the features
    # active at that level.  The losses accumulate as encoder_level increases:
    #
    #   Level 0: decorr_fused        (raw fused scene features)
    #   Level 1: + decorr_temporal   (history / temporal features)
    #   Level 2: + decorr_interaction(interaction features)
    #   Level 3: + decorr_scene      (scene-level agent features)
    #
    # All use the same Algorithm 2 mechanism but on different representations.

    def _decorr_algorithm2(self, features: torch.Tensor) -> torch.Tensor:
        """
        Paper Algorithm 2 core: decorrelation loss on a feature tensor.
        
        features: either [B, T_tokens, D]  (3-dim, token sequences)
                  or     [B, N, D]          (3-dim, per-agent)
                  or     [B, D]             (2-dim, already pooled)
        
        Returns scalar loss (NOT yet scaled by decorr_weight).
        """
        # Pool to [B, D] if needed
        if features.dim() == 3:
            M = features.mean(dim=1)        # [B, D]
        elif features.dim() == 2:
            M = features                    # [B, D]
        else:
            raise ValueError(f"_decorr_algorithm2 expects 2D or 3D input, got {features.dim()}D")

        B, D = M.shape
        if B < 2 or D < 2:
            return torch.tensor(0.0, device=features.device)

        # Normalize (Algorithm 2 step 1-2)
        M_centered = M - M.mean(dim=0, keepdim=True)
        M_std = M.std(dim=0, keepdim=True) + 1e-8
        M_normed = M_centered / M_std

        # Correlation matrix (Algorithm 2 step 3)
        corr = (M_normed.T @ M_normed) / B     # [D, D]

        # Off-diagonal penalty (Algorithm 2 step 4-5)
        mask = ~torch.eye(D, device=corr.device, dtype=torch.bool)
        off_diag = corr[mask]
        return (off_diag ** 2).mean()

    def compute_cumulative_decorrelation(
        self,
        encoder_level: int,
        fused_features: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
        interaction_features: Optional[torch.Tensor] = None,
        scene_features_agent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute cumulative decorrelation loss across encoder levels.
        
        Returns:
            total_decorr: weighted sum of all active level decorrelations
            breakdown:    dict with per-level losses for logging
        """
        device    = fused_features.device
        breakdown = {}
        total     = torch.tensor(0.0, device=device)

        # Level 0: always — decorrelation on raw fused scene features
        decorr_fused = self._decorr_algorithm2(fused_features)
        breakdown['decorr_fused'] = decorr_fused
        total = total + decorr_fused

        # Level 1: + temporal / history features
        if encoder_level >= 1 and temporal_features is not None:
            decorr_temporal = self._decorr_algorithm2(temporal_features)
            breakdown['decorr_temporal'] = decorr_temporal
            total = total + decorr_temporal

        # Level 2: + interaction features
        if encoder_level >= 2 and interaction_features is not None:
            decorr_interaction = self._decorr_algorithm2(interaction_features)
            breakdown['decorr_interaction'] = decorr_interaction
            total = total + decorr_interaction

        # Level 3: + scene-level agent features
        if encoder_level >= 3 and scene_features_agent is not None:
            decorr_scene = self._decorr_algorithm2(scene_features_agent)
            breakdown['decorr_scene'] = decorr_scene
            total = total + decorr_scene

        # Scale by beta (decorr_weight)
        total_scaled = total * self.decorr_weight
        # Scale breakdown too for consistent logging
        breakdown_scaled = {
            k: v * self.decorr_weight for k, v in breakdown.items()
        }

        return total_scaled, breakdown_scaled

    # ------------------------------------------------------------------
    # Scoring helpers (called by evaluate() in the wrapper)
    # ------------------------------------------------------------------

    def compute_goal_reaching_score(self, trajectory, goal_position):
        final_pos    = trajectory[:, :, -1, :2]
        distances    = torch.norm(final_pos - goal_position, dim=-1)
        avg_distance = distances.mean(dim=1)
        return torch.exp(-avg_distance / 10.0)

    def compute_collision_score(self, trajectory):
        B, N, T, _        = trajectory.shape
        positions         = trajectory[:, :, :, :2]
        collision_penalty = torch.zeros(B, device=trajectory.device)
        safe_distance     = 2.0
        for t in range(T):
            pos_t        = positions[:, :, t, :]
            diff         = pos_t.unsqueeze(2) - pos_t.unsqueeze(1)
            pairwise_dist= torch.norm(diff, dim=-1)
            mask         = ~torch.eye(N, device=trajectory.device, dtype=torch.bool)
            mask         = mask.unsqueeze(0).expand(B, -1, -1)
            collisions   = (pairwise_dist < safe_distance) & mask
            collision_penalty += collisions.float().sum(dim=(1, 2))
        return torch.exp(-collision_penalty / (N * T))

    def compute_kinematic_feasibility_score(self, trajectory):
        B, N, T, _   = trajectory.shape
        velocities   = trajectory[:, :, :, 2:4]
        speeds       = torch.norm(velocities, dim=-1)
        speed_pen    = torch.relu(speeds - 30.0).mean(dim=(1, 2))
        accel        = velocities[:, :, 1:, :] - velocities[:, :, :-1, :]
        accel_pen    = torch.relu(torch.norm(accel, dim=-1) - 8.0).mean(dim=(1, 2))
        positions    = trajectory[:, :, :, :2]
        jerk         = positions[:, :, 2:, :] - 2 * positions[:, :, 1:-1, :] + positions[:, :, :-2, :]
        jerk_pen     = torch.relu(torch.norm(jerk, dim=-1) - 5.0).mean(dim=(1, 2))
        return torch.exp(-(speed_pen + accel_pen + 0.5 * jerk_pen))

    def compute_smoothness_score(self, trajectory):
        positions = trajectory[:, :, :, :2]
        jerk      = positions[:, :, 2:, :] - 2 * positions[:, :, 1:-1, :] + positions[:, :, :-2, :]
        avg_jerk  = torch.norm(jerk, dim=-1).mean(dim=(1, 2))
        return torch.exp(-avg_jerk / 2.0)

    def compute_diversity_score(self, trajectory, other_proposals):
        if other_proposals is None or len(other_proposals) == 0:
            return torch.ones(trajectory.shape[0], device=trajectory.device)
        traj_flat   = trajectory.reshape(trajectory.shape[0], -1)
        sims        = [F.cosine_similarity(traj_flat, p.reshape(traj_flat.shape[0], -1), dim=1)
                       for p in other_proposals]
        max_sim     = torch.stack(sims).max(dim=0)[0]
        return torch.relu(1.0 - max_sim)

    def score_trajectory(self, trajectory, goal_positions, other_proposals=None, weights=None):
        weights = weights or {
            'goal': 1.0, 'collision': 2.0, 'kinematic': 0.8,
            'smoothness': 0.5, 'diversity': 0.3,
        }
        scores = {
            'goal':       self.compute_goal_reaching_score(trajectory, goal_positions),
            'collision':  self.compute_collision_score(trajectory),
            'kinematic':  self.compute_kinematic_feasibility_score(trajectory),
            'smoothness': self.compute_smoothness_score(trajectory),
            'diversity':  self.compute_diversity_score(trajectory, other_proposals),
        }
        total = sum(weights[k] * v for k, v in scores.items())
        scores['total'] = total
        return total, scores

    # ------------------------------------------------------------------
    # Decorrelation helpers (kept for encoder_level >= 3 path)
    # ------------------------------------------------------------------

    def apply_decorrelation(self, level1, level2, level3, B, N):
        l1f = level1.reshape(B * N, -1)
        l2f = level2.reshape(B * N, -1)
        l3f = level3.reshape(B * N, -1)
        decorr_loss, l1d, l2d, l3d = self.decorrelation(l1f, l2f, l3f)
        level1.copy_(l1d.reshape(B, N, -1))
        level2.copy_(l2d.reshape(B, N, -1))
        level3.copy_(l3d.reshape(B, N, -1))
        return decorr_loss

    def compute_block_decorrelation(self, traj_tokens, N):
        B, NT, D     = traj_tokens.shape
        agent_feats  = traj_tokens.reshape(B, N, -1, D).mean(dim=2)
        agent_feats  = F.normalize(agent_feats, dim=-1)
        corr         = torch.bmm(agent_feats, agent_feats.transpose(1, 2))
        identity     = torch.eye(N, device=corr.device).unsqueeze(0)
        return torch.mean((corr - identity) ** 2)

    # ------------------------------------------------------------------
    # Temporal helpers (only used when use_temporal_downsample=True)
    # ------------------------------------------------------------------

    def temporal_downsample(self, trajectory_emb, factor=2):
        B, N, T, D = trajectory_emb.shape
        if factor == 1:
            return trajectory_emb
        x = trajectory_emb.permute(0, 1, 3, 2).reshape(B * N, D, T)
        x = F.adaptive_avg_pool1d(x, T // factor)
        return x.reshape(B, N, D, T // factor).permute(0, 1, 3, 2)

    def temporal_upsample(self, trajectory_emb, target_length=None):
        B, N, T_c, D = trajectory_emb.shape
        target_length = target_length or T_c * 2
        if T_c == target_length:
            return trajectory_emb
        x = trajectory_emb.permute(0, 1, 3, 2).reshape(B * N, D, T_c)
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        return x.reshape(B, N, D, target_length).permute(0, 1, 3, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        adapted_batch,
        noisy_trajectory,
        t,
        encoder_level=0,
        complete_model=True,
        return_gate_info=False,
        use_draft_conditioning=False,
    ):
        B, N, T_future, C = noisy_trajectory.shape
        device            = noisy_trajectory.device

        agent_states  = adapted_batch['agent']
        agent_history = adapted_batch.get(
            'agent_history', torch.randn(B, N, 4, 5, device=device)
        )

        # timestep embedding + noise level
        t_emb       = self.t_embedder(t)
        noise_level = t.float() / self.max_timesteps
        alpha_global= torch.sigmoid(-5 * (noise_level - 0.5)).unsqueeze(-1)
        alpha_local = 1 - alpha_global

        # scene encoding
        scene_features, _ = self.encode_modalities(adapted_batch, return_gate_info)

        # agent tokens
        if self.use_agent_encoder:
            agent_out   = self.agent_encoder(
                agent_states=adapted_batch['agent'][:, :, :5],
                map_features=scene_features,
            )
            agent_tokens = torch.cat([
                agent_out.level1_temporal,
                agent_out.level2_interaction,
                agent_out.level3_scene,
            ], dim=1)
        else:
            agent_tokens = torch.zeros(
                B, 0, self.hidden_size, device=device, dtype=scene_features.dtype
            )

        context_encoded = torch.cat([scene_features, agent_tokens], dim=1)

        # trajectory embedding
        traj_emb = self.trajectory_embed(noisy_trajectory, self.training)
        if use_draft_conditioning and hasattr(self, '_draft_condition'):
            traj_emb = traj_emb + 0.5 * self.trajectory_embed(self._draft_condition, False)
        traj_emb = traj_emb + self.future_time_embedder(B, N)

        # coarse / fine scale — DISABLED by default (not in paper)
        T_active = T_future
        if self.use_temporal_downsample and noise_level.mean() > 0.3:
            traj_emb = self.temporal_downsample(traj_emb, factor=2)
            T_active = T_future // 2

        traj_flat    = traj_emb.reshape(B, N * T_active, -1)
        traj_summary = traj_emb.mean(dim=(1, 2))

        # ── Progressive encoder levels ───────────────────────────────
        # Collect features at each level for both tokens and decorrelation.
        encoder_tokens    = []
        encoder_summaries = []

        # Features that will be passed to cumulative decorrelation
        temporal_feats    = None   # level 1
        interaction_feats = None   # level 2
        scene_agent_feats = None   # level 3

        if encoder_level >= 1:
            if self.use_agent_encoder:
                enc_out = self.agent_encoder(
                    agent_states=agent_states[:, :, :5], map_features=scene_features
                )
                l1 = enc_out.level1_temporal * alpha_local
                encoder_tokens.append(l1)

            hist_enc = self.encode_history_temporal(agent_history) * alpha_global
            encoder_tokens.append(hist_enc)
            encoder_summaries.append(hist_enc.mean(dim=1))
            temporal_feats = hist_enc               # for decorrelation

            if encoder_level >= 2 and self.use_agent_encoder:
                l2 = enc_out.level2_interaction
                encoder_tokens.append(l2)
                interaction_feats = l2              # for decorrelation

                if encoder_level >= 3:
                    l3 = enc_out.level3_scene * alpha_global
                    encoder_tokens.append(l3)
                    scene_agent_feats = l3          # for decorrelation

        # ── Cumulative decorrelation (paper Algorithm 2 per level) ────
        # Level 0: decorr on fused features (always)
        # Level 1: + decorr on temporal features
        # Level 2: + decorr on interaction features
        # Level 3: + decorr on scene-level agent features
        decorr_loss     = torch.tensor(0.0, device=device)
        decorr_breakdown = {}

        if self.training and self.decorr_weight > 0:
            decorr_loss, decorr_breakdown = self.compute_cumulative_decorrelation(
                encoder_level        = encoder_level,
                fused_features       = context_encoded,
                temporal_features    = temporal_feats,
                interaction_features = interaction_feats,
                scene_features_agent = scene_agent_feats,
            )

        # conditioning vector
        c = t_emb + traj_summary
        for s in encoder_summaries:
            c = c + s

        # assemble tokens
        all_tokens = torch.cat([traj_flat, context_encoded, *encoder_tokens], dim=1)

        # positional embedding
        T_actual = all_tokens.shape[1]
        if T_actual <= self.pos_embed.shape[1]:
            all_tokens = all_tokens + self.pos_embed[:, :T_actual, :]
        else:
            pe = F.interpolate(
                self.pos_embed.permute(0, 2, 1), size=T_actual,
                mode='linear', align_corners=False
            ).permute(0, 2, 1)
            all_tokens = all_tokens + pe

        # transformer blocks
        token_select_list      = []
        attn_weight_masks_list = []
        mlp_weight_masks_list  = []

        for block_idx, block in enumerate(self.blocks):
            all_tokens, attn_mask, mlp_mask, token = block(
                all_tokens, c, t_emb, complete_model
            )
            # Block-level decorrelation on trajectory tokens (level >= 2)
            if self.training and encoder_level >= 2 and block_idx % 2 == 0:
                block_decorr = 0.1 * self.compute_block_decorrelation(
                    all_tokens[:, :N * T_active, :], N
                )
                decorr_loss = decorr_loss + block_decorr
            attn_weight_masks_list.append(attn_mask)
            mlp_weight_masks_list.append(mlp_mask)
            token_select_list.append(token)

        # extract trajectory tokens + upsample if needed
        traj_tokens = all_tokens[:, :N * T_active, :]
        if T_active != T_future:
            traj_tokens = self.temporal_upsample(traj_tokens.reshape(B, N, T_active, -1))
        else:
            traj_tokens = traj_tokens.reshape(B, N, T_future, -1)

        # output projection
        shift, scale    = self.output_adaLN(c).chunk(2, dim=1)
        traj_flat_out   = traj_tokens.reshape(B, N * T_future, -1)
        from model.dydittraj import modulate
        traj_flat_out   = modulate(self.output_norm(traj_flat_out), shift, scale)
        predicted_output= self.output_projection(traj_flat_out).reshape(B, N, T_future, -1)

        if self.learn_sigma:
            predicted_noise, predicted_var = predicted_output.chunk(2, dim=-1)
            predicted_output = torch.cat([predicted_noise, predicted_var], dim=-1)
        else:
            predicted_noise = predicted_output

        if not complete_model:
            from model.dydittraj import convert_list_to_tensor
            return (
                predicted_noise,
                decorr_loss,
                convert_list_to_tensor(attn_weight_masks_list),
                convert_list_to_tensor(mlp_weight_masks_list),
                convert_list_to_tensor(token_select_list),
                decorr_breakdown,       # NEW: per-level breakdown
            )
        return predicted_noise, decorr_loss, None, None, None, {}


# ============================================================
# TransDiffuserWithDiffusion
# ============================================================

class TransDiffuserWithDiffusion(nn.Module):
    """
    Wrapper that adds:
      - Training forward pass with noise generation via self.diffusion.q_sample()
      - Eval pass with full reverse-diffusion + ADE / PDMS metrics
      - Multi-proposal sampling pipeline
      - Configurable encoder_level (set from training args)
      - Action-space conversion (paper Eq. 1)
    """

    def __init__(
        self,
        transdiffuser_model,
        diffusion,
        num_proposals=20,
        selection_strategy='top_k_blend',
        top_k_for_blend=5,
        score_weights=None,
        use_refinement=True,
        refinement_noise_level=0.1,
        refirement_steps_ratio=0.3,
        # ── NEW: configurable from training args ──
        encoder_level=0,
    ):
        super().__init__()
        self.model               = transdiffuser_model
        self.diffusion           = diffusion
        self.learn_sigma         = transdiffuser_model.learn_sigma
        self.num_proposals       = num_proposals
        self.selection_strategy  = selection_strategy
        self.top_k_for_blend     = top_k_for_blend
        self.score_weights       = score_weights or {
            'goal': 1.0, 'collision': 2.0, 'kinematic': 0.8,
            'smoothness': 0.5, 'diversity': 0.3,
        }
        self.use_refinement          = use_refinement
        self.refinement_noise_level  = refinement_noise_level
        self.refinement_steps_ratio  = refirement_steps_ratio

        # ── encoder_level: configurable via training args ──
        # 0 = scene features only (no history, no decorr on encoder levels)
        # 1 = + history temporal encoding
        # 2 = + interaction features + block decorrelation
        # 3 = + scene-level features + full 3-level decorrelation
        # NOTE: fused-feature decorrelation (paper Algorithm 2) is ALWAYS active
        #       when decorr_weight > 0, regardless of encoder_level.
        self.encoder_level = encoder_level

    # ------------------------------------------------------------------
    # GT normalisation helper (shared by forward + evaluate)
    # ------------------------------------------------------------------

    def _normalise_gt(self, adapted_batch, B, device):
        """
        Return gt_trajectory normalised to [B, 1, T, traj_ch].
        Pads or trims channels as needed.
        Optionally converts to action-space (paper Eq. 1).
        """
        traj_ch = self.model.traj_channels

        if 'gt_trajectory' in adapted_batch:
            gt = adapted_batch['gt_trajectory']
        else:
            return torch.randn(B, 1, self.model.future_horizon, traj_ch, device=device)

        if gt.dim() == 2:               # [B, T]
            gt = gt.unsqueeze(1).unsqueeze(-1)
        elif gt.dim() == 3:             # [B, T, C]
            gt = gt.unsqueeze(1)        # → [B, 1, T, C]

        C_gt = gt.shape[3]
        if C_gt < traj_ch:
            pad = torch.zeros(B, 1, gt.shape[2], traj_ch - C_gt, device=device)
            gt  = torch.cat([gt, pad], dim=-1)
        elif C_gt > traj_ch:
            gt  = gt[..., :traj_ch]

        gt = gt[:, :1, :, :]          # ego only → [B, 1, T, traj_ch]

        # ── Convert to action-space if enabled (paper Eq. 1) ──
        if self.model.use_action_space:
            gt = waypoints_to_actions(gt)

        return gt

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(self, adapted_batch, goal_position=None, t=None):
        """
        Called as loss_dict = model(adapted_batch) from trainslurm.py.
        All noise generation lives here — do NOT add it in the training loop.
        """
        agent_states = adapted_batch['agent']
        B, N, _      = agent_states.shape
        device       = agent_states.device
        traj_ch      = self.model.traj_channels

        gt_trajectory = self._normalise_gt(adapted_batch, B, device)

        # one-time GT diagnostic on first forward call
        if not hasattr(self, '_gt_logged'):
            self._gt_logged = True
            xy = gt_trajectory[..., :2]
            C_gt = adapted_batch.get('gt_trajectory', gt_trajectory).shape[-1] \
                if 'gt_trajectory' in adapted_batch else traj_ch
            action_note = "action-space (deltas)" if self.model.use_action_space else "absolute waypoints"
            pad_note = (
                f"C_gt={C_gt} → padded to traj_ch={traj_ch} "
                f"(channels {C_gt}–{traj_ch-1} are zero — "
                f"loss on these channels teaches model to predict zeros)"
                if C_gt < traj_ch else f"C_gt={C_gt} → no padding needed"
            )
            print(
                f"[TransDiffuserWithDiffusion] GT diagnostic:\n"
                f"  shape      : {tuple(gt_trajectory.shape)}\n"
                f"  repr       : {action_note}\n"
                f"  xy mean    : {xy.mean():.4f}   xy std : {xy.std():.4f}\n"
                f"  xy min/max : {xy.min():.4f} / {xy.max():.4f}\n"
                f"  encoder_lvl: {self.encoder_level}\n"
                f"  decorr_wt  : {self.model.decorr_weight}\n"
                f"  diff_steps : {self.diffusion.num_timesteps}\n"
                f"  {pad_note}"
            )

        if t is None:
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)

        noise            = torch.randn_like(gt_trajectory)
        noisy_trajectory = self.diffusion.q_sample(gt_trajectory, t, noise)

        predicted_output, decorr_loss, attn_masks, mlp_masks, token_select, decorr_breakdown = self.model(
            adapted_batch    = adapted_batch,
            noisy_trajectory = noisy_trajectory,
            t                = t,
            encoder_level    = self.encoder_level,    # ← now configurable
            complete_model   = False,
        )

        # learn_sigma → output is [B, N, T, traj_ch*2]; take noise half only
        predicted_noise = predicted_output[..., :traj_ch] if self.learn_sigma else predicted_output
        diffusion_loss  = F.mse_loss(predicted_noise, noise)
        total_loss      = diffusion_loss + decorr_loss

        loss_dict = {
            'total_loss':         total_loss,
            'diffusion_loss':     diffusion_loss,
            'decorr_loss':        decorr_loss,
            'attn_channel_usage': attn_masks.mean()    if attn_masks    is not None else 0.0,
            'mlp_channel_usage':  mlp_masks.mean()     if mlp_masks     is not None else 0.0,
            'token_selection':    token_select.mean()  if token_select  is not None else 0.0,
        }

        # ── Per-level decorrelation breakdown for logging ──
        for level_name, level_loss in decorr_breakdown.items():
            loss_dict[level_name] = level_loss.item() if torch.is_tensor(level_loss) else float(level_loss)

        return loss_dict

    # ------------------------------------------------------------------
    # Evaluation  (every N epochs from trainslurm.py)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        adapter,
        device:       torch.device,
        n_batches:    int = 20,
        eval_steps:   int = 10,
        pdms_weights: Optional[Dict[str, float]] = None,
    ) -> EvalMetrics:
        """
        Called as:
            eval_model = model.module if hasattr(model, 'module') else model
            metrics = eval_model.evaluate(dataloader, adapter, device, n_batches=20)
            metrics.log(logger, epoch)

        Reuses _generate_single_proposal() so reverse-diffusion logic is
        written exactly once.

        ADE is evaluation-only — computing it every training step would
        require running the full reverse-diffusion loop each step (10× slower).
        The diffusion loss already implicitly minimises ADE: better noise
        prediction → better x0 reconstruction → lower ADE.
        """
        weights = pdms_weights or {
            'goal': 1.0, 'collision': 2.0, 'kinematic': 0.8, 'smoothness': 0.5,
        }
        self.eval()

        accum = {k: [] for k in [
            'ade', 'fde', 'goal', 'collision', 'smoothness', 'kinematic',
            'pred_xy_std', 'gt_xy_mean', 'gt_xy_std',
        ]}

        batches_done = 0
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            adapted = adapter.adapt_batch(batch)
            adapted = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in adapted.items()
            }

            B       = next(v for v in adapted.values() if isinstance(v, torch.Tensor)).shape[0]
            traj_ch = self.model.traj_channels

            gt_full = self._normalise_gt(adapted, B, device)   # [B, 1, T, traj_ch]
            T       = gt_full.shape[2]

            # ── predicted clean trajectory ────────────────────────────
            x0_pred = self._generate_single_proposal(
                adapted, num_inference_steps=eval_steps, seed=i
            )
            # _generate_single_proposal shape: [B, N, future_horizon, traj_ch]
            # clip to ego + correct T + traj_ch to match gt_full
            x0_pred = x0_pred[:, :1, :T, :traj_ch]             # [B, 1, T, traj_ch]

            # ── Convert back to waypoint-space for ADE/FDE ────────────
            if self.model.use_action_space:
                gt_waypoints   = actions_to_waypoints(gt_full)
                pred_waypoints = actions_to_waypoints(x0_pred)
            else:
                gt_waypoints   = gt_full
                pred_waypoints = x0_pred

            # ── ADE / FDE — x,y only ─────────────────────────────────
            pred_xy     = pred_waypoints[..., :2]               # [B, 1, T, 2]
            gt_xy       = gt_waypoints[..., :2]                 # [B, 1, T, 2]
            l2          = torch.norm(pred_xy - gt_xy, dim=-1)   # [B, 1, T]
            accum['ade'].append(l2.mean().item())
            accum['fde'].append(l2[:, :, -1].mean().item())

            # ── diagnostics ──────────────────────────────────────────
            accum['pred_xy_std'].append(pred_xy.std().item())
            accum['gt_xy_mean'].append(gt_xy.mean().item())
            accum['gt_xy_std'].append(gt_xy.std().item())

            # ── PDMS component scores (on waypoint-space) ─────────────
            goal_pos = gt_waypoints[:, 0, -1, :2]               # [B, 2]
            accum['goal'].append(
                self.model.compute_goal_reaching_score(pred_waypoints, goal_pos).mean().item()
            )
            accum['collision'].append(
                self.model.compute_collision_score(pred_waypoints).mean().item()
            )
            accum['smoothness'].append(
                self.model.compute_smoothness_score(pred_waypoints).mean().item()
            )
            accum['kinematic'].append(
                self.model.compute_kinematic_feasibility_score(pred_waypoints).mean().item()
            )
            batches_done = i + 1

        self.train()

        def avg(lst): return sum(lst) / len(lst) if lst else 0.0

        g = avg(accum['goal'])
        c = avg(accum['collision'])
        s = avg(accum['smoothness'])
        k = avg(accum['kinematic'])
        pdms = (
            weights['goal'] * g + weights['collision'] * c +
            weights['kinematic'] * k + weights['smoothness'] * s
        ) / sum(weights.values())

        return EvalMetrics(
            ade               = avg(accum['ade']),
            fde               = avg(accum['fde']),
            pdms              = pdms,
            goal_score        = g,
            collision_score   = c,
            smoothness_score  = s,
            kinematic_score   = k,
            pred_xy_std       = avg(accum['pred_xy_std']),
            gt_xy_mean        = avg(accum['gt_xy_mean']),
            gt_xy_std         = avg(accum['gt_xy_std']),
            n_batches_evaluated = batches_done,
        )

    # ------------------------------------------------------------------
    # Sampling pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        adapted_batch,
        goal_positions=None,
        num_inference_steps=50,
        return_all_proposals=False,
        return_scores=False,
    ):
        agent_states = adapted_batch['agent']
        B, N, _      = agent_states.shape
        device       = agent_states.device

        if goal_positions is None:
            goal_positions = agent_states[:, :, :2]

        all_proposals   = []
        all_score_dicts = []

        print(f"\n=== Stage 1: Generating {self.num_proposals} proposals ===")
        for k in range(self.num_proposals):
            proposal = self._generate_single_proposal(adapted_batch, num_inference_steps, seed=k)

            # Convert to waypoints for scoring if using action-space
            if self.model.use_action_space:
                proposal_wp = actions_to_waypoints(proposal)
            else:
                proposal_wp = proposal

            total_score, score_dict = self.model.score_trajectory(
                trajectory     = proposal_wp,
                goal_positions = goal_positions,
                other_proposals= [actions_to_waypoints(p) if self.model.use_action_space else p
                                  for p in all_proposals],
                weights        = self.score_weights,
            )
            all_proposals.append(proposal)
            all_score_dicts.append(score_dict)
            if (k + 1) % 5 == 0:
                print(f"  {k+1}/{self.num_proposals}  avg score={total_score.mean():.4f}")

        all_proposals = torch.stack(all_proposals, dim=0)
        all_scores    = torch.stack([sd['total'] for sd in all_score_dicts], dim=0)

        print(f"\n=== Stage 2a: Selection  (strategy={self.selection_strategy}) ===")
        if self.selection_strategy == 'best':
            selected = self._select_best(all_proposals, all_scores)
        elif self.selection_strategy == 'weighted_blend':
            selected = self._weighted_blend_all(all_proposals, all_scores)
        elif self.selection_strategy == 'top_k_blend':
            selected = self._top_k_blend(all_proposals, all_scores, k=self.top_k_for_blend)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

        if self.use_refinement:
            print("\n=== Stage 2b: Refinement ===")
            final = self._refine_trajectory(selected, adapted_batch, num_inference_steps)
        else:
            final = selected

        # Convert back to waypoints for output
        if self.model.use_action_space:
            final = actions_to_waypoints(final)

        print("\n=== Sampling complete ===\n")

        if return_all_proposals and return_scores:
            return final, all_proposals, all_scores
        if return_all_proposals:
            return final, all_proposals
        if return_scores:
            return final, all_scores
        return final

    @torch.no_grad()
    def _generate_single_proposal(self, adapted_batch, num_inference_steps, seed=None):
        """
        Full reverse-diffusion sampling — used by both sample() and evaluate().
        Shape is derived from model config so it is always correct.
        """
        agent_states = adapted_batch['agent']
        B, N, _      = agent_states.shape
        device       = agent_states.device

        if seed is not None:
            torch.manual_seed(seed)

        # initialise from pure noise — shape matches model config exactly
        trajectory = torch.randn(
            B, 1, self.model.future_horizon, self.model.traj_channels, device=device
        )

        def model_fn(x, t):
            pred, _, _, _, _, _ = self.model(
                adapted_batch    = adapted_batch,
                noisy_trajectory = x,
                t                = t,
                encoder_level    = self.encoder_level,    # ← pass through
                complete_model   = True,
            )
            return pred

        for t_idx in reversed(range(num_inference_steps)):
            t   = torch.full((B,), t_idx, device=device, dtype=torch.long)
            out = self.diffusion.p_sample(model_fn, trajectory, t, clip_denoised=True)
            trajectory = out['sample'] if isinstance(out, dict) else out

        return trajectory   # [B, 1, future_horizon, traj_ch]

    @torch.no_grad()
    def _refine_trajectory(self, trajectory_draft, adapted_batch, num_inference_steps):
        B, N, T, C  = trajectory_draft.shape
        device      = trajectory_draft.device
        ref_steps   = int(self.refinement_steps_ratio * num_inference_steps)
        t_ref       = torch.full((B,), ref_steps, device=device, dtype=torch.long)
        noisy_draft = self.diffusion.q_sample(trajectory_draft, t_ref, torch.randn_like(trajectory_draft))
        self.model.register_buffer('_draft_condition', trajectory_draft)

        def model_fn(x, t):
            pred, _, _, _, _, _ = self.model(
                adapted_batch          = adapted_batch,
                noisy_trajectory       = x,
                t                      = t,
                encoder_level          = self.encoder_level,
                complete_model         = True,
                use_draft_conditioning = True,
            )
            return pred

        refined = noisy_draft
        for t_idx in reversed(range(ref_steps)):
            t       = torch.full((B,), t_idx, device=device, dtype=torch.long)
            out     = self.diffusion.p_sample(model_fn, refined, t, clip_denoised=True)
            refined = out['sample'] if isinstance(out, dict) else out

        delattr(self.model, '_draft_condition')
        return refined

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_best(self, proposals, scores):
        K, B, N, T, C = proposals.shape
        best_k        = scores.argmax(dim=0)
        return torch.stack([proposals[best_k[b], b] for b in range(B)])

    def _weighted_blend_all(self, proposals, scores):
        w = torch.softmax(scores, dim=0).view(scores.shape[0], scores.shape[1], 1, 1, 1)
        return (proposals * w).sum(dim=0)

    def _top_k_blend(self, proposals, scores, k=5):
        K, B, N, T, C   = proposals.shape
        topk_s, topk_i  = torch.topk(scores, k=k, dim=0)
        blended = []
        for b in range(B):
            top_p  = proposals[topk_i[:, b], b]
            w      = torch.softmax(topk_s[:, b], dim=0).view(k, 1, 1, 1)
            blended.append((top_p * w).sum(dim=0))
        return torch.stack(blended)


# ============================================================
# Factory function
# ============================================================

def create_transdiffuser_adapted(
    adapter,
    hidden_size=256,
    depth=4,
    num_heads=4,
    # ── NEW: all configurable from training args ──
    decorr_weights=0.02,       # paper β=0.02
    max_agents=1,
    future_horizon=8,
    num_proposals=20,
    selection_strategy='top_k_blend',
    top_k_for_blend=5,
    use_refinement=True,
    # diffusion config
    diffusion_steps=10,        # paper T=10
    noise_schedule='cosine',   # cosine is better for low-dim trajectory data
    # encoder level
    encoder_level=1,
    # action-space
    use_action_space=True,     # paper Eq. 1
    # temporal downsample
    use_temporal_downsample=False,
):
    base_model = TransDiffuserIntegrated(
        adapter             = adapter,
        input_size          = 64,
        patch_size          = 4,
        traj_channels       = 5,          # NavSim GT is (x, y, vx, vy, heading)
        hidden_size         = hidden_size,
        depth               = depth,
        num_heads           = num_heads,
        decorr_weights      = decorr_weights,
        max_agents          = max_agents,
        future_horizon      = future_horizon,
        history_length      = 4,
        max_timesteps       = diffusion_steps,   # match diffusion T
        learn_sigma         = False,
        use_agent_encoder   = False,      # disable for shape debugging
        use_improved_encoder= False,      # raw concat, fewer moving parts
        use_modality_gating = False,
        use_action_space    = use_action_space,
        use_temporal_downsample = use_temporal_downsample,
    )

    # ── Diffusion: T=10 with cosine schedule (paper) ──
    # timestep_respacing="" means use all timesteps;
    # timestep_respacing="10" means respace to 10 steps.
    # Since we set num_timesteps=diffusion_steps via the schedule,
    # we pass the step count as the base schedule length.
    diffusion = create_diffusion(
        timestep_respacing = str(diffusion_steps),
        learn_sigma        = False,
        noise_schedule     = noise_schedule,
    )

    return TransDiffuserWithDiffusion(
        transdiffuser_model = base_model,
        diffusion           = diffusion,
        num_proposals       = num_proposals,
        selection_strategy  = selection_strategy,
        top_k_for_blend     = top_k_for_blend,
        use_refinement      = use_refinement,
        encoder_level       = encoder_level,
    )