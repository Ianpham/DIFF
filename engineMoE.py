
"""
engine.py — TransDiffuser with MoE Router Integration
======================================================
 
Changes from previous version:
    1. IMPORTS: Removed AgentEncoder, ModalityEncoder/ModalityGateInfo from
       hierachy_encoder/raw_router. Added AgentStateEncoder, MoE router.
    2. __init__: Removed self.improved_modality_encoder (with gating params),
       self.agent_encoder. Added self.moe_router, self.agent_state_encoder,
       individual Group C context encoders. Kept modality_embedders as
       pure Group A encoders.
    3. REMOVED: encode_modalities() method (gating logic),
       encode_history_temporal() (moved to HistoryEncoder usage),
       progressive encoder levels (L1/L2/L3 from AgentEncoder).
    4. forward(): Now calls pure encoders -> MoE router -> DiT blocks.
       The MoE router handles all cross-group attention, routing, and
       decorrelation instead of the old progressive encoder + MMRD.
    5. KEPT: _build_modality_config(), _wrap_adapter_encoders() (now produce
       pure Group A encoders), trajectory embed, DiT blocks, output heads,
       temporal_downsample/upsample, TransDiffuserWithDiffusion wrapper,
       create_transdiffuser_adapted factory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, List
from dataclasses import dataclass
 
from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType
 
from encode.modality_encoder import (
    LidarEmbedding, ResNetImageEncoder, BEVEncoder, MultiCameraEncoder,
    AgentstateEncoder,
    IntersectionEncoder, GoalIntentEncoder,
    PedestrianEncoder, BehaviorEncoder, OcclusionEncoder, TrafficControlEncoder,
    HistoryEncoder, FutureTimeEmbedder, TrajectoryEmbedder, TimestepEmbedder,
)
from encode.requirements import EncoderRequirements, StandardRequirements
 
# ── MoE backbone (replaces draft encode.moe_router) ───────────────────────────
from MoE import (
    MoEBlockConfig,
    StackedMoEBlocks,
    StackedMoEOutput,
    build_moe_backbone,
    PhaseTracker,
    DecorrConfig,
    TTYPE_BEV_STRUCT,
    TTYPE_UNKNOWN,
    NUM_C_TYPES,
)
# ──────────────────────────────────────────────────────────────────────────────
 
from datasets.navsim.navsim_utilize.data import NavsimDataset, EnhancedNavsimDataset, PhaseNavsimDataset
from model.dydittraj import DiTBlock, DiffRate, DynaLinear
from model.MMRD import MultiModalDecorrelation
from adapters import EncoderAdapter
from diffusion import create_diffusion

# Scene tokenizer

class SceneTokenizer(nn.Module):
    """Partitions heterogeneous scene features into Group A / B / C.

    Handles dimension alignment (projection to MoE embed_dim) and
    constructs the spatial_xyz tensor for Group A and the token_types
    tensor for Group C.

    Args:
        embed_dim:         shared MoE embedding dimension
        num_gaussian_tokens: N_gauss (from pooling stage)
        num_lidar_tokens:  N_lidar  (fixed-length lidar summary tokens)
        num_bev_tokens:    N_bev    (BEV feature map flattened tokens)
        num_img_tokens:    N_img    (image feature summary tokens)
        dim_gauss:         input dim of pooled Gaussian tokens
        dim_lidar:         input dim of f_lidar
        dim_bev:           input dim of f_bev
        dim_img:           input dim of f_img
        dim_motion:        input dim of motion (history/ego) tokens
    """

    def __init__(
        self,
        embed_dim: int,
        num_gaussian_tokens: int,
        num_lidar_tokens: int,
        num_bev_tokens: int,
        num_img_tokens: int,
        dim_gauss: int,
        dim_lidar: int,
        dim_bev: int,
        dim_img: int,
        dim_motion: int,
    ):
        super().__init__()
        D = embed_dim
        self.num_gaussian_tokens = num_gaussian_tokens
        self.num_lidar_tokens    = num_lidar_tokens
        self.num_bev_tokens      = num_bev_tokens
        self.num_img_tokens      = num_img_tokens

        # Group A projections
        self.proj_gauss = nn.Linear(dim_gauss,  D) if dim_gauss  != D else nn.Identity()
        self.proj_lidar = nn.Linear(dim_lidar,  D) if dim_lidar  != D else nn.Identity()

        # Group B projections (motion: history + ego)
        self.proj_motion = nn.Linear(dim_motion, D) if dim_motion != D else nn.Identity()

        # Group C projections
        self.proj_bev   = nn.Linear(dim_bev,    D) if dim_bev    != D else nn.Identity()
        self.proj_img   = nn.Linear(dim_img,    D) if dim_img    != D else nn.Identity()

        # Learnable lidar positional queries (N_lidar, 3) — proxy spatial coords
        # since flat lidar summary tokens don't have explicit xyz.
        self.lidar_xyz = nn.Parameter(torch.zeros(num_lidar_tokens, 3))

    def forward(
        self,
        gaussian_tokens: torch.Tensor,   # (B, N_gauss, dim_gauss)
        gaussian_means:  torch.Tensor,   # (B, N_gauss, 3) Gaussian mean coords
        f_lidar:         torch.Tensor,   # (B, N_lidar, dim_lidar)
        f_bev:           torch.Tensor,   # (B, N_bev,   dim_bev)
        f_img:           torch.Tensor,   # (B, N_img,   dim_img)
        emb_history:     torch.Tensor,   # (B, 1,       dim_motion)
        emb_ego:         torch.Tensor,   # (B, 1,       dim_motion)
    ) -> Tuple[
        torch.Tensor,   # tokens_A      (B, N_A, D)
        torch.Tensor,   # tokens_B      (B, N_B, D)
        torch.Tensor,   # tokens_C      (B, N_C, D)
        torch.Tensor,   # spatial_xyz   (B, N_A, 3)
        torch.Tensor,   # token_types_C (B, N_C) long
    ]:
        B = gaussian_tokens.shape[0]
        D = self.proj_gauss(gaussian_tokens).shape[-1] if isinstance(self.proj_gauss, nn.Linear) else gaussian_tokens.shape[-1]

        # Group A: Gaussian tokens + lidar summary 
        g_proj = self.proj_gauss(gaussian_tokens)   # (B, N_gauss, D)
        l_proj = self.proj_lidar(f_lidar)            # (B, N_lidar, D)
        tokens_A = torch.cat([g_proj, l_proj], dim=1)  # (B, N_A, D)

        # Spatial coords for Group A gate (Gaussian means + learnable lidar xyz)
        lidar_xyz_expand = self.lidar_xyz.unsqueeze(0).expand(B, -1, -1)  # (B, N_lidar, 3)
        spatial_xyz = torch.cat([gaussian_means, lidar_xyz_expand], dim=1)  # (B, N_A, 3)

        # Group B: history + ego state
        h_proj  = self.proj_motion(emb_history)   # (B, 1, D)
        e_proj  = self.proj_motion(emb_ego)        # (B, 1, D)
        tokens_B = torch.cat([h_proj, e_proj], dim=1)  # (B, 2, D)

        # Group C: BEV structural + image context
        bev_proj = self.proj_bev(f_bev)   # (B, N_bev, D)
        img_proj = self.proj_img(f_img)   # (B, N_img, D)
        tokens_C = torch.cat([bev_proj, img_proj], dim=1)  # (B, N_C, D)

        # Token type IDs for Group C structural router
        N_bev = bev_proj.shape[1]
        N_img = img_proj.shape[1]
        bev_types = torch.full((B, N_bev), TTYPE_BEV_STRUCT, dtype=torch.long, device=f_bev.device)
        img_types = torch.full((B, N_img), TTYPE_UNKNOWN,    dtype=torch.long, device=f_img.device)
        token_types_C = torch.cat([bev_types, img_types], dim=1)  # (B, N_C)

        return tokens_A, tokens_B, tokens_C, spatial_xyz, token_types_C
    

class TransDiffuserIntegrated(nn.Module):
    """
    TransDiffuser with full 3-group MoE backbone.
 
    Architecture:
        1. Pure encoders produce tokens per group (A, B, C)
        2. SceneTokenizer partitions and projects tokens into MoE groups
        3. StackedMoEBlocks: directed cross-attn A->C->B + per-group expert routing
        4. DiT transformer blocks process all tokens jointly
        5. Output head predicts trajectory noise
    """

    def __init__(
        self,
        # adapter
        adapter: EncoderAdapter,
        # DiT parameters
        input_size =64,
        patch_size = 5,
        traj_channels = 7,
        hidden_size = 768,
        depth = 12,
        num_heads = 12,
        mlp_ratio = 4.0,

        #decorrelation parameters (kept for MMRD on trajectory token)
        decorr_weights = 0.1,
        decorr_similiary = 'cosine',

        # agent parameters
        max_agents = 32,
        future_horizon = 8,
        history_length = 4,

        # other
        max_timesteps = 50,
        trajectory_dropout_prob = 0.1,
        learn_sigma = True,

        # === REMOVED parameters ===
        # use_improved_encoder, use_modality_gating, gate_type,
        # output_tokens_per_modality, parallel
        # These are now handled internally by the MoE router

        # New: MoE Router parameters
        moe_num_blocks: int = 4,
        moe_num_experts: int = 4,           # experts per group (A and B)
        moe_num_lidar_tokens: int = 64,     # fixed lidar token count after pooling
        moe_num_bev_tokens: int = 128,      # fixed BEV token count after pooling
        moe_num_img_tokens: int = 64,       # fixed image token count after pooling
        moe_step_start_A: int = 2_000,      # decorr warm-up steps for Group A
        moe_step_start_B: int = 6_000,
        moe_step_start_C: int = 10_000,
        stopgrad_relax_step: Optional[int] = None,  # step to relax C->B stop-grad
   ):
        super().__init__()
        self.adapter = adapter
        self.contract = adapter.contract
        self.hidden_size = hidden_size
        self.traj_channels = traj_channels
        self.max_agents = max_agents
        self.future_horizon = future_horizon
        self.max_timesteps = max_timesteps
        self.learn_sigma = learn_sigma
        self.stopgrad_relax_step = stopgrad_relax_step
        self._global_step: int = 0

        D = hidden_size

        # Group A: pure modality encoders (no gating)
        self.modality_config = self._build_modality_config()
        self.modality_embedders = self._wrap_adapter_encoders()
        self.group_a_modality_names = list(self.modality_config.keys())
 
        # Group B: agent state + history (pure encoders, no routing) ────
        self.agent_state_encoder = AgentStateEncoder(state_dim=5, hidden_size=D)
        self.history_encoder = HistoryEncoder(
            input_size=traj_channels,
            hidden_size=D,
            history_length=history_length,
            num_layers=2,
            batch_first=True,
        )
        self.history_attn_pool = nn.Sequential(
            nn.Linear(D, 1),
            nn.Softmax(dim=1),
        )
        self.intersection_encoder = BehaviorEncoder(hidden_size=D)
        self.pedestrian_ctx_encoder   = PedestrianEncoder(hidden_size=D)

        # group C: context encoder 
        self.intersection_encoder     = IntersectionEncoder(hidden_size=D)
        self.goal_encoder             = GoalIntentEncoder(hidden_size=D)
        self.traffic_control_encoder  = TrafficControlEncoder(hidden_size=D)
        self.occlusion_encoder        = OcclusionEncoder(hidden_size=D)
        
        self.group_c_semantic_names   = [
            'intersection', 'goal', 'traffic_control', 'occlusion', #pedestrian_ctx'
        ]

        self.lidar_pooler = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads, batch_first=True
        )
        self.lidar_pool_queries = nn.Parameter(
            torch.randn(1, moe_num_lidar_tokens, D) * 0.02
        )
        self.bev_pooler = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads, batch_first=True
        )
        self.bev_pool_queries = nn.Parameter(
            torch.randn(1, moe_num_bev_tokens, D) * 0.02
        )
        self.img_pooler = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads, batch_first=True
        )
        self.img_pool_queries = nn.Parameter(
            torch.randn(1, moe_num_img_tokens, D) * 0.02
        )

        # MoE: SceneTOkenizer
        # N_gauss_tokens = 1 per agent (state+history fused in _encode_group_b)
        # but the tokenizer's Group A is lidar_tokens only (sensory).
        # Group B comes from agent encoding above.
        # We treat the lidar pool output as "gaussian_tokens" for the tokenizer
        # (same slot in the token stream — raw dense sensory).

        N_A = moe_num_lidar_tokens          # sensory: lidar summary tokens
        N_B = max_agents                     # interaction: one token per agent
        N_C = moe_num_bev_tokens + moe_num_img_tokens  # map/context

        self.scene_tokenizer = SceneTokenizer(
            embed_dim=D,
            num_gaussian_tokens=moe_num_bev_tokens, # resuse lidar as "dense sensory"
            num_lidar_tokens= 0, # already folded above
            num_bev_tokens=moe_num_bev_tokens,
            num_img_tokens=moe_num_bev_tokens,
            dim_gauss=D, 
            dim_lidar=D,
            dim_bev=D,
            dim_img=D,
            dim_motion=D,
        )

        # MoE: StackedMoEBlocks
        moe_cfg = MoEBlockConfig(
            embed_dim=D,
            num_tokens_A=N_A,
            num_tokens_B=N_B,
            num_tokens_C=N_C,
            num_experts_A=moe_num_experts,
            num_experts_B=moe_num_experts,
            num_experts_C=max(moe_num_experts, NUM_C_TYPES),
            num_attn_heads=num_heads,
            expert_ff_mult=4,
            dim_A_in=D, dim_B_in=D, dim_C_in=D,
            step_start_A=moe_step_start_A,
            step_start_B=moe_step_start_B,
            step_start_C=moe_step_start_C,
            T_max=max_timesteps * 100,            
        )

        self.moe_backbone: StackedMoEBlocks = build_moe_backbone(
            moe_cfg, num_blocks=moe_num_blocks
        )

        # phase tracker for monitoring and stop-grad scheduling
        self.moe_phase_tracker = PhaseTracker(moe_cfg.to_decorr_config())

        # tracjectory token decoorelation 
        self.decorrelation = MultiModalDecorrelation(
            decorr_weight=decorr_weights,
            similarity_type= decorr_similiary,
        )

        # trajectory and timestep embedding
        self.trajectory_embed = TrajectoryEmbedder(traj_channels, D, trajectory_dropout_prob)

        self.t_embeder = TimestepEmbedder(D)
        self.future_time_embeder = FutureTimeEmbedder(D, future_horizon)

        # positional embedding
        self.num_patches = (input_size // patch_size) ** 2
        self.total_tokens = self._calculate_total_tokens(N_A, N_B, N_C)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.total_tokens, D), requires_grad = False
        )

        # DiT transformer blocks
        self.blocks = nn.ModuleList(
            DiTBlock(D, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        )

        # output heads
        self.output_norm = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        output_channels = traj_channels * 2 if learn_sigma else traj_channels

        self.output_projection = nn.Linear(D, output_channels)
        self.output_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D, 2 * D, bias = True)
        )

        self.initialize_weights()


    # config helper 
    def _build_modality_config(self) -> Dict[str, Dict[str, Any]]:
        modality_config = {}
        if self.contract.has(FeatureType.LIDAR_BEV):
            spec = self.contract.get_spec(FeatureType.LIDAR_BEV)
            modality_config['lidar'] = {
                'channels': spec.shape[0] if len(spec.shape) > 0 else 2,
                'shape': spec.shape, 'encoder_name': 'lidar',
            }
        if self.contract.has(FeatureType.BEV_LABELS):
            modality_config['bev'] = {
                'channels': self.contract.bev_channels,
                'shape': (self.contract.bev_channels, 200, 200), 'encoder_name': 'bev',
            }
        if self.contract.has(FeatureType.CAMERA_IMAGES):
            modality_config['img'] = {
                'channels': 3, 'shape': (3, 900, 1600), 'encoder_name': 'camera',
            }
        return modality_config
 
    def _wrap_adapter_encoders(self) -> nn.ModuleDict:
        wrapped = nn.ModuleDict()
        for name, info in self.modality_config.items():
            ch = info['channels']
            if name == 'lidar':
                wrapped[name] = LidarEmbedding(self.hidden_size)
            elif name == 'bev':
                wrapped[name] = BEVEncoder(hidden_size=self.hidden_size, in_channels=ch)
            elif name == 'img':
                wrapped[name] = ResNetImageEncoder(hidden_size=self.hidden_size)
        return wrapped
    
    def _calculate_total_tokens(self, N_A: int, N_B: int, N_C: int) -> int:
        """Token layout: [trajectory | group_A | group_B | group_C]"""
        traj_tokens = self.max_agents * self.future_horizon
        return traj_tokens + N_A + N_B + N_C
 
    def initialize_weights(self):
        from model.dydittraj import get_2d_sincos_pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True,
            extra_tokens=self.total_tokens - self.num_patches,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.constant_(self.output_adaLN[-1].weight, 0)
        nn.init.constant_(self.output_adaLN[-1].bias,  0)
        nn.init.constant_(self.output_projection.weight, 0)
        nn.init.constant_(self.output_projection.bias,   0)

    # group encoders

    def _encode_group_a(
            self, adapted_batch: Dict[str, Any]
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """Group A: raw sensory tokens. Return (tokens [B, T_a, D], types [B, T_a])"""
        all_tokens, all_types = [], []
        for idx, name in enumerate(self.group_a_modality_names):
            enc_name = self.modality_config[name]['encoder_name']
            if enc_name in adapted_batch and name in self.modality_embedders:
                feat = self.modality_embedders[name](adapted_batch[enc_name]) # [B, T_m, D]
                B, T_m, _ = feat.shape
                all_tokens.append(feat)
                all_types.append(torch.full((B, T_m), idx, device = feat.device, dtype = torch.long))

            if not all_tokens:
                B = list(adapted_batch.values())[0].shape[0]
                dev = list(adapted_batch.values())[0].device
                return torch.zeros(B, 1, self.hidden_size, device = dev), \
                        torch.zeros(B, 1, device = dev, dtype=torch.long)
            
            return torch.cat(all_tokens, dim = 1), torch.cat(all_types, dim = 1)
        
    def _encode_group_b(
            self, 
            agent_states: torch.Tensor, 
            agent_history: torch.Tensor, 
            agent_interaction: torch.Tensor,
            ped_interaction: torch.Tensor,

    )-> torch.Tensor:
        states_tokens = self.agent_state_encoder(agent_states) # (B, N, D)
        interaction_tokens = self.interaction_encoder(agent_interaction) # (B, N, D) # this is place holder for us after have interaction encoder
        B,N, T_hist, C = agent_history.shape

        history_seq = self.history_encoder(
            agent_history.reshape(B*N, T_hist, C)
        )                                               # (B*N, T_hist, D)

        attn_w = self.history_attn_pool(history_seq)    # [B*N, T_hist, 1]
        hist_vec = (history_seq * attn_w).sum(dim = 1).reshape(B, N, -1) # [B, N, D]
        return states_tokens + interaction_tokens + hist_vec
    
    def _encode_group_c(
        self, adapted_batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group C: context tokens.  Returns (tokens [B,T_c,D], types [B,T_c])."""
        B   = list(adapted_batch.values())[0].shape[0]
        dev = list(adapted_batch.values())[0].device
        encoder_map = {
            'intersection':   (self.intersection_encoder,    'intersection_features'),
            'goal':           (self.goal_encoder,             'goal_features'),
            'traffic_control':(self.traffic_control_encoder, 'traffic_control_features'),
            'occlusion':      (self.occlusion_encoder,        'occlusion_features'),
            'pedestrian_ctx': (self.pedestrian_ctx_encoder,  'pedestrian_features'), # consider we can remove pedestrain here, 
        }
        all_tokens, all_types = [], []
        for idx, sem_name in enumerate(self.group_c_semantic_names):
            enc, key = encoder_map[sem_name]
            if key in adapted_batch:
                feat = enc(adapted_batch[key])       # [B, K, D]
                B_f, K, _ = feat.shape
                all_tokens.append(feat)
                all_types.append(torch.full((B_f, K), idx, device=dev, dtype=torch.long))
        if not all_tokens:
            return torch.zeros(B, 1, self.hidden_size, device=dev), \
                   torch.zeros(B, 1, device=dev, dtype=torch.long)
        return torch.cat(all_tokens, dim=1), torch.cat(all_types, dim=1)
    
    # MoE context builder 
    def _pool_to_fixed(
            self,
            pooler: nn.MultiheadAttention,
            queries: nn.Parameter,
            src: torch.Tensor,          # (B, N_src, D)
    )-> torch.Tensor: 
        """Cross-attention compress src -> fixed N_query token count."""
        B = src.shape[0]
        q = queries.expand(B,-1, -1)
        out, _ = pooler(query = q, key = src, value = src)
        return out
    
    def _build_moe_context(
            self,
            group_a_raw: torch.Tensor,      # (B, T_a, D) - from _encode_group_a
            group_a_types: torch.Tensor,    # (B, T_a)
            group_b_tokens: torch.Tensor,   # (B, N_agents, D) - from _encode_groupb_b
            group_c_raw: torch.Tensor,      # (B, T_c, D) - from _encode_group_c
            group_c_types: torch.Tensor,    # (B, T_c)
            t: torch.Tensor,                # (B,) diffusion timesteps
    )-> Tuple[StackedMoEOutput, torch.Tensor]:
        """
        Pool raw group tokens to fixed sizes -> SceneTokenizer -> StackedMoEBlocks.
 
        Returns:
            moe_out:   StackedMoEOutput  (.tokens_A, .tokens_B, .tokens_C, .aux_loss)
            aux_loss:  scalar MoE auxiliary loss (decorr + capacity + ortho)
        """

        B = group_a_raw.shape[0]
        D = self.hidden_size

        # step 1: pool group A sensor features -> fixed lidar token count
        # we pool the concantenated raw sensory tokens as a single "dense sensory" set.
        # SceneTokenizer treats these as gassian_tokens (no separate lidar pass) or
        # do we have to make it become gaussian (I think we need to).
        pooled_sensory = self._pool_to_fixed(
            self.lidar_pooler, self.lidar_pool_queries, group_a_raw
        )   # (B, moe_num_lidar_tokens, D)

        # we fill it later
        # Placeholder spatial coords for pooled sensory tokens (no explicit xyz)
        sensory_xyz = torch.zeros(
            B, pooled_sensory.shape[1], 3,
            device=pooled_sensory.device,
        )

        # step 2: pool group c in BEV + image slots
        # splot group craw tokens into BEV-like and image-like halves.
        # if group c is short (context only, no BEV/img), we pad with zeros
        T_c = group_c_raw.shape[1]
        N_bev = self.bev_pool_queries.shape[1]
        N_img = self.img_pool_queries.shape[1]

        # use full group c pool for both bev and imge query sets 
        # the queries sepcialize them into different semantic subspaces.
        f_bev = self._pool_to_fixed(self.bev_pooler, self.bev_pool_queries, group_c_raw)
        f_img = self._pool_to_fixed(self.img_pooler, self.img_pool_queries, group_c_raw)

        # step 3: prepare b ego/history/interaction summary tokens
        # scene tokenizer experts emb_history, emb-ego, and interaction 
        emb_history = group_b_tokens.mean(dim=1, keepdim=True)          # (B, 1, D)
        emb_ego     = group_b_tokens[:, :1, :]                           # (B, 1, D)
        interaction = group_b_tokens[:, 1:2, :]                         # (B, 1 , D)

        # step 4: tokenizer (tokenA, tokenB, tokenC)
        # scenetokenizer lidar tokens slot is set to 0 in __init__
        # so only use gassian tokens
        tokens_A, tokens_B, tokens_C, spatial_xyz, token_types_C = (
            self.scene_tokenizer(
                gaussian_tokens=pooled_sensory,
                gaussian_means=sensory_xyz,
                f_lidar=torch.zeros(B, 0, D, device=group_a_raw.device),  # no separate lidar
                f_bev=f_bev,
                f_img=f_img,
                emb_history=emb_history,
                emb_ego=emb_ego,
            )
        )

        # maybe relax stop-gradient on c-> B path
        if (
            self.stopgrad_relax_step is not None
            and self._global_step >= self.stopgrad_relax_step
        ):
            self.moe_backbone.set_stopgrad_C_to_B(False)

        # step 6  run staked MoE blocks
        moe_out: StackedMoEOutput = self.moe_backbone(
            tokens_A=tokens_A,
            tokens_B=tokens_B,
            tokens_C=tokens_C,
            spatial_xyz=spatial_xyz,
            token_types_C=token_types_C,
            t=t,
            step=self._global_step,
        )
 
        return moe_out, moe_out.aux_loss
    
    def forward(
        self,
        adapted_batch,
        noisy_trajectory,
        t,
        complete_model=True,
        return_routing_info=False,
        use_draft_conditioning=False,
    ):
        """
        Forward pass with full MoE backbone.
 
        Flow:
            1. Encode tokens per group (pure encoders)
            2. Timestep embedding
            3. _build_moe_context: pool -> SceneTokenizer -> StackedMoEBlocks
            4. Embed trajectory + timestep
            5. Concatenate all tokens -> DiT blocks
            6. Extract trajectory tokens -> output projection
        """
        B, N, T_future, C = noisy_trajectory.shape
 
        agent_states  = adapted_batch['agent']
        agent_history = adapted_batch.get(
            'agent_history',
            torch.zeros(B, N, 4, self.traj_channels, device=agent_states.device),
        )
 
        # STEP 1: Timestep embedding 
        t_emb = self.t_embedder(t)   # [B, D]
 
        # STEP 2: Encode tokens per group (pure encoders) 
        group_a_tokens, group_a_types = self._encode_group_a(adapted_batch)
        group_b_tokens                = self._encode_group_b(agent_states, agent_history)
        group_c_tokens, group_c_types = self._encode_group_c(adapted_batch)
 
        # STEP 3: MoE backbone 
        # (replaces old: moe_output = self.moe_router(...))
        moe_out, moe_aux_loss = self._build_moe_context(
            group_a_raw=group_a_tokens,
            group_a_types=group_a_types,
            group_b_tokens=group_b_tokens,
            group_c_raw=group_c_tokens,
            group_c_types=group_c_types,
            t=t,
        )
        # moe_out.tokens_A  -> refined sensory tokens   [B, N_A, D]
        # moe_out.tokens_B  -> refined interaction tokens [B, N_B, D]
        # moe_out.tokens_C  -> refined map/context tokens [B, N_C, D]
 
        # STEP 4: Trajectory embedding (unchanged) 
        noise_level = t.float() / self.max_timesteps
        trajectory_emb_fine = self.trajectory_embed(noisy_trajectory, self.training)
 
        if use_draft_conditioning and hasattr(self, '_draft_condition'):
            draft_emb = self.trajectory_embed(self._draft_condition, False)
            trajectory_emb_fine = trajectory_emb_fine + 0.5 * draft_emb
 
        future_time_emb    = self.future_time_embedder(B, N)
        trajectory_emb_fine = trajectory_emb_fine + future_time_emb
 
        if noise_level.mean() > 0.3:
            trajectory_emb = self.temporal_downsample(trajectory_emb_fine, factor=2)
            T_active = T_future // 2
        else:
            trajectory_emb = trajectory_emb_fine
            T_active = T_future
 
        trajectory_flat = trajectory_emb.reshape(B, N * T_active, -1)
        traj_summary    = trajectory_emb.mean(dim=(1, 2))   # [B, D]
 
        # ── STEP 5: Assemble all tokens for DiT 
        c = t_emb + traj_summary   # conditioning vector
 
        all_tokens = torch.cat([
            trajectory_flat,      # [B, N*T_active, D]
            moe_out.tokens_A,     # [B, N_A, D]  ← was moe_output.group_a_tokens
            moe_out.tokens_B,     # [B, N_B, D]  ← was moe_output.group_b_tokens
            moe_out.tokens_C,     # [B, N_C, D]  ← was moe_output.group_c_tokens
        ], dim=1)
 
        if all_tokens.shape[1] <= self.pos_embed.shape[1]:
            all_tokens = all_tokens + self.pos_embed[:, :all_tokens.shape[1], :]
        else:
            all_tokens = all_tokens + self.pos_embed
 
        # STEP 6: DiT transformer blocks (unchanged) 
        decorr_loss = torch.tensor(0.0, device=noisy_trajectory.device)
        token_select_list        = []
        attn_weight_masks_list   = []
        mlp_weight_masks_list    = []
 
        for block_idx, block in enumerate(self.blocks):
            all_tokens, attn_mask, mlp_mask, token = block(
                all_tokens, c, t_emb, complete_model
            )
            if self.training and block_idx % 2 == 0:
                traj_tokens_block = all_tokens[:, :N * T_active, :]
                decorr_loss = decorr_loss + 0.1 * self.compute_block_decorrelation(
                    traj_tokens_block, N
                )
            attn_weight_masks_list.append(attn_mask)
            mlp_weight_masks_list.append(mlp_mask)
            token_select_list.append(token)
 
        # Add MoE auxiliary loss (decorr + capacity + ortho)
        # <- was: decorr_loss = decorr_loss + moe_output.orthogonal_loss
        decorr_loss = decorr_loss + moe_aux_loss
 
        # STEP 7: Extract & output (unchanged)
        traj_tokens = all_tokens[:, :N * T_active, :]
 
        if T_active != T_future:
            traj_tokens = self.temporal_upsample(traj_tokens.reshape(B, N, T_active, -1))
        else:
            traj_tokens = traj_tokens.reshape(B, N, T_future, -1)
 
        shift, scale = self.output_adaLN(c).chunk(2, dim=1)
        traj_flat    = traj_tokens.reshape(B, N * T_future, -1)
 
        from model.dydittraj import modulate
        traj_flat       = modulate(self.output_norm(traj_flat), shift, scale)
        predicted_output = self.output_projection(traj_flat).reshape(B, N, T_future, -1)
 
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
            )
        return predicted_noise, decorr_loss, None, None, None

    # MoE lifecycle helpers (NEW)

    def set_global_step(self, step: int) -> None:
        """Call from training loop at each gradient step."""
        self._global_step = step
 
    def invalidate_moe_caches(self) -> None:
        """Call at scene boundary during inference (clears DyDiT skip caches)."""
        self.moe_backbone.invalidate_caches()
 
    def reset_moe_decorr_buffers(self) -> None:
        """Call at training stage transitions (resets Group B anchor buffers)."""
        self.moe_backbone.reset_decorr_buffers()
 
    def moe_phase(self) -> int:
        return self.moe_phase_tracker.current_phase(self._global_step)
 
    def moe_phase_name(self) -> str:
        return self.moe_phase_tracker.phase_name(self._global_step)
    
    # utilize
    
    def apply_decorrelation(self, level1, level2, level3, B, N): 
        # pay attention, please this is when decorr loss with encoder do.
        # please consider with decorr attention, that is why we have layer and loss between layer in agent encoder.
        # this will be removed due to MoE that we prepared, so dcorrelation with depend on gate and in what scheme that we have.
        """Apply decorrelation with proper reshaping."""
        level1_flat = level1.reshape(B * N, -1)
        level2_flat = level2.reshape(B * N, -1)
        level3_flat = level3.reshape(B * N, -1)
        
        decorr_loss, l1_decorr, l2_decorr, l3_decorr = self.decorrelation(
            level1_flat, level2_flat, level3_flat
        )
        
        level1.copy_(l1_decorr.reshape(B, N, -1))
        level2.copy_(l2_decorr.reshape(B, N, -1))
        level3.copy_(l3_decorr.reshape(B, N, -1))
        
        return decorr_loss
    
    def compute_block_decorrelation(self, traj_tokens, N):
        """Prevent trajectory mode collapse within a block."""
        B, NT, D = traj_tokens.shape
        traj_per_agent = traj_tokens.reshape(B, N, -1, D)
        
        agent_features = traj_per_agent.mean(dim=2)  # [B, N, D]
        agent_features = F.normalize(agent_features, dim=-1)
        
        corr = torch.bmm(agent_features, agent_features.transpose(1, 2))
        
        identity = torch.eye(N, device=corr.device).unsqueeze(0)
        decorr_loss = torch.mean((corr - identity) ** 2)
        
        return decorr_loss
    

    def compute_goal_reaching_score(self, trajectory, goal_position):
        final_pos = trajectory[:, :, -1, :2]
        distances = torch.norm(final_pos - goal_position, dim=-1)
        avg_distance = distances.mean(dim=1)
        score = torch.exp(-avg_distance/10.0)
        return score
    
    def compute_collision_score(self, trajectory):
        B, N, T, _ = trajectory.shape
        positions = trajectory[:, :, :, :2]
        collision_penalty = torch.zeros(B, device=trajectory.device)
        safe_distance = 2.0
        
        for t in range(T):
            pos_t = positions[:, :, t, :]
            diff = pos_t.unsqueeze(2) - pos_t.unsqueeze(1)
            pairwise_dist = torch.norm(diff, dim=-1)
            
            mask = ~torch.eye(N, device=trajectory.device, dtype=torch.bool)
            mask = mask.unsqueeze(0).expand(B, -1, -1)
            
            collisions = (pairwise_dist < safe_distance) & mask
            collision_count = collisions.float().sum(dim=(1,2))
            collision_penalty += collision_count
        
        score = torch.exp(-collision_penalty / (N*T))
        return score
    
    def compute_kinematic_feasibility_score(self, trajectory):
        B, N, T, _ = trajectory.shape
        velocities = trajectory[:, :, :, 2:4]
        speeds = torch.norm(velocities, dim=-1)
        
        max_speed = 30.0
        speed_violations = torch.relu(speeds - max_speed)
        speed_penalty = speed_violations.mean(dim=(1, 2))
        
        accel = velocities[:, :, 1:, :] - velocities[:, :, :-1, :]
        accel_mag = torch.norm(accel, dim=-1)
        max_accel = 8.0
        accel_violations = torch.relu(accel_mag - max_accel)
        accel_penalty = accel_violations.mean(dim=(1, 2))
        
        positions = trajectory[:, :, :, :2]
        jerk = positions[:, :, 2:, :] - 2*positions[:, :, 1:-1, :] + positions[:, :, :-2, :]
        jerk_mag = torch.norm(jerk, dim=-1)
        max_jerk = 5.0
        jerk_violations = torch.relu(jerk_mag - max_jerk)
        jerk_penalty = jerk_violations.mean(dim=(1, 2))
        
        total_penalty = speed_penalty + accel_penalty + 0.5 * jerk_penalty
        score = torch.exp(-total_penalty)
        return score
    
    def compute_smoothness_score(self, trajectory):
        positions = trajectory[:, :, :, :2]
        jerk = positions[:, :, 2:, :] - 2*positions[:, :, 1:-1, :] + positions[:, :, :-2, :]
        jerk_mag = torch.norm(jerk, dim=-1)
        avg_jerk = jerk_mag.mean(dim=(1, 2))
        score = torch.exp(-avg_jerk / 2.0)
        return score
    
    def compute_diversity_score(self, trajectory, other_proposals):
        if other_proposals is None or len(other_proposals) == 0:
            return torch.ones(trajectory.shape[0], device=trajectory.device)
        
        B, N, T, _ = trajectory.shape
        K = len(other_proposals)
        
        traj_flat = trajectory.reshape(B, -1)
        
        similarities = []
        for k in range(K):
            other_flat = other_proposals[k].reshape(B, -1)
            similarity = F.cosine_similarity(traj_flat, other_flat, dim=1)
            similarities.append(similarity)
        
        similarities = torch.stack(similarities, dim=0)
        max_similarity = similarities.max(dim=0)[0]
        diversity = 1.0 - max_similarity
        
        return torch.relu(diversity)
    
    def score_trajectory(self, trajectory, goal_positions, other_proposals=None, weights=None):
        if weights is None:
            weights = {
                'goal': 1.0,
                'collision': 2.0,
                'kinematic': 0.8,
                'smoothness': 0.5,
                'diversity': 0.3,
            }
        
        goal_score = self.compute_goal_reaching_score(trajectory, goal_positions)
        collision_score = self.compute_collision_score(trajectory)
        kinematic_score = self.compute_kinematic_feasibility_score(trajectory)
        smoothness_score = self.compute_smoothness_score(trajectory)
        diversity_score = self.compute_diversity_score(trajectory, other_proposals)
        
        total_score = (
            weights['goal'] * goal_score +
            weights['collision'] * collision_score +
            weights['kinematic'] * kinematic_score +
            weights['smoothness'] * smoothness_score +
            weights['diversity'] * diversity_score
        )
        
        score_dict = {
            'goal': goal_score,
            'collision': collision_score,
            'kinematic': kinematic_score,
            'smoothness': smoothness_score,
            'diversity': diversity_score,
            'total': total_score,
        }
        
        return total_score, score_dict
    

class TransDiffuserWithDiffusion(nn.Module):
    """ Wrapper adding diffusion sampling to adapter Transdiffuser"""
    def __init__(
            self,
            transdiffuser_model,
            diffusion,
            num_proposals = 20,
            selection_strategy = 'top_k_blend',
            top_k_for_blend = 5,
            score_weights = None,
            use_refinement = True,
            refinement_noise_level = 0.1,
            refirement_steps_ratio = 0.3,
    ):
        super().__init__()
        self.model = transdiffuser_model
        self.diffusion = diffusion
        self.learn_sigma = transdiffuser_model.learn_sigma

        self.num_proposals = num_proposals
        self.selection_strategy = selection_strategy
        self.top_k_for_blend = top_k_for_blend
        self.score_weights = score_weights or {
            'goal': 1.0,
            'collision': 2.0,
            'kinematic': 0.8,
            'smoothness': 0.5,
            'diversity': 0.3,         
        }

        self.use_refinement = use_refinement
        self.refinement_noise_level = refinement_noise_level
        self.refinement_noise_ratio = refirement_steps_ratio

    def forward(self, adapted_batch, goal_position = None, t = None):
        """Training forward pass."""

        agent_states = adapted_batch['agent']
        B, N, _ = agent_states.shape
        device = agent_states.device

        if goal_position is None:
            goal_position = agent_states[:, :, :2]
        
        # get ground truth from batch

        if 'gt_trajectory' in adapted_batch:
            gt_trajectory = adapted_batch['gt_trajectory']

        else:
            # Fallback , just for fallback, 
            gt_trajectory = torch.randn(B, N, 20, 5, device=device) 
        if t is None:
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        
        noise = torch.randn_like(gt_trajectory)
        noisy_trajectory = self.diffusion.q_sample(gt_trajectory, t, noise)
        
        predicted_output, decorr_loss, attn_masks, mlp_masks, token_select = self.model(
            adapted_batch=adapted_batch,
            noisy_trajectory=noisy_trajectory,
            t=t,
            complete_model=False
        )
        
        if self.learn_sigma:
            predicted_noise = predicted_output[..., :5]
            diffusion_loss = F.mse_loss(predicted_noise, noise)
        else:
            diffusion_loss = F.mse_loss(predicted_output, noise)
        
        total_loss = diffusion_loss + decorr_loss
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'decorr_loss': decorr_loss,
            'attn_channel_usage': attn_masks.mean() if attn_masks is not None else 0,
            'mlp_channel_usage': mlp_masks.mean() if mlp_masks is not None else 0,
            'token_selection': token_select.mean() if token_select is not None else 0,
        }
    
    @torch.no_grad()
    def sample(
        self, 
        adapted_batch, 
        goal_positions=None,
        num_inference_steps=50,
        return_all_proposals=False,
        return_scores=False,
    ):
        """
        Complete sampling pipeline with multi-proposal and refinement.
        
        Args:
            adapted_batch: Pre-adapted batch from adapter
            goal_positions: [B, N, 2]
            ...
        """
        agent_states = adapted_batch['agent']
        B, N, _ = agent_states.shape
        device = agent_states.device
        
        if goal_positions is None:
            goal_positions = agent_states[:, :, :2]
        
        # Stage 1: Generate proposals
        print(f"\n=== Stage 1: Generating {self.num_proposals} proposals ===")
        
        all_proposals = []
        all_score_dicts = []
        
        for k in range(self.num_proposals):
            proposal = self._generate_single_proposal(
                adapted_batch, num_inference_steps, seed=k
            )
            
            total_score, score_dict = self.model.score_trajectory(
                trajectory=proposal,
                goal_positions=goal_positions,
                other_proposals=all_proposals,
                weights=self.score_weights
            )
            
            all_proposals.append(proposal)
            all_score_dicts.append(score_dict)
            
            if (k + 1) % 5 == 0:
                print(f"  Progress: {k+1}/{self.num_proposals}, "
                      f"avg score: {total_score.mean().item():.4f}")
        
        all_proposals = torch.stack(all_proposals, dim=0)
        all_scores = torch.stack([sd['total'] for sd in all_score_dicts], dim=0)
        
        # Stage 2a: Selection
        print(f"\n=== Stage 2a: Selecting trajectory (strategy: {self.selection_strategy}) ===")
        
        if self.selection_strategy == 'best':
            selected_trajectory = self._select_best(all_proposals, all_scores)
        elif self.selection_strategy == 'weighted_blend':
            selected_trajectory = self._weighted_blend_all(all_proposals, all_scores)
        elif self.selection_strategy == 'top_k_blend':
            selected_trajectory = self._top_k_blend(
                all_proposals, all_scores, k=self.top_k_for_blend
            )
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        # Stage 2b: Refinement
        if self.use_refinement:
            print(f"\n=== Stage 2b: Two-pass refinement ===")
            final_trajectory = self._refine_trajectory(
                selected_trajectory, adapted_batch, num_inference_steps
            )
        else:
            final_trajectory = selected_trajectory
        
        print("\n=== Sampling complete ===\n")
        
        if return_all_proposals and return_scores:
            return final_trajectory, all_proposals, all_scores
        elif return_all_proposals:
            return final_trajectory, all_proposals
        elif return_scores:
            return final_trajectory, all_scores
        else:
            return final_trajectory
    
    @torch.no_grad()
    def _generate_single_proposal(self, adapted_batch, num_inference_steps, seed=None):
        """Generate a single trajectory proposal."""
        agent_states = adapted_batch['agent']
        B, N, _ = agent_states.shape
        device = agent_states.device
        
        if seed is not None:
            torch.manual_seed(seed)
        
        trajectory = torch.randn(B, N, 20, 5, device=device)
        
        def model_fn(x, t):
            predicted_noise, _, _, _, _ = self.model(
                adapted_batch=adapted_batch,
                noisy_trajectory=x,
                t=t,
                complete_model=True
            )
            return predicted_noise
        
        for t_idx in reversed(range(num_inference_steps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            
            out = self.diffusion.p_sample(
                model_fn,
                trajectory,
                t,
                clip_denoised=True
            )
            
            if isinstance(out, dict):
                trajectory = out['sample']
            else:
                trajectory = out
        
        return trajectory
    
    @torch.no_grad()
    def _refine_trajectory(self, trajectory_draft, adapted_batch, num_inference_steps):
        """Two-pass refinement."""
        B, N, T, C = trajectory_draft.shape
        device = trajectory_draft.device
        
        refinement_steps = int(self.refinement_steps_ratio * num_inference_steps)
        
        t_refine = torch.full((B,), refinement_steps, device=device, dtype=torch.long)
        noise = torch.randn_like(trajectory_draft)
        noisy_draft = self.diffusion.q_sample(trajectory_draft, t_refine, noise)
        
        self.model.register_buffer('_draft_condition', trajectory_draft)
        
        def model_fn_refine(x, t):
            predicted_noise, _, _, _, _ = self.model(
                adapted_batch=adapted_batch,
                noisy_trajectory=x,
                t=t,
                complete_model=True,
                use_draft_conditioning=True
            )
            return predicted_noise
        
        trajectory_refined = noisy_draft
        for t_idx in reversed(range(refinement_steps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            
            out = self.diffusion.p_sample(
                model_fn_refine,
                trajectory_refined,
                t,
                clip_denoised=True
            )
            
            if isinstance(out, dict):
                trajectory_refined = out['sample']
            else:
                trajectory_refined = out
        
        delattr(self.model, '_draft_condition')
        return trajectory_refined
    
    def _select_best(self, proposals, scores):
        K, B, N, T, C = proposals.shape
        best_indices = scores.argmax(dim=0)
        best_trajectories = []
        for b in range(B):
            best_k = best_indices[b]
            best_trajectories.append(proposals[best_k, b])
        return torch.stack(best_trajectories, dim=0)
    
    def _weighted_blend_all(self, proposals, scores):
        K, B, N, T, C = proposals.shape
        weights = torch.softmax(scores, dim=0)
        weights_expanded = weights.view(K, B, 1, 1, 1)
        blended = (proposals * weights_expanded).sum(dim=0)
        return blended
    
    def _top_k_blend(self, proposals, scores, k=5):
        K, B, N, T, C = proposals.shape
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=0)
        
        blended_trajectories = []
        for b in range(B):
            top_proposals = proposals[topk_indices[:, b], b]
            top_scores = topk_scores[:, b]
            weights = torch.softmax(top_scores, dim=0)
            blended = (top_proposals * weights.view(k, 1, 1, 1)).sum(dim=0)
            blended_trajectories.append(blended)
        
        return torch.stack(blended_trajectories, dim=0)
    


# factory (update MoE parameter)

def create_transdiffuser_adapted(
    adapter: EncoderAdapter,
    hidden_size=768,
    depth=12,
    num_heads=12,
    decorr_weights=0.1,
    max_agents=8,
    future_horizon=20,
    num_proposals=20,
    selection_strategy='top_k_blend',
    top_k_for_blend=5,
    use_refinement=True,
    # MoE params
    moe_num_blocks: int = 4,
    moe_num_experts: int = 4,
    moe_num_lidar_tokens: int = 64,
    moe_num_bev_tokens: int = 128,
    moe_num_img_tokens: int = 64,
    moe_step_start_A: int = 2_000,
    moe_step_start_B: int = 6_000,
    moe_step_start_C: int = 10_000,
    stopgrad_relax_step: Optional[int] = None,
) -> TransDiffuserWithDiffusion:
    """Create TransDiffuser with full 3-group MoE backbone."""
    base_model = TransDiffuserIntegrated(
        adapter=adapter,
        input_size=64,
        patch_size=4,
        traj_channels=5,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        decorr_weights=decorr_weights,
        max_agents=max_agents,
        future_horizon=future_horizon,
        history_length=4,
        moe_num_blocks=moe_num_blocks,
        moe_num_experts=moe_num_experts,
        moe_num_lidar_tokens=moe_num_lidar_tokens,
        moe_num_bev_tokens=moe_num_bev_tokens,
        moe_num_img_tokens=moe_num_img_tokens,
        moe_step_start_A=moe_step_start_A,
        moe_step_start_B=moe_step_start_B,
        moe_step_start_C=moe_step_start_C,
        stopgrad_relax_step=stopgrad_relax_step,
    )
    diffusion = create_diffusion(
        timestep_respacing="", learn_sigma=False, noise_schedule="linear"
    )
    return TransDiffuserWithDiffusion(
        transdiffuser_model=base_model,
        diffusion=diffusion,
        num_proposals=num_proposals,
        selection_strategy=selection_strategy,
        top_k_for_blend=top_k_for_blend,
        use_refinement=use_refinement,
    )