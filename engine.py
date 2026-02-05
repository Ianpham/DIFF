import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType
from encode.hierachy_encoder import ModalityEncoder, ModalityGateInfo
from encode.modality_encoder import (LidarEmbedding, ResNetImageEncoder, BEVEncoder, MultiCameraEncoder,
                                     AgentEncoder, IntersectionEncoder, GoalIntentEncoder, 
                                     PedestrianEncoder, BehaviorEncoder,OcclusionEncoder, TrafficControlEncoder,
                                     HistoryEncoder, FutureTimeEmbedder, TrajectoryEmbedder, TimestepEmbedder,
                                    ) # please consider to encoder vectorfeatuere map (which lane feature)
from encode.requirements import EncoderRequirements, StandardRequirements

from datasets.navsim.navsim_utilize.data import NavsimDataset, EnhancedNavsimDataset, PhaseNavsimDataset
from model.dydittraj import DiTBlock, DiffRate, DynaLinear
from model.MMRD import MultiModalDecorrelation

from adapters import EncoderAdapter

from diffusion import create_diffusion

# class ModalityConfig(Enum):
#     # set different phase base on contract and dataset
#     LIDAR = 2
#     BEV_HDMAP = 12
#     BEV_FEATURE = 256
#     IMG = 3


class TransDiffuserIntegrated(nn.Module):
    """
    Complete TransDiffuser model integrating:
    1. Multi-modal encoder (from DiT)
    2. Decorrelation mechanism - MMRD
    3. Diffusion-based trajectory decoder (DiT)
    4. Contract dataset and encoder, then compile with encoder to decorrelation
    """
        
    def __init__(
        self,
        # adapter
        adapter: EncoderAdapter,
        #DiT parameters
        input_size = 64,
        patch_size = 5,
        traj_channels = 7, # previous is 5 but we will add to config later, then see whether we can keep 7 or just 5
        hidden_size = 768,
        depth = 12,
        num_heads = 12,
        mlp_ratio = 4.0,

        # decorrlation parameters
        decorr_weights = 0.1,
        decorr_similiary = 'cosine',

        # agent parameters
        max_agents = 32,
        future_horizon= 8,  # previous is 20, but in this dataset, what we can do is to increase and see how confidence is lower by timestep
        history_length = 4, # previous is 30, but in this dataset, what we can hafve is just 4, navsim is for short predict, if we enhance the encode, longer prediction can be

        # other
        max_timesteps = 50, # 10 is enough for correcrt, but we tried for best result then decrease
        trajectory_dropout_prob = 0.1,
        learn_sigma = True,
        use_modality_specific = True,
        parallel = True, # whether we use hierarchy encoder or use parallel
        # modality_config = ModalityConfig, 

        # parameter for encoder
        use_improved_encoder = True,
        use_modality_gating = True,
        gate_type = 'soft',
        output_tokens_per_modality = 16,

    ):
        super().__init__()
        
        self.adpater = adapter
        self.contract = adapter.contract
        self.hidden_size = hidden_size
        self.traj_channels = traj_channels
        self.max_agents = max_agents
        self.future_horizon = future_horizon
        self.max_timesteps = max_timesteps
        self.learn_sigma = learn_sigma
        self.use_improve_encoder = use_improved_encoder

        #  dynamic modality configuration (from adapters)
        self.modality_config = self._build_modality_config()

        print(f"\n Dettect modality configuration:")
        for name, info in self.modality_config.items():
            print(f" {name}: {info['channels']} channels, shae = {info['shape']}")

        # multi-modal encoder (from adapter-built encoders)

        # get encoders from adapter
        self.modality_embedders = self._wrap_adapter_encoders()

        # create improved modality encoder if enabled
        if use_improved_encoder:
            self.improved_modality_encoder = ModalityEncoder(
                modality_embedders = self.modality_embedders,
                modality_config= {k: v['channels'] for k, v in self.modality_config.items()},
                hidden_size= hidden_size, 
                num_heads= num_heads,
                dropout=0.1,
                parallel= parallel,
                use_gating= use_modality_gating,
                gate_type= gate_type,
                output_tokens_per_modality= output_tokens_per_modality
            )

            self.context_token_fixed = self.improved_modality_encoder.get_output_token_count()

        else:
            self.context_token_fixed = None

        # agent_encoder

        # from encode.modality_encoder import (
        #     AgentEncoder, HistoryEncoder, 
        #     TimestepEmbedder, TrajectoryEmbedder,
        #     FutureTimeEmbedder
        # )

        self.agent_encoder = AgentEncoder(hidden_size=hidden_size)

        self.history_encoder_temporal = HistoryEncoder(
            input_size=traj_channels, 
            hidden_size= hidden_size,
            history_length= history_length,
            num_layers=2,
            batch_first=True
        )

        # decorrelation module
        self.decorrelation = MultiModalDecorrelation(
            decorr_weight = decorr_weights,
            similarity_type= decorr_similiary
        )

        # trajectory and timestep embedders
        self.trajectory_embed = TrajectoryEmbedder(
            traj_channels, hidden_size, trajectory_dropout_prob
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.future_time_embedder = FutureTimeEmbedder(hidden_size, future_horizon)

        # positional embedding
        self.num_patches = (input_size // patch_size) ** 2
        self.total_tokens = self._calculate_total_tokens()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.total_tokens, hidden_size),
            requires_grad=False
        )

        # DiT transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio= mlp_ratio)
            for _ in range(depth)
        ])

        # output heads
        self.output_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps = 1e-6)

        output_channels = traj_channels * 2 if learn_sigma else traj_channels
        self.output_projection = nn.Linear(hidden_size, output_channels)

        self.output_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias = True)
        )

        self.initialize_weights()

    # this is where we update more encoder and modality for upcoming phase with more loading modalities
    def _build_modality_config(self) -> Dict[str, Dict[str, Any]]:
        """
        build modality configuration fromd dataset contract.
        Maps contract features to modality embeeder requirements

        """
        modality_config = {}

        # LIDAR BEV
        if self.contract.has(FeatureType.LIDAR_BEV):
            spec = self.contract.get_spec(FeatureType.LIDAR_BEV)
            modality_config['lidar'] = {
                'channels': spec.shape[0] if len(spec.shape) > 0 else 2,
                'shape': spec.shape,
                'encoder_name': 'lidar'
            }


        # BEV labels
        if self.contract.has(FeatureType.BEV_LABELS):
            modality_config['BEV'] = {
                'channels': self.contract.bev_channels,
                'shape': (self.contract.bev_channels, 200, 200),
                'encoder_name': 'bev'
            }

        # camera images
        if self.contract.has(FeatureType.CAMERA_IMAGES):
            modality_config['img'] = {
                'channels': 3,
                'shape': (3, 900, 1600),
                'encoder_name': 'camera'
            }

        return modality_config

    def _wrap_adapter_encoders(self) -> nn.ModuleDict:
        """
        Wrap adapter-built encoders to match Trandiffusers's expected interface.

        THis adapter builds encoders, but transdiffuser expects specific name
        This function creates a mapping layer
        """
    
        wrapped_encoders = nn.ModuleDict()

        for modality_name, modality_info in self.modality_config.items():
            encoder_name = modality_info['encoder_name']
            channels = modality_info['channels']

            # create embedder (these are lightweight, just define archictecture)
            # the adapter will provide the actual weights during forward pass

            if modality_name == 'lidar':
                wrapped_encoders[modality_name] = LidarEmbedding(
                    self.hidden_size
                )

            elif modality_name == 'BEV':
                wrapped_encoders[modality_name] = BEVEncoder(
                    channels, self.hidden_size, patch_size= 4
                )

            elif modality_name == 'img':
                wrapped_encoders[modality_name] = ResNetImageEncoder(
                    channels, self.hidden_size, patch_size= 4
                )

        return wrapped_encoders


    def _calculate_total_tokens(self):
        """Calculate total tokens for positional embedding."""

        # context tokens (from modalities)
        context_tokens = 0
        for modality_name in self.modality_config.keys():
            if modality_name in self.modality_config.keys():
                if modality_name in ['lidar', 'img', 'BEV']:
                    context_tokens += self.num_patches +1

                else:
                    context_tokens += 1

        # agent tokens per agent: 1 state + 1 history + 20 (8) future
        tokens_per_agent = 1 + 1 + self.future_horizon

        total = context_tokens + (self.max_agents * tokens_per_agent)

        return total

    def initialize_weights(self):
        """Initialize model weights"""
        from model.dydittraj import get_2d_sincos_pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True,
            extra_tokens=self.total_tokens - self.num_patches
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Zero-out output layers
        nn.init.constant_(self.output_adaLN[-1].weight, 0)
        nn.init.constant_(self.output_adaLN[-1].bias, 0)
        nn.init.constant_(self.output_projection.weight, 0)
        nn.init.constant_(self.output_projection.bias, 0)

    def encode_modalities(self, adapted_batch, return_gate_info = False):
        """
        Encode multi-modal context using adapter-prepared batch.
        
        Args:
            adapted_batch: Already adapted by adapter.adapt_batch()
            return_gate_info: Whether to return gate information
            
        Returns:
            encoded: [B, T_context, D]
            gate_info: Optional[ModalityGateInfo]
        """

        # build context dict from adapted batch
        context = {}

        for modality_name, modality_info in self.modality_config.items():
            encoder_name = modality_info['encoder_name']

            if encoder_name in adapted_batch:
                context[modality_name] = adapted_batch[encoder_name]

        if self.use_improve_encoder:
            # use gating and cross-attention
            encoded, gate_info = self.improved_modality_encoder.encode_modalities(
                context, return_gate_info=return_gate_info
            )

            return encoded, gate_info
        
        else:
            # naive concatenation
            modality_features = []

            for modality_name in self.modality_config.keys():
                if modality_name in context:
                    modality_input = context[modality_name]
                    features = self.modality_embedders[modality_name](modality_input)
                    modality_features.append(features)

            encoded = torch.cat(modality_features, dim = 1) # (B, T_context, D)

            return encoded, None
        
    def encode_history_temporal(self, history):
        """
        Compress temporal history into single vector per agent.
        [B, N, 30, 5] → [B, N, D]
        """
        B, N, T, C = history.shape
        
        # Reshape for LSTM: [B*N, T, C]
        history_flat = history.reshape(B * N, T, C)
        
        # LSTM encoding
        _, (h_n, _) = self.history_encoder_temporal(history_flat)  # h_n: [2, B*N, D]
        
        # Take last layer output
        history_encoded = h_n[-1]  # [B*N, D]
        
        # Reshape back: [B, N, D]
        history_encoded = history_encoded.reshape(B, N, -1)
        
        return history_encoded
    
    def forward(
            self,
            adapted_batch,
            noisy_trajectory,
            t, 
            encoder_level = 0,
            complete_model = True,
            return_gate_info = False,
            use_draft_conditioning = False,
    ):
        """
        Forward pass with adapter-prepared inputs.
        
        Args:
            adapted_batch: Output from adapter.adapt_batch()
            noisy_trajectory: [B, N, T, C]
            t: Diffusion timestep
            ...
        """

        B, N, T_future, C = noisy_trajectory.shape

        # Extract required data from adapted batch
        agent_states = adapted_batch['agent'] # already padded to 7D if needed
        agent_history = adapted_batch.get('agent_history', torch.randn(B, N, 30, 5, device = agent_states.device))

        # noise scheduling analysis
        t_emb = self.t_embedder(t) # B, D
        noise_level = t.float() / self.max_timesteps # [B] in [0,1]

        alpha_global = torch.sigmoid(-5 * (noise_level - 0.5)).unsqueeze(-1) # [B, 1]
        alpha_local = 1 - alpha_global

        # encode context (scene understanding)

        context_encoded, gate_info = self.encode_modalities(
            adapted_batch, return_gate_info=return_gate_info
        )

        # multi-scale trajectory embedding
        trajectory_emb_fine = self.trajectory_embed(noisy_trajectory, self.training)
        if use_draft_conditioning and hasattr(self, '_draft_contition'):
            draft_emb = self.trajectory_embed(self._draft_condition, False)
            trajectory_emb_fine = trajectory_emb_fine + 0.5 * draft_emb
        future_time_emb = self.future_time_embedder(B, N)
        trajectory_emb_fine = trajectory_emb_fine + future_time_emb

        # coarse scale (reduced temporal resolution for early diffusion)
        if noise_level.mean() > 0.3: # high noise  - use coarse
            trajectory_emb = self.temporal_downsample(trajectory_emb_fine, factor = 2)
            T_active = T_future // 2

        else:
            trajectory_emb = trajectory_emb_fine
            T_active =  T_future

        trajectory_flat = trajectory_emb.reshape(B, N * T_active, -1)       
        traj_summary = trajectory_emb.mean(dim = (1,2)) # B, D

        # Progressive encoder
        decorr_loss = torch.tensor(0.0, device = noisy_trajectory.device)
        encoder_tokens = []
        encoder_summaries = []

        if encoder_level >= 1:
            # temporal
            encoder_output = self.agent_encoder(agent_states)
            level1_features = encoder_output.level1_temporal # B, N, D
            history_encoded = self.encode_history_temporal(agent_history)

            level1_features = level1_features * alpha_local
            history_encoded = history_encoded * alpha_global

            encoder_tokens.extend([level1_features, history_encoded])
            encoder_summaries.append(history_encoded.mean(dim = 2))

            if encoder_level >= 2:
                # interaction
                level2_features = encoder_output.level2_interaction # [B, N, D]
                level1_features = level1_features * alpha_local
                encoder_tokens.append(level2_features)

                if encoder_level >= 3:
                #level 3: scene
                    level3_features = encoder_output.level3_scene # B, N, D
                    level3_features = level3_features * alpha_global

                    encoder_tokens.append(level3_features)

                    # decoorelation
                    decorr_loss = self.apply_decorrelation(
                        level1_features, level2_features, level3_features, B, N
                    )

        # conditioning vector
        c = t_emb + traj_summary
        for summary in encoder_summaries:
            c = c + summary

        # step  6: trajectory-centric token ordering
        token_list = [
            trajectory_flat, # B, N* T_active, D
            context_encoded,   # B, T_context, D
        ]

        token_list.extend(encoder_tokens)

        all_tokens = torch.cat(token_list, dim = 1)

        # positional embedding
        if all_tokens.shape[1] <= self.pos_embed.shape[1]:
            all_tokens = all_tokens + self.pos_embed[:, :all_tokens.shape[1], :]
        else:
            all_tokens = all_tokens + self.pos_embed
        
        # transformer
        token_select_list = []
        attn_weight_masks_list = []
        mlp_weight_masks_list = []
        
        for block_idx, block in enumerate(self.blocks):
            all_tokens, attn_mask, mlp_mask, token = block(
                all_tokens, c, t_emb, complete_model
            )
            
            if self.training and encoder_level >= 2 and block_idx % 2 == 0:
                traj_tokens_block = all_tokens[:, :N*T_active, :]
                block_decorr = self.compute_block_decorrelation(traj_tokens_block, N)
                decorr_loss = decorr_loss + 0.1 * block_decorr
            
            attn_weight_masks_list.append(attn_mask)
            mlp_weight_masks_list.append(mlp_mask)
            token_select_list.append(token)
        
        #EXTRACT & UPSAMPLE TRAJECTORY
        traj_tokens = all_tokens[:, :N*T_active, :]  # [B, N*T_active, D]
        
        if T_active != T_future:
            traj_tokens = self.temporal_upsample(
                traj_tokens.reshape(B, N, T_active, -1)
            )  # [B, N, T_future, D]
        else:
            traj_tokens = traj_tokens.reshape(B, N, T_future, -1)
        
        # OUTPUT PROJECTION
        shift, scale = self.output_adaLN(c).chunk(2, dim=1)
        traj_tokens_flat = traj_tokens.reshape(B, N * T_future, -1)
        
        from model.dydittraj import modulate
        traj_tokens_flat = modulate(
            self.output_norm(traj_tokens_flat),
            shift, scale
        )
        
        predicted_output = self.output_projection(traj_tokens_flat)
        predicted_output = predicted_output.reshape(B, N, T_future, -1)
        
        if self.learn_sigma:
            predicted_noise, predicted_var = predicted_output.chunk(2, dim=-1)
            predicted_output = torch.cat([predicted_noise, predicted_var], dim=-1)
        else:
            predicted_noise = predicted_output
        

        if not complete_model:
            from model.dydittraj import convert_list_to_tensor
            attn_masks = convert_list_to_tensor(attn_weight_masks_list)
            mlp_masks = convert_list_to_tensor(mlp_weight_masks_list)
            token_select = convert_list_to_tensor(token_select_list)
            return predicted_noise, decorr_loss, attn_masks, mlp_masks, token_select
        else:
            return predicted_noise, decorr_loss, None, None, None
        
    def temporal_downsample(self, trajectory_emb, factor=2):
        """Downsample trajectory embeddings along temporal dimension."""
        B, N, T, D = trajectory_emb.shape
        
        if factor == 1:
            return trajectory_emb
        
        trajectory_emb = trajectory_emb.permute(0, 1, 3, 2)  # [B, N, D, T]
        trajectory_emb = trajectory_emb.reshape(B * N, D, T)
        
        downsampled = torch.nn.functional.adaptive_avg_pool1d(
            trajectory_emb, 
            output_size=T // factor
        )
        
        downsampled = downsampled.reshape(B, N, D, T // factor)
        downsampled = downsampled.permute(0, 1, 3, 2)  # [B, N, T//factor, D]
        
        return downsampled
    
    def temporal_upsample(self, trajectory_emb, target_length=None):
        """Upsample trajectory embeddings along temporal dimension."""
        B, N, T_coarse, D = trajectory_emb.shape
        
        if target_length is None:
            target_length = T_coarse * 2
        
        if T_coarse == target_length:
            return trajectory_emb
        
        trajectory_emb = trajectory_emb.permute(0, 1, 3, 2)  # [B, N, D, T_coarse]
        trajectory_emb = trajectory_emb.reshape(B * N, D, T_coarse)
        
        upsampled = torch.nn.functional.interpolate(
            trajectory_emb,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        upsampled = upsampled.reshape(B, N, D, target_length)
        upsampled = upsampled.permute(0, 1, 3, 2)  # [B, N, T_fine, D]
        
        return upsampled
    
    def apply_decorrelation(self, level1, level2, level3, B, N):
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


# Factory function
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
):
    """
    Create adapted TransDiffuser with adapter system.
    
    Args:
        adapter: EncoderAdapter instance
        ... (other args same as before)
    
    Returns:
        TransDiffuserWithDiffusionAdapted model
    """
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
        history_length=30,
    )
    
    diffusion = create_diffusion(
        timestep_respacing="",
        learn_sigma=False,
        noise_schedule="linear"
    )
    
    model = TransDiffuserWithDiffusion(
        transdiffuser_model=base_model,
        diffusion=diffusion,
        num_proposals=num_proposals,
        selection_strategy=selection_strategy,
        top_k_for_blend=top_k_for_blend,
        use_refinement=use_refinement,
    )
    
    return model