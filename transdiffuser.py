"""
TransDiffuser: Integrated Model with Decorrelation
Connects DiT architecture with multi-modal decorrelation mechanism
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from your existing files
from model.MMRD import MultiModalDecorrelation
from encode.hierachy_encoder import (
    ModalityEncoder, 
    create_improved_modality_encoder, 
    ModalityGateInfo
)
from datasets.navsim.navsim_utilize.navsimdataset import NavsimDataset
from datasets.navsim.navsim_utilize.enhancenavsim import EnhancedNavsimDataset


from diffusion import create_diffusion

class TransDiffuserIntegrated(nn.Module):
    """
    Complete TransDiffuser model integrating:
    1. Multi-modal encoder (from DiT)
    2. Decorrelation mechanism - MMRD
    3. Diffusion-based trajectory decoder (DiT)
    """
    
    def __init__(
        self,
        # DiT parameters
        input_size=64,
        patch_size=4,
        traj_channels=5,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        
        # Decorrelation parameters
        decorr_weight=0.1,
        decorr_similarity='cosine',
        
        # Agent parameters
        max_agents=32,
        future_horizon=20,
        history_length=30,
        
        # Other
        max_timesteps = 50,
        trajectory_dropout_prob=0.1,
        learn_sigma=True,
        use_modality_specific=True,
        parallel=True, # whether we use hierachy encoder or use parallel
        modality_config=None,

        # paramter for encoder
        use_improved_encoder = True,
        use_modality_gating = True,
        gate_type = 'soft',
        output_tokens_per_modality = 16,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.traj_channels = traj_channels
        self.max_agents = max_agents
        self.future_horizon = future_horizon
        self.max_timesteps = max_timesteps
        self.learn_sigma = learn_sigma
        self.use_improve_encoder = use_improved_encoder

        
        # 1. MULTI-MODAL ENCODERS (Context - Shared across agents)
        
        if modality_config is None:
            self.modality_config = {
                'lidar': 2,
                'BEV': 7,
                'img': 3,
            }
        else:
            self.modality_config = modality_config
        
        # Import or define your modality embedders here
        # (LidarEmbedding, BEVEncoder, etc. from document 2)
        from encode.modality_encoder import (
            LidarEmbedding, BEVEncoder, ImgCNN,
            AgentEncoder, HistoryEncoder, 
            TimestepEmbedder, TrajectoryEmbedder,
            FutureTimeEmbedder
        )
        # start with lidar, embedding, BEV, Img, we will gradually to increase what to put in this, make encoder understand where to select.
        self.modality_embedders = nn.ModuleDict()
        for modality_name, channels in self.modality_config.items():
            if modality_name == 'lidar':
                self.modality_embedders[modality_name] = LidarEmbedding(
                    channels, hidden_size, patch_size
                )
            elif modality_name == 'BEV':
                self.modality_embedders[modality_name] = BEVEncoder(
                    channels, hidden_size, patch_size
                )
            elif modality_name == 'img':
                self.modality_embedders[modality_name] = ImgCNN(
                    channels, hidden_size, patch_size
                )
        
        # NEW: Create improved modality encoder
        if use_improved_encoder:
            self.improved_modality_encoder = ModalityEncoder(
                modality_embedders=self.modality_embedders,
                modality_config=self.modality_config,
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=0.1,
                parallel=parallel,
                use_gating=use_modality_gating,
                gate_type=gate_type,
                output_tokens_per_modality=output_tokens_per_modality
            )
            # Get fixed context token count
            self.context_tokens_fixed = self.improved_modality_encoder.get_output_token_count()
        else:
            self.context_tokens_fixed = None
        # 2. AGENT ENCODER (Produces level1/level2/level3)
        
        self.agent_encoder = AgentEncoder(hidden_size)
        
        
        # 3. HISTORY ENCODER (Temporal Compression: [B,N,30,5] → [B,N,D])
        
        self.history_encoder_temporal = HistoryEncoder(
            input_size=traj_channels,
            hidden_size=hidden_size,
            history_length= history_length,
            num_layers=2,
            batch_first=True
        )
        
        # 4. DECORRELATION MODULE
        
        self.decorrelation = MultiModalDecorrelation(
            decorr_weight=decorr_weight,
            similarity_type=decorr_similarity
        )
        
        
        # 5. TRAJECTORY & TIMESTEP EMBEDDERS
        
        self.trajectory_embed = TrajectoryEmbedder(
            traj_channels, hidden_size, trajectory_dropout_prob
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.future_time_embedder = FutureTimeEmbedder(hidden_size, future_horizon)
        
        
        # 6. POSITIONAL EMBEDDING
        
        # Calculate total sequence length
        self.num_patches = (input_size // patch_size) ** 2
        self.total_tokens = self._calculate_total_tokens()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.total_tokens, hidden_size), 
            requires_grad=False
        )
        
        
        # 7. DiT TRANSFORMER BLOCKS
        
        from model.dydittraj import DiTBlock
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) 
            for _ in range(depth)
        ])
        
        
        # 8. OUTPUT HEAD (Trajectory prediction)
        
        self.output_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # output channels: if learn_sigma, output both mean and variance
        output_channels = traj_channels * 2 if learn_sigma else traj_channels
        self.output_projection = nn.Linear(hidden_size, output_channels)
        
        # AdaLN modulation for output
        self.output_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )



        
        self.initialize_weights()
    
    def _calculate_total_tokens(self):
        """Calculate total tokens for positional embedding"""
        # Context tokens (shared)
        context_tokens = 0
        for modality_name in self.modality_config.keys():
            if modality_name in ['lidar', 'img', 'BEV']:
                context_tokens += self.num_patches + 1
            else:
                context_tokens += 1
        
        # Agent tokens per agent: 1 state + 1 history + 20 future
        tokens_per_agent = 1 + 1 + self.future_horizon
        
        total = context_tokens + (self.max_agents * tokens_per_agent)
        return total
    
    def initialize_weights(self):
        """Initialize model weights"""
        # Positional embedding
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
    
    def encode_history_temporal(self, history):
        """
        Compress temporal history into single vector per agent.
        [B, N, 30, 5] → [B, N, D]
        
        This is what you meant by "eliminate temporal"
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
    
    def encode_modalities(self, context, return_gate_info=False):
        """
        Encode multi-modal context (shared across agents).
        
        IMPROVED VERSION: Uses gating and cross-attention if enabled.
        
        Args:
            context: Dict of modality tensors
            return_gate_info: Whether to return gate information
            
        Returns:
            encoded: [B, T_context, D]
            gate_info: Optional[ModalityGateInfo]
        """
        if self.use_improve_encoder:
            # encoder with gating and cross-attention
            encoded, gate_info = self.improved_modality_encoder.encode_modalities(
                context, return_gate_info=return_gate_info
            )
            return encoded, gate_info
        else:
            # OLD: Naive concatenation
            modality_features = []
            
            for modality_name in self.modality_config.keys():
                if modality_name in context:
                    modality_input = context[modality_name]
                    features = self.modality_embedders[modality_name](modality_input)
                    modality_features.append(features)
            
            # Concatenate all modalities
            encoded = torch.cat(modality_features, dim=1)  # [B, T_context, D]
            return encoded, None
        
    def forward(
        self,
        context,
        agent_states,
        noisy_trajectory,
        agent_history,
        t,
        encoder_level=0,
        complete_model=True,
        return_gate_info = False,
        use_draft_conditioning=False,
    ):
        """
        Improved diffusion-centric forward pass.
        """
        B, N, T_future, C = noisy_trajectory.shape
        
        
        # STEP 1: NOISE SCHEDULE ANALYSIS
        
        t_emb = self.t_embedder(t)  # [B, D]
        noise_level = t.float() / self.max_timesteps  # [B] in [0, 1]
        
        # Adaptive weighting based on diffusion timestep
        # High noise → focus on global context
        # Low noise → focus on local details
        alpha_global = torch.sigmoid(-5 * (noise_level - 0.5)).unsqueeze(-1)  # [B, 1]
        alpha_local = 1 - alpha_global
        
        
        # STEP 2: ENCODE CONTEXT (Scene Understanding)
        
        context_encoded, gate_info = self.encode_modalities(context, return_gate_info=return_gate_info)  # [B, T_context, D]
        
        
        # STEP 3: MULTI-SCALE TRAJECTORY EMBEDDING
        
        # Fine scale (all 20 timesteps)
        trajectory_emb_fine = self.trajectory_embed(noisy_trajectory, self.training)
        if use_draft_conditioning and hasattr(self, '_draft_condition'):
            draft_emb = self.trajectory_embed(self._draft_condition, False)
            trajectory_emb_fine = trajectory_emb_fine + 0.5 * draft_emb
        future_time_emb = self.future_time_embedder(B, N)
        trajectory_emb_fine = trajectory_emb_fine + future_time_emb
        
        # Coarse scale (reduced temporal resolution for early diffusion)
        if noise_level.mean() > 0.3:  # High noise - use coarse
            trajectory_emb = self.temporal_downsample(trajectory_emb_fine, factor=2)
            T_active = T_future // 2
        else:  # Low noise - use fine
            trajectory_emb = trajectory_emb_fine
            T_active = T_future
        
        trajectory_flat = trajectory_emb.reshape(B, N * T_active, -1)
        traj_summary = trajectory_emb.mean(dim=(1, 2))  # [B, D]
        
        
        # STEP 4: PROGRESSIVE ENCODER (Conditional)
        
        decorr_loss = torch.tensor(0.0, device=noisy_trajectory.device)
        encoder_tokens = []
        encoder_summaries = []
        
        if encoder_level >= 1:
            # Level 1: Temporal
            encoder_output = self.agent_encoder(agent_states)
            level1_features = encoder_output.level1_temporal  # [B, N, D]
            history_encoded = self.encode_history_temporal(agent_history)
            
            # Weight by noise level (more history in early steps)
            level1_features = level1_features * alpha_local
            history_encoded = history_encoded * alpha_global
            
            encoder_tokens.extend([level1_features, history_encoded])
            encoder_summaries.append(history_encoded.mean(dim=1))
            
            if encoder_level >= 2:
                # Level 2: Interaction
                level2_features = encoder_output.level2_interaction  # [B, N, D]
                level2_features = level2_features * alpha_local  # Local details
                encoder_tokens.append(level2_features)
                
                if encoder_level >= 3:
                    # Level 3: Scene
                    level3_features = encoder_output.level3_scene  # [B, N, D]
                    level3_features = level3_features * alpha_global  # Global context
                    encoder_tokens.append(level3_features)
                    
                    # Decorrelation across all levels
                    decorr_loss = self.apply_decorrelation(
                        level1_features, level2_features, level3_features, B, N
                    )
        
        
        # STEP 5: CONDITIONING VECTOR
        
        c = t_emb + traj_summary
        for summary in encoder_summaries:
            c = c + summary
        
        
        # STEP 6: TRAJECTORY-CENTRIC TOKEN ORDERING
        
        # CRITICAL: Trajectory first for better denoising attention
        token_list = [
            trajectory_flat,     # [B, N*T_active, D] - PRIMARY (what we denoise)
            context_encoded,     # [B, T_context, D] - CONTEXT (guides denoising)
        ]
        token_list.extend(encoder_tokens)  # Additional agent features
        
        all_tokens = torch.cat(token_list, dim=1)
        
        # Positional embedding
        if all_tokens.shape[1] <= self.pos_embed.shape[1]:
            all_tokens = all_tokens + self.pos_embed[:, :all_tokens.shape[1], :]
        else:
            all_tokens = all_tokens + self.pos_embed
        
        
        # STEP 7: TRANSFORMER with BLOCK-WISE DECORRELATION
        
        token_select_list = []
        attn_weight_masks_list = []
        mlp_weight_masks_list = []
        
        for block_idx, block in enumerate(self.blocks):
            all_tokens, attn_mask, mlp_mask, token = block(
                all_tokens, c, t_emb, complete_model
            )
            
            # Block-wise decorrelation (prevent mode collapse during denoising)
            if self.training and encoder_level >= 2 and block_idx % 2 == 0:
                traj_tokens_block = all_tokens[:, :N*T_active, :]
                block_decorr = self.compute_block_decorrelation(traj_tokens_block, N)
                decorr_loss = decorr_loss + 0.1 * block_decorr
            
            attn_weight_masks_list.append(attn_mask)
            mlp_weight_masks_list.append(mlp_mask)
            token_select_list.append(token)
        
        
        # STEP 8: EXTRACT & UPSAMPLE TRAJECTORY
        
        # Extract trajectory tokens (first N*T_active tokens now!)
        traj_tokens = all_tokens[:, :N*T_active, :]  # [B, N*T_active, D]
        
        # Upsample if we used coarse scale
        if T_active != T_future:
            traj_tokens = self.temporal_upsample(
                traj_tokens.reshape(B, N, T_active, -1)
            )  # [B, N, T_future, D]
        else:
            traj_tokens = traj_tokens.reshape(B, N, T_future, -1)
        
        
        # STEP 9: OUTPUT PROJECTION with RESIDUAL
        
        shift, scale = self.output_adaLN(c).chunk(2, dim=1)
        traj_tokens_flat = traj_tokens.reshape(B, N * T_future, -1)
        
        from model.dydittraj import modulate
        traj_tokens_flat = modulate(
            self.output_norm(traj_tokens_flat),
            shift, scale
        )
        
        predicted_output = self.output_projection(traj_tokens_flat)
        predicted_output = predicted_output.reshape(B, N, T_future, -1) ## [B, N, T, C*2 or C]
        
        # if learn_sigma, split noise and variance

        if self.learn_sigma:
            predicted_noise, predicted_var = predicted_output.chunk(2, dim = -1)

            # concatenate diffusion model format [B, N, T, C] + [B, N, T, C] along channel dim
            predicted_output = torch.cat([predicted_noise, predicted_var], dim = -1)
        else:
            predicted_noise = predicted_output
        
        # STEP 10: RETURN
        
        if not complete_model:
            from model.dydittraj import convert_list_to_tensor
            attn_masks = convert_list_to_tensor(attn_weight_masks_list)
            mlp_masks = convert_list_to_tensor(mlp_weight_masks_list)
            token_select = convert_list_to_tensor(token_select_list)
            return predicted_noise, decorr_loss, attn_masks, mlp_masks, token_select
        else:
            return predicted_noise, decorr_loss, None, None, None

    def temporal_downsample(self, trajectory_emb, factor=2):
        """
        Downsample trajectory embeddings along the temporal dimension.
        
        Args:
            trajectory_emb: [B, N, T, D] - trajectory embeddings
            factor: downsampling factor (default=2, keeps every 2nd timestep)
        
        Returns:
            downsampled_emb: [B, N, T//factor, D]
        """
        B, N, T, D = trajectory_emb.shape
        
        if factor == 1:
            return trajectory_emb
        
        # Method 1: Simple strided sampling (fastest, good for early diffusion)
        # downsampled = trajectory_emb[:, :, ::factor, :]
        
        # Method 2: Average pooling (smoother, preserves more information)
        # Reshape to enable pooling
        trajectory_emb = trajectory_emb.permute(0, 1, 3, 2)  # [B, N, D, T]
        trajectory_emb = trajectory_emb.reshape(B * N, D, T)
        
        # Apply adaptive average pooling
        downsampled = torch.nn.functional.adaptive_avg_pool1d(
            trajectory_emb, 
            output_size=T // factor
        )
        
        # Reshape back
        downsampled = downsampled.reshape(B, N, D, T // factor)
        downsampled = downsampled.permute(0, 1, 3, 2)  # [B, N, T//factor, D]
        
        return downsampled
    
    def temporal_upsample(self, trajectory_emb, target_length=None):
        """
        Upsample trajectory embeddings along the temporal dimension.
        
        Args:
            trajectory_emb: [B, N, T_coarse, D] - coarse trajectory embeddings
            target_length: target temporal length (if None, doubles the length)
        
        Returns:
            upsampled_emb: [B, N, T_fine, D]
        """
        B, N, T_coarse, D = trajectory_emb.shape
        
        if target_length is None:
            target_length = T_coarse * 2
        
        if T_coarse == target_length:
            return trajectory_emb
        
        # Reshape for interpolation
        trajectory_emb = trajectory_emb.permute(0, 1, 3, 2)  # [B, N, D, T_coarse]
        trajectory_emb = trajectory_emb.reshape(B * N, D, T_coarse)  # [B*N, D, T_coarse] <- MUST include D!
        
        # Apply linear interpolation
        upsampled = torch.nn.functional.interpolate(
            trajectory_emb,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        # Reshape back
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
        
        # Update in-place
        level1.copy_(l1_decorr.reshape(B, N, -1))
        level2.copy_(l2_decorr.reshape(B, N, -1))
        level3.copy_(l3_decorr.reshape(B, N, -1))
        
        return decorr_loss


    def compute_block_decorrelation(self, traj_tokens, N):
        """
        Prevent trajectory mode collapse within a block.
        Encourages diversity across agents.
        """
        B, NT, D = traj_tokens.shape
        # Reshape to [B, N, T, D]
        traj_per_agent = traj_tokens.reshape(B, N, -1, D)
        
        # Compute correlation between agents
        agent_features = traj_per_agent.mean(dim=2)  # [B, N, D]
        
        # Normalize
        agent_features = F.normalize(agent_features, dim=-1)
        
        # Correlation matrix [B, N, N]
        corr = torch.bmm(agent_features, agent_features.transpose(1, 2))
        
        # Penalize high off-diagonal correlation
        identity = torch.eye(N, device=corr.device).unsqueeze(0)
        decorr_loss = torch.mean((corr - identity) ** 2)
        
        return decorr_loss

    # scoring function
    def compute_goal_reaching_score(self, trajectory, goal_position):
        """
        Score based on how close the final position is to the goal.
        
        Args:
            trajectory: [B, N, T, 5] - predicted trajectories
            goal_positions: [B, N, 2] - goal positions for each agent
        
        Returns:
            score: [B] - higher is better
        """
        final_pos = trajectory[:, :, -1, :2] # [B, N, 2]
        distances = torch.norm(final_pos - goal_position, dim = -1) #[B, N]

        # avg distance
        avg_distance = distances.mean(dim = 1) #[B]
        
        #convert to score(closer = higher score)
        score = torch.exp(-avg_distance/10.0) # exponential decay
        return score
    
    def compute_collision_score(self, trajectory):
        """
        Score based on collision avoidance between agents.
        
        Args:
            trajectory: [B, N, T, 5]
        
        Returns:
            score: [B] - higher is better (no collisions)
        """
        B, N, T, _ = trajectory.shape
        positions = trajectory[:, :, :, :2] # [B, N, T, 2]

        collision_penalty = torch.zeros(B, device= trajectory.device)
        safe_distance = 2.0 # meters

        for t in range(T):
            pos_t = positions[:, :, t, :] # [B, N, 2]

            # compute pairwise distance: [B, N, N]
            diff = pos_t.unsqueeze(2) - pos_t.unsqueeze(1) # [B, N, N, 2]
            pairwise_dist = torch.norm(diff, dim = -1) #[B, N, N]

            #create mask to ignore self-distance
            mask = ~torch.eye(N, device = trajectory.device, dtype = torch.bool)
            mask = mask.unsqueeze(0).expand(B, -1, -1) # [B, N, N]

            #check collisions
            collisions = (pairwise_dist < safe_distance) & mask
            collision_count = collisions.float().sum(dim = (1,2)) # [B]
            collision_penalty += collision_count

        score = torch.exp(-collision_penalty / (N*T))

        return score   


    def compute_kinematic_feasibility_score(self, trajectory):
        """
        Score based on kinematic constraints (velocity, acceleration limits).
        
        Args:
            trajectory: [B, N, T, 5] - (x, y, vx, vy, heading)
        
        Returns:
            score: [B] - higher is better
        """
        B, N, T, _ = trajectory.shape
        
        # Extract velocities
        velocities = trajectory[:, :, :, 2:4]  # [B, N, T, 2]
        speeds = torch.norm(velocities, dim=-1)  # [B, N, T]
        
        # Velocity constraint
        max_speed = 30.0  # m/s (adjust based on your scenario)
        speed_violations = torch.relu(speeds - max_speed)  # [B, N, T]
        speed_penalty = speed_violations.mean(dim=(1, 2))  # [B]
        
        # Acceleration constraint
        accel = velocities[:, :, 1:, :] - velocities[:, :, :-1, :]  # [B, N, T-1, 2]
        accel_mag = torch.norm(accel, dim=-1)  # [B, N, T-1]
        max_accel = 8.0  # m/s^2
        accel_violations = torch.relu(accel_mag - max_accel)
        accel_penalty = accel_violations.mean(dim=(1, 2))  # [B]
        
        # Compute positions from trajectory for jerk calculation
        positions = trajectory[:, :, :, :2]  # [B, N, T, 2]
        
        # Jerk (rate of change of acceleration)
        jerk = positions[:, :, 2:, :] - 2*positions[:, :, 1:-1, :] + positions[:, :, :-2, :]
        jerk_mag = torch.norm(jerk, dim=-1)  # [B, N, T-2]
        max_jerk = 5.0
        jerk_violations = torch.relu(jerk_mag - max_jerk)
        jerk_penalty = jerk_violations.mean(dim=(1, 2))  # [B]
        
        # Combined penalty
        total_penalty = speed_penalty + accel_penalty + 0.5 * jerk_penalty
        
        # Convert to score
        score = torch.exp(-total_penalty)
        return score


    def compute_smoothness_score(self, trajectory):
        """
        Score based on trajectory smoothness (lower jerk = smoother).
        
        Args:
            trajectory: [B, N, T, 5]
        
        Returns:
            score: [B] - higher is better
        """
        positions = trajectory[:, :, :, :2]  # [B, N, T, 2]
        
        # Compute jerk (third derivative of position)
        # Using finite differences: jerk ≈ Δ²position
        jerk = positions[:, :, 2:, :] - 2*positions[:, :, 1:-1, :] + positions[:, :, :-2, :]
        jerk_mag = torch.norm(jerk, dim=-1)  # [B, N, T-2]
        
        # Average jerk magnitude
        avg_jerk = jerk_mag.mean(dim=(1, 2))  # [B]
        
        # Convert to score (lower jerk = higher score)
        score = torch.exp(-avg_jerk / 2.0)
        return score


    def compute_diversity_score(self, trajectory, other_proposals):
        """
        Encourage diversity among proposals.
        
        Args:
            trajectory: [B, N, T, 5] - current proposal
            other_proposals: [K, B, N, T, 5] - other proposals to compare against
        
        Returns:
            score: [B] - higher means more diverse
        """
        if other_proposals is None or len(other_proposals) == 0:
            return torch.ones(trajectory.shape[0], device=trajectory.device)
        
        B, N, T, _ = trajectory.shape
        K = len(other_proposals)
        
        # Flatten trajectory for comparison
        traj_flat = trajectory.reshape(B, -1)  # [B, N*T*5]
        
        # Compute similarity with each existing proposal
        similarities = []
        for k in range(K):
            other_flat = other_proposals[k].reshape(B, -1)  # [B, N*T*5]
            
            # Cosine similarity
            similarity = F.cosine_similarity(traj_flat, other_flat, dim=1)  # [B]
            similarities.append(similarity)
        
        similarities = torch.stack(similarities, dim=0)  # [K, B]
        
        # Diversity = 1 - max_similarity (want to be different from all others)
        max_similarity = similarities.max(dim=0)[0]  # [B]
        diversity = 1.0 - max_similarity
        
        return torch.relu(diversity)  # Ensure non-negative


    def score_trajectory(self, trajectory, goal_positions, other_proposals=None, weights=None):
        """
        Comprehensive scoring function combining all criteria.
        
        Args:
            trajectory: [B, N, T, 5]
            goal_positions: [B, N, 2]
            other_proposals: Optional[List] - for diversity scoring
            weights: Optional[Dict] - weights for each criterion
        
        Returns:
            total_score: [B]
            score_dict: Dict with individual scores
        """
        if weights is None:
            weights = {
                'goal': 1.0,
                'collision': 2.0,  # Prioritize safety
                'kinematic': 0.8,
                'smoothness': 0.5,
                'diversity': 0.3,
            }
        
        # Compute individual scores
        goal_score = self.compute_goal_reaching_score(trajectory, goal_positions)
        collision_score = self.compute_collision_score(trajectory)
        kinematic_score = self.compute_kinematic_feasibility_score(trajectory)
        smoothness_score = self.compute_smoothness_score(trajectory)
        diversity_score = self.compute_diversity_score(trajectory, other_proposals)
        
        # Weighted combination
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
    """
    Complete implementation following the paper:
    1. Generate K proposals (Stage 1)
    2. Select/blend best proposals (Stage 2a)
    3. Refine selected trajectory (Stage 2b - two-pass)
    """
    
    def __init__(
        self,
        transdiffuser_model,
        diffusion,
        # Multi-proposal parameters
        num_proposals=20,
        selection_strategy='top_k_blend',
        top_k_for_blend=5,
        score_weights=None,
        # Refinement parameters
        use_refinement=True,
        refinement_noise_level=0.1,
        refinement_steps_ratio=0.3,  # Use 30% of original steps for refinement
    ):
        super().__init__()
        self.model = transdiffuser_model
        self.diffusion = diffusion
        self.learn_sigma = transdiffuser_model.learn_sigma
        
        # Multi-proposal parameters
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
        
        # Refinement parameters
        self.use_refinement = use_refinement
        self.refinement_noise_level = refinement_noise_level
        self.refinement_steps_ratio = refinement_steps_ratio
    
    def forward(self, context, agent_states, agent_history, goal_positions=None, t=None):
        """Training forward pass (unchanged)."""
        B, N, _ = agent_states.shape
        device = agent_states.device
        
        if goal_positions is None:
            goal_positions = agent_states[:, :, :2]
        
        # Sample ground truth trajectory (REPLACE WITH YOUR ACTUAL DATA)
        gt_trajectory = torch.randn(B, N, 20, 5, device=device)
        
        if t is None:
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        
        noise = torch.randn_like(gt_trajectory)
        noisy_trajectory = self.diffusion.q_sample(gt_trajectory, t, noise)
        
        predicted_output, decorr_loss, attn_masks, mlp_masks, token_select = self.model(
            context=context,
            agent_states=agent_states,
            noisy_trajectory=noisy_trajectory,
            agent_history=agent_history,
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
    def generate_single_proposal(
        self, 
        context, 
        agent_states, 
        agent_history, 
        num_inference_steps=50,
        seed=None
    ):
        """Generate a single trajectory proposal (one denoising pass)."""
        B, N, _ = agent_states.shape
        device = agent_states.device
        
        if seed is not None:
            torch.manual_seed(seed)
        
        trajectory = torch.randn(B, N, 20, 5, device=device)
        
        def model_fn(x, t):
            predicted_noise, _, _, _, _ = self.model(
                context=context,
                agent_states=agent_states,
                noisy_trajectory=x,
                agent_history=agent_history,
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
    def refine_trajectory(
        self,
        trajectory_draft,
        context,
        agent_states,
        agent_history,
        num_inference_steps=50,
    ):
        """
        Two-pass refinement: re-denoise the draft trajectory.
        
        Args:
            trajectory_draft: [B, N, T, 5] - selected/blended trajectory from proposals
            context, agent_states, agent_history: same as before
            num_inference_steps: original number of steps
        
        Returns:
            refined_trajectory: [B, N, T, 5]
        """
        B, N, T, C = trajectory_draft.shape
        device = trajectory_draft.device
        
        # Calculate refinement parameters
        refinement_steps = int(self.refinement_steps_ratio * num_inference_steps)
        
        # Add noise to draft (partial noise, not full noise)
        t_refine = torch.full((B,), refinement_steps, device=device, dtype=torch.long)
        noise = torch.randn_like(trajectory_draft)
        noisy_draft = self.diffusion.q_sample(trajectory_draft, t_refine, noise)
        
        print(f"Refining trajectory with {refinement_steps} steps...")
        
        # Store draft as conditioning
        self.model.register_buffer('_draft_condition', trajectory_draft)
        
        def model_fn_refine(x, t):
            """Model function with draft conditioning."""
            predicted_noise, _, _, _, _ = self.model(
                context=context,
                agent_states=agent_states,
                noisy_trajectory=x,
                agent_history=agent_history,
                t=t,
                complete_model=True,
                use_draft_conditioning=True  # Signal to use draft
            )
            return predicted_noise
        
        # Denoise from refinement_steps -> 0
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
        
        # Clean up buffer
        delattr(self.model, '_draft_condition')
        
        print("Refinement complete.")
        return trajectory_refined
    
    @torch.no_grad()
    def sample(
        self, 
        context, 
        agent_states, 
        agent_history, 
        goal_positions=None,
        num_inference_steps=50,
        return_all_proposals=False,
        return_scores=False,
    ):
        """
        Complete sampling pipeline following the paper:
        1. Generate K proposals
        2. Score and select/blend
        3. Refine selected trajectory (two-pass)
        """
        B, N, _ = agent_states.shape
        device = agent_states.device
        
        if goal_positions is None:
            goal_positions = agent_states[:, :, :2]
        
        # === STAGE 1: Multi-Proposal Generation ===
        print(f"\n=== Stage 1: Generating {self.num_proposals} proposals ===")
        
        all_proposals = []
        all_score_dicts = []
        
        for k in range(self.num_proposals):
            proposal = self.generate_single_proposal(
                context=context,
                agent_states=agent_states,
                agent_history=agent_history,
                num_inference_steps=num_inference_steps,
                seed=k
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
        
        all_proposals = torch.stack(all_proposals, dim=0)  # [K, B, N, T, 5]
        all_scores = torch.stack([sd['total'] for sd in all_score_dicts], dim=0)  # [K, B]
        
        # === STAGE 2a: Selection ===
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
        
        best_scores = all_scores.max(dim=0)[0]
        print(f"Selection complete. Best scores: {best_scores.cpu().numpy()}")
        
        # === STAGE 2b: Refinement (Two-Pass) ===
        if self.use_refinement:
            print(f"\n=== Stage 2b: Two-pass refinement ===")
            final_trajectory = self.refine_trajectory(
                trajectory_draft=selected_trajectory,
                context=context,
                agent_states=agent_states,
                agent_history=agent_history,
                num_inference_steps=num_inference_steps,
            )
        else:
            final_trajectory = selected_trajectory
            print("\n=== Refinement disabled, using selected trajectory ===")
        
        print("\n=== Sampling complete ===\n")
        
        # Return based on flags
        if return_all_proposals and return_scores:
            return final_trajectory, all_proposals, all_scores
        elif return_all_proposals:
            return final_trajectory, all_proposals
        elif return_scores:
            return final_trajectory, all_scores
        else:
            return final_trajectory
    
    def _select_best(self, proposals, scores):
        """Select single best proposal."""
        K, B, N, T, C = proposals.shape
        best_indices = scores.argmax(dim=0)
        best_trajectories = []
        for b in range(B):
            best_k = best_indices[b]
            best_trajectories.append(proposals[best_k, b])
        return torch.stack(best_trajectories, dim=0)
    
    def _weighted_blend_all(self, proposals, scores):
        """Blend all proposals with softmax weights."""
        K, B, N, T, C = proposals.shape
        weights = torch.softmax(scores, dim=0)
        weights_expanded = weights.view(K, B, 1, 1, 1)
        blended = (proposals * weights_expanded).sum(dim=0)
        return blended
    
    def _top_k_blend(self, proposals, scores, k=5):
        """Blend top-k proposals."""
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


# pack to interface
def create_transdiffuser(
    hidden_size=768,
    depth=12,
    num_heads=12,
    decorr_weight=0.1,
    max_agents=8,
    future_horizon=20,
    # Multi-proposal parameters
    num_proposals=20,
    selection_strategy='top_k_blend',
    top_k_for_blend=5,
    # Refinement parameters
    use_refinement=True,
    refinement_noise_level=0.1,
    refinement_steps_ratio=0.3,
):
    """
    Factory function following the paper:
    - Multi-proposal generation (K=20)
    - Selection/blending
    - Two-pass refinement
    """
    base_model = TransDiffuserIntegrated(
        input_size=64,
        patch_size=4,
        traj_channels=5,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        decorr_weight=decorr_weight,
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
        refinement_noise_level=refinement_noise_level,
        refinement_steps_ratio=refinement_steps_ratio,
    )
    
    return model

def create_dataset(
        dataset = 'navsim', # right now is navsim, but we will test into other dataset (H3D, ONCE) and PNK dataset(we will convert into Nuplan style)
        phase = 'phase0', # shall be list as phase 0-4, depend on, we will make 
        difficulty_level = 'easy',

):
    if dataset == 'navsim':
        import os
        from DDPM.datasets.navsim.navsim_utilize.navsimdataset import NavsimDataset
        from DDPM.datasets.navsim.navsim_utilize.enhancenavsim import EnhancedNavsimDataset, TrajectoryConfig, DifficultyLevel
        from DDPM.datasets.navsim.navsim_utilize.navsimwithphase import EnhancedNavsimDatasetWithPhases, precompute_phase_0, precompute_phase_1, precompute_phase_2
        
        map_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
        os.environ['NUPLAN_MAPS_ROOT'] = map_root
        
        print(f"\n✓ Set NUPLAN_MAPS_ROOT to: {map_root}")
        print(f"✓ Verifying map directory exists...")
        
        if not Path(map_root).exists():
            print(f"❌ ERROR: Map directory not found at {map_root}")
            print("Please check the path and try again.")
            exit(1)
        
        print(f"✓ Map directory found!")
        
        # List available maps
        print(f"\n📁 Maps in directory:")
        for item in Path(map_root).iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        
        # Create dataset
        if phase == 'phase0':        
            dataset = NavsimDataset(
                bev_size=(200, 200),
                bev_range=50.0,
                use_uniad_bev=False,
                extract_labels=True,
                use_cache=False,
                map_root=map_root
            )
        elif phase == 'phase1':
            dataset = EnhancedNavsimDataset(
                data_split= "mini", # we will consider to do full trainval (2TB) of nuplan dataset if facility required is sufficient.
                bev_size= (200, 200),
                bev_range= 50,
                trajectory_sampling= TrajectoryConfig.PLANNING_TRAJECTORY_SAMPLING,
                difficulty_filter= DifficultyLevel.EASY,
                extract_labels=True,
                extract_route_info=True,
                use_cache=False,
                map_root= map_root
            )

        elif phase == 'phase2':
            # phase 2 here is what we consider from phase 1 (phase 0)
                dataset = EnhancedNavsimDatasetWithPhases(
                data_split='mini',
                enable_phase_0=False,
                enable_phase_1=True,
                enable_phase_2=False,
                use_cache=True,
                force_recompute=True,
            )
    
        elif phase == 'phase3':
            dataset = EnhancedNavsimDatasetWithPhases(
                data_split='mini',
                enable_phase_0=False,
                enable_phase_1=False,
                enable_phase_2=True,
                use_cache=True,
                force_recompute=True,
            )

        else:
            return ValueError(f'{phase} is not available, current selection shall be "phase0-4" ')
            
    else:
        # place holder for current situation
        return ValueError(f'for now only navsim availabel for testing, another will be Waymo, H3D, Once, and so on. final is PNK. Only {dataset} is available')
    

    return dataset
if __name__ == "__main__":


    model = create_transdiffuser(
        hidden_size=768,
        depth=12,
        num_heads=12,
        max_agents=8
    )
    
    # Example input
    B, N = 2, 8
    context = {
        'BEV': torch.randn(B, 7, 64, 64),
        'lidar': torch.randn(B, 2, 64, 64),
    }
    agent_states = torch.randn(B, N, 5)
    agent_history = torch.randn(B, N, 30, 5)
    
    # Training
    model.train()
    loss_dict = model(context, agent_states, agent_history)
    print("Training losses:", loss_dict)
    
    # Inference
    model.eval()
    trajectory, all_proposals, all_scores = model.sample(context, agent_states, agent_history, num_inference_steps=50)

    # Analyze best proposal
    # best_idx = all_scores[:, 0].argmax()
    # best_proposal = all_proposals[best_idx, 0]  # [N, T, 5]
    
    # print("\n=== Detailed Score Breakdown ===")
    # _, score_dict = model.model.score_trajectory(
    #     best_proposal.unsqueeze(0),  # Add batch dim
    #     goal_positions[:1],
    #     weights=model.score_weights
    # )
    
    # for key, value in score_dict.items():
    #     print(f"{key:15s}: {value.item():.4f}")
    
    # # Check trajectory statistics
    # print("\n=== Trajectory Statistics ===")
    # velocities = best_proposal[:, :, 2:4]
    # speeds = torch.norm(velocities, dim=-1)
    # print(f"Speed: min={speeds.min():.2f}, max={speeds.max():.2f}, mean={speeds.mean():.2f} m/s")
    
    # accel = velocities[:, 1:, :] - velocities[:, :-1, :]
    # accel_mag = torch.norm(accel, dim=-1)
    # print(f"Accel: min={accel_mag.min():.2f}, max={accel_mag.max():.2f}, mean={accel_mag.mean():.2f} m/s²")
    
    # final_pos = best_proposal[:, -1, :2]
    # distances = torch.norm(final_pos - goal_positions[0], dim=-1)
    # print(f"Goal distance: {distances.cpu().numpy()}")
    
    # # Collision check
    # positions = best_proposal[:, :, :2]  # [N, T, 2]
    # for t in [0, 10, 19]:  # Check beginning, middle, end
    #     pos_t = positions[:, t, :]
    #     dists = torch.cdist(pos_t, pos_t)
    #     min_dist = dists[dists > 0].min()
    #     print(f"Min inter-agent distance at t={t}: {min_dist:.2f} m")
    print(f"Generated trajectory shape: {trajectory.shape}")  # [2, 8, 20, 5]