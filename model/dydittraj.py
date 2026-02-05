import torch
import torch.nn as nn
import numpy as np
import math
from torch.jit import Final 
from timm.layers.helpers import to_2tuple
from functools import partial
from einops import rearrange

# ========================================
# UTILITY FUNCTIONS
# ========================================

def convert_list_to_tensor(list_convert):
    if len(list_convert) and list_convert[0] is not None:
        result = torch.stack(list_convert, dim=1)
    else:
        result = None
    return result


def _gumbel_sigmoid(
        logits, tau=1, hard=False, eps=1e-10, training=True, threshold=0.5
):
    if training:
        gumbles1 = (
            -torch.empty_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).exponential_().log()
        )
        gumbles2 = (
            -torch.empty_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).exponential_().log()
        )
        gumbles1 = (logits + gumbles1 - gumbles2) / tau
        y_soft = gumbles1.sigmoid()
    else:
        y_soft = logits.sigmoid()

    if hard:
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret


# ========================================
# DYNAMIC MODULES
# ========================================

class TokenSelect(nn.Module):
    def __init__(
            self,
            in_channels,
            num_sub_layer=1, 
            tau=5,
            is_hard=True,
            threshold=0.5,
            bias=True,
    ):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=bias),
            nn.ReLU(),
            nn.Linear(in_channels // 16, 1, bias=bias)
        )
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        """
        Args:
            x: (N, T, D)
        Returns:
            token_select: (N, T, 1)
            logits: (N, T, 1)
        """
        logits = self.mlp_head(x)  # (N, T, 1)
        token_select = _gumbel_sigmoid(
            logits, self.tau, self.is_hard, 
            threshold=self.threshold, training=self.training
        )
        return token_select, logits


class DiffRate(nn.Module):
    def __init__(
            self,
            dim=768,
            channel_number=196, 
            tau=5, 
            is_hard=True, 
            threshold=0.5
    ) -> None:
        super().__init__()
        self.dim = dim
        self.channel_number = channel_number
        self.router = nn.Linear(dim, channel_number)
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def forward(self, x):
        """
        Args:
            x: (N, D) or (N, T, D)
        Returns:
            channel_select: (N, channel_number)
        """
        if x.dim() == 3:
            x = x.mean(dim=1)  # (N, D)
        
        logits = self.router(x)  # (N, channel_number)
        channel_select = _gumbel_sigmoid(
            logits, self.tau, self.is_hard, 
            threshold=self.threshold, training=self.training
        )
        return channel_select


# ========================================
# DYNAMIC LINEAR LAYERS
# ========================================

def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
    new_width_mult = round(num_heads * width_mult) * 1.0 / num_heads
    input_size = int(new_width_mult * input_size)
    new_input_size = max(min_value, input_size)
    return new_input_size


class DynaLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_heads, head_dim, 
                 bias=True, dyna_dim=[True, True], width_mult=1.):
        super(DynaLinear, self).__init__(in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.width_mult = width_mult
        self.dyna_dim = dyna_dim

    def forward(self, input_x):
        if self.dyna_dim[0]:
            self.in_features = round_to_nearest(
                self.in_features_max, self.width_mult, self.num_heads
            )
        if self.dyna_dim[1]:
            self.out_features = round_to_nearest(
                self.out_features_max, self.width_mult, self.num_heads
            )

        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        
        return nn.functional.linear(input_x, weight, bias)


class DynaQKVLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_heads, head_dim, 
                 bias=True, dyna_dim=[True, True], width_mult=1.):
        super(DynaQKVLinear, self).__init__(in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features // 3
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.width_mult = width_mult
        self.dyna_dim = dyna_dim

    def forward(self, input_x):
        if self.dyna_dim[0]:
            self.in_features = round_to_nearest(
                self.in_features_max, self.width_mult, self.num_heads
            )
        if self.dyna_dim[1]:
            self.out_features = round_to_nearest(
                self.out_features_max, self.width_mult, self.num_heads
            )

        weight = rearrange(self.weight, "(qkv in_c) out_c -> qkv in_c out_c", qkv=3)
        weight = weight[:, :self.out_features, :self.in_features]
        weight = rearrange(weight, "qkv in_c out_c -> (qkv in_c) out_c")

        if self.bias is not None:
            bias = rearrange(self.bias, "(qkv out_c) -> qkv out_c", qkv=3)
            bias = bias[:, :self.out_features]
            bias = rearrange(bias, "qkv out_c -> (qkv out_c)")
        else:
            bias = self.bias
            
        return nn.functional.linear(input_x, weight, bias)


# ========================================
# ATTENTION AND MLP
# ========================================

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = DynaQKVLinear(
            in_features=dim, out_features=dim * 3, num_heads=self.num_heads, 
            head_dim=self.head_dim, bias=qkv_bias, dyna_dim=[False, True]
        )
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DynaLinear(
            in_features=dim, out_features=dim, num_heads=self.num_heads,  
            head_dim=self.head_dim, bias=True, dyna_dim=[True, False]
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, channel_mask=None) -> torch.Tensor:
        """
        Args:
            x: (N, T, D)
            channel_mask: (N, num_heads) or None
        Returns:
            x: (N, T, D)
        """
        B, N, C = x.shape
        qkv = self.qkv(x)  # (N, T, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (N, num_heads, T, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (N, num_heads, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v  # (N, num_heads, T, head_dim)

        x = x.transpose(1, 2).reshape(B, N, -1)  # (N, T, D)
        
        if channel_mask is not None:
            # channel_mask: (N, num_heads) -> (N, 1, num_heads * head_dim)
            channel_mask = channel_mask.unsqueeze(dim=2).repeat(1, 1, self.head_dim)
            channel_mask = channel_mask.flatten(1).unsqueeze(1)  # (N, 1, D)
            x = channel_mask * x
                    
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            num_heads=8
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = DynaLinear(
            in_features=in_features, out_features=hidden_features, 
            num_heads=num_heads, head_dim=(hidden_features // num_heads), 
            bias=bias[0], dyna_dim=[False, True]
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = DynaLinear(
            in_features=hidden_features, out_features=out_features, 
            num_heads=num_heads, head_dim=(hidden_features // num_heads), 
            bias=bias[1], dyna_dim=[True, False]
        )
        self.drop2 = nn.Dropout(drop_probs[1])
        self.head_dim = hidden_features // num_heads

    def forward(self, x, channel_mask=None):
        """
        Args:
            x: (N, T, D)
            channel_mask: (N, num_heads) or None
        Returns:
            x: (N, T, D)
        """
        x = self.fc1(x)  # (N, T, hidden_features)
        
        if channel_mask is not None:
            channel_mask = channel_mask.unsqueeze(dim=2).repeat(1, 1, self.head_dim)
            channel_mask = channel_mask.flatten(1).unsqueeze(1)  # (N, 1, hidden_features)
            x = channel_mask * x
            
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def modulate(x, shift, scale):
    """
    Args:
        x: (N, T, D)
        shift: (N, D)
        scale: (N, D)
    Returns:
        x: (N, T, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ========================================
# EMBEDDERS
# ========================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        """
        Args:
            t: (N,) timesteps
        Returns:
            t_emb: (N, hidden_size)
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TrajectoryEmbedder(nn.Module):
    def __init__(self, trajectory_dim=5, hidden_size=768, dropout_prob=0.1):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.mlp = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dropout_prob = dropout_prob
        self.null_trajectory = nn.Parameter(torch.zeros(trajectory_dim))
        
    def forward(self, trajectory, training=True):
        """
        Args:
            trajectory: (B, N, T, 5) - Multi-agent noisy trajectories
            training: bool
        Returns:
            y_emb: (B, N, T, hidden_size)
        """
        B, N, T, C = trajectory.shape
        
        if training and self.dropout_prob > 0:
            # Dropout entire agents randomly for classifier-free guidance
            mask = torch.rand(B, N, device=trajectory.device) < self.dropout_prob
            mask = mask.unsqueeze(2).unsqueeze(3)  # (B, N, 1, 1)
            null_traj = self.null_trajectory.view(1, 1, 1, -1).expand(B, N, T, -1)
            trajectory = torch.where(mask, null_traj, trajectory)
        
        # Reshape for MLP: (B, N, T, 5) → (B*N*T, 5)
        trajectory = trajectory.reshape(B * N * T, C)
        emb = self.mlp(trajectory)  # (B*N*T, hidden_size)
        emb = emb.reshape(B, N, T, -1)  # (B, N, T, hidden_size)
        
        return emb


class HistoryEncoder(nn.Module):
    """
    Encodes agent history by pooling temporal frames.
    """
    def __init__(self, in_channels=5, hidden_size=768, history_length=30, pool_size=5):
        super().__init__()
        self.pool_size = pool_size
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * pool_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, 30, 5) - Multi-agent history
        Returns:
            out: (B, N, 6, hidden_size) - Pooled history per agent
        """
        B, N, T, C = x.shape
        # Pool every 5 frames: (B, N, 30, 5) → (B, N, 6, 25)
        x = x.reshape(B, N, T // self.pool_size, self.pool_size * C)
        
        # Reshape for MLP: (B, N, 6, 25) → (B*N*6, 25)
        x = x.reshape(B * N * (T // self.pool_size), self.pool_size * C)
        x = self.mlp(x)  # (B*N*6, hidden_size)
        x = x.reshape(B, N, T // self.pool_size, -1)  # (B, N, 6, hidden_size)
        
        return x


class FutureTimeEmbedder(nn.Module):
    """
    Learnable temporal embeddings for future timesteps.
    """
    def __init__(self, hidden_size=768, max_len=20):
        super().__init__()
        self.time_embed = nn.Parameter(torch.randn(max_len, hidden_size) * 0.02)
    
    def forward(self, batch_size, num_agents):
        """
        Returns:
            (B, N, 20, hidden_size) - Broadcast temporal embeddings
        """
        # (20, D) → (1, 1, 20, D) → (B, N, 20, D)
        return self.time_embed.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_agents, -1, -1
        )


# ========================================
# ENCODER OUTPUT CONTAINER
# ========================================

class EncoderOutput:
    """
    Container for encoder outputs following the responsibility matrix.
    Medium-term decoder (diffusion) uses level2 (primary) + level3 (secondary).
    """
    def __init__(self):
        # Core representations (from matrix)
        self.level1_temporal = None      # [B, N, D] - for short decoder
        self.level2_interaction = None   # [B, N, D] - PRIMARY for medium decoder
        self.level3_scene = None         # [B, N, D] - SECONDARY for medium decoder
        
        # Lane hierarchy
        self.lanes_fine = None           # [B, 500, D] - for short decoder
        self.lanes_groups = None         # [B, 100, D] - for medium decoder  
        self.lanes_network = None        # [B, 20, D] - for long decoder
        
        # Future context (geometric, no parameters)
        self.curvature = None            # [B, N, 1]
        self.reachable_2s = None         # [B, N, K]
        self.reachable_5s = None         # [B, N, K] - for medium decoder
        self.reachable_8s = None         # [B, N, K]
        self.signal_timing = None        # [B, N, 3]
        self.conflicts = None            # [B, N, N]
        
        # Intersection module
        self.in_intersection = None      # [B, N, 1]
        self.approach_lane = None        # [B, N, 4]
        self.turn_intention = None       # [B, N, 3]
        self.right_of_way = None         # [B, N, 1]
        self.conflict_points = None      # [B, N, N, K]
        
        # Pedestrian module
        self.crossing_intention = None   # [B, N_ped, 1]
        self.vehicle_proximity = None    # [B, N_ped, N_veh]
        self.group_behavior = None       # [B, N_ped, N_ped]
        
        # Occlusion module
        self.occlusion_map = None        # [B, H, W]
        self.agent_risk = None           # [B, N, 1]
        
        # Interaction semantics


        # Interaction semantics
        self.interaction_types = None    # [B, N, N, 6]
        self.interaction_features = None # [B, N, N, F]
        
        # Optional
        self.goals = None                # [B, N, 2] if available


# ========================================
# AGENT ENCODER PLACEHOLDER
# ========================================

class AgentEncoder(nn.Module):
    """
    Placeholder for multi-level agent encoder.
    Will be replaced with actual encoder following responsibility matrix.
    
    This encoder should produce:
    - level1_temporal: Raw kinematics features
    - level2_interaction: Social context with agent-agent attention
    - level3_scene: Full scene understanding with map
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Simple placeholder: just embed current agent states
        self.state_mlp = nn.Sequential(
            nn.Linear(5, hidden_size),  # [x, y, vx, vy, heading]
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, agent_states):
        """
        Args:
            agent_states: (B, N, 5) - Current agent states
        Returns:
            encoder_output: EncoderOutput object with placeholders
        """
        B, N, _ = agent_states.shape
        D = self.hidden_size
        device = agent_states.device
        
        # Create encoder output container
        output = EncoderOutput()
        
        # Embed agent states: (B, N, 5) → (B, N, D)
        agent_emb = self.state_mlp(agent_states.reshape(B * N, -1))
        agent_emb = agent_emb.reshape(B, N, D)
        
        # For now, use same embedding for all levels (placeholder)
        output.level1_temporal = agent_emb
        output.level2_interaction = agent_emb  # PRIMARY for medium decoder
        output.level3_scene = agent_emb
        
        # Placeholder values for other required outputs
        output.lanes_groups = torch.zeros(B, 100, D, device=device)
        output.turn_intention = torch.zeros(B, N, 3, device=device)
        output.right_of_way = torch.zeros(B, N, 1, device=device)
        output.interaction_types = torch.zeros(B, N, N, 6, device=device)
        
        return output


# ========================================
# MODALITY EMBEDDERS
# ========================================

class LidarEmbedding(nn.Module):
    def __init__(self, in_channels=2, hidden_size=768, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.modality_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
    def forward(self, x):
        """
        Args:
            x: (B, 2, H, W)
        Returns:
            out: (B, T_lidar, hidden_size) where T_lidar = (H//patch_size)*(W//patch_size) + 1
        """
        B = x.shape[0]
        x = self.proj(x)  # (B, hidden_size, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        modality_token = self.modality_token.expand(B, -1, -1)  # (B, 1, hidden_size)
        x = torch.cat([modality_token, x], dim=1)  # (B, T_lidar, hidden_size)
        return x


class ImgCNN(nn.Module):
    def __init__(self, in_channels=3, hidden_size=768, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.modality_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            out: (B, T_img, hidden_size)
        """
        B = x.shape[0]
        x = self.proj(x)  # (B, hidden_size, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        modality_token = self.modality_token.expand(B, -1, -1)
        x = torch.cat([modality_token, x], dim=1)  # (B, T_img, hidden_size)
        return x


class BEVEncoder(nn.Module):
    def __init__(self, in_channels=7, hidden_size=768, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.modality_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
    def forward(self, x):
        """
        Args:
            x: (B, 7, H, W)
        Returns:
            out: (B, T_bev, hidden_size)
        """
        B = x.shape[0]
        x = self.proj(x)  # (B, hidden_size, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        modality_token = self.modality_token.expand(B, -1, -1)
        x = torch.cat([modality_token, x], dim=1)  # (B, T_bev, hidden_size)
        return x


class ActionEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_size=768, max_seq_len=10):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.modality_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
    def forward(self, x):
        """
        Args:
            x: (B, 4, T_seq) or (B, 4)
        Returns:
            out: (B, T_action, hidden_size)
        """
        B = x.shape[0]
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (B, T_seq, 4)
            x = self.mlp(x)  # (B, T_seq, hidden_size)
        else:
            x = self.mlp(x).unsqueeze(1)  # (B, 1, hidden_size)
        
        modality_token = self.modality_token.expand(B, -1, -1)
        x = torch.cat([modality_token, x], dim=1)  # (B, T_action, hidden_size)
        return x


class EgoEmbedding(nn.Module):
    def __init__(self, in_channels=7, hidden_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 7)
        Returns:
            out: (B, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (B, 1, hidden_size)
        return x


class BehaviorEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 6)
        Returns:
            out: (B, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (B, 1, hidden_size)
        return x


class IntersectionEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 5)
        Returns:
            out: (B, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (B, 1, hidden_size)
        return x


class PedestrianEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 5)
        Returns:
            out: (B, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (B, 1, hidden_size)
        return x


class TrafficEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 5)
        Returns:
            out: (B, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (B, 1, hidden_size)
        return x


class Occlusion(nn.Module):
    def __init__(self, in_channels=5, hidden_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 5)
        Returns:
            out: (B, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (B, 1, hidden_size)
        return x


class GoalIntent(nn.Module):
    def __init__(self, in_channels=5, hidden_size=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 5)
        Returns:
            out: (B, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (B, 1, hidden_size)
        return x


# ========================================
# CROSS-MODAL ATTENTION
# ========================================

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query_modality, key_value_modality):
        """
        Args:
            query_modality: (B, T1, D)
            key_value_modality: (B, T2, D)
        Returns:
            out: (B, T1, D)
        """
        B, T1, D = query_modality.shape
        T2 = key_value_modality.shape[1]
        
        q = self.q_proj(query_modality).reshape(B, T1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value_modality).reshape(B, T2, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value_modality).reshape(B, T2, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T1, D)
        out = self.out_proj(out)
        
        return out


# ========================================
# DiT BLOCK
# ========================================

class DiTBlock(nn.Module):
    def __init__(
            self, 
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            use_modality_specific=False,
            **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.use_modality_specific = use_modality_specific
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, 
            act_layer=approx_gelu, drop=0, num_heads=num_heads
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.attn_rate = DiffRate(dim=hidden_size, channel_number=num_heads)
        self.mlp_rate = DiffRate(dim=hidden_size, channel_number=num_heads)
        self.token_selection = TokenSelect(in_channels=hidden_size, num_sub_layer=1)

    def forward(self, x, c, t_emb, complete_model):
        """
        Args:
            x: (B, T, D) - input tokens
            c: (B, D) - conditioning (timestep + trajectory)
            t_emb: (B, D) - timestep embedding for rate controllers
            complete_model: bool
        Returns:
            x: (B, T, D)
            attn_weight_mask: (B, num_heads) or None
            mlp_weight_mask: (B, num_heads) or None
            token_select: (B, T, 1) or None
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention branch
        if not complete_model:
            attn_weight_mask = self.attn_rate(t_emb)  # (B, num_heads)
        else:
            attn_weight_mask = None

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), 
            attn_weight_mask
        )
        
        # MLP branch
        if not complete_model:
            token_select, _ = self.token_selection(x)  # (B, T, 1)
            mlp_weight_mask = self.mlp_rate(x)  # (B, num_heads)
            
            mlp_out = gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp),
                mlp_weight_mask
            )
            x = x + token_select * mlp_out
        else:
            token_select = None
            mlp_weight_mask = None
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        
        return x, attn_weight_mask, mlp_weight_mask, token_select


# ========================================
# FINAL LAYER
# ========================================

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x: (B, T, D)
            c: (B, D)
        Returns:
            x: (B, T, patch_size^2 * out_channels)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ========================================
# POSITIONAL EMBEDDING
# ========================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ========================================
# MAIN MODEL
# ========================================

class TransDiffuserDiT(nn.Module):
    """
    Multi-modal Multi-agent Diffusion Transformer for trajectory prediction.
    
    Supports:
    - Multi-agent trajectory prediction (B, N, T, 5)
    - Multiple fusion strategies (parallel/hierarchical)
    - Encoder-decoder separation following responsibility matrix
    """
    
    def __init__(
        self,
        input_size=64,
        patch_size=4,
        traj_channels=5,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        trajectory_dropout_prob=0.1,
        learn_sigma=True,
        use_modality_specific=True,
        parallel=True,
        modality_config=None,
        max_agents=32,
        future_horizon=20,
        history_length=30,
    ):
        super().__init__()
        
        self.learn_sigma = learn_sigma
        self.hidden_size = hidden_size
        self.traj_channels = traj_channels
        self.out_channels = traj_channels * 2 if learn_sigma else traj_channels
        self.num_heads = num_heads
        self.depth = depth
        self.use_modality_specific = use_modality_specific
        self.parallel = parallel
        self.input_size = input_size
        self.patch_size = patch_size
        self.max_agents = max_agents
        self.future_horizon = future_horizon
        self.history_length = history_length
        
        # Default modality configuration
        if modality_config is None:
            self.modality_config = {
                'lidar': 2,
                'img': 3,
                'BEV': 7,
                'action': 4,
                'ego': 7,
                'behavior': 6,
                'intersection': 5,
                'pedestrian': 5,
                'traffic_control': 5,
                'occlusion': 5,
                'goal_intent': 5,
            }
        else:
            self.modality_config = modality_config
        
        # Create modality-specific embedders (for context - shared across agents)
        if use_modality_specific:
            self.modality_embedders = nn.ModuleDict()
            
            for modality_name, modality_channels in self.modality_config.items():
                if modality_name == 'lidar':
                    self.modality_embedders[modality_name] = LidarEmbedding(
                        modality_channels, hidden_size, patch_size
                    )
                elif modality_name == 'BEV':
                    self.modality_embedders[modality_name] = BEVEncoder(
                        modality_channels, hidden_size, patch_size
                    )
                elif modality_name == 'img':
                    self.modality_embedders[modality_name] = ImgCNN(
                        modality_channels, hidden_size, patch_size
                    )
                elif modality_name == 'action':
                    self.modality_embedders[modality_name] = ActionEncoder(
                        modality_channels, hidden_size
                    )
                elif modality_name == 'ego':
                    self.modality_embedders[modality_name] = EgoEmbedding(
                        modality_channels, hidden_size
                    )
                elif modality_name == 'behavior':
                    self.modality_embedders[modality_name] = BehaviorEncoder(
                        modality_channels, hidden_size
                    )
                elif modality_name == 'intersection':
                    self.modality_embedders[modality_name] = IntersectionEncoder(
                        modality_channels, hidden_size
                    )
                elif modality_name == 'pedestrian':
                    self.modality_embedders[modality_name] = PedestrianEncoder(
                        modality_channels, hidden_size
                    )
                elif modality_name == 'traffic_control':
                    self.modality_embedders[modality_name] = TrafficEncoder(
                        modality_channels, hidden_size
                    )
                elif modality_name == 'occlusion':
                    self.modality_embedders[modality_name] = Occlusion(
                        modality_channels, hidden_size
                    )
                elif modality_name == 'goal_intent':
                    self.modality_embedders[modality_name] = GoalIntent(
                        modality_channels, hidden_size
                    )
                else:
                    raise ValueError(f"Unknown modality: {modality_name}")
            
            # Cross-modal attention for hierarchical fusion
            if not parallel:
                self.cross_modal_attentions = nn.ModuleList([
                    CrossModalAttention(hidden_size, num_heads) 
                    for _ in range(len(self.modality_config) - 1)
                ])
        
        # Agent encoder (placeholder for actual multi-level encoder)
        self.agent_encoder = AgentEncoder(hidden_size)
        
        # History encoder (multi-agent)
        self.history_encoder = HistoryEncoder(
            traj_channels, hidden_size, history_length, pool_size=5
        )
        
        # Future time embedder (multi-agent)
        self.future_time_embedder = FutureTimeEmbedder(hidden_size, max_len=future_horizon)
        
        # Trajectory and timestep embedders
        self.trajectory_embed = TrajectoryEmbedder(
            traj_channels, hidden_size, trajectory_dropout_prob
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Calculate total sequence length
        self.num_patches = (input_size // patch_size) ** 2
        self.total_tokens = self._calculate_total_tokens()
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.total_tokens, hidden_size), requires_grad=False
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) 
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        # Trajectory-specific output head
        self.trajectory_head = nn.Linear(hidden_size, traj_channels)
        
        self.initialize_weights()
    
    def _calculate_total_tokens(self):
        """
        Calculate total token count for positional embedding.
        
        Tokens include:
        - Context tokens (shared across agents): BEV, lidar, etc.
        - Agent-specific tokens: state + history + future trajectory
        """
        # Context tokens (shared)
        context_tokens = 0
        for modality_name in self.modality_config.keys():
            if modality_name in ['lidar', 'img', 'BEV']:
                context_tokens += self.num_patches + 1
            elif modality_name == 'action':
                context_tokens += 11
            else:
                context_tokens += 1
        
        # Agent-specific tokens (per agent)
        # - 1 agent state token (from encoder)
        # - 6 history tokens (30 frames pooled to 6)
        # - 20 future trajectory tokens
        tokens_per_agent = 1 + 6 + self.future_horizon
        
        # Total tokens
        total = context_tokens + (self.max_agents * tokens_per_agent)
        return total
    
    def initialize_weights(self):
        # Initialize positional embedding
        # Note: This is a simplified version; you may want more sophisticated init
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True,
            extra_tokens=self.total_tokens - self.num_patches
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize modality embedders
        if self.use_modality_specific:
            for embedder in self.modality_embedders.values():
                if hasattr(embedder, 'proj'):
                    w = embedder.proj.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                    if embedder.proj.bias is not None:
                        nn.init.constant_(embedder.proj.bias, 0)
                if hasattr(embedder, 'modality_token'):
                    nn.init.normal_(embedder.modality_token, std=0.02)
        
        # Initialize trajectory embedding
        nn.init.normal_(self.trajectory_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.trajectory_embed.mlp[2].weight, std=0.02)
        
        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.trajectory_head.weight, 0)
        nn.init.constant_(self.trajectory_head.bias, 0)
        
        # Initialize rate controllers
        for block in self.blocks:
            if hasattr(block, 'attn_rate'):
                nn.init.constant_(block.attn_rate.router.bias, 15.0)
            if hasattr(block, 'mlp_rate'):
                nn.init.constant_(block.mlp_rate.router.bias, 15.0)
            if hasattr(block, 'token_selection'):
                nn.init.constant_(block.token_selection.mlp_head[2].bias, 35.0)
    
    def encode_modalities(self, x):
        """
        Encode multi-modal context (shared across all agents).
        
        Args:
            x: Dict of modality tensors, e.g.,
               {
                   'lidar': (B, 2, H, W),
                   'img': (B, 3, H, W),
                   'BEV': (B, 7, H, W),
                   ...
               }
        
        Returns:
            encoded: (B, T_context, hidden_size)
        """
        if not self.use_modality_specific:
            raise NotImplementedError("Early fusion not implemented in this version")
        
        modality_features = {}
        
        for modality_name in self.modality_config.keys():
            if modality_name in x:
                modality_input = x[modality_name]
                modality_features[modality_name] = self.modality_embedders[modality_name](
                    modality_input
                )
        
        if self.parallel:
            # PARALLEL FUSION: Concatenate all modalities
            encoded = torch.cat(list(modality_features.values()), dim=1)
        else:
            # HIERARCHICAL FUSION: Sequential cross-attention
            modality_names = list(modality_features.keys())
            encoded = modality_features[modality_names[0]]
            
            for i in range(1, len(modality_names)):
                current_modality = modality_features[modality_names[i]]
                enhanced = self.cross_modal_attentions[i-1](current_modality, encoded)
                current_modality = current_modality + enhanced
                encoded = torch.cat([encoded, current_modality], dim=1)
        
        return encoded
    
    def forward(self, context, agent_states, noisy_trajectory, agent_history, t, complete_model=False):
        """
        Forward pass for multi-agent diffusion-based trajectory prediction.
        
        Args:
            context: Dict of multi-modal inputs (SHARED CONTEXT)
                    {
                        'lidar': (B, 2, H, W),
                        'BEV': (B, 7, H, W),
                        'img': (B, 3, H, W),
                        ...
                    }
            
            agent_states: (B, N, 5) - Current agent states [x, y, vx, vy, heading]
                         This goes to encoder to produce level2/level3 features
            
            noisy_trajectory: (B, N, 20, 5) - Noisy future trajectories to denoise
            
            agent_history: (B, N, 30, 5) - Historical trajectories
            
            t: (B,) - Diffusion timesteps
            
            complete_model: bool
            
        Returns:
            out: (B, N, 20, 5) - Predicted clean trajectories
            attn_weight_masks: (B, depth, num_heads) or None
            mlp_weight_masks: (B, depth, num_heads) or None
            token_select: (B, depth, Total_T, 1) or None
        """
        B, N, T_future, C = noisy_trajectory.shape
        
        # ========== STEP 1: ENCODE CONTEXT (SHARED) ==========
        context_encoded = self.encode_modalities(context)  # (B, T_context, D)
        
        # ========== STEP 2: ENCODE AGENTS (PER-AGENT) ==========
        # This is where the encoder from the matrix would be called
        encoder_output = self.agent_encoder(agent_states)  # EncoderOutput object
        
        # Extract level2_interaction (primary for medium decoder)
        agent_features = encoder_output.level2_interaction  # (B, N, D)
        
        # ========== STEP 3: ENCODE HISTORY (PER-AGENT) ==========
        history_tokens = self.history_encoder(agent_history)  # (B, N, 6, D)
        
        # ========== STEP 4: EMBED NOISY TRAJECTORY (PER-AGENT) ==========
        trajectory_emb = self.trajectory_embed(
            noisy_trajectory,  # (B, N, 20, 5)
            self.training
        )  # (B, N, 20, D)
        
        # Add temporal embeddings for future timesteps
        future_time_emb = self.future_time_embedder(B, N)  # (B, N, 20, D)
        trajectory_emb = trajectory_emb + future_time_emb  # (B, N, 20, D)
        
        # ========== STEP 5: TIMESTEP EMBEDDING ==========
        t_emb = self.t_embedder(t)  # (B, D)
        
        # Conditioning: timestep + average trajectory info
        # Average over agents and time for global conditioning
        traj_summary = trajectory_emb.mean(dim=(1, 2))  # (B, D)
        c = t_emb + traj_summary  # (B, D)
        
        # ========== STEP 6: COMBINE ALL TOKENS ==========
        # Flatten agent-specific tokens
        B, N, T_hist, D = history_tokens.shape
        history_flat = history_tokens.reshape(B, N * T_hist, D)  # (B, N*6, D)
        
        B, N, T_fut, D = trajectory_emb.shape
        trajectory_flat = trajectory_emb.reshape(B, N * T_fut, D)  # (B, N*20, D)
        
        # Agent state tokens: (B, N, D)
        agent_flat = agent_features
        
        # Concatenate: context + agent_states + history + trajectory
        all_tokens = torch.cat([
            context_encoded,           # (B, T_context, D)
            agent_flat,                # (B, N, D)
            history_flat,              # (B, N*6, D)
            trajectory_flat            # (B, N*20, D)
        ], dim=1)  # (B, Total_T, D)
        
        # Add positional embeddings
        # Handle variable sequence length
        if all_tokens.shape[1] <= self.pos_embed.shape[1]:
            all_tokens = all_tokens + self.pos_embed[:, :all_tokens.shape[1], :]
        else:
            # If exceeds max, truncate pos_embed (should not happen if max_agents set correctly)
            all_tokens = all_tokens + self.pos_embed
        
        # ========== STEP 7: TRANSFORMER BLOCKS ==========
        token_select_list = []
        attn_weight_masks_list = []
        mlp_weight_masks_list = []
        
        for block in self.blocks:
            all_tokens, attn_mask, mlp_mask, token = block(
                all_tokens, c, t_emb, complete_model
            )
            attn_weight_masks_list.append(attn_mask)
            mlp_weight_masks_list.append(mlp_mask)
            token_select_list.append(token)
        
        # ========== STEP 8: EXTRACT TRAJECTORY TOKENS ==========
        # Extract only the trajectory tokens (last N*20 tokens)
        traj_start_idx = -N * T_fut
        traj_tokens = all_tokens[:, traj_start_idx:, :]  # (B, N*20, D)
        
        # Reshape back: (B, N*20, D) → (B, N, 20, D)
        traj_tokens = traj_tokens.reshape(B, N, T_fut, D)
        
        # ========== STEP 9: PROJECT TO OUTPUT ==========
        # Modulate using final layer
        shift, scale = self.final_layer.adaLN_modulation(c).chunk(2, dim=1)
        
        # Reshape for modulation: (B, N, 20, D) → (B, N*20, D)
        traj_tokens_flat = traj_tokens.reshape(B, N * T_fut, D)
        
        # Modulate
        traj_tokens_flat = modulate(
            self.final_layer.norm_final(traj_tokens_flat),
            shift, scale
        )
        
        # Project: (B, N*20, D) → (B, N*20, 5)
        out_flat = self.trajectory_head(traj_tokens_flat)  # (B, N*20, 5)
        
        # Reshape: (B, N*20, 5) → (B, N, 20, 5)
        out = out_flat.reshape(B, N, T_fut, 5)
        
        # ========== STEP 10: RETURN ==========
        if not complete_model:
            attn_weight_masks = convert_list_to_tensor(attn_weight_masks_list)
            mlp_weight_masks = convert_list_to_tensor(mlp_weight_masks_list)
            token_select = convert_list_to_tensor(token_select_list)
            return out, attn_weight_masks, mlp_weight_masks, token_select
        else:
            return out, None, None, None
    
    def forward_with_cfg(self, context, agent_states, noisy_trajectory, agent_history, t, cfg_scale=7.5):
        """
        Forward pass with Classifier-Free Guidance (CFG).
        
        Args:
            context: Dict of multi-modal inputs
            agent_states: (B, N, 5) - Current agent states
            noisy_trajectory: (B, N, 20, 5) - Noisy trajectories
            agent_history: (B, N, 30, 5) - Historical trajectories
            t: (B,) - Timesteps
            cfg_scale: float - Guidance scale
            
        Returns:
            out: (B, N, 20, 5) with CFG applied
        """
        # Get batch size
        B = noisy_trajectory.shape[0]
        half = B // 2
        
        # Create null context for unconditional prediction
        context_null = {}
        for key, value in context.items():
            if value is not None:
                context_null[key] = torch.zeros_like(value)
        
        # Combine conditional and unconditional
        context_combined = {}
        for key in context.keys():
            context_combined[key] = torch.cat(
                [context[key][:half], context_null[key][:half]], 
                dim=0
            )
        
        # Duplicate agent inputs
        agent_states_combined = torch.cat(
            [agent_states[:half], agent_states[:half]], dim=0
        )
        noisy_trajectory_combined = torch.cat(
            [noisy_trajectory[:half], noisy_trajectory[:half]], dim=0
        )
        agent_history_combined = torch.cat(
            [agent_history[:half], agent_history[:half]], dim=0
        )
        t_combined = torch.cat([t[:half], t[:half]], dim=0)
        
        # Run model
        model_out, attn_masks, mlp_masks, token_select = self.forward(
            context_combined, 
            agent_states_combined,
            noisy_trajectory_combined,
            agent_history_combined,
            t_combined,
            complete_model=False
        )
        
        # Print statistics
        if attn_masks is not None:
            print(f"Attn Channel: {attn_masks.mean():.4f}")
        if mlp_masks is not None:
            print(f"MLP Channel: {mlp_masks.mean():.4f}")
        if token_select is not None:
            print(f"Token Selection: {token_select.mean():.4f}")
        
        # Apply CFG
        cond_out, uncond_out = torch.split(model_out, half, dim=0)
        cfg_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return cfg_out


# ========================================
# FACTORY FUNCTIONS
# ========================================

def create_transdiffuser_dit_small(**kwargs):
    """Small model for testing"""
    return TransDiffuserDiT(
        input_size=64, patch_size=4, hidden_size=384, depth=12, num_heads=6, **kwargs
    )


def create_transdiffuser_dit_base(**kwargs):
    """Base model"""
    return TransDiffuserDiT(
        input_size=64, patch_size=4, hidden_size=768, depth=12, num_heads=12, **kwargs
    )


def create_transdiffuser_dit_large(**kwargs):
    """Large model"""
    return TransDiffuserDiT(
        input_size=64, patch_size=4, hidden_size=1024, depth=24, num_heads=16, **kwargs
    )


# ========================================
# EXAMPLE USAGE
# ========================================

if __name__ == "__main__":
    # Create model
    model = create_transdiffuser_dit_base(
        traj_channels=5,
        use_modality_specific=True,
        parallel=True,
        max_agents=8,
        future_horizon=20,
        history_length=30,
    )
    
    # Example input
    batch_size = 2
    num_agents = 8
    H, W = 64, 64
    
    # Context (shared across agents)
    context = {
        'lidar': torch.randn(batch_size, 2, H, W),
        'BEV': torch.randn(batch_size, 7, H, W),
    }
    
    # Agent-specific inputs
    agent_states = torch.randn(batch_size, num_agents, 5)
    agent_history = torch.randn(batch_size, num_agents, 30, 5)
    noisy_trajectory = torch.randn(batch_size, num_agents, 20, 5)
    
    # Diffusion timestep
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    out, attn_masks, mlp_masks, token_select = model(
        context, agent_states, noisy_trajectory, agent_history, t,
        complete_model=False
    )
    
    print(f"Output shape: {out.shape}")  # Should be (2, 8, 20, 5)
    print(f"Attn masks shape: {attn_masks.shape if attn_masks is not None else None}")
    print(f"MLP masks shape: {mlp_masks.shape if mlp_masks is not None else None}")
    print(f"Token select shape: {token_select.shape if token_select is not None else None}")