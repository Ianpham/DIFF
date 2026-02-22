# for later change in modality encoder, we will let model learn independently and frozen when it need,
# pruning and lower bit for lattern

# change on in_channels will require the change of modality_config at dydittraj (line 879 - 893)

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from typing import List, Dict,Optional
# SCENE MODALITY ENCODERS (Environment Context - Shared across agents)
# These encode the ENVIRONMENT: LiDAR, BEV, Images, Traffic, etc.
# Output: Shared scene context for all agents
# 
@dataclass
class BEVConfig: # this is from BEV HD map, if segmentation do their job, we need also make it BEV
    """ configuration for encoder which bev provided in HDMAP and annotation, not segmentation from perception (that task will be later)"""
    hidden_size: 768
    patch_size: 4
    num_heads: int = 12
    dropout: float = 0.1

    #bev_configuration
    bev_5_channels: List[str] = None  # ['drivable_area', 'lane_boundaries', 'vehicle_occupancy', 'velocity_x', 'velocity_y']
    bev_7_channels: List[str] = None # + ['ego_mask', 'traffic_lights']
    bev_12_channels: List[str] = None

    def __post_init__(self):
        if self.bev_5_channels is None:
            self.bev_5_channels = [
                'drivable_area',
                'lane_boundaries', 
                'vehicle_occupancy',
                'velocity_x',
                'velocity_y'
            ]
        
        if self.bev_7_channels is None:
            self.bev_7_channels = self.bev_5_channels + [
                'ego_mask',
                'traffic_lights'
            ]
        
        if self.bev_12_channels is None:
            self.bev_12_channels = [
                'drivable_area',
                'lane_boundaries',
                'lane_dividers',
                'vehicle_occupancy',
                'pedestrian_occupancy',
                'velocity_x',
                'velocity_y',
                'ego_mask',
                'traffic_lights',
                'vehicle_classes',
                'crosswalks',
                'stop_lines'
            ]


# Lidar and Lidar BEV encoder 
class LidarBEVBackbone(nn.Module):
    """
    CNN backbone for pre-rasterized LiDAR BEV images.
    
    Input: (B, C, H, W) - BEV image (e.g., B, 2, 200, 200)
    Output: (B, T, hidden_size) - sequence of spatial tokens
    """
    def __init__(
        self, 
        hidden_size=768,
        in_channels=2,  # Your BEV has 2 channels
        bev_size=(200, 200)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        
        # CNN backbone to process BEV image
        self.backbone = nn.Sequential(
            # (B, 2, 200, 200) -> (B, 64, 100, 100)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # (B, 64, 100, 100) -> (B, 128, 50, 50)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # (B, 128, 50, 50) -> (B, 256, 25, 25)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # (B, 256, 25, 25) -> (B, 512, 13, 13)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # (B, 512, 13, 13) -> (B, hidden_size, 7, 7)
            nn.Conv2d(512, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )
        
        # Modality token (prepended to sequence)
        self.modality_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
    def forward(self, lidar_bev: torch.Tensor):
        """
        Args:
            lidar_bev: (B, C, H, W) - BEV image, e.g., (B, 2, 200, 200)
            
        Returns:
            features: (B, T+1, hidden_size) - spatial tokens + modality token
        """
        B = lidar_bev.shape[0]
        
        # Apply CNN backbone
        features = self.backbone(lidar_bev)  # (B, hidden_size, H', W')
        
        # Flatten spatial dimensions to create token sequence
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', hidden_size)
        
        # Add modality token
        modality_token = self.modality_token.expand(B, -1, -1)
        features = torch.cat([modality_token, features], dim=1)  # (B, T+1, hidden_size)
        
        return features


class LidarEmbedding(nn.Module):
    """Embedding module for LiDAR BEV images."""
    
    def __init__(self, hidden_size=768, in_channels=2, bev_size=(200, 200)):
        super().__init__()
        self.proj = LidarBEVBackbone(
            hidden_size=hidden_size,
            in_channels=in_channels,
            bev_size=bev_size
        )

    def forward(self, lidar_bev: torch.Tensor):
        """
        Args:
            lidar_bev: (B, C, H, W) - BEV image tensor
            
        Returns:
            features: (B, T, hidden_size)
        """
        return self.proj(lidar_bev)
# image encoder 

class ResNetImageEncoder(nn.Module):
    """
    Image encoder using ResNet-50 backbone (pretrained on ImageNet).
    Extracts features from multiple camera views.
    
    Reference: ResNet (He et al., 2016)
    """
    def __init__(
        self, 
        hidden_size=768, 
        patch_size=4,
        use_pretrained=True,
        freeze_backbone=False
    ):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        
        # Load pretrained ResNet-50
        if use_pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)
        
        # Remove final classification layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ResNet-50 output: 2048 channels
        self.proj = nn.Conv2d(2048, hidden_size, kernel_size=1)
        self.modality_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - RGB image (already normalized)
        Returns:
            out: (B, T_img, hidden_size)
        """
        B = x.shape[0]
        
        # ResNet backbone
        features = self.backbone(x)  # (B, 2048, H', W')
        
        # Project to hidden_size
        features = self.proj(features)  # (B, hidden_size, H', W')
        
        # Flatten spatial dimensions
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', hidden_size)
        
        # Add modality token
        modality_token = self.modality_token.expand(B, -1, -1)
        features = torch.cat([modality_token, features], dim=1)
        
        return features

class MultiCameraEncoder(nn.Module):
    """
    Encodes multiple camera views and fuses them.
    Handles the dictionary of camera images from dataset.
    """
    def __init__(
        self, 
        hidden_size=768,
        use_pretrained=True,
        freeze_backbone=False,
        camera_names=None
    ):
        super().__init__()
        
        if camera_names is None:
            self.camera_names = [
                'front', 'front_left', 'side_left', 'back_left',
                'front_right', 'side_right', 'back_right', 'back'
            ]
        else:
            self.camera_names = camera_names
        
        # Shared backbone for all cameras
        self.image_encoder = ResNetImageEncoder(
            hidden_size=hidden_size,
            use_pretrained=use_pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # Camera-specific embeddings
        self.camera_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            for name in self.camera_names
        })
        
        # Fusion attention
        self.fusion_attention = nn.MultiheadAttention(
            hidden_size, num_heads=12, dropout=0.1, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, camera_images: Dict[str, torch.Tensor]):
        """
        Args:
            camera_images: Dict[str, (B, 3, H, W)] - Multiple camera views
            
        Returns:
            out: (B, T_img, hidden_size) - Fused camera features
        """
        if not camera_images:
            # No camera images available
            B = 1  # Fallback
            return torch.zeros(B, 1, self.image_encoder.proj.out_channels, 
                             device=next(self.parameters()).device)
        
        # Get batch size from first available camera
        B = next(iter(camera_images.values())).shape[0]
        device = next(iter(camera_images.values())).device
        
        # Encode each camera view
        camera_features = []
        for cam_name in self.camera_names:
            if cam_name in camera_images:
                img = camera_images[cam_name]
                features = self.image_encoder(img)  # (B, T, D)
                
                # Add camera-specific embedding
                cam_emb = self.camera_embeddings[cam_name].expand(B, features.shape[1], -1)
                features = features + cam_emb
                
                camera_features.append(features)
        
        if not camera_features:
            return torch.zeros(B, 1, self.image_encoder.proj.out_channels, device=device)
        
        # Concatenate all camera features
        all_features = torch.cat(camera_features, dim=1)  # (B, T_total, D)
        
        # Self-attention fusion
        fused_features, _ = self.fusion_attention(all_features, all_features, all_features)
        fused_features = self.fusion_norm(all_features + fused_features)
        
        return fused_features




class BEVEncoder(nn.Module):
    """
    Bird's Eye View encoder - multi-channel BEV representation.
    Pretrained: Can use BEVFormer, LSS (Lift-Splat-Shoot) features.
    Channels typically:[    
    1. Drivable area -> selected
    2. Lane boundaries -> lane marking 
    3. Lane dividers -> road boundary
    4. Vehicle occupancy -> vehicel velocity field
    5. Pedestrian occupancy -
    6. Agent velocity X -> selected
    7. Agent velocity Y -> slected as footprint
    8. Ego vehicle mask
    9. Traffic light status
    10. Multiple vehicle classes
    11. Crosswalks
    12. Stop lines
    Road boundaries -> latter
    Static obstacles] -) later

    # my plan is separate, take only ego vehicle footfrint for ovelap due to bev benefit of correct agent state
    # will consider 7 or 12 based on performance of model, in cut-in, take over scenario.
    """
    def __init__(self,
                 in_channels=7, # 5, 7 or 12
                 hidden_size=768,
                 patch_size=4,
                 channel_names: Optional[List[str]] = None
                 ): 
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.channels_names = channel_names

        # validate channels count
        if in_channels not in [5, 7, 12]:
            raise ValueError(f"num channels must be in 5, 7, 12, got{in_channels}")
        
        # we currently use BEVformer from Uniad approach
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.modality_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
    
    def prepare_bev_input(self, labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack label channels according to configuration.
        
        Args: 
            labels: Dict of label tensor from dataset
        Returns:
            bev_input: (B, num_channels, H, W)
        """
        config = BEVConfig(hidden_size=self.hidden_size, patch_size=self.patch_size)
        
        # Fix: Match in_channels to correct config
        if self.in_channels == 5:
            channel_list = config.bev_5_channels
        elif self.in_channels == 7:
            channel_list = config.bev_7_channels
        else:  # 12
            channel_list = config.bev_12_channels
        
        # Get reference tensor for device and shape
        ref_tensor = next(iter(labels.values()))
        device = ref_tensor.device
        B = ref_tensor.shape[0]
        H, W = ref_tensor.shape[-2:]
        
        # Stack channels 
        channels = []
        for name in channel_list:
            if name in labels:
                tensor = labels[name]
                
                # Handle different input dimensions
                if tensor.dim() == 2:  # (H, W)
                    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    tensor = tensor.expand(B, -1, -1, -1)  # (B, 1, H, W)
                elif tensor.dim() == 3:  # (B, H, W)
                    tensor = tensor.unsqueeze(1)  # (B, 1, H, W)
                elif tensor.dim() == 4:  # (B, C, H, W) - already correct
                    pass
                
                channels.append(tensor)  # <-- THIS WAS MISSING!
            else:
                # Channel not found, create zeros
                print(f"Warning: Channel '{name}' not found in labels, using zeros")
                channels.append(torch.zeros(B, 1, H, W, device=device))
        
        bev_input = torch.cat(channels, dim=1)  # [B, num_channels, H, W]
        return bev_input

    


    def forward(self, labels: Dict[str, torch.Tensor]):
        """
        Args:
            labels: Dict of BEV label tensors from dataset
            
        Returns:
            out: (B, T_bev, hidden_size)
        """
        x = self.prepare_bev_input(labels)
        B = x.shape[0]
        
        # project 
        x = self.proj(x)  # (N, hidden_size, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, hidden_size)

        # add modality token
        modality_token = self.modality_token.expand(B, -1, -1)
        x = torch.cat([modality_token, x], dim=1)  # (N, T_bev, hidden_size)
        return x

# auxilary for phase 2 or phase 3 encoding when we fully develop
class TrafficControlEncoder(nn.Module):
    """
    Traffic control elements: lights, signs, crosswalks.
    Encodes discrete elements as tokens.
    """
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
            x: (N, 5) or (N, K, 5) - Traffic control features
        Returns:
            out: (N, K, hidden_size) or (N, 1, hidden_size)
        """
        if x.dim() == 2:
            x = self.mlp(x).unsqueeze(1)  # (N, 1, hidden_size)
        else:
            N, K, C = x.shape
            x = x.reshape(N * K, C)
            x = self.mlp(x).reshape(N, K, -1)  # (N, K, hidden_size)
        return x


class OcclusionEncoder(nn.Module):
    """
    Occlusion map encoder - indicates uncertain/hidden regions.
    Important for safety-critical scenarios. # here latter we will do Surround Occ, this help use to one more step close to understand scene from occlusion view
    """
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
            x: (N, 5) - Occlusion features
        Returns:
            out: (N, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (N, 1, hidden_size)
        return x



# AGENT-SPECIFIC ENCODERS (Per-agent features)
# These encode AGENT properties: kinematics, actions, behaviors
# NOT scene context - that comes from modality encoders above


class ActionEncoder(nn.Module):
    """
    Ego vehicle action encoder (steering, throttle, brake).
    Used for Level 1 (Temporal) agent encoding.
    """
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
            x: (N, 4, T_seq) or (N, 4) - [steering, throttle, brake, gear]
        Returns:
            out: (N, T_action, hidden_size)
        """
        N = x.shape[0]
        if x.dim() == 3:
            x = x.transpose(1, 2)  # (N, T_seq, 4)
            x = self.mlp(x)  # (N, T_seq, hidden_size)
        else:
            x = self.mlp(x).unsqueeze(1)  # (N, 1, hidden_size)
        
        modality_token = self.modality_token.expand(N, -1, -1)
        x = torch.cat([modality_token, x], dim=1)  # (N, T_action, hidden_size)
        return x


class EgoStateEncoder(nn.Module):
    """
    Ego vehicle state encoder (speed, acceleration, yaw rate).
    Used for Level 1 (Temporal) - immediate kinematics.
    Features: [x, y, vx, vy, ax, ay, heading]
    """
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
            x: (N, 7) - [x, y, vx, vy, ax, ay, heading]
        Returns:
            out: (N, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (N, 1, hidden_size)
        return x

# mean that here we use MoE for select right intent and behavior for planning,
class BehaviorEncoder(nn.Module):
    """
    Agent behavior encoder (lane keeping, following, changing, etc.).
    Used for Level 2 (Interaction) - tactical behaviors.
    Features: [behavior_class, aggressiveness, lane_offset, speed_relative, ...]
    """
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
            x: (N, 6) - Behavior features
        Returns:
            out: (N, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (N, 1, hidden_size)
        return x


class IntersectionEncoder(nn.Module):
    """
    Intersection-specific features for agents in intersections.
    Used for Level 3 (Scene) - strategic reasoning.
    Features: [in_intersection, approach_lane, turn_intention, right_of_way, distance_to_center]
    """
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
            x: (N, 5) - Intersection features
        Returns:
            out: (N, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (N, 1, hidden_size)
        return x


class PedestrianEncoder(nn.Module):
    """
    Pedestrian-specific features.
    Used for Level 2/3 - different interaction dynamics than vehicles.
    Features: [crossing_intention, gaze_direction, group_size, distance_to_crosswalk, ...]
    """
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
            x: (N, 5) - Pedestrian features
        Returns:
            out: (N, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (N, 1, hidden_size)
        return x


class GoalIntentEncoder(nn.Module):
    """
    Goal/destination encoder - where agent is trying to go.
    Used for Level 3 (Scene) - long-term planning.
    Features: [goal_x, goal_y, route_lane_ids, distance_to_goal, ...]
    """
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
            x: (N, 5) - Goal/intent features
        Returns:
            out: (N, 1, hidden_size)
        """
        x = self.mlp(x).unsqueeze(1)  # (N, 1, hidden_size)
        return x



# EMBEDDERS (Diffusion-specific)


class TimestepEmbedder(nn.Module):
    """
    Diffusion timestep embedder using sinusoidal encoding.
    Standard in diffusion models (DDPM, DiT).
    """
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
        """
        Sinusoidal timestep embedding from DDPM.
        """
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
    """
    Noisy trajectory embedder for diffusion process.
    Supports classifier-free guidance via dropout.
    Features: [x, y, vx, vy, heading] per timestep
    """
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
    Agent history encoder with temporal pooling.
    
    SIGNATURE MUST MATCH 
        HistoryEncoder(
            history_length=30,
            input_size=5,
            hidden_size=768,
            num_layers=2,
            batch_first=True
        )
    """
    def __init__(
        self, 
        input_size=5,           
        hidden_size=768, 
        history_length=30, 
        num_layers=2,          
        batch_first=True,       
        pool_size=5             
    ):
        super().__init__()
        self.pool_size = pool_size
        self.history_length = history_length
        self.batch_first = batch_first
        
        # Temporal encoding with LSTM/GRU (simpler than full transformer)
 
        self.temporal_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Learnable positional encoding (sinusoidal)
        self.pos_encoding = nn.Parameter(
            self._create_sinusoidal_positions(history_length, input_size),
            requires_grad=False
        )
        
        # Attention pooling (Doc 5 recommendation)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def _create_sinusoidal_positions(self, length, dim):
        """Create sinusoidal positional encodings"""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pos_encoding = torch.zeros(length, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 0:
            pos_encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            pos_encoding[:, 1::2] = torch.cos(position * div_term[:-1])
        return pos_encoding.unsqueeze(0)  # (1, length, dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, 30, 5) - Multi-agent history
        Returns:
            out: (B, N, hidden_size) - Compressed history per agent
        """
        B, N, T, C = x.shape
        device = x.device
        
        # Reshape: (B, N, 30, 5) → (B*N, 30, 5)
        x = x.reshape(B * N, T, C)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :].to(device)
        
        # LSTM encoding
        x, (h_n, c_n) = self.temporal_encoder(x)  # x: (B*N, 30, hidden_size)
        
        # Attention pooling
        attn_weights = self.attention_pool(x)  # (B*N, 30, 1)
        x = (attn_weights * x).sum(dim=1)  # (B*N, hidden_size)
        
        # Reshape back
        x = x.reshape(B, N, -1)  # (B, N, hidden_size)
        
        return x



class FutureTimeEmbedder(nn.Module):
    """
    Learnable temporal embeddings for future timesteps.
    Helps model distinguish different prediction horizons (1s vs 2s vs 5s).
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


# 
# ENCODER OUTPUT CONTAINER
# 

class EncoderOutput:
    """
    Container for encoder outputs following the responsibility matrix 
    
    THREE ENCODING LEVELS:
    - Level 1 (Temporal): Immediate kinematics, short-term patterns
    - Level 2 (Interaction): Agent-agent social reasoning
    - Level 3 (Scene): Map-grounded strategic planning
    
    SEPARATION OF CONCERNS:
    - Modality encoders → scene context (shared)
    - Agent encoder → per-agent features (individual)
    - They meet at Level 3 via cross-attention
    """
    def __init__(self):
        #  AGENT ENCODER LEVELS (Per-agent features) 
        self.level1_temporal = None      # [B, N, D] - Raw kinematics, immediate state
        self.level2_interaction = None   # [B, N, D] - Social context, agent-agent
        self.level3_scene = None         # [B, N, D] - Map-grounded, strategic
        
        #  LANE/MAP FEATURES (From modality encoders, shared context) 
        self.lanes_fine = None           # [B, 500, D] - Fine lane segments
        self.lanes_groups = None         # [B, 100, D] - Lane groups/roads
        self.lanes_network = None        # [B, 20, D] - High-level topology
        
        #  GEOMETRIC CONTEXT (No parameters, computed from map) 
        self.curvature = None            # [B, N, 1] - Lane curvature at agent
        self.reachable_2s = None         # [B, N, K] - Reachable set @ 2s
        self.reachable_5s = None         # [B, N, K] - Reachable set @ 5s
        self.reachable_8s = None         # [B, N, K] - Reachable set @ 8s
        self.signal_timing = None        # [B, N, 3] - Traffic light timing
        self.conflicts = None            # [B, N, N] - Conflict matrix
        
        #  INTERSECTION MODULE (Level 3 specific) 
        self.in_intersection = None      # [B, N, 1] - Binary flag
        self.approach_lane = None        # [B, N, 4] - Which approach
        self.turn_intention = None       # [B, N, 3] - Left/straight/right
        self.right_of_way = None         # [B, N, 1] - Priority score
        self.conflict_points = None      # [B, N, N, K] - Spatial conflicts
        
        #  PEDESTRIAN MODULE (Level 2/3 specific) 
        self.crossing_intention = None   # [B, N_ped, 1] - Will cross?
        self.vehicle_proximity = None    # [B, N_ped, N_veh] - Nearby vehicles
        self.group_behavior = None       # [B, N_ped, N_ped] - Pedestrian groups
        
        #  OCCLUSION MODULE (Level 3 safety) 
        self.occlusion_map = None        # [B, H, W] - Visibility map
        self.agent_risk = None           # [B, N, 1] - Occlusion risk score
        
        #  INTERACTION SEMANTICS (Level 2) 
        self.interaction_types = None    # [B, N, N, 6] - Interaction classification
        self.interaction_features = None # [B, N, N, F] - Pairwise features
        
        #  OPTIONAL 
        self.goals = None                # [B, N, 2] - Goal positions if available


# 
# MULTI-LEVEL AGENT ENCODER
# 
# working as router for agent, not scene as hierarchical encoder gate
class AgentEncoder(nn.Module):
    """
    CRITICAL: This is SEPARATE from modality encoders
    - Modality encoders handle SCENE (LiDAR, BEV, Images)
    - Agent encoder handles PER-AGENT features (kinematics, behavior)
    - They connect at Level 3 via cross-attention
    
    LEVEL 1 (TEMPORAL): 
        - Input: Agent states [x, y, vx, vy, heading], history
        - Processing: Temporal attention on 3-second history
        - Output: level1_temporal [B, N, D]
        - Purpose: Immediate kinematics, motion patterns
        
    LEVEL 2 (INTERACTION):
        - Input: Level 1 + agent-agent relationships
        - Processing: Factorized attention (spatial, relational, semantic)
        - Output: level2_interaction [B, N, D]
        - Purpose: Social reasoning, yielding, following, conflicts
        
    LEVEL 3 (SCENE):
        - Input: Level 2 + map context from modality encoders
        - Processing: Cross-attention with lanes, traffic, intersections
        - Output: level3_scene [B, N, D]
        - Purpose: Map-grounded strategic planning
    
    """
    
    def __init__(
        self, 
        hidden_size=768, 
        num_heads=12,
        use_pretrained=False,
        pretrained_path=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        #  LEVEL 1: TEMPORAL (Current state encoding) 
        self.state_encoder = nn.Sequential(
            nn.Linear(5, hidden_size),  # [x, y, vx, vy, heading]
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        #  LEVEL 2: INTERACTION (Agent-agent attention) 
        
        # Spatial attention (distance-aware)
        self.spatial_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(hidden_size)
        
        # Distance bias encoder
        self.distance_bias_mlp = nn.Sequential(
            nn.Linear(4, hidden_size // 4),  # [d, d^2, log(d), 1/d]
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_heads)
        )
        
        # Relational attention
        self.relational_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.relational_norm = nn.LayerNorm(hidden_size)
        
        # Relation feature encoder
        self.relation_encoder = nn.Sequential(
            nn.Linear(6, hidden_size),  # [Δx, Δy, Δvx, Δvy, TTC, bearing]
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # FFN for level 2
        self.level2_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.level2_ffn_norm = nn.LayerNorm(hidden_size)
        
        #  LEVEL 3: SCENE (Map cross-attention) 
        
        # Agent-to-Lane attention
        self.agent_to_lane_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )
        self.a2l_norm = nn.LayerNorm(hidden_size)
        
        # FFN for level 3
        self.level3_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.level3_ffn_norm = nn.LayerNorm(hidden_size)
        
        #  AUXILIARY 
        self.interaction_classifier = nn.Linear(hidden_size, 6)
        
    def compute_distance_features(self, agent_states):
        """
        Compute pairwise distance features.
        Args:
            agent_states: [B, N, 5]
        Returns:
            dist_features: [B, N, N, 4]
        """
        B, N, _ = agent_states.shape
        pos = agent_states[:, :, :2]  # [B, N, 2]
        
        pos_i = pos.unsqueeze(2)  # [B, N, 1, 2]
        pos_j = pos.unsqueeze(1)  # [B, 1, N, 2]
        diff = pos_i - pos_j  # [B, N, N, 2]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        
        eps = 1e-3
        dist_features = torch.cat([
            dist,
            dist ** 2,
            torch.log(dist + eps),
            1.0 / (dist + eps)
        ], dim=-1)  # [B, N, N, 4]
        
        return dist_features
    
    def compute_relational_features(self, agent_states):
        """
        Compute pairwise relational features.
        Args:
            agent_states: [B, N, 5]
        Returns:
            rel_features: [B, N, N, 6]
        """
        B, N, _ = agent_states.shape
        
        pos = agent_states[:, :, :2]  # [B, N, 2]
        vel = agent_states[:, :, 2:4]  # [B, N, 2]
        heading = agent_states[:, :, 4:5]  # [B, N, 1]
        
        pos_i = pos.unsqueeze(2)
        pos_j = pos.unsqueeze(1)
        vel_i = vel.unsqueeze(2)
        vel_j = vel.unsqueeze(1)
        heading_i = heading.unsqueeze(2)
        
        delta_pos = pos_j - pos_i  # [B, N, N, 2]
        delta_vel = vel_j - vel_i  # [B, N, N, 2]
        
        dist = torch.norm(delta_pos, dim=-1, keepdim=True)
        rel_speed = torch.norm(delta_vel, dim=-1, keepdim=True)
        
        approaching = (delta_pos * delta_vel).sum(dim=-1, keepdim=True) < 0
        ttc = torch.where(
            approaching,
            dist / (rel_speed + 1e-3),
            torch.full_like(dist, 10.0)
        )
        ttc = torch.clamp(ttc, 0, 10.0)
        
        bearing = torch.atan2(delta_pos[..., 1:2], delta_pos[..., 0:1])
        bearing = bearing - heading_i
        
        rel_features = torch.cat([delta_pos, delta_vel, ttc, bearing], dim=-1)
        
        return rel_features
    
    def forward(
        self, 
        agent_states: torch.Tensor,
        map_features: Optional[torch.Tensor] = None # that would be 
    ) -> EncoderOutput:
        """
        SIMPLIFIED SIGNATURE to match Document 4 usage.
        
        Args:
            agent_states: [B, N, 5] - Current agent states [x, y, vx, vy, heading]
            map_features: [B, T_map, D] - Optional map context from modality encoder
            
        Returns:
            EncoderOutput with level1, level2, level3 populated
        """
        B, N, _ = agent_states.shape
        D = self.hidden_size
        device = agent_states.device
        
        output = EncoderOutput()
        
        #  LEVEL 1: TEMPORAL 
        # Encode current state only (history handled separately in TransDiffuserIntegrated)
        state_emb = self.state_encoder(agent_states.reshape(B * N, -1))
        level1_features = state_emb.reshape(B, N, D)
        
        output.level1_temporal = level1_features
        
        #  LEVEL 2: INTERACTION 
        x = level1_features
        
        # Spatial attention
        dist_features = self.compute_distance_features(agent_states)
        spatial_out, _ = self.spatial_attention(x, x, x)
        x = self.spatial_norm(x + spatial_out)
        
        # Relational attention
        rel_features = self.compute_relational_features(agent_states)
        rel_emb = self.relation_encoder(rel_features.reshape(B * N * N, 6))
        rel_emb = rel_emb.reshape(B, N, N, D)
        rel_context = rel_emb.mean(dim=2)
        
        relational_out, _ = self.relational_attention(x + rel_context, x, x)
        x = self.relational_norm(x + relational_out)
        
        # FFN
        x = x + self.level2_ffn(self.level2_ffn_norm(x))
        
        output.level2_interaction = x
        
        # Interaction types
        output.interaction_types = self.interaction_classifier(
            x.unsqueeze(2).expand(-1, -1, N, -1).reshape(B * N * N, D)
        ).reshape(B, N, N, 6)
        
        #  LEVEL 3: SCENE 
        if map_features is not None:
            # Cross-attention with map
            map_expand = map_features.unsqueeze(1).expand(-1, N, -1, -1)
            map_expand = map_expand.reshape(B * N, -1, D)
            x_flat = x.reshape(B * N, 1, D)
            
            lane_context, _ = self.agent_to_lane_attn(x_flat, map_expand, map_expand)
            lane_context = lane_context.reshape(B, N, D)
            
            x = self.a2l_norm(x + lane_context)
        
        # FFN
        x = x + self.level3_ffn(self.level3_ffn_norm(x))
        
        output.level3_scene = x
        
        #  PLACEHOLDERS 
        output.lanes_groups = torch.zeros(B, 100, D, device=device)
        output.turn_intention = torch.zeros(B, N, 3, device=device)
        output.right_of_way = torch.zeros(B, N, 1, device=device)
        
        return output