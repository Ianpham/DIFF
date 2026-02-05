"""
TransDiffuser: Multi-Modal Representation Decorrelation and Action Decoder
Based on arXiv:2505.09315
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalDecorrelation(nn.Module):
    """
    Multi-modal representation decorrelation mechanism.
    Regularizes the correlation matrix to be diagonal, reducing redundancy
    across different dimensions of multi-modal representations.
    """

    def __init__(self, decorr_weight=0.1, similarity_type='cosine'):  # Fixed typo
        """
        Args:
            decorr_weight: weight for the decorrelation loss
            similarity_type: type of similarity metric ('cosine' or 'correlation')  # Fixed typo
        """
        super().__init__()
        self.decorr_weight = decorr_weight
        self.similarity_type = similarity_type  # Fixed typo

    def compute_correlation_matrix(self, features):
        """
        Compute correlation matrix for a batch of features
        Args:
            features: [batch_size, feature_dim]

        Returns:
            correlation_matrix: [feature_dim, feature_dim]
        """
        # Normalize across batch dimension
        feature_normalized = F.normalize(features, p=2, dim=0)

        # Compute correlation matrix
        if self.similarity_type == 'cosine':
            corr_matrix = torch.mm(feature_normalized.t(), feature_normalized)
        else:  # correlation
            # Mean-center the features
            feature_centered = features - features.mean(dim=0, keepdim=True)
            
            corr_matrix = torch.mm(feature_centered.t(), feature_centered) / features.size(0)
            
            # Normalize to get correlation coefficients
            std = torch.sqrt(torch.diag(corr_matrix)).unsqueeze(1)
            corr_matrix = corr_matrix / (std @ std.t() + 1e-8)

        return corr_matrix
    
    def decorrelation_loss(self, corr_matrix):
        """
        Compute decorrelation loss to push correlation matrix toward diagonal/identity.
        
        Args:
            corr_matrix: [feature_dim, feature_dim]
        
        Returns:
            loss: scalar
        """
        # Identity matrix target
        identity = torch.eye(corr_matrix.size(0), device=corr_matrix.device)
        
        # Frobenius norm of difference
        loss = torch.norm(corr_matrix - identity, p='fro') ** 2
        
        return loss
    
    def forward(self, scene_features, motion_features, agent_features):
        """
        Apply decorrelation to multi-modal features.
        
        Args:
            scene_features: [batch_size, scene_dim]
            motion_features: [batch_size, motion_dim]
            agent_features: [batch_size, agent_dim]
        
        Returns:
            decorr_loss: scalar
            scene_features: unchanged
            motion_features: unchanged
            agent_features: unchanged
        """
        # Compute correlation matrices for each modality
        scene_corr = self.compute_correlation_matrix(scene_features)
        motion_corr = self.compute_correlation_matrix(motion_features)
        agent_corr = self.compute_correlation_matrix(agent_features)

        # Compute decorrelation losses
        scene_decorr_loss = self.decorrelation_loss(scene_corr)
        motion_decorr_loss = self.decorrelation_loss(motion_corr)
        agent_decorr_loss = self.decorrelation_loss(agent_corr)  # FIXED!

        # Total decorrelation loss (weighted)
        total_decorr_loss = self.decorr_weight * (  # FIXED: use * not ()
            scene_decorr_loss + motion_decorr_loss + agent_decorr_loss
        )

        return total_decorr_loss, scene_features, motion_features, agent_features


class ActionDecoder(nn.Module):
    """
    Action decoder that takes decorrelated multi-modal representations
    and outputs trajectory waypoints for autonomous driving.
    """
    
    def __init__(
        self,
        scene_dim,
        motion_dim,
        agent_dim=None,  # Added for compatibility
        hidden_dim=256,
        num_future_steps=12,
        output_dim=2,
        num_modes=10,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_future_steps = num_future_steps
        self.output_dim = output_dim
        self.num_modes = num_modes
        
        # Calculate total input dimension
        total_dim = scene_dim + motion_dim
        if agent_dim is not None:
            total_dim += agent_dim
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-mode trajectory decoder
        self.trajectory_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_future_steps * output_dim)
            ) for _ in range(num_modes)
        ])
        
        # Mode confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_modes)
        )
    
    def forward(self, scene_features, motion_features, agent_features=None):
        """
        Decode multi-modal features into trajectory predictions.
        
        Args:
            scene_features: [batch_size, scene_dim]
            motion_features: [batch_size, motion_dim]
            agent_features: [batch_size, agent_dim] (optional)
        
        Returns:
            trajectories: [batch_size, num_modes, num_future_steps, output_dim]
            confidences: [batch_size, num_modes]
        """
        batch_size = scene_features.size(0)
        
        # Fuse multi-modal features
        if agent_features is not None:
            fused_features = torch.cat([scene_features, motion_features, agent_features], dim=-1)
        else:
            fused_features = torch.cat([scene_features, motion_features], dim=-1)
            
        fused_features = self.fusion(fused_features)
        
        # Generate multiple trajectory candidates
        trajectories = []
        for mode_head in self.trajectory_head:
            traj = mode_head(fused_features)
            traj = traj.view(batch_size, self.num_future_steps, self.output_dim)
            trajectories.append(traj)
        
        trajectories = torch.stack(trajectories, dim=1)
        
        # Predict confidence for each mode
        confidences = self.confidence_head(fused_features)
        confidences = F.softmax(confidences, dim=-1)
        
        return trajectories, confidences


class TransDiffuserDecoder(nn.Module):
    """
    Complete decoder module combining decorrelation and action decoding.
    """
    
    def __init__(
        self,
        scene_dim,
        motion_dim,
        agent_dim,  # Added
        hidden_dim=256,
        num_future_steps=12,
        output_dim=2,
        num_modes=10,
        decorr_weight=0.1,
        dropout=0.1
    ):
        super().__init__()
        
        # Multi-modal decorrelation module
        self.decorrelation = MultiModalDecorrelation(
            decorr_weight=decorr_weight,
            similarity_type='cosine'
        )
        
        # Action decoder
        self.action_decoder = ActionDecoder(
            scene_dim=scene_dim,
            motion_dim=motion_dim,
            agent_dim=agent_dim,  # Added
            hidden_dim=hidden_dim,
            num_future_steps=num_future_steps,
            output_dim=output_dim,
            num_modes=num_modes,
            dropout=dropout
        )
    
    def forward(self, scene_features, motion_features, agent_features):
        """
        Forward pass through decorrelation and action decoding.
        
        Args:
            scene_features: [batch_size, scene_dim]
            motion_features: [batch_size, motion_dim]
            agent_features: [batch_size, agent_dim]
        
        Returns:
            trajectories: [batch_size, num_modes, num_future_steps, output_dim]
            confidences: [batch_size, num_modes]
            decorr_loss: scalar
        """
        # Apply decorrelation (FIXED: pass all 3 features)
        decorr_loss, scene_features, motion_features, agent_features = self.decorrelation(
            scene_features, motion_features, agent_features
        )
        
        # Decode actions
        trajectories, confidences = self.action_decoder(
            scene_features, motion_features, agent_features
        )
        
        return trajectories, confidences, decorr_loss