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
        # # Normalize across batch dimension
        # feature_normalized = F.normalize(features, p=2, dim=0)

        # # Compute correlation matrix
        # if self.similarity_type == 'cosine':
        #     corr_matrix = torch.mm(feature_normalized.t(), feature_normalized)
        # else:  # correlation
        #     # Mean-center the features
        #     feature_centered = features - features.mean(dim=0, keepdim=True)
            
        #     corr_matrix = torch.mm(feature_centered.t(), feature_centered) / features.size(0)
            
        #     # Normalize to get correlation coefficients
        #     std = torch.sqrt(torch.diag(corr_matrix)).unsqueeze(1)
        #     corr_matrix = corr_matrix / (std @ std.t() + 1e-8)

        # return corr_matrix

        # fix the problem of that decorrelation getting 0 with since beggining of epoch
        feature_normalize = F.normalize(features, p = 2, dim = 1)

        # correlation between feature dimension: [feature_dim, feature_dim]
        corr_matrix = torch.mm(feature_normalize.t(), feature_normalize)/ features.size(0)

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
