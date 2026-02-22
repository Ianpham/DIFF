"""
Confidence-Weighted Fusion for Multi-Modal Trajectory Prediction

Inspired by QASA (Quality-Aware Slot Attention) but adapted for 
multi-modal encoder fusion rather than object detection.

Key Idea: Weight modality contributions by both:
1. Relevance (slot attention weights) - "Should we use this modality?"
2. Confidence (quality scores) - "How reliable is this modality encoding?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ModalityConfidenceEstimator(nn.Module):
    """
    Estimates confidence/quality scores for each modality encoding.
    
    Methods:
    1. Feature-based: Predict confidence from encoding statistics
    2. Uncertainty-based: Use prediction variance as proxy
    3. Hybrid: Combine both approaches
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int,
        method: str = 'feature_based',  # 'feature_based', 'uncertainty', 'hybrid'
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.method = method
        
        # Feature-based confidence prediction
        if method in ['feature_based', 'hybrid']:
            self.confidence_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Confidence in [0, 1]
            )
        
        # Uncertainty-based (requires additional forward passes or ensemble)
        if method in ['uncertainty', 'hybrid']:
            self.uncertainty_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Softplus()  # Positive uncertainty values
            )
    
    def forward(
        self, 
        modality_encodings: torch.Tensor,
        return_uncertainties: bool = False
    ) -> torch.Tensor:
        """
        Args:
            modality_encodings: [B, num_modalities, hidden_dim]
            
        Returns:
            confidence_scores: [B, num_modalities, 1]
                Values in [0, 1] indicating encoding quality
        """
        B, K, D = modality_encodings.shape
        
        if self.method == 'feature_based':
            # Predict confidence directly from encoding features
            confidence = self.confidence_net(modality_encodings)  # [B, K, 1]
            
        elif self.method == 'uncertainty':
            # Use predicted uncertainty to derive confidence
            uncertainty = self.uncertainty_predictor(modality_encodings)  # [B, K, 1]
            confidence = 1.0 / (1.0 + uncertainty)  # High uncertainty → low confidence
            
        elif self.method == 'hybrid':
            # Combine feature-based and uncertainty-based
            feature_conf = self.confidence_net(modality_encodings)
            uncertainty = self.uncertainty_predictor(modality_encodings)
            uncertainty_conf = 1.0 / (1.0 + uncertainty)
            confidence = 0.7 * feature_conf + 0.3 * uncertainty_conf  # Weighted combination
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if return_uncertainties and self.method in ['uncertainty', 'hybrid']:
            return confidence, uncertainty
        return confidence


class ConfidenceWeightedSlotAttention(nn.Module):
    """
    Slot Attention with confidence-weighted fusion.
    
    Combines:
    1. Standard slot attention weights (relevance)
    2. Modality confidence scores (reliability)
    
    Final weight = attention_weight × confidence_score
    """
    
    def __init__(
        self,
        num_slots: int,
        hidden_dim: int,
        num_modalities: int,
        num_iterations: int = 3,
        confidence_method: str = 'feature_based',
        use_confidence: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.num_iterations = num_iterations
        self.use_confidence = use_confidence
        self.temperature = temperature
        
        # Slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, hidden_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, hidden_dim))
        
        # Standard slot attention components
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_inputs = nn.LayerNorm(hidden_dim)
        
        # Confidence estimator
        if use_confidence:
            self.confidence_estimator = ModalityConfidenceEstimator(
                hidden_dim=hidden_dim,
                num_modalities=num_modalities,
                method=confidence_method
            )
        
        # Output projection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def initialize_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample initial slot vectors from learned distribution."""
        mu = self.slot_mu.expand(batch_size, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, -1, -1)
        return mu + sigma * torch.randn_like(mu)
    
    def forward(
        self,
        modality_encodings: torch.Tensor,
        return_attention: bool = False,
        return_confidence: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            modality_encodings: [B, num_modalities, hidden_dim]
                Outputs from different modality encoders
                
        Returns:
            slots: [B, num_slots, hidden_dim]
                Fused multi-modal slot representations
            info_dict: Dictionary containing:
                - attention_weights: [B, num_slots, num_modalities]
                - confidence_scores: [B, num_modalities, 1] (if confidence enabled)
                - combined_weights: [B, num_slots, num_modalities]
        """
        B, K, D = modality_encodings.shape
        device = modality_encodings.device
        
        # Normalize inputs
        inputs = self.norm_inputs(modality_encodings)  # [B, K, D]
        
        # Get confidence scores for each modality
        if self.use_confidence:
            confidence_scores = self.confidence_estimator(modality_encodings)  # [B, K, 1]
        else:
            confidence_scores = torch.ones(B, K, 1, device=device)
        
        # Initialize slots
        slots = self.initialize_slots(B, device)  # [B, num_slots, D]
        
        # Compute keys and values once
        k = self.to_k(inputs)  # [B, K, D]
        v = self.to_v(inputs)  # [B, K, D]
        
        # Iterative slot attention refinement
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Compute attention weights
            q = self.to_q(slots)  # [B, num_slots, D]
            
            # Scaled dot-product attention
            attn_logits = torch.einsum('bsd,bkd->bsk', q, k)  # [B, num_slots, K]
            attn_logits = attn_logits / (D ** 0.5)
            
            # Standard attention normalization (across modalities)
            attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)  # [B, num_slots, K]
            
            # Confidence weighting: Multiply attention weights by confidence scores
            # confidence_scores: [B, K, 1] → [B, 1, K] for broadcasting
            confidence_broadcast = confidence_scores.squeeze(-1).unsqueeze(1)  # [B, 1, K]
            combined_weights = attn_weights * confidence_broadcast  # [B, num_slots, K]
            
            # Renormalize after confidence weighting
            combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Weighted sum of values
            updates = torch.einsum('bsk,bkd->bsd', combined_weights, v)  # [B, num_slots, D]
            
            # GRU-based update
            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                slots_prev.reshape(B * self.num_slots, D)
            )
            slots = slots.reshape(B, self.num_slots, D)
            
            # Residual MLP
            slots = slots + self.mlp(self.norm_slots(slots))
        
        # Prepare info dictionary
        info = {
            'attention_weights': attn_weights.detach() if return_attention else None,
            'confidence_scores': confidence_scores.detach() if return_confidence else None,
            'combined_weights': combined_weights.detach() if return_attention else None
        }
        
        return slots, info


class ConfidenceWeightedFusion(nn.Module):
    """
    Simplified version: Just confidence-weighted sum without iterative attention.
    
    Use this if you want a drop-in replacement for simple averaging.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int,
        confidence_method: str = 'feature_based'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        self.confidence_estimator = ModalityConfidenceEstimator(
            hidden_dim=hidden_dim,
            num_modalities=num_modalities,
            method=confidence_method
        )
        
        # Learnable fusion weights (optional)
        self.fusion_weights = nn.Parameter(torch.ones(num_modalities))
    
    def forward(
        self,
        modality_encodings: torch.Tensor,
        return_info: bool = False
    ) -> torch.Tensor:
        """
        Args:
            modality_encodings: [B, num_modalities, hidden_dim]
            
        Returns:
            fused: [B, hidden_dim] - Confidence-weighted fusion
        """
        B, K, D = modality_encodings.shape
        
        # Get confidence scores
        confidence = self.confidence_estimator(modality_encodings)  # [B, K, 1]
        
        # Combine with learnable fusion weights
        fusion_weights = F.softmax(self.fusion_weights, dim=0)  # [K]
        combined_weights = confidence.squeeze(-1) * fusion_weights.unsqueeze(0)  # [B, K]
        
        # Normalize
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Weighted sum
        fused = torch.einsum('bk,bkd->bd', combined_weights, modality_encodings)
        
        if return_info:
            return fused, {'confidence': confidence, 'weights': combined_weights}
        return fused


# ============================================================================
# Training utilities
# ============================================================================

def confidence_regularization_loss(
    confidence_scores: torch.Tensor,
    min_confidence: float = 0.3,
    target_mean: float = 0.7
) -> torch.Tensor:
    """
    Regularization to prevent confidence collapse.
    
    Args:
        confidence_scores: [B, K, 1] - Predicted confidences
        min_confidence: Minimum allowed confidence per modality
        target_mean: Target average confidence across batch
        
    Returns:
        loss: Scalar regularization loss
    """
    # Prevent overconfidence collapse (all confidences → 1)
    mean_confidence = confidence_scores.mean()
    mean_loss = F.mse_loss(mean_confidence, torch.tensor(target_mean, device=confidence_scores.device))
    
    # Prevent any modality from being consistently ignored
    min_loss = F.relu(min_confidence - confidence_scores).mean()
    
    return mean_loss + min_loss


def confidence_supervised_loss(
    confidence_scores: torch.Tensor,
    prediction_errors: torch.Tensor
) -> torch.Tensor:
    """
    Supervise confidence scores using prediction errors.
    
    Idea: Low confidence should correlate with high prediction error.
    
    Args:
        confidence_scores: [B, K, 1] - Predicted confidences per modality
        prediction_errors: [B, K] - Per-modality prediction errors
        
    Returns:
        loss: Scalar supervision loss
    """
    # Normalize errors to [0, 1] range
    errors_normalized = prediction_errors / (prediction_errors.max() + 1e-8)
    
    # Target: confidence = 1 - error
    target_confidence = 1.0 - errors_normalized.unsqueeze(-1)
    
    return F.mse_loss(confidence_scores, target_confidence)


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    num_modalities = 5
    hidden_dim = 128
    num_slots = 3
    
    # Dummy modality encodings
    encodings = torch.randn(batch_size, num_modalities, hidden_dim)
    
    # Test 1: Confidence-weighted slot attention
    print("Testing ConfidenceWeightedSlotAttention...")
    model = ConfidenceWeightedSlotAttention(
        num_slots=num_slots,
        hidden_dim=hidden_dim,
        num_modalities=num_modalities,
        use_confidence=True
    )
    
    slots, info = model(encodings, return_attention=True, return_confidence=True)
    print(f"Slots shape: {slots.shape}")
    print(f"Attention weights shape: {info['attention_weights'].shape}")
    print(f"Confidence scores shape: {info['confidence_scores'].shape}")
    print(f"Combined weights shape: {info['combined_weights'].shape}")
    print(f"Confidence values: {info['confidence_scores'][0].squeeze()}")
    
    # Test 2: Simple confidence-weighted fusion
    print("\nTesting ConfidenceWeightedFusion...")
    simple_model = ConfidenceWeightedFusion(
        hidden_dim=hidden_dim,
        num_modalities=num_modalities
    )
    
    fused, info = simple_model(encodings, return_info=True)
    print(f"Fused shape: {fused.shape}")
    print(f"Confidence: {info['confidence'][0].squeeze()}")
    print(f"Weights: {info['weights'][0]}")