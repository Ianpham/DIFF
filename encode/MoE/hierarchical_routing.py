"""
Phase 3: Hierarchical Routing

This builds on Phases 1 & 2 to create a structured, multi-level decision system
that mimics how humans process driving scenes.

Key Idea: Route in stages, from coarse to fine
- L1 (Scene-level): What type of scene? (highway, urban, parking)
- L2 (Modality-level): Which modality groups? (interaction-focused vs. navigation-focused)
- L3 (Slot-level): How to fuse selected modalities? (slot attention with confidence)

Benefits:
1. Interpretable: Clear decision hierarchy
2. Efficient: Early pruning of unnecessary computation  
3. Structured: Leverages domain knowledge
4. Robust: Graceful degradation with missing modalities

Inspired by:
1. Hierarchical MoE (H-MoE, Google 2022)
2. Conditional Computation (Bengio et al., 2013)
3. Task-specific routing (Meta's MOAT, 2023)
4. Human driving cognition (scene → attention → action)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from enum import Enum



# Scene Type Classification (L1)


class SceneType(Enum):
    """Predefined scene types for autonomous driving."""
    HIGHWAY = "highway"
    URBAN_INTERSECTION = "urban_intersection"
    PARKING = "parking"
    RESIDENTIAL = "residential"
    UNKNOWN = "unknown"


class SceneClassifier(nn.Module):
    """
    L1: Scene-level classification.
    
    Predicts high-level scene type from context features.
    This guides downstream modality selection.
    
    Design: Lightweight, runs once per scene.
    """
    
    def __init__(
        self,
        context_dim: int,
        num_scene_types: int = 5,
        hidden_dim: int = 64,
        use_uncertainty: bool = True
    ):
        super().__init__()
        self.num_scene_types = num_scene_types
        self.use_uncertainty = use_uncertainty
        
        # Scene classifier
        self.classifier = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_scene_types)
        )
        
        # Optional uncertainty estimation
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(context_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Positive uncertainty
            )
    
    def forward(
        self, 
        context: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            context: [B, context_dim] - Scene context features
            
        Returns:
            scene_probs: [B, num_scene_types] - Scene type probabilities
            uncertainty: [B, 1] - Classification uncertainty (optional)
        """
        logits = self.classifier(context)
        scene_probs = F.softmax(logits, dim=-1)
        
        if return_uncertainty and self.use_uncertainty:
            uncertainty = self.uncertainty_head(context)
            return scene_probs, uncertainty
        
        return scene_probs, None



# Modality Grouping & Selection (L2)


class ModalityGroup(Enum):
    """Functional grouping of modalities."""
    INTERACTION = "interaction"      # pedestrian, intersection
    NAVIGATION = "navigation"        # goal, traffic
    BEHAVIOR = "behavior"            # behavior (always needed)
    ALL = "all"


# Scene-specific modality importance (learned or hand-crafted)
SCENE_MODALITY_PRIORS = {
    SceneType.HIGHWAY: {
        'behavior': 0.9,
        'pedestrian': 0.1,
        'intersection': 0.2,
        'goal': 0.8,
        'traffic': 0.7
    },
    SceneType.URBAN_INTERSECTION: {
        'behavior': 0.9,
        'pedestrian': 0.9,
        'intersection': 0.95,
        'goal': 0.8,
        'traffic': 0.9
    },
    SceneType.PARKING: {
        'behavior': 0.8,
        'pedestrian': 0.9,
        'intersection': 0.3,
        'goal': 0.85,
        'traffic': 0.5
    },
    SceneType.RESIDENTIAL: {
        'behavior': 0.8,
        'pedestrian': 0.85,
        'intersection': 0.6,
        'goal': 0.8,
        'traffic': 0.6
    },
    SceneType.UNKNOWN: {
        'behavior': 0.7,
        'pedestrian': 0.7,
        'intersection': 0.7,
        'goal': 0.7,
        'traffic': 0.7
    }
}


class HierarchicalModalityRouter(nn.Module):
    """
    L2: Hierarchical modality selection.
    
    Uses scene type (L1) to guide modality selection.
    Two-stage routing:
    1. Scene-conditioned base routing
    2. Content-based refinement (from Phase 2)
    """
    
    def __init__(
        self,
        context_dim: int,
        num_modalities: int,
        modality_names: List[str],
        hidden_dim: int = 64,
        use_scene_priors: bool = True,
        learnable_priors: bool = False,
        top_k: Optional[int] = None
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_names = modality_names
        self.use_scene_priors = use_scene_priors
        self.top_k = top_k
        
        # Content-based router (from Phase 2)
        self.content_router = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities)
        )
        
        # Scene priors (learned or fixed)
        if use_scene_priors:
            if learnable_priors:
                # Learnable scene-modality affinity matrix
                num_scene_types = len(SceneType)
                self.scene_priors = nn.Parameter(
                    torch.randn(num_scene_types, num_modalities)
                )
            else:
                # Fixed priors from domain knowledge
                self.scene_priors = self._create_fixed_priors()
        
        # Combination weights (how much to trust scene priors vs. content)
        self.prior_weight = nn.Parameter(torch.tensor(0.5))
    
    def _create_fixed_priors(self) -> torch.Tensor:
        """Create fixed scene-modality prior matrix from domain knowledge."""
        num_scene_types = len(SceneType)
        priors = torch.zeros(num_scene_types, self.num_modalities)
        
        scene_type_list = list(SceneType)
        for i, scene_type in enumerate(scene_type_list):
            prior_dict = SCENE_MODALITY_PRIORS.get(scene_type, {})
            for j, modality in enumerate(self.modality_names):
                priors[i, j] = prior_dict.get(modality, 0.5)
        
        return priors
    
    def forward(
        self,
        context: torch.Tensor,
        scene_probs: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: [B, context_dim] - Scene context
            scene_probs: [B, num_scene_types] - Scene type probabilities from L1
            training: If True, soft selection; if False, hard selection
            
        Returns:
            selection_mask: [B, K] - Binary selection mask
            selection_scores: [B, K] - Selection scores (for analysis)
        """
        B = context.size(0)
        
        # 1. Content-based routing (from Phase 2)
        content_logits = self.content_router(context)  # [B, K]
        content_probs = torch.sigmoid(content_logits)
        
        # 2. Scene-based prior
        if self.use_scene_priors:
            # Get scene priors: [B, num_scene_types] @ [num_scene_types, K] = [B, K]
            if isinstance(self.scene_priors, nn.Parameter):
                priors = torch.sigmoid(self.scene_priors)
            else:
                priors = self.scene_priors.to(context.device)
            
            scene_priors = torch.matmul(scene_probs, priors)  # [B, K]
        else:
            scene_priors = torch.ones_like(content_probs) * 0.5
        
        # 3. Combine content and scene priors
        alpha = torch.sigmoid(self.prior_weight)  # In [0, 1]
        combined_probs = alpha * scene_priors + (1 - alpha) * content_probs
        
        # 4. Selection
        if self.top_k is not None:
            # Top-K selection
            if training:
                # Soft top-K (differentiable)
                top_k_vals, top_k_idx = torch.topk(combined_probs, self.top_k, dim=-1)
                selection_mask = torch.zeros_like(combined_probs)
                selection_mask.scatter_(1, top_k_idx, 1.0)
                
                # Add some probability mass to non-selected (for gradient flow)
                selection_mask = 0.9 * selection_mask + 0.1 * combined_probs
            else:
                # Hard top-K
                top_k_vals, top_k_idx = torch.topk(combined_probs, self.top_k, dim=-1)
                selection_mask = torch.zeros_like(combined_probs)
                selection_mask.scatter_(1, top_k_idx, 1.0)
        else:
            # Threshold-based selection
            threshold = 0.5
            if training:
                # Soft selection
                selection_mask = combined_probs
            else:
                # Hard selection
                selection_mask = (combined_probs > threshold).float()
        
        return selection_mask, combined_probs



# Hierarchical Routing System (L1 + L2 + L3)


class HierarchicalRoutingSystem(nn.Module):
    """
    Complete 3-level hierarchical routing system.
    
    L1: Scene classification (highway vs. urban vs. parking)
    L2: Modality selection (which encoders to use)
    L3: Slot-based fusion (how to combine selected modalities)
    
    This is the complete Phase 3 system that integrates with Phases 1 & 2.
    """
    
    def __init__(
        self,
        context_dim: int,
        num_modalities: int,
        modality_names: List[str],
        hidden_dim: int,
        num_slots: int,
        num_scene_types: int = 5,
        use_scene_priors: bool = True,
        use_confidence_weighting: bool = True,
        top_k_modalities: Optional[int] = 3
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_names = modality_names
        self.use_confidence_weighting = use_confidence_weighting
        
        # L1: Scene classifier
        self.scene_classifier = SceneClassifier(
            context_dim=context_dim,
            num_scene_types=num_scene_types,
            hidden_dim=64,
            use_uncertainty=True
        )
        
        # L2: Hierarchical modality router
        self.modality_router = HierarchicalModalityRouter(
            context_dim=context_dim,
            num_modalities=num_modalities,
            modality_names=modality_names,
            hidden_dim=64,
            use_scene_priors=use_scene_priors,
            top_k=top_k_modalities
        )
        
        # L3: Confidence-weighted slot attention (from Phase 1)
        if use_confidence_weighting:
            from confidence_weighting import ConfidenceWeightedSlotAttention
            self.fusion = ConfidenceWeightedSlotAttention(
                num_slots=num_slots,
                hidden_dim=hidden_dim,
                num_modalities=num_modalities,
                use_confidence=True
            )
        else:
            # Simple slot attention
            from confidence_weighting import ConfidenceWeightedSlotAttention
            self.fusion = ConfidenceWeightedSlotAttention(
                num_slots=num_slots,
                hidden_dim=hidden_dim,
                num_modalities=num_modalities,
                use_confidence=False
            )
    
    def forward(
        self,
        context: torch.Tensor,
        modality_encodings: torch.Tensor,
        training: bool = True,
        return_routing_info: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Complete hierarchical routing forward pass.
        
        Args:
            context: [B, context_dim] - Scene context features
            modality_encodings: [B, K, hidden_dim] - Pre-computed modality encodings
            training: If True, use soft routing; if False, use hard routing
            
        Returns:
            fused_output: [B, num_slots, hidden_dim] - Fused slot representations
            info: Dict with routing decisions at each level
        """
        B = context.size(0)
        
        # === L1: Scene Classification ===
        scene_probs, scene_uncertainty = self.scene_classifier(
            context,
            return_uncertainty=True
        )
        scene_pred = torch.argmax(scene_probs, dim=-1)  # [B]
        
        # === L2: Hierarchical Modality Selection ===
        selection_mask, selection_scores = self.modality_router(
            context,
            scene_probs,
            training=training
        )
        
        # Mask out unselected modalities
        masked_encodings = modality_encodings * selection_mask.unsqueeze(-1)
        
        # === L3: Confidence-Weighted Fusion ===
        fused_output, fusion_info = self.fusion(
            masked_encodings,
            return_attention=True,
            return_confidence=True if self.use_confidence_weighting else False
        )
        
        # Prepare routing info
        info = {
            # L1 info
            'scene_probs': scene_probs.detach(),
            'scene_pred': scene_pred.detach(),
            'scene_uncertainty': scene_uncertainty.detach() if scene_uncertainty is not None else None,
            
            # L2 info
            'selection_mask': selection_mask.detach(),
            'selection_scores': selection_scores.detach(),
            'num_modalities_selected': selection_mask.sum(dim=-1).mean().item(),
            'per_modality_selection_rate': selection_mask.mean(dim=0).detach(),
            
            # L3 info (from fusion)
            **fusion_info
        }
        
        return fused_output, info



# Training Utilities


def compute_scene_classification_loss(
    scene_logits: torch.Tensor,
    scene_labels: torch.Tensor,
    uncertainty: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Scene classification loss with optional uncertainty.
    
    Args:
        scene_logits: [B, num_scene_types]
        scene_labels: [B] - Ground truth scene type indices
        uncertainty: [B, 1] - Predicted uncertainty
        
    Returns:
        loss: Scalar classification loss
    """
    # Standard cross-entropy
    ce_loss = F.cross_entropy(scene_logits, scene_labels)
    
    # Optional: Uncertainty-aware loss
    # High uncertainty should correlate with high loss
    if uncertainty is not None:
        # Normalize uncertainty
        uncertainty = uncertainty.squeeze(-1)  # [B]
        
        # Get per-sample loss
        per_sample_loss = F.cross_entropy(
            scene_logits, 
            scene_labels, 
            reduction='none'
        )
        
        # Uncertainty regularization
        # Encourage: high uncertainty when loss is high
        uncertainty_target = per_sample_loss / (per_sample_loss.max() + 1e-8)
        uncertainty_loss = F.mse_loss(uncertainty, uncertainty_target)
        
        return ce_loss + 0.1 * uncertainty_loss
    
    return ce_loss


def compute_hierarchical_routing_loss(
    selection_mask: torch.Tensor,
    scene_probs: torch.Tensor,
    modality_names: List[str],
    target_sparsity: float = 0.6
) -> torch.Tensor:
    """
    Hierarchical routing-specific loss.
    
    Encourages scene-appropriate sparsity:
    - Highway: sparse (use fewer modalities)
    - Urban: dense (use more modalities)
    
    Args:
        selection_mask: [B, K]
        scene_probs: [B, num_scene_types]
        modality_names: List of modality names
        target_sparsity: Base target sparsity
        
    Returns:
        loss: Scalar routing loss
    """
    B, K = selection_mask.shape
    
    # Scene-specific target sparsity
    # Highway (type 0): target = 0.5 (use 50%)
    # Urban (type 1): target = 0.9 (use 90%)
    # Parking (type 2): target = 0.6 (use 60%)
    scene_sparsity_targets = torch.tensor([0.5, 0.9, 0.6, 0.7, 0.7], 
                                         device=selection_mask.device)
    
    # Compute expected sparsity per sample based on scene probs
    expected_sparsity = torch.matmul(
        scene_probs[:, :len(scene_sparsity_targets)], 
        scene_sparsity_targets
    )  # [B]
    
    # Actual sparsity per sample
    actual_sparsity = selection_mask.mean(dim=-1)  # [B]
    
    # Loss: match scene-appropriate sparsity
    sparsity_loss = F.mse_loss(actual_sparsity, expected_sparsity)
    
    return sparsity_loss


def compute_prior_alignment_loss(
    selection_mask: torch.Tensor,
    scene_probs: torch.Tensor,
    modality_names: List[str]
) -> torch.Tensor:
    """
    Encourage selection to align with scene priors.
    
    E.g., if scene is highway, penalize selecting pedestrian encoder.
    
    Args:
        selection_mask: [B, K]
        scene_probs: [B, num_scene_types]
        modality_names: List of modality names
        
    Returns:
        loss: Scalar alignment loss
    """
    B, K = selection_mask.shape
    device = selection_mask.device
    
    # Build prior matrix: [num_scene_types, K]
    num_scene_types = scene_probs.size(1)
    prior_matrix = torch.zeros(num_scene_types, K, device=device)
    
    scene_type_list = list(SceneType)[:num_scene_types]
    for i, scene_type in enumerate(scene_type_list):
        prior_dict = SCENE_MODALITY_PRIORS.get(scene_type, {})
        for j, modality in enumerate(modality_names):
            prior_matrix[i, j] = prior_dict.get(modality, 0.5)
    
    # Expected selection based on scene priors
    expected_selection = torch.matmul(scene_probs, prior_matrix)  # [B, K]
    
    # Alignment loss: selection should match priors
    alignment_loss = F.mse_loss(selection_mask, expected_selection)
    
    return alignment_loss



# Analysis & Visualization


class HierarchicalRoutingAnalyzer:
    """
    Analyze hierarchical routing decisions across all levels.
    """
    
    @staticmethod
    def analyze_routing_hierarchy(
        model,
        dataloader,
        device,
        modality_names: List[str]
    ) -> Dict:
        """
        Analyze routing patterns at each hierarchy level.
        """
        model.eval()
        
        # Collect statistics
        stats = {
            'L1_scene_distribution': {},
            'L2_modality_by_scene': {},
            'L3_confidence_by_scene': {},
            'L3_attention_patterns': []
        }
        
        scene_counts = torch.zeros(5)  # 5 scene types
        scene_modality_usage = torch.zeros(5, len(modality_names))
        
        with torch.no_grad():
            for batch in dataloader:
                # Assuming batch has context and encodings
                context = batch['context'].to(device)
                encodings = batch['modality_encodings'].to(device)
                
                # Forward pass
                _, info = model(context, encodings, training=False)
                
                # L1: Scene distribution
                scene_pred = info['scene_pred']
                for pred in scene_pred:
                    scene_counts[pred] += 1
                
                # L2: Modality usage by scene
                selection_mask = info['selection_mask']
                for i, pred in enumerate(scene_pred):
                    scene_modality_usage[pred] += selection_mask[i].cpu()
        
        # Normalize
        for i in range(5):
            if scene_counts[i] > 0:
                scene_modality_usage[i] /= scene_counts[i]
        
        # Format results
        scene_names = ['Highway', 'Urban', 'Parking', 'Residential', 'Unknown']
        stats['L1_scene_distribution'] = {
            scene_names[i]: (scene_counts[i] / scene_counts.sum()).item()
            for i in range(5)
        }
        
        stats['L2_modality_by_scene'] = {}
        for i, scene_name in enumerate(scene_names):
            stats['L2_modality_by_scene'][scene_name] = {
                modality_names[j]: scene_modality_usage[i, j].item()
                for j in range(len(modality_names))
            }
        
        return stats
    
    @staticmethod
    def print_hierarchy_analysis(stats: Dict):
        """Pretty print hierarchical routing analysis."""
        print("\n" + "="*70)
        print("Hierarchical Routing Analysis")
        print("="*70)
        
        print("\nL1: Scene Distribution")
        print("-" * 50)
        for scene, prob in stats['L1_scene_distribution'].items():
            bar = "█" * int(prob * 40)
            print(f"  {scene:20s} [{prob:5.1%}] {bar}")
        
        print("\nL2: Modality Selection by Scene Type")
        print("-" * 50)
        for scene, modalities in stats['L2_modality_by_scene'].items():
            print(f"\n{scene}:")
            for modality, rate in modalities.items():
                bar = "█" * int(rate * 30)
                print(f"  {modality:15s} [{rate:5.1%}] {bar}")


if __name__ == "__main__":
    print("Phase 3: Hierarchical Routing")
    print("=" * 70)
    print("\nThree-level decision hierarchy:")
    print("L1: Scene classification → What type of scene?")
    print("L2: Modality selection → Which encoders to use?")
    print("L3: Fusion → How to combine selected modalities?")
    print("\nSee phase3_integration.py for full integration guide")