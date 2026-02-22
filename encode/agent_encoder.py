import torch
import torch.nn as nn

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from einops import rearrange
"""
Agent Encoder with Semantic Slot Routing and Decorrelation Loss
================================================================

Three-level hierarchical encoding:
- Level 1 (Temporal): Immediate kinematics + history
- Level 2 (Interaction): Agent-agent social reasoning  
- Level 3 (Scene): Map-grounded strategic planning

Features:
- Slot Attention-based semantic routing
- Decorrelation loss for attention diversity
- Compatible with DyDiT token selection
- Handles behavior/pedestrian/intersection/goal semantics


Implements:
1. AdaSlot-inspired adaptive slot selection
2. QASA-inspired quality-guided routing
3. Combined system with semantic confidence integration

Based on:
- Adaptive Slot Attention (Fan et al., CVPR 2024)
- Quality-Guided Slot Attention (QASA, 2025)
- Original Slot Attention (Locatello et al., 2020)
"""

# here my effort is trying to exclude agentencoder out of modality (may take pedesstrian or behavior from this as well) 
# then make router from it,

# for raw dataset, I use hiearrchy_encoder is enough.

# let separate thing like this
# and let see what we lack from this need of agent encoder
# interaction model, weather, lane change
@dataclass
class SlotRoutingOutput:
    slots: torch.Tensor # [B, K, D] - refined slots
    selection_mask: torch.Tensor # [B, K] - binary mask of active slots
    quality_scores: torch.Tensor # [B, K] - quality per slot
    attention_weigths: torch.Tensor # [B, N, K] - per-agent importance
    attention_maps: Optional[torch.Tensor] # [B, K, N] - full attention

    # losses
    sparsity_loss: Optional[torch.Tensor] = None
    quality_loss: Optional[torch.Tensor] = None

    # metadata 
    num_active_slots: Optional[torch.Tensor] = None # [B] - count per batch

# fromthere is what we do with slot
    
@dataclass
class AgentEncoderOutput:
    """Output from agent encoder with all levels."""
    level1_temporal: torch.Tensor # [B, N, D]
    level2_interaction: torch.Tensor # [B, N, D]
    level3_semantic: torch.Tensor # [B, N, D]

    # semantic tokens if routing is using
    semantic_token_l2: Optional[torch.Tensor] = None # [B, N, K2, D]
    semantic_token_l3: Optional[torch.Tensor] = None # [B, N, K3, D]

    # flatten for DiT
    all_tokens: Optional[torch.Tensor] = None # [B, T_total, D]

    # routing information
    routing: Optional[torch.Tensor] = None

    # decolation losses (for training)
    decor_loss = Optional[Dict[str, torch.tensor]] = None



######## decorlation loss #############

class DecorrelationLoss(nn.Module):
    """
    Decorrelation loss to encourage diversity in attention heads.
    
    Based on:
    - DeCAtt (CVPR 2023): Minimizes cross-correlation among heads
    - TransDiffuser: Reduces redundancy in representations
    
    Loss encourages attention heads to focus on different aspects.
    """
    def __inir__(
            self,
            loss_type: str = 'cross_correlation', # 'cosine' 'mse'
            normalize: bool = True
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

    
    def forward(
            self,
            attention_maps: torch.Tensor, # [B, num_heads, T, T] or [B, num_heads, T, D]
            eps: 1e-6
    ):
        """
        Compute decorrelation loss for attention heads.
        Args:
            attention_maps: Attention weights or features per head
            eps: small constant for numerical stability

        Returns:
            loss: scalar decorrlation loss
        """

        if self.loss_type == 'cross_correlation':
            return self._cross_correlation_loss(attention_maps, eps)
        
        elif self.loss_type == 'cosine':
            return self._cosine_similarity_loss(attention_maps, eps)
        
        elif self.loss_type == 'mse':
            return self._mse_diversity_loss(attention_maps)
        
        else:
            raise ValueError(f"Unknow loss_type {self.loss_type}")
        

    def _cross_correlation_loss(
            self, 
            attention_maps: torch.Tensor,
            eps: float
    ):
        """Minimize off-diagonal elements of cross-correlation matrix."""

        B, H, T, D = attention_maps.shape

        # flatten spatial dimension 
        attn_flat = attention_maps.flatten(2)

        # normalize if requested
        if self.normalize:
            attn_flat = torch.nn.functional.normalize(attn_flat, p = 2, dim =2)

        # compute cross_relation matrix [B, H, H]
        # corr_ij = <head_i, head_j>
        corr_matrix = torch.bmm(attn_flat, attn_flat.transpose(-2, -1)) # B, H, H

        # create identifier matrix
        identity = torch.eye(H, device = corr_matrix.device).unsqueeze(1) # [1, H, H]
        off_diagonal_mask = 1 - identity
        # corr_ij ==/= 0 for i =/= j
        loss = (corr_matrix * off_diagonal_mask).pow(2).sum(dim = [1, 2]).mean()

        return loss
    
    def _cross_similiarity_loss(
            self,
            attention_maps: torch.Tensor,
            eps: float
    ):
        """Minimize pair wise cosine similarity between heads"""
        B, H, T, D = attention_maps.shape

        # flatten
        attn_flat = attention_maps.flatten(2)

        # normalize
        attn_norm = nn.functional.normalize(attn_flat, p = 2, dim = 2)

        #pair wise
        cos_sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))

        # Mask diagonal
        identity = torch.eye(H, device=cos_sim.device).unsqueeze(0)
        cos_sim = cos_sim * (1 - identity)
        
        # Loss: minimize similarity (maximize diversity)
        loss = cos_sim.abs().sum(dim=[1, 2]).mean()
        
        return loss
    
    def _mse_diversity_loss(
        self,
        attention_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE-based diversity loss.
        Penalize similar attention patterns.
        """
        B, H, T, D = attention_maps.shape
        
        # Mean attention pattern across heads
        mean_pattern = attention_maps.mean(dim=1, keepdim=True)  # [B, 1, T, D]
        
        # Variance from mean (want high variance)
        variance = (attention_maps - mean_pattern).pow(2).mean()
        
        # Loss: maximize variance = minimize negative variance
        loss = -variance
        
        return loss
    
# slot attention (for senmatic routing)

class SlotAttention(nn.Module):
    """
    Slot Attention for semantic encoder routing

    Slots compete to explain agent features.
    Each slot represents a semantic concept (behavior, pedestrian, etc)

    Based on: Locatello et al., "Object-Centric Learning with Slot Attention"
    """

    def __init__(
            self,
            hidden_size: int, 
            num_slots: int,
            num_iterations: int = 3,
            slot_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_names = slot_names or [f"slot_{i}" for i in range(num_slots)]

        # learnable slot initializations  (one per semantic concept)
        self.slot_inits = nn.Parameter(
            torch.randn(num_slots, hidden_size) * 0.02
        )

        # attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias = False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias = False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias = False)

        # scale for attention
        self.scale  = hidden_size ** -0.5

        # gru for iteractive refinement
        self.gru = nn.GRU(hidden_size, hidden_size)

        # MLP for slot update
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # layer norm
        self.norm_input = nn.LayerNorm(hidden_size)
        self.norm_slots = nn.LayerNorm(hidden_size)

    def forward(
            self,
            inputs: torch.Tensor, # [B, N, D] - agent features
            return_attention: bool = False,
    )-> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route agent features to semantic slots.

        Args:
            inputs: [B, N, D] - Agent features
            return_attention: Whether to return attention maps
            
        Returns:
            slots: [B, K, D] - Refined slot representations
            attn_weights: [B, N, K] - Attention weights (per-agent importance)
            attn_maps: [B, K, N, N] - Full attention maps (if requested)

        """
        B, N, D = inputs.shape

        K = self.num_slots

        # normalize
        inputs = self.norm_input(inputs)

        # Initilize slots [B, K, D]
        slots = self.slot_inits.unsqueeze(0).expand(B, -1, -1)
        slots = self.norm_slots(slots)

        # iterative refinement
        attn_maps_all = []
        for iteration in range(self.num_iterations):
            slots_prev = slots

            # attention, slot is querry and input sizes as key/value
            q = self.q_proj(slots) # [B,K, D]
            k =  self.k_proj(inputs) # [B, N, D]
            v = self.v_proj(inputs) # [B, N, D]

            # compute attention score
            attn_logits = torch.einsum('bkd, bnd -> bkn', q, k)  * self.scale

            # softmax over slots

            attn = nn.functional.softmax(attn_logits, dim = 1) # normalize over K

            # weight sum: [B, K, D]
            updates = torch.einsum('bkn, bnd -> bkd', attn, v)

            # gru update
            slots_flat = slots.reshape(B*K, D)
            updates_flat = updates_flat.reshape(B*K, D)
            slots_flat = self.gru(updates_flat, slots_flat)
            slots = slots_flat.reshape(B, K, D)

            # MLP refinement
            slots = slots + self.mlp(slots)

            if return_attention:
                attn_maps_all.append(attn)

            # Final attention weights: [B, N, K] (transpose for per-agent view)
            attn_weights = attn.transpose(1, 2)  # [B, N, K]
            
            # Stack attention maps if requested
            attn_maps = torch.stack(attn_maps_all, dim=0) if return_attention else None
            # attn_maps: [num_iterations, B, K, N]
            
            return slots, attn_weights, attn_maps
        
# SEMANTIC ENCODER BANK (Placeholder)
# ============================================================================

class SemanticEncoderBank(nn.Module):
    """
    Bank of semantic encoders for different concepts.
    
    NOTE: These are PLACEHOLDERS. Real implementations will come from
    your detection/labeling models later.
    
    Current encoders:
    - Level 2: Behavior, Pedestrian
    - Level 3: Intersection, Goal, Traffic Control
    """
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Level 2 encoders (interaction semantics)
        self.level2_encoders = nn.ModuleDict({
            'behavior': self._create_placeholder_encoder(
                in_features=6,  # Placeholder: [behavior_class, aggressiveness, ...]
                hidden_size=hidden_size,
                num_layers=4  # HEAVY - models intent
            ),
            'pedestrian': self._create_placeholder_encoder(
                in_features=5,  # [crossing_intention, gaze, group_size, ...]
                hidden_size=hidden_size,
                num_layers=2
            )
        })
        
        # Level 3 encoders (scene semantics)
        self.level3_encoders = nn.ModuleDict({
            'intersection': self._create_placeholder_encoder(
                in_features=5,  # [in_intersection, approach_lane, ...]
                hidden_size=hidden_size,
                num_layers=3
            ),
            'goal': self._create_placeholder_encoder(
                in_features=4,  # [goal_x, goal_y, distance, ...]
                hidden_size=hidden_size,
                num_layers=2
            ),
            'traffic_control': self._create_placeholder_encoder(
                in_features=3,  # [light_state, distance_to_signal, ...]
                hidden_size=hidden_size,
                num_layers=1
            )
        })
    
    def _create_placeholder_encoder(
        self,
        in_features: int,
        hidden_size: int,
        num_layers: int
    ) -> nn.Module:
        """Create a simple MLP encoder."""
        layers = []
        current_dim = in_features
        
        for i in range(num_layers):
            next_dim = hidden_size if i == num_layers - 1 else hidden_size // 2
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(next_dim))
            current_dim = next_dim
        
        return nn.Sequential(*layers)


# ============================================================================
# MAIN AGENT ENCODER
# ============================================================================

class AgentEncoderWithRouting(nn.Module):
    """
    Complete agent encoder with semantic slot routing and decorrelation.
    
    Architecture:
    1. Level 1: State + History encoding
    2. Level 2: Agent-Agent interaction + Semantic routing (behavior/pedestrian)
    3. Level 3: Agent-Scene cross-attention + Semantic routing (intersection/goal)
    
    Features:
    - Slot attention-based semantic routing
    - Decorrelation loss for attention diversity
    - Compatible with your DyDiT blocks
    - Handles missing semantic inputs gracefully
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        history_length: int = 30,
        dropout: float = 0.1,
        # Semantic routing config
        use_semantic_routing: bool = True,
        slot_iterations: int = 2,
        # Decorrelation config
        use_decorr_loss: bool = True,
        decorr_weight: float = 0.01,
        decorr_type: str = 'cross_correlation'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_semantic_routing = use_semantic_routing
        self.use_decorr_loss = use_decorr_loss
        self.decorr_weight = decorr_weight
        
        # ====================================================================
        # LEVEL 1: TEMPORAL
        # ====================================================================
        
        # Current state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(5, hidden_size),  # [x, y, vx, vy, heading]
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # History encoder (LSTM)
        self.history_encoder = nn.LSTM(
            input_size=5,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0
        )
        
        # History attention pooling
        self.history_attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Fuse state + history
        self.level1_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # ====================================================================
        # LEVEL 2: INTERACTION
        # ====================================================================
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(hidden_size)
        
        # Relational features
        self.relation_encoder = nn.Sequential(
            nn.Linear(6, hidden_size),  # [Δx, Δy, Δvx, Δvy, TTC, bearing]
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # FFN
        self.level2_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.level2_norm = nn.LayerNorm(hidden_size)
        
        # Semantic routing for Level 2
        if use_semantic_routing:
            self.slot_router_l2 = SlotAttention(
                hidden_size=hidden_size,
                num_slots=2,  # behavior, pedestrian
                num_iterations=slot_iterations,
                slot_names=['behavior', 'pedestrian']
            )
            self.semantic_bank = SemanticEncoderBank(hidden_size)
        
        # ====================================================================
        # LEVEL 3: SCENE
        # ====================================================================
        
        # Agent-scene cross-attention
        self.agent_scene_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.level3_norm = nn.LayerNorm(hidden_size)
        
        # FFN
        self.level3_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.level3_ffn_norm = nn.LayerNorm(hidden_size)
        
        # Semantic routing for Level 3
        if use_semantic_routing:
            self.slot_router_l3 = SlotAttention(
                hidden_size=hidden_size,
                num_slots=3,  # intersection, goal, traffic
                num_iterations=slot_iterations,
                slot_names=['intersection', 'goal', 'traffic_control']
            )
        
        # ====================================================================
        # DECORRELATION LOSS
        # ====================================================================
        
        if use_decorr_loss:
            self.decorr_criterion = DecorrelationLoss(
                loss_type=decorr_type,
                normalize=True
            )
    
    def forward(
        self,
        agent_states: torch.Tensor,      # [B, N, 5]
        agent_history: torch.Tensor,     # [B, N, T_hist, 5]
        scene_features: Optional[torch.Tensor] = None,  # [B, T_scene, D]
        semantic_inputs: Optional[Dict[str, torch.Tensor]] = None,
        agent_types: Optional[torch.Tensor] = None,
        scene_context: Optional[Dict] = None,
        return_all_tokens: bool = True,
        return_routing_info: bool = False
    ) -> AgentEncoderOutput:
        """
        Complete agent encoding with semantic routing.
        
        Args:
            agent_states: [B, N, 5] - Current states
            agent_history: [B, N, T_hist, 5] - Historical trajectories
            scene_features: [B, T_scene, D] - Scene context from modality encoder
            semantic_inputs: Dict of semantic features for routing
            agent_types: [B, N] - Agent type indices (optional)
            scene_context: Dict with scene metadata
            return_all_tokens: Whether to flatten all tokens for DiT
            return_routing_info: Whether to return routing decisions
            
        Returns:
            AgentEncoderOutput with all levels + optional routing info
        """
        B, N, _ = agent_states.shape
        device = agent_states.device
        
        routing_info = {}
        decorr_losses = {}
        
        # ====================================================================
        # LEVEL 1: TEMPORAL
        # ====================================================================
        
        # Encode current state
        state_emb = self.state_encoder(
            agent_states.reshape(B * N, -1)
        ).reshape(B, N, -1)  # [B, N, D]
        
        # Encode history with attention pooling
        history_flat = agent_history.reshape(B * N, agent_history.shape[2], 5)
        history_seq, _ = self.history_encoder(history_flat)  # [B*N, T_hist, D]
        
        # Attention pooling over time
        attn_weights = self.history_attention(history_seq)  # [B*N, T_hist, 1]
        history_emb = (history_seq * attn_weights).sum(dim=1)  # [B*N, D]
        history_emb = history_emb.reshape(B, N, -1)  # [B, N, D]
        
        # Fuse
        level1_features = self.level1_fusion(
            torch.cat([state_emb, history_emb], dim=-1)
        )  # [B, N, D]
        
        # ====================================================================
        # LEVEL 2: INTERACTION
        # ====================================================================
        
        x = level1_features
        
        # Spatial attention with decorr loss
        if self.use_decorr_loss and self.training:
            # Store attention maps for decorr loss
            spatial_out, spatial_attn_weights = self.spatial_attention(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            # spatial_attn_weights: [B, num_heads, N, N]
            decorr_losses['spatial_attn'] = self.decorr_criterion(
                spatial_attn_weights.unsqueeze(-1)  # [B, H, N, N, 1]
            )
        else:
            spatial_out, _ = self.spatial_attention(x, x, x)
        
        x = self.spatial_norm(x + spatial_out)
        
        # Relational features
        rel_features = self._compute_relational_features(agent_states)  # [B, N, N, 6]
        rel_emb = self.relation_encoder(
            rel_features.reshape(B * N * N, 6)
        ).reshape(B, N, N, -1)
        rel_context = rel_emb.mean(dim=2)  # [B, N, D]
        
        # FFN
        x = x + self.level2_ffn(self.level2_norm(x + rel_context))
        level2_features = x  # [B, N, D]
        
        # Semantic routing for Level 2
        semantic_tokens_l2 = None
        if self.use_semantic_routing and semantic_inputs is not None:
            slots_l2, attn_l2, attn_maps_l2 = self.slot_router_l2(
                level2_features,
                return_attention=self.use_decorr_loss and self.training
            )
            
            # Decode semantic tokens
            semantic_tokens_l2 = self._decode_semantic_tokens(
                slots=slots_l2,
                slot_names=self.slot_router_l2.slot_names,
                level=2,
                semantic_inputs=semantic_inputs,
                agent_features=level2_features
            )  # [B, N, K2, D] or None
            
            if return_routing_info:
                routing_info['level2'] = {
                    'slot_attention': attn_l2,  # [B, N, K]
                    'slot_names': self.slot_router_l2.slot_names
                }
            
            # Decorr loss for slot attention
            if self.use_decorr_loss and self.training and attn_maps_l2 is not None:
                # attn_maps_l2: [num_iters, B, K, N]
                decorr_losses['slot_attn_l2'] = self.decorr_criterion(
                    attn_maps_l2[-1].unsqueeze(-1)  # Last iteration
                )
        
        # ====================================================================
        # LEVEL 3: SCENE
        # ====================================================================
        
        if scene_features is not None:
            # Cross-attention with decorr loss
            if self.use_decorr_loss and self.training:
                scene_out, scene_attn_weights = self.agent_scene_attn(
                    level2_features, scene_features, scene_features,
                    need_weights=True, average_attn_weights=False
                )
                decorr_losses['scene_attn'] = self.decorr_criterion(
                    scene_attn_weights.unsqueeze(-1)
                )
            else:
                scene_out, _ = self.agent_scene_attn(
                    level2_features, scene_features, scene_features
                )
            
            x = self.level3_norm(level2_features + scene_out)
        else:
            x = level2_features
        
        # FFN
        x = x + self.level3_ffn(self.level3_ffn_norm(x))
        level3_features = x  # [B, N, D]
        
        # Semantic routing for Level 3
        semantic_tokens_l3 = None
        if self.use_semantic_routing and semantic_inputs is not None:
            slots_l3, attn_l3, attn_maps_l3 = self.slot_router_l3(
                level3_features,
                return_attention=self.use_decorr_loss and self.training
            )
            
            semantic_tokens_l3 = self._decode_semantic_tokens(
                slots=slots_l3,
                slot_names=self.slot_router_l3.slot_names,
                level=3,
                semantic_inputs=semantic_inputs,
                agent_features=level3_features
            )
            
            if return_routing_info:
                routing_info['level3'] = {
                    'slot_attention': attn_l3,
                    'slot_names': self.slot_router_l3.slot_names
                }
            
            if self.use_decorr_loss and self.training and attn_maps_l3 is not None:
                decorr_losses['slot_attn_l3'] = self.decorr_criterion(
                    attn_maps_l3[-1].unsqueeze(-1)
                )
        
        # ====================================================================
        # FLATTEN FOR DIT (if requested)
        # ====================================================================
        
        all_tokens = None
        if return_all_tokens:
            token_list = [level1_features, level2_features, level3_features]
            
            # Add semantic tokens if present
            if semantic_tokens_l2 is not None:
                # [B, N, K2, D] → [B, N*K2, D]
                token_list.append(semantic_tokens_l2.reshape(B, -1, self.hidden_size))
            
            if semantic_tokens_l3 is not None:
                token_list.append(semantic_tokens_l3.reshape(B, -1, self.hidden_size))
            
            all_tokens = torch.cat(token_list, dim=1)  # [B, T_total, D]
        
        # ====================================================================
        # OUTPUT
        # ====================================================================
        
        output = AgentEncoderOutput(
            level1_temporal=level1_features,
            level2_interaction=level2_features,
            level3_scene=level3_features,
            semantic_tokens_l2=semantic_tokens_l2,
            semantic_tokens_l3=semantic_tokens_l3,
            all_tokens=all_tokens,
            routing_info=routing_info if return_routing_info else None,
            decorr_losses=decorr_losses if self.use_decorr_loss and self.training else None
        )
        
        return output
    
    def _compute_relational_features(
        self,
        agent_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise relational features."""
        B, N, _ = agent_states.shape
        
        pos = agent_states[:, :, :2]
        vel = agent_states[:, :, 2:4]
        heading = agent_states[:, :, 4:5]
        
        pos_i = pos.unsqueeze(2)
        pos_j = pos.unsqueeze(1)
        delta_pos = pos_j - pos_i  # [B, N, N, 2]
        
        vel_i = vel.unsqueeze(2)
        vel_j = vel.unsqueeze(1)
        delta_vel = vel_j - vel_i
        
        dist = torch.norm(delta_pos, dim=-1, keepdim=True)
        rel_speed = torch.norm(delta_vel, dim=-1, keepdim=True)
        
        approaching = (delta_pos * delta_vel).sum(dim=-1, keepdim=True) < 0
        ttc = torch.where(
            approaching,
            dist / (rel_speed + 1e-3),
            torch.full_like(dist, 10.0)
        )
        ttc = torch.clamp(ttc, 0, 10.0)
        
        heading_i = heading.unsqueeze(2)
        bearing = torch.atan2(delta_pos[..., 1:2], delta_pos[..., 0:1])
        bearing = bearing - heading_i
        
        return torch.cat([delta_pos, delta_vel, ttc, bearing], dim=-1)
    
    def _decode_semantic_tokens(
        self,
        slots: torch.Tensor,  # [B, K, D]
        slot_names: List[str],
        level: int,
        semantic_inputs: Dict[str, torch.Tensor],
        agent_features: torch.Tensor  # [B, N, D]
    ) -> Optional[torch.Tensor]:
        """
        Decode semantic tokens from slots.
        
        For each slot that has corresponding semantic input,
        encode the semantic features and weight by slot attention.
        
        Returns: [B, N, K_active, D] or None
        """
        B, N, D = agent_features.shape
        K = len(slot_names)
        
        encoder_dict = getattr(self.semantic_bank, f'level{level}_encoders')
        
        active_tokens = []
        for i, slot_name in enumerate(slot_names):
            if slot_name in semantic_inputs and slot_name in encoder_dict:
                # Encode semantic input
                semantic_feat = semantic_inputs[slot_name]  # Could be various shapes
                
                # Simple placeholder: assume it's [B, N, F]
                if semantic_feat.dim() == 3:
                    B_s, N_s, F = semantic_feat.shape
                    semantic_feat_flat = semantic_feat.reshape(B_s * N_s, F)
                    encoded = encoder_dict[slot_name](semantic_feat_flat)  # [B*N, D]
                    encoded = encoded.reshape(B_s, N_s, D)
                elif semantic_feat.dim() == 2:
                    # [B, F] → expand to [B, N, D]
                    encoded = encoder_dict[slot_name](semantic_feat).unsqueeze(1).expand(-1, N, -1)
                else:
                    continue
                
                active_tokens.append(encoded)
        
        if not active_tokens:
            return None
        
        # Stack: [B, N, K_active, D]
        semantic_tokens = torch.stack(active_tokens, dim=2)
        
        return semantic_tokens
    
    def get_decorr_loss(self, agent_output: AgentEncoderOutput) -> torch.Tensor:
        """
        Compute total decorrelation loss.
        
        Call this in your training loop to add to main loss.
        """
        if agent_output.decorr_losses is None or not self.use_decorr_loss:
            return torch.tensor(0.0, device=agent_output.level1_temporal.device)
        
        total_loss = sum(agent_output.decorr_losses.values())
        return self.decorr_weight * total_loss


# ============================================================================
# TESTING / DEBUGGING
# ============================================================================

if __name__ == "__main__":
    """Test agent encoder."""
    
    # Config
    B, N, T_hist = 4, 10, 30
    T_scene = 48
    hidden_size = 768
    
    # Create encoder
    encoder = AgentEncoderWithRouting(
        hidden_size=hidden_size,
        num_heads=12,
        history_length=T_hist,
        use_semantic_routing=True,
        use_decorr_loss=True,
        decorr_weight=0.01
    )
    
    # Mock inputs
    agent_states = torch.randn(B, N, 5)
    agent_history = torch.randn(B, N, T_hist, 5)
    scene_features = torch.randn(B, T_scene, hidden_size)
    
    # Mock semantic inputs (placeholders)
    semantic_inputs = {
        'behavior': torch.randn(B, N, 6),
        'pedestrian': torch.randn(B, N, 5),
        'intersection': torch.randn(B, N, 5),
        'goal': torch.randn(B, N, 4)
    }
    
    # Forward pass
    output = encoder(
        agent_states=agent_states,
        agent_history=agent_history,
        scene_features=scene_features,
        semantic_inputs=semantic_inputs,
        return_all_tokens=True,
        return_routing_info=True
    )
    
    print("=" * 70)
    print("Agent Encoder Output")
    print("=" * 70)
    print(f"Level 1 (Temporal):    {output.level1_temporal.shape}")
    print(f"Level 2 (Interaction): {output.level2_interaction.shape}")
    print(f"Level 3 (Scene):       {output.level3_scene.shape}")
    
    if output.semantic_tokens_l2 is not None:
        print(f"Semantic L2:           {output.semantic_tokens_l2.shape}")
    if output.semantic_tokens_l3 is not None:
        print(f"Semantic L3:           {output.semantic_tokens_l3.shape}")
    
    if output.all_tokens is not None:
        print(f"All tokens (for DiT):  {output.all_tokens.shape}")
    
    if output.routing_info:
        print("\nRouting Info:")
        for level, info in output.routing_info.items():
            print(f"  {level}: {info['slot_names']}")
            print(f"    Attention shape: {info['slot_attention'].shape}")
    
    if output.decorr_losses:
        print("\nDecorrelation Losses:")
        for name, loss in output.decorr_losses.items():
            print(f"  {name}: {loss.item():.6f}")
        
        total_decorr = encoder.get_decorr_loss(output)
        print(f"  Total (weighted): {total_decorr.item():.6f}")
    
    print("=" * 70)