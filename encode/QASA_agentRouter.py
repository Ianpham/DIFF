import torch
import torch.nn as nn

from typing import Tuple, Optional, List, Any, Dict
from dataclasses import dataclass
# gumbel slot selector(adaslot - inspired)
# here we implement the update versionm of slot attention

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


class GumbelSlotSelector(nn.Module):
    """
    Differentiable slot selector using Gumbel-Softmax.
    
    Based on: Adaptive Slot Attention (Fan et al., CVPR 2024)
    
    Instead of selecting a fixed number of slots, dynamically determine
    which slots should be active based on:
    1. Slot content (learned selector)
    2. Coverage (ensure enough tokens are explained)     
    """

    def __init__(
            self,
            hidden_size: int,
            temperature: float = 0.5,
            min_active_slots: int = 1,
            coverage_threshold: float = 0.8
    ):
        super().__init__()
        self.hidden_size =hidden_size
        self.temperature = temperature
        self.converage_threshold = coverage_threshold
        self.min_active_slots = min_active_slots

        # selection network: slot -> selection probability
        self.selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(
            self,
            slots: torch.Tensor, #[B, K, D]
            attention_weights: torch.Tensor, # [B, N, K]
            use_gumbel: bool = True
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select which slots should be active.

        Returns:
            selection_mask: [B, K] - Binary mask (1 = active, 0 = inactive)
            selection_logits: [B, K] - Raw selection scores
        """

        B, K, D = slots.shape

        #compute selection logits from slot content
        selection_logits = self.selector(slots).squeeze(-1) # B, K

        if use_gumbel and self.training:
            # gumbel-softmax for differentiable selection
            # this allows gradients to flow while maintaining discreteness

            selection_probs = nn.functional.gumbel_softmax(
                selection_logits,
                tau = self.temperature,
                hard = True # hard selection (0 or 1)
            )

        else:
            # hard selection at inference
            selection_probs = torch.sigmoid(selection_logits)
            selection_mask = (selection_probs > 0.5).float()

            # ensure minimum number of active slots
            if selection_mask.sum(dim = 1).min() < self.min_active_slots:
                # select top-k slots if too few are active
                _, top_indices = torch.topk(
                    selection_logits, k = self.min_active_slots, dim = 1
                )
                selection_mask = torch.zeros_like(selection_mask)
                selection_mask.scatter_()

            selection_probs = selection_mask

        return selection_probs, selection_logits
    
    def compute_converage(
            self,
            attention_weights: torch.Tensor, # [ B, N, K]
            selection_mask: torch.Tensor,
    )-> torch.Tensor:
        """
        Compute converage: fraction of token explainted by active slots
        """
        action_attention = attention_weights * selection_mask.unsqueeze(1) # [B, N, K]
        coverage = action_attention.sum(dim = 2) / (attention_weights.sum(dim = 2) + 1e-8)

        return coverage.mean(dim = 1) # [B]
    
#### quality estimator (QASA inspired)

class SlotQualityEstimator(nn.Module):
    """
    Based on: QASA (Quality-Guided K-Adaptive Slot Attention, 2025)
    
    Key idea: High-quality slot wins decisively on its tokens.
    Poor-quality slot has diffuse attention (ambiguous binding).
    """

    def __init__(
            self, 
            hidden_size: int,
            use_learned_quality: bool = True
    ):
        super().__init__()
        self.use_learned_quality = use_learned_quality

        if use_learned_quality:
            # learned quality estimator (in addtion to attention-based)
            self.quality_net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )

    def compute_attention_quality(
            self,
            attention_maps: torch.Tensor #[B, K, N]
    )-> torch.Tensor:
        """
        Compute attention-based quality metric.

        Quality = proportion of attention mass in winning region.

        for each slot k:
        - find tokens where slot k wins (argmax over slots)
        - measure what fraction of slot k's attention is on those tokens
        - high quality -> concentrated attention on winning tokens
        - low quality -> diffuse attetnion across all tokens
        
        """

        B, K, N = attention_maps.shape
        
        # finding wining slots per tokens [B, N]
        winners = attention_maps.argmax(dim = 1)

        # compute quality for each slots
        quality_score = []
        for k in range(K):
            #binary mask : which tokens does slot k win?
            winning_mask = (winners == k).float() # [B, N]

            # slot k's attention on all tokens
            slot_k_attn = attention_maps[:,k,:] #[B, N]

            # attention on winning tokens
            winning_attn = (slot_k_attn * winning_mask).sum(dim = 1) # [B]
            total_attn = slot_k_attn.sum(dim = 1) # [B]

            # quality = winning/total
            quality = winning_attn/ (total_attn + 1e-8)
            quality_score.append(quality)

        return torch.stack(quality_score, dim = 1) # [B, K]
    
    def compute_learned_quality(
            self,
            slots: torch.Tensor, # [B, K, D]
        )-> torch.Tensor:
        """Learned quality from slot representation"""
        if not self.use_learned_quality:
            return torch.ones(
                slots.shape[0], slots.shape[1],
                device = slots.device
            )
        
        return self.quality_net(slots).squeeze(-1) # [B, K]
    
    def forward(
            self,
            slots: torch.Tensor,
            attention_maps: torch.Tensor,
            semantic_confidence: Optional[torch.Tensor] = None,
            combine_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    )-> torch.Tensor:
        """
        Compute combined quality score.
        
        Args:
            slots: [B, K, D]
            attention_maps: [B, K, N]
            semantic_confidence: [B, K] - Optional detector confidence
            combine_weights: (attn_weight, learned_weight, confidence_weight)
            
        Returns:
            quality_scores: [B, K] in [0, 1]
        """

        # attention-based quality (unsupervised)
        attn_quality = self.compute_attention_quality(attention_maps)

        # learned quality (trained end-to-end)
        learned_quality = self.compute_learned_quality(slots)

        # combine
        w_attn, w_learned, w_conf = combine_weights
        quality = w_attn * attn_quality + w_learned * learned_quality

        # incorporate semantic dectector confidence if available
        if semantic_confidence is not None:
            quality = quality * w_conf + semantic_confidence * (1 - w_conf)


        return quality

# adaptive quality-guided slot router, upgrade version of slot attention

class AdaptiveQualityGuidedSlotRouter(nn.Module):
    """
    Combined adaptive + quality-guided slot attention routing.
    
    Combines:
    1. AdaSlot: Dynamic slot selection
    2. QASA: Quality-based filtering
    3. Base Slot Attention: Competitive binding
    
    Features:
    - Variable number of active slots per instance
    - Quality-based slot filtering
    - Semantic confidence integration
    - Graceful handling of missing semantic concepts
    """

    def __init__(
            self,
            hidden_size: int, 
            max_slots: int, # maxium possible slots
            slot_names: List[str],
            num_iterations: int = 3,
            # adaptive selection config
            use_adaptive_selection: bool = True,
            gumbel_temperature: float = 0.5,
            min_active_slots: int = 1,
            # quality guidance config
            use_quality_guidance: bool = True,
            quality_threshold: float = 0.3,
            quality_combine_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_slots = max_slots
        self.num_iterations = num_iterations
        self.use_adaptive = use_adaptive_selection
        self.use_quality = use_quality_guidance
        self.quality_threshold = quality_threshold

        # base slot attention
        self.slot_inits = nn.Parameter(
            torch.randn(max_slots, hidden_size) * 0.02
        )

        # attention projections # same as 
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
        
        # next is we add adaptive selection through this
        if use_adaptive_selection:
            self.slot_selector = GumbelSlotSelector(
                hidden_size = hidden_size, 
                temperature = gumbel_temperature,
                min_active_slots= min_active_slots
            )

        # quality estimation
        if use_quality_guidance:
            self.quality_estimator = SlotQualityEstimator(
                hidden_size= hidden_size, 
                use_learned_quality= True
            )

            self.quality_combine_weights = quality_combine_weights
    
    def forward(
            self,
            inputs: torch.Tensor, # [B, N, D]
            semantic_confidence: Optional[torch.Tensor] = None, # [B, K]
            return_all_info: bool = False
    )-> SlotRoutingOutput:
        """
        Route agent features to semantic slots with adaptive selections
        Args:
            inputs: [B, N, D] - Agent features
            semantic_confidence: [B, K] - Confidence from semantic detectors
            return_all_info: Return attention maps for analysis
            
        Returns:
            SlotRoutingOutput with slots, masks, quality, losses

        """
        B, N, D = inputs.shape
        K = self.max_slots

        # normalize input
        inputs = self.norm_input(inputs)

        # initialize all candidate slots
        slots = self.slot_inits.unsqueeze(0).expand(B, -1, -1) # [B, K, D]

        # iterative slots attention
        attn_maps_all = []

        for iteration in range(self.num_iterations):
            slot_prev = slots

            # attention: slot (query) attend to inputs (key/value)
            q = self.q_proj(slots) # [B, K, D]
            k = self.k_proj(inputs) # [B, N, D]
            v = self.v_proj(inputs) # [B, N, D]

            # attention scores : [B, K, N]
            attn_logits = torch.einsum('bkd, bnd -> bkn', q, k) * self.scale

            # competitive softmax (over slots)
            attn = nn.functional.softmax(attn_logits, dim = 1)

            # weighted sum: [B, K, D]
            updates = torch.einsum('bkn, bnd -> bkd', attn, v)

            # GRU update
            slots_flat = slots.reshape(B*K, D)
            updates_flat = updates.reshape(B*K, D)

            slots_flat = self.gru(updates_flat, slots_flat)

            slots = slots + self.mlp(slots)

            attn_maps_all.append(attn)

        # final attention: [B, K, N] -> transpose to [B, N, K]
        final_attn = attn_maps_all[-1]
        attn_weights = final_attn.transpose(1,2) # [B, N, K]

        # quality esitmation
        if self.use_quality:
            quality_scores = self.quality_estimator(
                slots = slots,
                attention_maps = final_attn,
                semantic_confidence = semantic_confidence, 
                conbine_weights = self.quality_combine_weights
            ) # [B, K]
        
        else:
            quality_scores = torch.ones(B, K, device = inputs.device)

        #adaptive slot selection
        if self.use_adaptive:
            selection_mask, selection_logits = self.slot_selector(
                slots = slots,
                attn_weights = attn_weights,
                use_gumbel = self.training
            ) # [B, K]

            # combine with quality: only select high-quality slots
            if self.use_quality:
                quality_mask = (quality_scores > self.quality_threshold).float()
                selection_mask = selection_mask * quality_mask

            # ensure at least one slot is active
            if selection_mask.sum(dim = 1).min() < 1:
                # select best quality slot
                best_slot_idx = quality_scores.argmax(dim = 1)
                selection_mask.scatter_(
                    1, best_slot_idx.unsqueeze(1), 1
                )

        else:
             # No adaptive selection: use quality threshold only
            if self.use_quality:
                selection_mask = (quality_scores > self.quality_threshold).float()
            else:
                selection_mask = torch.ones(B, K, device=inputs.device)

        # apply gating to slots        
        gated_slots = slots * selection_mask.unsqueeze(-1)
        

        # compute losses

        # Sparsity loss: encourage using fewer slots
        if self.use_adaptive and self.training:
            sparsity_loss = selection_mask.sum() / (B * K)
        else:
            sparsity_loss = None
        
        # Quality loss: encourage high quality slots
        if self.use_quality and self.training:
            # Only penalize active slots
            active_quality = quality_scores * selection_mask
            quality_loss = -active_quality.sum() / (selection_mask.sum() + 1e-8)
        else:
            quality_loss = None
        
        #gating
        
        num_active = selection_mask.sum(dim=1)  # [B]
        
        output = SlotRoutingOutput(
            slots=gated_slots,
            selection_mask=selection_mask,
            quality_scores=quality_scores,
            attention_weights=attn_weights,
            attention_maps=final_attn if return_all_info else None,
            sparsity_loss=sparsity_loss,
            quality_loss=quality_loss,
            num_active_slots=num_active
        )
        
        return output      
    
class AgentEncoderWithEnhancedRouting(nn.Module):
    """
    Drop-in replacement for AgentEncoderWithRouting.
    
    Changes:
    1. Uses AdaptiveQualityGuidedSlotRouter instead of basic SlotAttention
    2. Handles variable number of semantic concepts
    3. Integrates detector confidence scores
    4. Adds quality and sparsity losses
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        # Semantic routing config
        use_semantic_routing: bool = True,
        max_slots_l2: int = 5,
        max_slots_l3: int = 4,
        slot_names_l2: List[str] = None,
        slot_names_l3: List[str] = None,
        # Quality/adaptive config
        use_adaptive_selection: bool = True,
        use_quality_guidance: bool = True,
        quality_threshold: float = 0.3,
        gumbel_temperature: float = 0.5,
        sparsity_weight: float = 0.01,
        quality_weight: float = 0.01,

        # decor loss config:
        use_decorr_loss: bool = True,
        decorr_weight: float = 0.01, 
        decorr_type: str = 'cross_correlation',
        
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sparsity_weight = sparsity_weight
        self.quality_weight = quality_weight
        self.num_heads = num_heads
        self.use_decorr_loss = use_decorr_loss
        self.decorr_weight = decorr_weight
        self.use_senmatic_routing = use_semantic_routing
        #level 1 temporal
        # current state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(5, hidden_size), # please consider that 5 would be 7 if we want to have acceleration
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # history encoder (LTSM)
        self.history_encoder = nn.LTSM(
            input_size= 5,
            hidden_size = hidden_size,
            num_layers = 2,
            batch_first = True,
            dropout = dropout if dropout > 0 else 0.0
        )

        # history attention pooling
        self.history_attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim = 1)
        )

        # fuse state + history
        self.level1_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # level 2 interaction
        # spatial attention

        self.spatial_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout= dropout, batch_first= True
        )

        self.spatial_norm = nn.LayerNorm(hidden_size)

        # relation feature
        # please gonna use relation that we already design (between agent-agent, agent-lane, agent-pedestrian)
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
            from .agent_encoder import SlotAttention, SemanticEncoderBank
            slot_iterations = 3
            self.slot_router_l2 = SlotAttention(
                hidden_size=hidden_size,
                num_slots=2,  # behavior, pedestrian
                num_iterations=slot_iterations,
                slot_names=['behavior', 'pedestrian']
            )
            self.semantic_bank = SemanticEncoderBank(hidden_size)
        
        # Default slot names
        if slot_names_l2 is None:
            slot_names_l2 = ['behavior', 'pedestrian', 'vehicle_type', 'maneuver', 'intent']
        if slot_names_l3 is None:
            slot_names_l3 = ['intersection', 'goal', 'traffic_control', 'lane_change']
        
        #level 3: scene
        
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
        
        # decorrelation loss 
        from .agent_encoder import DecorrelationLoss    
        if use_decorr_loss:
            self.decorr_criterion = DecorrelationLoss(
                loss_type=decorr_type,
                normalize=True
            )
        
        # Enhanced slot routers
        self.slot_router_l2 = AdaptiveQualityGuidedSlotRouter(
            hidden_size=hidden_size,
            max_slots=max_slots_l2,
            slot_names=slot_names_l2,
            use_adaptive_selection=use_adaptive_selection,
            use_quality_guidance=use_quality_guidance,
            quality_threshold=quality_threshold,
            gumbel_temperature=gumbel_temperature
        )
        
        self.slot_router_l3 = AdaptiveQualityGuidedSlotRouter(
            hidden_size=hidden_size,
            max_slots=max_slots_l3,
            slot_names=slot_names_l3,
            use_adaptive_selection=use_adaptive_selection,
            use_quality_guidance=use_quality_guidance,
            quality_threshold=quality_threshold,
            gumbel_temperature=gumbel_temperature
        )
    
    def forward(
        self,
        agent_states: torch.Tensor,
        agent_history: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
        semantic_inputs: Optional[Dict[str, torch.Tensor]] = None,
        semantic_confidence: Optional[Dict[str, torch.Tensor]] = None,  # NEW
        return_all_tokens: bool = True,
        return_routing_info: bool = False,
        **kwargs
    ):
        """Forward pass with enhanced routing."""
        
        # ... (Level 1 and Level 2 base encoding same as before) ...
        B, N, _ = agent_states.shape
        device = agent_states.device
        
        routing_info = {}
        decorr_losses = {}
        # level 1 temporal 
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
        

        
        # Level 2 routing with quality guidance

          
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
        if semantic_confidence is not None:
            conf_l2 = semantic_confidence.get('level2')
        else:
            conf_l2 = None
        
        routing_output_l2 = self.slot_router_l2(
            inputs=level2_features,
            semantic_confidence=conf_l2,
            return_all_info=False
        )
        
        # Extract results
        slots_l2 = routing_output_l2.slots
        selection_mask_l2 = routing_output_l2.selection_mask
        quality_l2 = routing_output_l2.quality_scores
        
        # ... (Level 3 similar) ...
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
        #Collect losses
        losses = {}
        if routing_output_l2.sparsity_loss is not None:
            losses['sparsity_l2'] = self.sparsity_weight * routing_output_l2.sparsity_loss
        if routing_output_l2.quality_loss is not None:
            losses['quality_l2'] = self.quality_weight * routing_output_l2.quality_loss
        
        #... (same for L3) ...
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
        
        # Add to output
        output = AgentEncoderOutput(
            level1_temporal=level1_features,
            level2_interaction=level2_features,
            level3_scene=level3_features,
                    semantic_tokens_l2=semantic_tokens_l2,
            semantic_tokens_l3=semantic_tokens_l3,
            routing_info={
                'level2': {
                    'selection_mask': selection_mask_l2,
                    'quality_scores': quality_l2,
                    'num_active': routing_output_l2.num_active_slots
                },
                # 
            },
            decorr_losses=losses
        )
        
        return output


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """Test enhanced routing."""
    
    B, N, D = 4, 10, 768
    max_slots = 5
    
    # Create router
    router = AdaptiveQualityGuidedSlotRouter(
        hidden_size=D,
        max_slots=max_slots,
        slot_names=['behavior', 'pedestrian', 'vehicle_type', 'maneuver', 'intent'],
        use_adaptive_selection=True,
        use_quality_guidance=True,
        quality_threshold=0.3
    )
    
    # Mock inputs
    features = torch.randn(B, N, D)
    
    # Mock semantic confidence from detectors
    # Shape: [B, K] with confidence per semantic concept
    semantic_conf = torch.tensor([
        [0.95, 0.45, 0.88, 0.72, 0.31],  # High behavior, low pedestrian
        [0.23, 0.91, 0.12, 0.08, 0.89],  # Low behavior, high pedestrian
        [0.87, 0.15, 0.94, 0.81, 0.76],  # High most, low pedestrian
        [0.91, 0.88, 0.92, 0.85, 0.79],  # All high confidence
    ])
    
    # Forward pass
    output = router(
        inputs=features,
        semantic_confidence=semantic_conf,
        return_all_info=True
    )
    
    print("=" * 70)
    print("Enhanced Slot Routing Output")
    print("=" * 70)
    print(f"Slots shape: {output.slots.shape}")
    print(f"Selection mask: {output.selection_mask}")
    print(f"Quality scores: {output.quality_scores}")
    print(f"Active slots per batch: {output.num_active_slots}")
    print()
    
    # Show which slots are active per batch
    slot_names = router.slot_names
    for b in range(B):
        active_slots = [
            name for i, name in enumerate(slot_names)
            if output.selection_mask[b, i] > 0.5
        ]
        print(f"Batch {b}: Active slots = {active_slots}")
        print(f"  Confidence: {semantic_conf[b].tolist()}")
        print(f"  Quality:    {output.quality_scores[b].tolist()}")
        print()
    
    if output.sparsity_loss is not None:
        print(f"Sparsity loss: {output.sparsity_loss.item():.4f}")
    if output.quality_loss is not None:
        print(f"Quality loss: {output.quality_loss.item():.4f}")
    
    print("=" * 70)
    print("\nUtilization Statistics:")
    print(f"Average active slots: {output.num_active_slots.float().mean().item():.2f} / {max_slots}")
    print(f"Utilization rate: {output.num_active_slots.float().mean().item() / max_slots * 100:.1f}%")
    print(f"Average quality (active): {(output.quality_scores * output.selection_mask).sum() / output.selection_mask.sum():.3f}")