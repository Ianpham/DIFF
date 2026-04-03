"""
Set Encoders
============
Implements the shared SetEncoder base class and three concrete encoders:
 
    TrafficControlEncoder  — spec §4
    PedestrianEncoder      — spec §5
    IntersectionEncoder    — spec §6
 
All three follow the Deep Sets + pairwise distance metadata pattern.
The pairwise bias tensors are NOT consumed inside the encoder; they are
returned alongside the token sequence and injected into the DyDiT
backbone's cross-attention logits.
 
NavsimDataset inputs
--------------------
traffic_control_features  : (B, 8, 5)   [rel_x, rel_y, is_red, distance, bearing]
pedestrian_features       : (B, 10, 5)  [rel_x, rel_y, speed, heading_alignment, crosswalk_proximity]
intersection_features     : (B, 4, 5)   [in_intersection, approach_dist, turn_angle, row_proxy, dist_to_center]
 
Design notes
------------
  SetEncoder base class is parameterised by input_dim, d_hidden, d_model,
  num_heads, and pairwise_dim (1 for simple distance, 3 for pedestrian group).

  Pairwise bias MLP output shape: (B, K, K, num_heads) — ready to be
  permuted to (B, num_heads, K, K) and added to backbone attention logits.

  All encoders add a learnable modality embedding after projection.

  PedestrianEncoder extends pairwise features with heading_diff and
  speed_diff (group proximity bias, spec).
  
  IntersectionEncoder uses (rel_x, rel_y) derived from approach_dist +
  turn_angle as positional proxy for pairwise distance .

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Tuple


# share MLP builder

def build_mlp(in_channels: int, hidden_dim: int, out_channels: int) -> nn.Sequential:
    """Return two layer MLP: Linear -> GELU -> Linear"""

    return nn.Sequential(
        nn.Linear(in_channels, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_channels)
    )

# set encoder 

class SetEncoder(nn.Module):
    """Shared Deep Sets encoder for unordered set-valued modalities.
 
    Stage 1 : Per-element MLP  (B, K, input_dim) → (B, K, d_model)
    Stage 2 : Pairwise bias MLP  (B, K, K, pairwise_dim) → (B, K, K, num_heads)
    Stage 3 : Modality embedding added to all K tokens
 
    The pairwise bias is returned as a second output and is NOT applied
    inside the encoder — the DyDiT backbone adds it to attention logits.
 
    Parameters
    ----------
    input_dim    : Feature dimension of each element.
    d_hidden     : Hidden dim for the per-element MLP.
    d_model      : Output token dimension.
    num_heads    : Number of attention heads in the backbone (needed to
                   size the bias MLP output).
    pairwise_dim : Dimension of the pairwise feature vector fed into the
                   bias MLP. 1 = scalar distance only; 3 = [dist, Δheading, Δspeed].
    bias_hidden  : Hidden dim for the bias MLP. Default 16 (32 for ped).
    """
    def __init__(
          self,
          in_channels: int,
          hidden_dim: int, 
          model_dim: int,
          num_heads: int,
          pairwise_dim: int = 1,
          bias_hidden: int = 16,
    ):
      super().__init__()
      

      # per element embedding (deep set stype)
      self.element_mlp = build_mlp(in_channels, hidden_dim, model_dim)

      # pairwise bias mlp
      self.bias_mlp = build_mlp(pairwise_dim, bias_hidden, num_heads)

      # learnable modality token
      self.modality_emb = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

      self.model_dim = model_dim
      self.num_heads = num_heads

    # pairewise feature
    def _compute_pairwise_feature(
          self,
          xy: torch.Tensor
    ) -> torch.Tensor:
       
       """
       Compute scalr pairwise L2 distances from (x, y) columns.

       Parameters

       xy: (B, K, 2) - position columsn from the input features

       Returns

       dist: (B, K, K, 1)
       """

       diff = xy.unsqueeze(2) - xy.unsqueeze(1) # (B, K, K, 2)

       dist = diff.norm(dim = -1, keepdim=True) # (B, K, K, 1)

       return dist
    

    def forward(
          self, 
          x:   torch.Tensor,
          pairwise_feats: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, torch.Tensor]:
       
        """
        Parameters
        ----------
        x              : (B, K, input_dim)  set of element features
        pairwise_feats : (B, K, K, pairwise_dim) or None.
                         If None, computed from x[:, :, :2] as L2 distance.
 
        Returns
        -------
        tokens      : (B, K, d_model)
        bias        : (B, K, K, num_heads)  — add to backbone attention logits
        """      
        B, K, _ = x.shape

        # per-element MLP
        tokens = self.element_mlp(x) # (B, K, d_model)

        # pair wise bias
        if pairwise_feats is None:
           xy = x[:, :, :2]
           pairwise_feats = self._compute_pairwise_feature(xy) # (B, K, K, 1)


        bias = self.bias_mlp(pairwise_feats)   # (B, K, K, num_heads)

        # modality embedding
        tokens = tokens + self.modality_emb # broadcast (1, 1, d_model)

        return tokens, bias
    


# Now apply to traffic control encoder 

class TrafficControlEncoder(nn.Module):
   """
   Encode up to K = 8, traffic light features
   input feature (5D): [rel_x, rel_y, is_red, distance, bearing]
   pairwise bias: scalar L2 distance between light pairs

   Parameters

   d_model: backbone token dimension.
   d_hidden: per-element MLP hidden dim, spec: 128
   num_heads: backbone attention heads
   max_k: maximum number of traffic lights (for masking). current default 8.
   
   """

   def __init__(
         self, 
         d_model: int  = 256,
         d_hidden: int = 128,
         num_heads: int = 8,
         max_k    : int = 8,
   ):
      super().__init__()

      self.encoder = SetEncoder(
         in_channels=5,
         hidden_dim=d_hidden,
         model_dim=d_model,
         num_heads=num_heads, 
         pairwise_dim=1, 
         bias_hidden=16,
      )
      self.max_k = max_k

   def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, torch.Tensor]:
             
        """
        Parameters
        ----------
        x            : (B, K, 5)  K ≤ 8 traffic light features
        padding_mask : (B, K) bool — True where element is padding (absent light)
 
        Returns
        -------
        tokens : (B, K, d_model)
        bias   : (B, K, K, num_heads)
        """  
        tokens, bias = self.encoder(x)

        # zero out padding positioon so absent lights dont contribute
        if padding_mask is not None:
           
           # tokens: zero padded rows
           tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)

           # bias: zero rows and cols corresponding to padded elements
           pad2d = padding_mask.unsqueeze(2) | padding_mask.unsqueeze(1)  # (B, K, K)
           bias = bias.masked_fill(pad2d.unsqueeze(-1), float("-inf"))

        return tokens, bias
   


# PedestrianEncoder 

class PedestrianEncoder(nn.Module):

    """Encode up to K=10 pedestrian features.
 
    Input features (5D): [rel_x, rel_y, speed, heading_alignment, crosswalk_proximity]
 
    Pairwise features (3D per pair):
        [pairwise_dist, |heading_diff|, |speed_diff|]
 
    This richer pairwise feature encodes group formation — pedestrians
    that are close, moving at similar speeds in similar directions tend
    to cross together (Social Force Model, Helbing & Molnár 1995).
 
    Bias MLP hidden dim is 32 (vs 16 for other encoders) because the
    input is 3D rather than scalar.
    """
    def __init__(
          self, 
          d_model   : int = 256,
          d_hidden  : int = 128,
          num_heads : int = 8,
          max_k     : int = 10,
    ):
       super().__init__()
       self.encoder = SetEncoder(
          in_channels = 5,
          hidden_dim = d_hidden,
          model_dim = d_model,
          num_heads=num_heads,
          pairwise_dim=3, # dist + heading_diff + speed_diff
          bias_hidden=32, # wider hidden for 3D pairwise input
       )   

       self.max_k = max_k

    @staticmethod
    def _group_pairwise_features(x: torch.Tensor) -> torch.Tensor:
        """Build the 3D pairwise feature tensor for group proximity bias.

        Parameters
        ----------
        x : (B, K, 5)  [rel_x, rel_y, speed, heading_alignment, crosswalk_proximity]

        Returns
        -------
        feats : (B, K, K, 3)  [pairwise_dist, |Δheading|, |Δspeed|] 
        """
        xy = x[:, :, 0:2] # (B, K, 2)
        speed = x[:, :, 2]  # (B, K)
        heading = x[:, :, 3]  # (B, K)

        # pairwise L2 distance
        diff = xy.unsqueeze(2) - xy.unsqueeze(1)  # (B, K, K, 2)
        dist = diff.norm(dim = -1, keepdim= True) # (B, K, K, 1)

        # absolute pairwise heading difference
        hdiff = (heading.unsqueeze(2) - heading.unsqueeze(1)).abs().unsqueeze(-1) # (B, K, K, 1)

        # absolute pairwise speed difference
        sdiff = (speed.unsqueeze(2) - speed.unsqueeze(1)).abs().unsqueeze(-1)   # (B, K, K, 1)

        return torch.cat([dist, hdiff, sdiff], dim = -1)    # (B, K, K, 3)
    
    def forward(
          self, 
          x: torch.Tensor,
          padding_mask: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x            : (B, K, 5)  K ≤ 10 pedestrian features
        padding_mask : (B, K) bool — True for absent pedestrians
 
        Returns
        -------
        tokens : (B, K, d_model)
        bias   : (B, K, K, num_heads)
        """
        group_feats = self._group_pairwise_features(x) # (B, K, K, 3)
        tokens, bias = self.encoder(x, pairwise_feats = group_feats)

        if padding_mask is not None:
            tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            pad2d = padding_mask.unsqueeze(2) | padding_mask.unsqueeze(1)
            bias = bias.masked_fill(pad2d.unsqueeze(-1), float("-inf"))

        return tokens, bias
    

# intersection encoder

class IntersectionEncoder(nn.Module):
    """Encode up to K=4 intersection features.
 
    Input features (5D):
        [in_intersection, approach_dist, turn_angle, row_proxy, dist_to_center]
 
    Positional proxy for pairwise distance:
        The input has no direct (x, y); we derive an approximate 2D position
        from (approach_dist, turn_angle) using polar → Cartesian conversion:
            proxy_x = approach_dist * cos(turn_angle)
            proxy_y = approach_dist * sin(turn_angle)
        This gives a reasonable spatial layout of intersections around the ego.
 
    Bias MLP: Linear(1, 16) → GELU → Linear(16, num_heads)  (same as traffic).
    """
    
    def __init__(
            self, 
            d_model: int = 256,
            d_hidden: int = 128,
            num_heads: int = 8,
            max_k:     int = 4,
            input_dim: int = 5,
    ):
        super().__init__()
        self.encoder = SetEncoder(
            in_channels = input_dim,
            hidden_dim = d_hidden, 
            model_dim = d_model,
            num_heads=num_heads,
            pairwise_dim=1,
            bias_hidden=16,
        )
        self.max_k = max_k


    @staticmethod
    def _intersection_pairwise_dist(x: torch.Tensor) -> torch.Tensor:
        """Derive pairwise distance using polar → Cartesian from (approach_dist, turn_angle).
 
        NavsimDataset intersection_features channels:
            0: in_intersection
            1: approach_dist
            2: turn_angle
            3: row_proxy
            4: dist_to_center
 
        Returns
        -------
        dist : (B, K, K, 1)
        """
        approach = x[:, :, 1] # (B, K)
        angle    = x[:, :, 2] # (B, K)

        proxy_x = approach * torch.cos(angle)   # (B, K)
        proxy_y = approach * torch.sin(angle)   # (B, K)

        proxy = torch.stack([proxy_x, proxy_y], dim = -1) # (B, K, 2)

        diff = proxy.unsqueeze(2) - proxy.unsqueeze(1)  # (B, K, K, 2)

        dist = diff.norm(dim = -1, keepdim=True)  # (B, K, K, 1)

        return dist
    
    def forward(
            self,
            x: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Parameters
        ----------
        x            : (B, K, 5)  K ≤ 4 intersection features
        padding_mask : (B, K) bool — True for absent intersections
 
        Returns
        -------
        tokens : (B, K, d_model)
        bias   : (B, K, K, num_heads)
        """
        pairwise_dist = self._intersection_pairwise_dist(x)   # (B, K, K, 1)
        tokens, bias  = self.encoder(x, pairwise_feats=pairwise_dist)
 
        if padding_mask is not None:
            tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            pad2d  = padding_mask.unsqueeze(2) | padding_mask.unsqueeze(1)
            bias   = bias.masked_fill(pad2d.unsqueeze(-1), float("-inf"))
 
        return tokens, bias

# NavsimDataset adapter
# ---------------------------------------------------------------------------
 
def prepare_set_encoder_inputs(
    batch    : dict,
    device   : Optional[torch.device] = None,
) -> dict:
    """Extract set-encoder inputs from a NavsimDataset batch.
 
    Returns
    -------
    dict with keys:
        traffic_x        : (B, 8, 5)
        traffic_mask     : (B, 8) bool — True = padding
        pedestrian_x     : (B, 10, 5)
        pedestrian_mask  : (B, 10) bool
        intersection_x   : (B, 4, 5)
        intersection_mask: (B, 4) bool
    """
    def _get(key: str, max_k: int, feat_dim: int):
        t = batch[key]                         # (B, K, F) or (K, F)
        if device is not None:
            t = t.to(device)
        t = t.float()
        if t.dim() == 2:                       # unbatched → (1, K, F)
            t = t.unsqueeze(0)
        B, K, F = t.shape
        # Padding mask: rows that are all-zero are treated as absent
        mask = (t.abs().sum(dim=-1) == 0)      # (B, K)
        return t, mask
 
    tc_x,   tc_mask   = _get("traffic_control_features", 8,  5)
    ped_x,  ped_mask  = _get("pedestrian_features",      10, 5)
    int_x,  int_mask  = _get("intersection_features",    4,  5)
 
    return {
        "traffic_x"        : tc_x,
        "traffic_mask"     : tc_mask,
        "pedestrian_x"     : ped_x,
        "pedestrian_mask"  : ped_mask,
        "intersection_x"   : int_x,
        "intersection_mask": int_mask,
    }
 
 
# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    torch.manual_seed(0)
    B         = 2
    D         = 256
    NUM_HEADS = 8
 
    # ------------------------------------------------------------------
    # TrafficControlEncoder
    # ------------------------------------------------------------------
    print("=" * 55)
    print("TrafficControlEncoder")
    print("=" * 55)
 
    tc_enc = TrafficControlEncoder(d_model=D, d_hidden=128, num_heads=NUM_HEADS)
 
    # K=8 traffic lights; last 2 are absent (zero-padded)
    tc_x    = torch.randn(B, 8, 5)
    tc_x[:, 6:, :] = 0.0                           # simulate missing lights
    tc_mask = torch.zeros(B, 8, dtype=torch.bool)
    tc_mask[:, 6:] = True
 
    tc_tokens, tc_bias = tc_enc(tc_x, padding_mask=tc_mask)
    print(f"  Input  : {tc_x.shape}")
    print(f"  Tokens : {tc_tokens.shape}")          # (2, 8, 256)
    print(f"  Bias   : {tc_bias.shape}")            # (2, 8, 8, 8)
    # Padded token rows should be zero
    assert tc_tokens[:, 6:, :].abs().max() == 0, "Padded tokens not zeroed"
    total = sum(p.numel() for p in tc_enc.parameters())
    print(f"  Params : {total:,}")
 
    # ------------------------------------------------------------------
    # PedestrianEncoder
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("PedestrianEncoder")
    print("=" * 55)
 
    ped_enc = PedestrianEncoder(d_model=D, d_hidden=128, num_heads=NUM_HEADS)
 
    ped_x    = torch.randn(B, 10, 5)
    ped_x[:, 8:, :] = 0.0
    ped_mask = torch.zeros(B, 10, dtype=torch.bool)
    ped_mask[:, 8:] = True
 
    ped_tokens, ped_bias = ped_enc(ped_x, padding_mask=ped_mask)
    print(f"  Input  : {ped_x.shape}")
    print(f"  Tokens : {ped_tokens.shape}")          # (2, 10, 256)
    print(f"  Bias   : {ped_bias.shape}")            # (2, 10, 10, 8)
    assert ped_tokens[:, 8:, :].abs().max() == 0, "Padded tokens not zeroed"
    total = sum(p.numel() for p in ped_enc.parameters())
    print(f"  Params : {total:,}")
 
    # Verify group pairwise features are non-symmetric in general
    # (dist is symmetric but heading/speed diffs are |·| so also symmetric — correct)
    gf = PedestrianEncoder._group_pairwise_features(ped_x)
    print(f"  Group pairwise feats : {gf.shape}")    # (2, 10, 10, 3)
    # Diagonal distances should be zero
    diag_dist = gf[:, :, :, 0].diagonal(dim1=1, dim2=2)
    assert diag_dist.abs().max() < 1e-5, "Self-distances not zero"
 
    # ------------------------------------------------------------------
    # IntersectionEncoder
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("IntersectionEncoder")
    print("=" * 55)
 
    int_enc = IntersectionEncoder(d_model=D, d_hidden=128, num_heads=NUM_HEADS)
 
    int_x    = torch.randn(B, 4, 5)
    int_mask = torch.zeros(B, 4, dtype=torch.bool)  # all present
 
    int_tokens, int_bias = int_enc(int_x, padding_mask=int_mask)
    print(f"  Input  : {int_x.shape}")
    print(f"  Tokens : {int_tokens.shape}")          # (2, 4, 256)
    print(f"  Bias   : {int_bias.shape}")            # (2, 4, 4, 8)
    total = sum(p.numel() for p in int_enc.parameters())
    print(f"  Params : {total:,}")
 
    # ------------------------------------------------------------------
    # Bias shape ready for backbone injection
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("Backbone bias injection check")
    print("=" * 55)
    # The backbone needs (B, num_heads, K, K)
    tc_bias_for_attn = tc_bias.permute(0, 3, 1, 2)   # (B, H, K, K)
    print(f"  Traffic bias for attn_mask : {tc_bias_for_attn.shape}")
 
    # ------------------------------------------------------------------
    # NavsimDataset batch adapter
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("NavsimDataset adapter")
    print("=" * 55)
    fake_batch = {
        "traffic_control_features": torch.randn(B, 8, 5),
        "pedestrian_features"     : torch.randn(B, 10, 5),
        "intersection_features"   : torch.randn(B, 4, 5),
    }
    # Simulate 2 absent traffic lights
    fake_batch["traffic_control_features"][:, 6:, :] = 0.0
 
    inputs = prepare_set_encoder_inputs(fake_batch)
    print(f"  traffic_x shape    : {inputs['traffic_x'].shape}")
    print(f"  traffic_mask       : {inputs['traffic_mask'][0].tolist()}")
    print(f"  pedestrian_x shape : {inputs['pedestrian_x'].shape}")
    print(f"  intersection_x     : {inputs['intersection_x'].shape}")
 
    # Full pipeline test
    tc_t, tc_b   = tc_enc(inputs["traffic_x"],  inputs["traffic_mask"])
    ped_t, ped_b = ped_enc(inputs["pedestrian_x"], inputs["pedestrian_mask"])
    int_t, int_b = int_enc(inputs["intersection_x"], inputs["intersection_mask"])
 
    print(f"\n  All tokens produced successfully:")
    print(f"    Traffic      tokens: {tc_t.shape}  bias: {tc_b.shape}")
    print(f"    Pedestrian   tokens: {ped_t.shape} bias: {ped_b.shape}")
    print(f"    Intersection tokens: {int_t.shape} bias: {int_b.shape}")
    print("\n✓ All checks passed")  


      

    
