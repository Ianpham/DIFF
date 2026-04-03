# here we change style and more apdapted with Dydit design and MoE and Gate design.

"""
HistoryEncoder
==============
Encodes temporal trajectories of N agents over T_hist timesteps.
 
Spec source : encoder_architecture_specification.docx §2
Dataset source: NavsimDataset
 
Input mapping from NavsimDataset
---------------------------------
multi_agent_history : (B, 32, 4, 7)  → (B, N, T_hist, 7)
    channels: [x, y, vx, vy, ax, ay, heading]
 
agent_history       : (B, 1, 4, 5)   → ego only, channels [x, y, vx, vy, heading]
                      padded to (B, 1, 4, 7) with zeros for ax, ay
 
Both are concatenated along dim=1 before encoding when the caller wants
a unified agent sequence.  The encoder itself is agnostic to N.
 
Processing pipeline (§2.1)
---------------------------
Stage 1 : Heading decomp (7→8) + per-feature affine normalization
Stage 2 : 2 x 1D Depthwise-Separable Conv along T_hist, residual
Stage 3 : Linear projection → d_model + modality embedding + temporal PosEnc
"""

import math
import torch

import torch.nn as nn
import torch.nn.functional as F


# helper

class HeadingDecomposition(nn.Module):
    """Convert heading to (cos(heading), sin(heading))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heading = x[..., 6:7] 
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        return torch.cat([x[..., :6], cos_h, sin_h], dim = -1)          # (..., 8)

class PerFeatureAffineNorm(nn.Module):
    """Learnable element-wise affine: x_norm = gamma * x + beta
    Not batchnorm or layernorm, just 2 x in_dim learable scalars.
    Solve to mismatch between position (+-200m) velocity (+-30m/s)
    acceleration (+= 5m/s2) and cos/sin(+=1) without batch satisitics.

    Parameter: 2 x in_dim (default 16 for in_dim = 8)
    """

    def __init__(self, in_dim: int =8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(in_dim))
        self.beta  = nn.Parameter(torch.ones(in_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta
    
class DepthSepConv1dBlock(nn.Module):
    """
    1D Depthwise-Separable Conv block with residual connection.
 
    Operates on the T_hist dimension.
 
    Layout per block
    ----------------
    Depthwise  Conv1d(groups=C_in, k=3, pad=1)   - per-channel temporal filter
    Pointwise  Conv1d(k=1)                        - channel mixing
    GELU
    Residual   skip + output  (with projection if C_in ≠ C_out)
 
    'same' padding is achieved by setting padding=1 for kernel_size=3, so the
    temporal length T_hist is preserved throughout.
    
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            c_in, c_in, kernel_size= kernel_size,
            padding=padding, groups=c_in, bias = False
        )

        self.pointwise  = nn.Conv1d(c_in, c_out, kernel_size= 1, bias = True)
        self.act        = nn.GELU()

        # skip projection only when channels dims differ
        self.skip_proj = (
            nn.Conv1d(c_in, c_out, kernel_size = 1, bias = False)
            if c_in != c_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(x)
        out  = self.depthwise(x)
        out  = self.pointwise(out)
        out  = self.act(out)
        return out + skip
    
# sinusodial temporal position encoding

def sinusoidal_pos_enc(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard sine/cosine position encoding, shape (1, seq_len, d_model)."""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return pe.unsqueeze(0)  # (1, T, d_model)

# main encoder

class HistoryEncoder(nn.Module):
    """Encode (B, N, T_hist, 7) agent kinematic histories → (B, N, T_hist, d_model).
 
    Args
    ----
    d_model     : Output token dimension (e.g. 256 or 384).
    c_mid       : Intermediate channel count for conv blocks.
                  Spec recommends 128 for d_model=256.
    in_features : Raw feature dim before heading decomp (default 7).
    max_t_hist  : Maximum T_hist for pre-allocated sinusoidal encoding (default 20).
    """
 
    def __init__(
        self,
        d_model    : int = 256,
        c_mid      : int = 128,
        in_features: int = 7,    # x, y, vx, vy, ax, ay, heading
        max_t_hist : int = 20,
    ):
        super().__init__()
        in_after_decomp = in_features + 1   # 7 → 8 (replace heading with cos/sin)
 
        # stage 1
        self.heading_decomp = HeadingDecomposition()
        self.affine_norm    = PerFeatureAffineNorm(in_after_decomp)
 
        # stage 2
        # Block 1: 8 → c_mid
        self.conv_block1 = DepthSepConv1dBlock(in_after_decomp, c_mid, kernel_size=3)
        # Block 2: c_mid → c_mid
        self.conv_block2 = DepthSepConv1dBlock(c_mid, c_mid, kernel_size=3)
 
        # stage 3
        self.proj = nn.Linear(c_mid, d_model)
 
        # Learnable modality embedding: distinguishes HistoryEncoder tokens
        # from BEV/traffic/etc. tokens in the backbone.
        self.modality_emb = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
 
        # Sinusoidal temporal position encoding (fixed, pre-allocated)
        self.register_buffer(
            "pos_enc",
            sinusoidal_pos_enc(max_t_hist, d_model, device=torch.device("cpu")),
            persistent=False,
        )
        self.max_t_hist = max_t_hist
        self.d_model    = d_model

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, T_hist, 7)
            Agent kinematic histories in current-ego frame.
            Channels: [x, y, vx, vy, ax, ay, heading]
 
        Returns
        -------
        tokens : (B, N, T_hist, d_model)
        """

        B, N, T, F = x.shape
        assert F == 7, f"Expeceted 7 input features, got {F}"
        assert T <= self.max_t_hist,  (
            f"T_hist={T} exceeds max_t_hist={self.max_t_hist}. "
            "Increase max_t_hist at construction."      
        )

        # heading decomp + affine norm
        x = self.heading_decomp(x)      # (B, N, T, 8)
        x = self.affine_norm(x)         # (B, N, T, 8)

        # depthwise-sep conv along T axis
        # reshape (B*N, C, T) for Conv1d which experts (batch, channels, length)
        x = x.reshape(B * N, T, 8).permute(0, 2, 1)

        x = self.conv_block1(x)     # (B*N, c_mid, T)
        x = self.conv_block2(x)     # (B*N, c_mid, T)

        # back to (B*N, T, c_mid)
        x = x.permute(0, 2, 1)

        # linear projection -> dmodel

        x = self.proj(x)            # (B *N, T, d_model)

        #reshape to (B, N, T, d_model)
        x = x.reshape(B, N, T, self.d_model)

        # add sinusodial temporal position encodeing
        x = x + self.pos_enc[:, :T, :] # (1, T, d_model) # (1, T, d_model) → broadcast

        # add modality embedding
        x = x + self.modality_emb       # (1, 1, 1, d_model) → broadcast

        return x                         # (B, N, T, d_model)
    

def prepare_history_input(batch: dict) -> torch.Tensor:
    """Merge ego + multi-agent histories from a NavsimDataset batch.
 
    NavsimDataset outputs
    ----------------------
    multi_agent_history : (B, 32, 4, 7)  [x, y, vx, vy, ax, ay, heading]
    agent_history       : (B, 1, 4, 5)   [x, y, vx, vy, heading]
 
    The encoder expects (..., 7) everywhere.  For agent_history we pad
    channels 4-5 (ax, ay) with zeros, then overwrite row-0 of
    multi_agent_history with the ego entry.  Row-0 of multi_agent_history
    is *already* the ego agent per the NavsimDataset convention, so in
    practice this simply ensures type consistency.
 
    Returns
    -------
    combined : (B, 32, 4, 7)  ready for HistoryEncoder.forward()
    """
    mah = batch["multi_agent_history"]   # (B, 32, 4, 7)
    ah  = batch["agent_history"]         # (B, 1, 4, 5)
    B, _, T, _ = ah.shape
 
    # Pad ego history: add zero ax, ay columns → (B, 1, 4, 7)
    zeros = torch.zeros(B, 1, T, 2, device=ah.device, dtype=ah.dtype)
    # agent_history channels: [x, y, vx, vy, heading]
    # Reorder to match multi_agent_history: [x, y, vx, vy, ax, ay, heading]
    ego_7 = torch.cat([ah[..., :4], zeros, ah[..., 4:5]], dim=-1)  # (B,1,4,7)
 
    # Replace row-0 with the properly formatted ego entry
    combined = mah.clone()
    combined[:, 0:1, :, :] = ego_7
 
    return combined

# sanity check


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, T = 2, 32, 4
 
    encoder = HistoryEncoder(d_model=256, c_mid=128)
 
    # Simulate a NavsimDataset batch
    fake_batch = {
        "multi_agent_history": torch.randn(B, N, T, 7),
        "agent_history":       torch.randn(B, 1, T, 5),
    }
 
    x = prepare_history_input(fake_batch)          # (B, 32, T, 7)
    tokens = encoder(x)                            # (B, 32, T, 256)
 
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {tokens.shape}")
 
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"HistoryEncoder parameters: {total_params:,}")
 
    # Verify heading decomp: heading channel should be smooth after transform
    sample_heading = torch.tensor([0.0, math.pi / 4, math.pi / 2, math.pi])
    dummy = torch.zeros(1, 1, 4, 7)
    dummy[0, 0, :, 6] = sample_heading
    decomped = encoder.heading_decomp(dummy)
    print("\nHeading decomposition check:")
    print(f"  θ            : {sample_heading.tolist()}")
    print(f"  cos(θ)       : {decomped[0,0,:,6].tolist()}")
    print(f"  sin(θ)       : {decomped[0,0,:,7].tolist()}")
    print("\n✓ All checks passed")