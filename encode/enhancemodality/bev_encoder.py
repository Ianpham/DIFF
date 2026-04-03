"""
BEVEncoder
==========
Encodes the Bird's Eye View representation into a spatial token sequence
for the DyDiT backbone.
 
Spec source : encoder_architecture_specification.docx §3
Dataset source: NavsimDataset
 
NavsimDataset BEV inputs
------------------------
labels       : Dict[str -> Tensor(200, 200)]  - 12 HD map channels
lidar_bev    : (2, 200, 200)                  - ch0=density, ch1=max-height
camera_bev   : (C, 64, 64) or (C, 200, 200)  - UniAD BEV features (C≈256)
 
Stacking into the 12-channel spec input
----------------------------------------
The spec §3 defines 12 BEV channels split into 4 semantic groups.
NavsimDataset supplies them via `labels` dict.  This file provides a
`stack_bev_labels()` helper that assembles the canonical 12-channel
tensor in the correct order.
 
The optional camera_bev and lidar_bev are fused *after* the grouped stems
via a lightweight fusion head (see §3.3 and the FusionHead class below).
When not present they are simply skipped.
 
Architecture (§3)
-----------------
Group A  Topology  (4 ch)  -> 2-layer CNN stem -> 64 ch
Group B  Occupancy (5 ch)  -> 2-layer CNN stem -> 64 ch   (3x3 then 5x5)
Group C  Dynamics  (2 ch)  -> 2-layer dilated CNN stem -> 64 ch  (dil=2)
Group D  Signals   (1 ch)  -> 1-layer CNN stem -> 32 ch
 
Concatenate -> (224, H, W)
-> FusionConv2d (224 + optional_extras -> C_fuse, k=3)
-> PatchTokenizer Conv2d (C_fuse -> d_model, k=patch_size, s=patch_size)
-> Modality Embedding + 2D Sinusoidal Position Encoding
-> (B, N_patches, d_model)
"""

import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# label keys from Navsim

GROUP_A_KEYS = [
    "drivable_area",
    "lane_boundaries",
    "lane_dividers",
    "stop_lines",
]  # 4 channels — binary topology
 
GROUP_B_KEYS = [
    "vehicle_occupancy",
    "pedestrian_occupancy",
    "crosswalks",
    "ego_mask",
    "vehicle_classes",
]  # 5 channels — occupancy / region
 
GROUP_C_KEYS = [
    "velocity_x",
    "velocity_y",
]  # 2 channels — continuous velocity fields
 
GROUP_D_KEYS = [
    "traffic_lights",
]  # 1 channel — sparse signals

ALL_LABEL_KEYS: List[str] = GROUP_A_KEYS + GROUP_B_KEYS + GROUP_C_KEYS + GROUP_D_KEYS

def stack_bev_labels(
    labels: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Assemble the canonical 12-channel BEV tensor from the labels dict.
 
    Parameters
    ----------
    labels : dict mapping NavsimDataset label key -> Tensor(H, W)  or  (B, H, W)
    device : optional target device
 
    Returns
    -------
    bev : (12, H, W)  or  (B, 12, H, W) - float32
    """
    tensors = []
    for key in ALL_LABEL_KEYS:
        if key not in labels:
            raise KeyError(
                f"BEV label '{key}' not found in labels dict. "
                f"Expected keys: {ALL_LABEL_KEYS}"
            )
        t = labels[key]
        if device is not None:
            t = t.to(device)
        t = t.float()
        # Normalise to (B, 1, H, W)
        if t.dim() == 2:          # (H, W) -> (1, 1, H, W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:        # (B, H, W) -> (B, 1, H, W)
            t = t.unsqueeze(1)
        # t is now (B, 1, H, W)
        tensors.append(t)
 
    bev = torch.cat(tensors, dim=1)  # (B, 12, H, W)
    return bev

# per group CNN stems
def _make_bn_gelu(c: int) -> nn.Sequential:
    return nn.Sequential(nn.BatchNorm2d(c), nn.GELU())

class TopologyStem(nn.Module):
    """
    Group A - binary topology channels -> 64 channels

    two 3x3 convolutions preservng spatial reosultion (no downsampling)

    sharp edges, fine lane- level detail. BatchNorm is safe here cause binary mask have stable statistics.

    """

    def __init__(self, in_channels: int = 4, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size= 3, stride = 1, padding = 1, bias = False),
            _make_bn_gelu(32),
            nn.Conv2d(32, out_channels, kernel_size= 3, stride = 1, padding = 1, bias = False),
            _make_bn_gelu(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class OccupancyStem(nn.Module):
    """
    Group B — 5 occupancy channels -> 64 channels.
 
    First conv 3x3, second conv 5x5 to capture blob-like region extent.
    y masks have stable statistics.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding= 1, bias=False),
            _make_bn_gelu(32),
            nn.Conv2d(32, out_channels, kernel_size=5, stride = 1, padding=2, bias =False),
            _make_bn_gelu(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class DynamicsStem(nn.Module):
    """Group C — 2 velocity-field channels -> 64 channels.
 
    Both convolutions use dilation=2 (padding=2 for 3x3 dilated, same output size).
    Effective RF after 2 layers = 9x9 pixels, capturing broad flow structure.

    """

    def __init__(self, in_channels: int = 2, out_channels: int = 64):
        super().__init__()

        # dilation = 2, k = 3 -> padding = dilation * (k - 1)/ 2 = 2

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride = 1, padding=2, dilation= 2, bias=False ),
            _make_bn_gelu(32),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=2, dilation= 2, bias= False),
            _make_bn_gelu(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class SignalsStem(nn.Module):
    """
    Group D: traffic light channels -> 32 channels
    Single conv layer: sparse point-like activations need minimal processing.
    No BatchNorm — sparse inputs with a single channel can have degenerate
    batch statistics; GELU only.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) 



# task here we add sceneEncoder here, make it options for.
# but it is just adapter
class CameraBEVAdapter(nn.Module):
    """Lightweight adapter that projects UniAD camera BEV features into C_fuse space.
 
    NavsimDataset: camera_bev is (C, 64, 64) or (C, 200, 200).
    When interpolate_bev=False it stays at (C, 64, 64) and needs upsampling to HxW.
    When interpolate_bev=True it's already (C, 200, 200).
 
    The adapter bilinearly upsamples to (H, W), then uses a 1x1 conv to
    project C -> c_camera_out channels for fusion concatenation.
    """
    def __init__(self, c_in: int, c_out: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.GELU()
 
    def forward(self, camera_bev: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        H, W = target_hw
        if camera_bev.shape[-2:] != (H, W):
            camera_bev = F.interpolate(
                camera_bev, size=(H, W), mode="bilinear", align_corners=False
            )
        return self.act(self.bn(self.proj(camera_bev)))
 
 
class LidarBEVAdapter(nn.Module):
    """Project 2-channel LiDAR BEV (density + max-height) -> c_lidar_out channels."""
    def __init__(self, c_in: int = 2, c_out: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.GELU()
 
    def forward(self, lidar_bev: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.proj(lidar_bev)))
 
# 2D sinusodial position encoding
def sinusoidal_2d_pos_enc(
    h_patches: int, w_patches: int, d_model: int, device: torch.device
) -> torch.Tensor:
    """2D sinusoidal position encoding.
 
    Generates independent sin/cos encodings for row and column indices,
    each using d_model/2 dimensions, then concatenates.
 
    Returns
    -------
    pe : (1, h_patches * w_patches, d_model)
    """
    assert d_model % 2 == 0, "d_model must be even for 2D sinusoidal encoding"
    d_half = d_model // 2
 
    def _1d_enc(length: int, d: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / d)
        )
        pe = torch.zeros(length, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d // 2])
        return pe  # (length, d)
 
    row_enc = _1d_enc(h_patches, d_half)   # (H, d/2)
    col_enc = _1d_enc(w_patches, d_half)   # (W, d/2)
 
    # Broadcast: each patch (i, j) gets cat(row_enc[i], col_enc[j])
    row_enc = row_enc.unsqueeze(1).expand(-1, w_patches, -1)  # (H, W, d/2)
    col_enc = col_enc.unsqueeze(0).expand(h_patches, -1, -1)  # (H, W, d/2)
 
    pe_2d = torch.cat([row_enc, col_enc], dim=-1)             # (H, W, d_model)
    pe_2d = pe_2d.reshape(1, h_patches * w_patches, d_model)  # (1, N_patches, d_model)
    return pe_2d

class BEVEncoder(nn.Module):
    """Encode multi-channel BEV representation -> spatial token sequence.
 
    Parameters
    ----------
    d_model        : Output token dimension (e.g. 256).
    c_fuse         : Fusion channel count after concatenating group stems.
                     Spec recommends 128-192.  Default 160.
    patch_size     : Spatial patch size for tokenization.
                     8x8 for 200x200 BEV (-> 25x25 = 625 tokens).
                     4x4 for lower-res BEV (-> 50x50 = 2500 tokens).
    camera_c_in    : UniAD camera BEV channel count (detected from cache,
                     e.g. 256).  None -> camera BEV not used.
    camera_c_adapt : Channels after camera adapter projection.  Default 64.
    use_lidar      : Whether to fuse LiDAR BEV (2 ch).  Default True.
    lidar_c_adapt  : Channels after LiDAR adapter projection.  Default 32.
    bev_h, bev_w   : Expected BEV spatial size.  Used for pre-allocating
                     sinusoidal position encoding.  Default 200x200.
    """
 
    def __init__(
        self,
        d_model        : int = 256,
        c_fuse         : int = 160,
        patch_size     : int = 8,
        camera_c_in    : Optional[int] = 256,
        camera_c_adapt : int = 64,
        use_lidar      : bool = True,
        lidar_c_adapt  : int = 32,
        bev_h          : int = 200,
        bev_w          : int = 200,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model    = d_model
 
        # ---------- Group stems ----------
        self.stem_a = TopologyStem( in_channels=4, out_channels=64)   # Group A
        self.stem_b = OccupancyStem(in_channels=5, out_channels=64)   # Group B
        self.stem_c = DynamicsStem( in_channels=2, out_channels=64)   # Group C
        self.stem_d = SignalsStem( in_channels=1, out_channels=32)   # Group D
 
        # Concatenated stem channels: 64+64+64+32 = 224
        stem_total = 64 + 64 + 64 + 32  # 224
 
        # ---------- Optional modality adapters ----------
        self.camera_adapter: Optional[CameraBEVAdapter] = None
        camera_contrib = 0
        if camera_c_in is not None:
            self.camera_adapter = CameraBEVAdapter(c_in=camera_c_in, c_out=camera_c_adapt)
            camera_contrib = camera_c_adapt
 
        self.lidar_adapter: Optional[LidarBEVAdapter] = None
        lidar_contrib = 0
        if use_lidar:
            self.lidar_adapter = LidarBEVAdapter(c_in=2, c_out=lidar_c_adapt)
            lidar_contrib = lidar_c_adapt
 
        fusion_in = stem_total + camera_contrib + lidar_contrib  # e.g. 224+64+32=320
 
        # ---------- Fusion conv ----------
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, c_fuse, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_fuse),
            nn.GELU(),
        )
 
        # ---------- Patch tokenization ----------
        # Non-overlapping conv stride=patch_size collapses each patch into one token
        self.patch_tokenizer = nn.Conv2d(
            c_fuse, d_model, kernel_size=patch_size, stride=patch_size, bias=True
        )
 
        # ---------- Modality embedding ----------
        self.modality_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
 
        # ---------- 2D sinusoidal position encoding (pre-allocated) ----------
        assert bev_h % patch_size == 0 and bev_w % patch_size == 0, (
            f"BEV size ({bev_h}x{bev_w}) must be divisible by patch_size={patch_size}"
        )
        h_p = bev_h // patch_size
        w_p = bev_w // patch_size
        self.register_buffer(
            "pos_enc",
            sinusoidal_2d_pos_enc(h_p, w_p, d_model, device=torch.device("cpu")),
            persistent=False,
        )
        self.bev_h = bev_h
        self.bev_w = bev_w
 
    # ------------------------------------------------------------------
    def forward(
        self,
        bev_labels  : torch.Tensor,
        camera_bev  : Optional[torch.Tensor] = None,
        lidar_bev   : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        bev_labels : (B, 12, H, W)
            12-channel BEV assembled by stack_bev_labels().
            Groups A/B/C/D occupy the first 4/5/2/1 channels respectively.
        camera_bev : (B, C_cam, H_cam, W_cam)  or None
            UniAD BEV features.  Upsampled to (H, W) if needed.
        lidar_bev  : (B, 2, H, W)  or None
            LiDAR density + max-height channels.
 
        Returns
        -------
        tokens : (B, N_patches, d_model)
            N_patches = (H // patch_size) x (W // patch_size)
        """
        B, _, H, W = bev_labels.shape
 
        # ---------- Split into groups ----------
        grp_a = bev_labels[:, 0:4,   :, :]   # topology   (B,4,H,W)
        grp_b = bev_labels[:, 4:9,   :, :]   # occupancy  (B,5,H,W)
        grp_c = bev_labels[:, 9:11,  :, :]   # dynamics   (B,2,H,W)
        grp_d = bev_labels[:, 11:12, :, :]   # signals    (B,1,H,W)
 
        # ---------- Per-group stems ----------
        feat_a = self.stem_a(grp_a)   # (B,64,H,W)
        feat_b = self.stem_b(grp_b)   # (B,64,H,W)
        feat_c = self.stem_c(grp_c)   # (B,64,H,W)
        feat_d = self.stem_d(grp_d)   # (B,32,H,W)
 
        feats = [feat_a, feat_b, feat_c, feat_d]
 
        # ---------- Optional modality adapters ----------
        if camera_bev is not None and self.camera_adapter is not None:
            feats.append(self.camera_adapter(camera_bev, (H, W)))  # (B,64,H,W)
 
        if lidar_bev is not None and self.lidar_adapter is not None:
            feats.append(self.lidar_adapter(lidar_bev))            # (B,32,H,W)
 
        # ---------- Concatenate + fusion ----------
        x = torch.cat(feats, dim=1)   # (B, fusion_in, H, W)
        x = self.fusion(x)            # (B, C_fuse, H, W)
 
        # ---------- Patch tokenization ----------
        x = self.patch_tokenizer(x)   # (B, d_model, H/p, W/p)
 
        # Flatten spatial dims -> token sequence
        B2, D, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1)    # (B, Hp, Wp, d_model)
        x = x.reshape(B2, Hp * Wp, D) # (B, N_patches, d_model)
 
        # ---------- Position + modality embeddings ----------
        # Recompute pos_enc if BEV size changed (flexible inference)
        if self.pos_enc.shape[1] != Hp * Wp:
            pos = sinusoidal_2d_pos_enc(Hp, Wp, D, device=x.device)
        else:
            pos = self.pos_enc
 
        x = x + pos                  # (B, N_patches, d_model)
        x = x + self.modality_emb    # broadcast (1, 1, d_model)
 
        return x                     # (B, N_patches, d_model)
 
 

# NavsimDataset batch adapter

 
def prepare_bev_input(
    batch      : dict,
    encoder    : BEVEncoder,
    device     : Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract and assemble BEV inputs from a NavsimDataset batch.
 
    Returns
    -------
    bev_labels : (B, 12, H, W)
    camera_bev : (B, C, H_cam, W_cam)  or None
    lidar_bev  : (B, 2, H, W)          or None
    """
    # labels dict may hold batched (B,H,W) or unbatched (H,W) tensors
    labels = batch["labels"]
    bev_labels = stack_bev_labels(labels, device=device)
 
    # NavsimDataset may return (H,W) per-sample without batch dim from DataLoader
    # DataLoader with default collate will give (B,H,W); add channel dim above handles it.
    # If collated, shape is already (B,12,H,W).
 
    camera_bev = batch.get("camera_bev", None)
    lidar_bev  = batch.get("lidar_bev",  None)
 
    if device is not None:
        if camera_bev is not None:
            camera_bev = camera_bev.to(device)
        if lidar_bev is not None:
            lidar_bev = lidar_bev.to(device)
 
    return bev_labels, camera_bev, lidar_bev
 
 

# Sanity check

 
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 200, 200
    C_cam   = 256
 
    # --- Build fake NavsimDataset batch ---
    label_keys = ALL_LABEL_KEYS
    fake_labels = {k: torch.randn(B, H, W) for k in label_keys}
 
    # camera_bev at UniAD native resolution (64x64), lidar at full (200x200)
    fake_camera_bev = torch.randn(B, C_cam, 64, 64)
    fake_lidar_bev  = torch.randn(B, 2, H, W)
 
    # --- Assemble 12-channel BEV tensor ---
    bev_labels = stack_bev_labels(fake_labels)
    print(f"Stacked BEV labels shape  : {bev_labels.shape}")  # (B,12,200,200)
 
    # --- Encoder ---
    encoder = BEVEncoder(
        d_model     = 256,
        c_fuse      = 160,
        patch_size  = 8,
        camera_c_in = C_cam,
        use_lidar   = True,
        bev_h       = H,
        bev_w       = W,
    )
 
    tokens = encoder(bev_labels, camera_bev=fake_camera_bev, lidar_bev=fake_lidar_bev)
    print(f"Output tokens shape       : {tokens.shape}")  # (B, 625, 256)
 
    n_patches = (H // 8) * (W // 8)
    print(f"Expected N_patches (8x8)  : {n_patches}")     # 625
 
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"BEVEncoder parameters     : {total_params:,}")
 
    # --- Without optional modalities ---
    encoder_base = BEVEncoder(
        d_model     = 256,
        c_fuse      = 160,
        patch_size  = 8,
        camera_c_in = None,
        use_lidar   = False,
        bev_h       = H,
        bev_w       = W,
    )
    tokens_base = encoder_base(bev_labels)
    print(f"\nWithout camera/lidar      : {tokens_base.shape}")
    base_params = sum(p.numel() for p in encoder_base.parameters())
    print(f"Base encoder parameters   : {base_params:,}")


