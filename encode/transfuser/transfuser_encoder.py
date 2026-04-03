"""
TransDiffuser Scene Encoder
============================
Wraps TransfuserBackbone to produce the three feature streams the
Denoising Decoder needs, plus the two motion embeddings.
 
Requires ONE change to transfuser.py (see patch below).
Supports two-stage training via freeze_backbone() / unfreeze_backbone().
 
──────────────────────────────────────────────────────────────────────
REQUIRED PATCH TO transfuser.py
──────────────────────────────────────────────────────────────────────
In TransfuserBackbone.forward(), replace the last ~6 lines:
 
BEFORE
------
image_features = self.image_encoder.features.global_pool(image_features)
image_features = torch.flatten(image_features, 1)
lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
lidar_features = torch.flatten(lidar_features, 1)

fused_features = image_features + lidar_features

features = self.top_down(x4)
return features, image_features_grid, fused_features

AFTER
-----
image_features = self.image_encoder.features.global_pool(image_features)
image_features = torch.flatten(image_features, 1)
lidar_features = self.lidar_encoder._model.global_pool(lidar_features)
lidar_features = torch.flatten(lidar_features, 1)

features = self.top_down(x4)
return features, image_features_grid, image_features, lidar_features
 

That is the only edit. fused_features is no longer returned; callers that
need it can simply sum image_features + lidar_features themselves.
──────────────────────────────────────────────────────────────────────
 
Output dict keys
────────────────
  "Fbev"       (B, C_bev, H', W')   spatial BEV  — from FPN p2
  "Fimg"       (B, C_feat)          global image  — for cross-attention
  "FLiDAR"     (B, C_feat)          global LiDAR  — for cross-attention
  "Emb_action" (B, embed_dim)       historical action embedding
  "Emb_ego"    (B, embed_dim)       current ego-status embedding
 
Two-stage training
──────────────────
  Stage 1  joint:   encoder.unfreeze_backbone()   ← default at init
  Stage 2  frozen:  encoder.freeze_backbone()     ← call when ready
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


# minimum config that statisfy transfuser backbone 

class TransfuserConfig:
    """
    Field names replicate the backbone's exact (sometimes misspelled) attrs:
      bev_features_chanels, use_point_pillars, lidar_horz_anchors
    """
 
    img_vert_anchors:   int = 8
    img_horz_anchors:   int = 8
    lidar_vert_anchors: int = 8
    lidar_horz_anchors: int = 8
    seq_len:            int = 1
 
    n_head:      int   = 4
    block_exp:   int   = 4
    n_layer:     int   = 2
    embd_pdrop:  float = 0.1
    attn_pdrop:  float = 0.1
    resid_pdrop: float = 0.1
 
    perception_output_features: int = 512
    bev_features_chanels:       int = 256
    bev_upsample_factor:        int = 2
 
    use_point_pillars:      bool = False
    num_features:           list = None
    lidar_seq_len:          int  = 1
    use_target_point_image: bool = False
 
    gpt_linear_layer_init_mean:  float = 0.0
    gpt_linear_layer_init_std:   float = 0.02
    gpt_layer_norm_init_weight:  float = 1.0
 
    def __init__(self, **overrides):
        if self.num_features is None:
            self.num_features = [32, 64, 128, 256, 512]
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise ValueError(f"TransfuserConfig: unknown field '{k}'")
            setattr(self, k, v)

class SceneEncoder(nn.Module):
    """
    Args:
        backbone:       Patched TransfuserBackbone.
        num_cameras:    Number of cameras in the dataset (any positive int).
        action_dim:     Per-step action dim (default 2: Δx, Δy).
        num_history:    History steps for action encoder.
        ego_state_dim:  Ego state vector length.
        embed_dim:      Motion MLP output size.
        freeze_backbone: Start frozen (Stage 2) or not (Stage 1).
    """
 
    def __init__(
        self,
        backbone: nn.Module,
        num_cameras: int,
        action_dim: int = 2,
        num_history: int = 4,
        ego_state_dim: int = 5,
        embed_dim: int = 256,
        freeze_backbone: bool = False,
    ):
        super().__init__()
 
        self.backbone = backbone
        self.num_cameras = num_cameras
        self.embed_dim = embed_dim
        self.num_history = num_history
        self.action_dim = action_dim
 
        # Merge N cameras into backbone's expected [B, 3, H, W].
        # For N=1 this is essentially a passthrough (1×1 conv, 3→3).
        self.camera_merge = nn.Conv2d(num_cameras * 3, 3, kernel_size=1, bias=False)
 
        # Motion encoders (always trainable)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * num_history, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
 
        self.ego_status_encoder = nn.Sequential(
            nn.Linear(ego_state_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
 
        if freeze_backbone:
            self.freeze_backbone()
 

    # Freeze / unfreeze

 
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        print("[SceneEncoder] Backbone frozen — Stage 2 active")
 
    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        self.backbone.train()
        print("[SceneEncoder] Backbone unfrozen — Stage 1 active")
 
    def backbone_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())
 

    # Param groups

 
    def param_groups(self, backbone_lr: float = 1e-5, head_lr: float = 1e-4):
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.camera_merge.parameters())
            + list(self.action_encoder.parameters())
            + list(self.ego_status_encoder.parameters())
        )
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
 
    
    # Forward

 
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch dict keys (after DataLoader collation):
                camera_images   [B, N_cam, 3, H, W]  
                lidar_bev       [B, 2, H_l, W_l]
                agent_states    [B, 1, 5]             — [x, y, vx, vy, heading]
                agent_history   [B, 1, T_hist, 5]
        """
        # --- 1. Cameras [B, N, 3, H, W] → [B, 3, H, W] ---
        cam = batch["camera_images"]                       # [B, N, 3, H, W]
        B, N, C, H, W = cam.shape
        cam_flat = cam.view(B, N * C, H, W)                # [B, N*3, H, W]
        image = self.camera_merge(cam_flat)                 # [B, 3, H, W]
 
        # --- 2. LiDAR ---
        lidar = batch["lidar_bev"]                          # [B, 2, H_l, W_l]
 
        # --- 3. Velocity scalar [B, 1] ---
        agent_states = batch["agent_states"]                # [B, 1, 5]
        vx = agent_states[:, 0, 2]
        vy = agent_states[:, 0, 3]
        velocity = torch.sqrt(vx ** 2 + vy ** 2).unsqueeze(-1)
 
        # --- 4. Backbone ---
        if self.backbone_is_frozen():
            self.backbone.eval()
 
        fpn_features, _, Fimg, Flidar = self.backbone(image, lidar, velocity)
        Fbev = fpn_features[0]
 
        # --- 5. Action embedding (position deltas) ---
        history_xy = batch["agent_history"][:, 0, :, :2]   # [B, T_hist, 2]
        deltas = history_xy[:, 1:, :] - history_xy[:, :-1, :]
 
        T_delta = deltas.shape[1]
        if T_delta < self.num_history:
            pad = torch.zeros(
                B, self.num_history - T_delta, self.action_dim,
                device=deltas.device,
            )
            deltas = torch.cat([pad, deltas], dim=1)
        else:
            deltas = deltas[:, -self.num_history:]
 
        emb_action = self.action_encoder(deltas.reshape(B, -1))
 
        # --- 6. Ego status embedding ---
        emb_ego = self.ego_status_encoder(agent_states[:, 0, :])
 
        return {
            "bev":        Fbev,
            "img":        Fimg,
            "lidar":      Flidar,
            "emb_action": emb_action,
            "emb_ego":    emb_ego,
        }

# factory
def build_scene_encoder(
    image_architecture: str = "resnet34",
    lidar_architecture: str = "resnet18",
    num_cameras: int = 3,
    freeze_backbone: bool = False,
    embed_dim: int = 256,
    config_overrides: Optional[Dict] = None,
    checkpoint_path: Optional[str] = None,
) -> SceneEncoder:
    """
    Build a SceneEncoder.  num_cameras must match your dataset.
 
    Example
    -------
    encoder = build_scene_encoder(num_cameras=dataset.num_cameras)
    """
    from encode.transfuser.backbone import TransfuserBackbone
 
    cfg = TransfuserConfig(**(config_overrides or {}))
 
    backbone = TransfuserBackbone(
        config=cfg,
        image_architecture=image_architecture,
        lidar_architecture=lidar_architecture,
        use_velocity=True,
    )
 
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        state = state.get("state_dict", state)
        state = {k.removeprefix("backbone."): v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            print(f"[build] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[build] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"[build] Loaded checkpoint: {checkpoint_path}")
 
    return SceneEncoder(
        backbone=backbone,
        num_cameras=num_cameras,
        freeze_backbone=freeze_backbone,
        embed_dim=embed_dim,
    )
 
 

# Sanity test

 
if __name__ == "__main__":
    import sys
 
    B = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
    for N_CAM in [1, 3, 5, 8]:
        print("=" * 60)
        print(f"Sanity test: {N_CAM} camera(s)  (device: {DEVICE})")
        print("=" * 60)
 
        fake_batch = {
            "camera_images": torch.rand(B, N_CAM, 3, 224, 448, device=DEVICE),
            "lidar_bev":     torch.rand(B, 2, 200, 200, device=DEVICE),
            "agent_states":  torch.tensor([
                [[0.0, 0.0, 5.0, 0.1, 0.0]],
                [[0.0, 0.0, 3.0, 0.5, 0.0]],
            ], device=DEVICE),
            "agent_history": torch.rand(B, 1, 4, 5, device=DEVICE),
        }
 
        try:
            encoder = build_scene_encoder(
                num_cameras=N_CAM, freeze_backbone=False,
            ).to(DEVICE)
        except ImportError as e:
            print(f"  SKIP: {e}")
            sys.exit(0)
 
        encoder.train()
        out = encoder(fake_batch)
 
        print(f"  camera_merge weight: {tuple(encoder.camera_merge.weight.shape)}")
        for k, v in out.items():
            print(f"  {k:<14} {tuple(v.shape)}")
 
        loss = sum(v.mean() for v in out.values())
        loss.backward()
        encoder.zero_grad()
        print(f"  OK\n")
 
    print("All checks passed.")