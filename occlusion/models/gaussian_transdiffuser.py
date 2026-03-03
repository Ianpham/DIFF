"""
GaussianTransDiffuser: Full Pipeline (Strategy C — Hybrid).

Combines:
1. TransFuser backbone (frozen) for proven scene features
2. GaussianFormer3D for occlusion-aware 3D scene understanding
3. TransDiffuser denoising decoder for multi-mode trajectory generation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoders.gaussian_lifter import VoxelToGaussianLifter
from .encoders.gaussian_encoder import GaussianOccEncoder
from .decoders.gaussian_splatting import GaussianSplattingDecoder
from .decoders.diffusion_decoder import (
    DiffusionTrajectoryDecoder,
    MotionEncoder,
)
from .losses.occ_loss import CombinedLoss
from utils.depth_map import build_depth_maps_batch, create_multiscale_depth_maps
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.pooling import build_gaussian_pooling


# ============================================================
# Placeholder backbones (replace with actual implementations)
# ============================================================

class TransFuserBackbone(nn.Module):
    """Frozen TransFuser backbone placeholder.

    Replace with actual TransFuser from:
    https://github.com/autonomousvision/navsim

    Outputs: F_img (B,16,D), F_LiDAR (B,16,D), F_bev (B,25,D)
    """

    def __init__(self, dim: int = 512, frozen: bool = True):
        super().__init__()
        self.dim = dim
        # Minimal placeholders — swap for real TransFuser layers
        self.img_proj = nn.Linear(3 * 7 * 7, dim)
        self.lid_proj = nn.Linear(5, dim)
        self.bev_proj = nn.Linear(dim, dim)
        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, images, points):
        B = images.shape[0]
        device = images.device
        f_img = torch.randn(B, 16, self.dim, device=device)
        f_lidar = torch.randn(B, 16, self.dim, device=device)
        f_bev = torch.randn(B, 25, self.dim, device=device)
        return {"f_img": f_img, "f_lidar": f_lidar, "f_bev": f_bev}


class ImageBackbone(nn.Module):
    """Multi-scale image feature extractor for GaussianFormer3D.

    Placeholder: replace with ResNet101-DCN + FPN from mmdet3d.
    """

    def __init__(self, embed_dims: int = 128, num_levels: int = 4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.fpn = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, embed_dims, 1), nn.BatchNorm2d(embed_dims), nn.ReLU(True))
            for _ in range(num_levels)
        ])

    def forward(self, images: torch.Tensor) -> list:
        """
        Args:
            images: (B, V, 3, H, W).
        Returns:
            list of L tensors, each (B, V, C, H_l, W_l).
        """
        B, V, C, H, W = images.shape
        x = self.base(images.reshape(B * V, C, H, W))
        feats = []
        for i, fpn in enumerate(self.fpn):
            if i > 0:
                x = nn.functional.avg_pool2d(x, 2)
            f = fpn(x)
            _, Co, Hl, Wl = f.shape
            feats.append(f.reshape(B, V, Co, Hl, Wl))
        return feats


# ============================================================
# Main Model
# ============================================================

class GaussianTransDiffuser(nn.Module):
    """Full GaussianFormer3D × TransDiffuser pipeline (Strategy C).

    Phase 1: GaussianFormer3D + occ loss only       → call forward_phase1()
    Phase 2: frozen Gaussians + diffusion planning   → call forward_phase2()
    Phase 3: joint fine-tuning, all losses           → call forward_phase3()
    Inference: generate trajectory candidates         → call forward_inference()
    """

    def __init__(
        self,
        # Gaussian config
        num_gaussians: int = 25600,
        embed_dims: int = 128,
        num_classes: int = 17,
        num_encoder_blocks: int = 6,
        num_gaussian_tokens: int = 256,
        pooling_method: str = "fps",
        # Decoder config
        decoder_embed_dim: int = 256,
        decoder_num_heads: int = 8,
        decoder_num_layers: int = 6,
        decoder_ff_dim: int = 1024,
        trajectory_length: int = 8,
        action_dim: int = 2,
        num_diffusion_steps: int = 10,
        num_candidates: int = 30,
        # TransFuser
        transfuser_dim: int = 512,
        # Scene geometry
        point_cloud_range: list = None,
        occ_size: list = None,
        # Loss
        loss_weights: dict = None,
    ):
        super().__init__()

        if point_cloud_range is None:
            point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        if occ_size is None:
            occ_size = [200, 200, 16]

        self.num_gaussians = num_gaussians
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.decoder_embed_dim = decoder_embed_dim

        # ---- GaussianFormer3D branch ----
        self.img_backbone = ImageBackbone(embed_dims=embed_dims)

        self.gaussian_lifter = VoxelToGaussianLifter(
            num_gaussians=num_gaussians,
            embed_dims=embed_dims,
            point_cloud_range=point_cloud_range,
            num_classes=num_classes,
        )

        self.gaussian_encoder = GaussianOccEncoder(
            num_blocks=num_encoder_blocks,
            embed_dims=embed_dims,
            num_classes=num_classes,
        )

        self.gaussian_splatting = GaussianSplattingDecoder(
            occ_size=occ_size,
            point_cloud_range=point_cloud_range,
            num_classes=num_classes,
            embed_dims=embed_dims,
        )

        # ---- Gaussian → Decoder token pooling ----
        self.gaussian_pooling = build_gaussian_pooling(
            method=pooling_method,
            num_tokens=num_gaussian_tokens,
            in_dims=embed_dims,
            out_dims=decoder_embed_dim,
        )

        # ---- TransFuser backbone (frozen) ----
        self.transfuser = TransFuserBackbone(dim=transfuser_dim, frozen=True)

        # Projections: TransFuser features → decoder dimension
        self.proj_img = nn.Linear(transfuser_dim, decoder_embed_dim)
        self.proj_lidar = nn.Linear(transfuser_dim, decoder_embed_dim)
        self.proj_bev = nn.Linear(transfuser_dim, decoder_embed_dim)

        # ---- Motion encoders ----
        self.motion_encoder = MotionEncoder(
            trajectory_length=trajectory_length,
            action_dim=action_dim,
            embed_dim=decoder_embed_dim,
        )

        # ---- Diffusion trajectory decoder ----
        # 5 cross-attention targets: bev, img, lidar, gauss, motion(action+ego concat)
        self.diffusion_decoder = DiffusionTrajectoryDecoder(
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            ff_dim=decoder_ff_dim,
            trajectory_length=trajectory_length,
            action_dim=action_dim,
            num_diffusion_steps=num_diffusion_steps,
            num_candidates=num_candidates,
            num_cross_attn_targets=5,
        )

        # ---- Loss ----
        if loss_weights is None:
            loss_weights = dict(
                occ_ce=1.0, occ_lovasz=1.0,
                diffusion=1.0, decorrelation=0.02,
            )
        self.loss_fn = CombinedLoss(
            num_classes=num_classes,
            loss_weights=loss_weights,
        )

    
    # Shared sub-forward helpers
    

    def _run_gaussian_branch(
        self, batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run GaussianFormer3D: lifter → encoder → splatting.

        Returns:
            gaussian_feats: (B, N, D) refined Gaussian features.
            gaussian_props: (B, N, prop_dim) refined properties.
            occ_pred: (B, X, Y, Z, C) occupancy logits.
        """
        images = batch["images"]          # (B, V, 3, H, W)
        points = batch["points"]          # list of (N_i, 5)
        lidar2img = batch["lidar2img"]    # (B, V, 4, 4)
        img_shape = tuple(batch["img_shape"][0].tolist())

        # Image features (multi-scale)
        ms_feats = self.img_backbone(images)

        # V2G initialization from LiDAR
        gaussian_props, gaussian_feats = self.gaussian_lifter(points)

        # Depth maps placeholder (in production: project LiDAR to camera depth maps)
        # depth_maps = torch.zeros(
        #     images.shape[0], images.shape[1], 1,
        #     ms_feats[0].shape[-2], ms_feats[0].shape[-1],
        #     device=images.device,
        # )
        depth_maps_full = build_depth_maps_batch(
        points, lidar2img, img_shape,
        num_bins=64, depth_range=(1.0, 60.0), mode="hard",
        )
        fpn_shapes = [(f.shape[-2], f.shape[-1]) for f in ms_feats]
        depth_maps = create_multiscale_depth_maps(depth_maps_full, fpn_shapes)
        # Iterative refinement
        gaussian_feats, gaussian_props, intermediates = self.gaussian_encoder(
            gaussian_feats, gaussian_props,
            ms_feats, depth_maps, lidar2img, img_shape,
        )

        # Gaussian → Voxel splatting for occupancy
        occ_pred = self.gaussian_splatting(gaussian_props, gaussian_feats)

        return gaussian_feats, gaussian_props, occ_pred

    def _run_transfuser(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run frozen TransFuser and project features."""
        with torch.no_grad():
            tf_out = self.transfuser(batch["images"], batch["points"])

        return {
            "f_bev": self.proj_bev(tf_out["f_bev"]),
            "f_img": self.proj_img(tf_out["f_img"]),
            "f_lidar": self.proj_lidar(tf_out["f_lidar"]),
        }

    def _build_conditional_features(
        self,
        tf_feats: Dict[str, torch.Tensor],
        gaussian_tokens: torch.Tensor,
        motion_tokens: torch.Tensor,
    ) -> list:
        """Assemble the 5 conditional feature groups for the decoder.

        Order must match cross-attention layers:
        [f_bev, f_img, f_lidar, f_gauss, motion]
        """
        return [
            tf_feats["f_bev"],     # (B, N_bev, D)
            tf_feats["f_img"],     # (B, N_img, D)
            tf_feats["f_lidar"],   # (B, N_lid, D)
            gaussian_tokens,       # (B, K, D)
            motion_tokens,         # (B, 2, D)   action + ego concatenated
        ]

    
    # Phase-specific forward passes
    

    def forward_phase1(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Phase 1: Occupancy-only training.

        Train GaussianFormer3D on OpenScene occupancy labels.
        No planning, no TransFuser.
        """
        gaussian_feats, gaussian_props, occ_pred = self._run_gaussian_branch(batch)

        losses = self.loss_fn(
            occ_pred=occ_pred,
            occ_target=batch["occ_label"],
        )

        return {
            "losses": losses,
            "occ_pred": occ_pred,
        }

    def forward_phase2(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Phase 2: Frozen Gaussians + diffusion planning.

        GaussianFormer3D is frozen. Only the decoder + projections train.
        """
        # Gaussian branch (frozen)
        with torch.no_grad():
            gaussian_feats, gaussian_props, occ_pred = self._run_gaussian_branch(batch)

        # Pool Gaussians → decoder tokens
        gaussian_means = gaussian_props[..., :3]
        gaussian_tokens = self.gaussian_pooling(gaussian_feats, gaussian_means)

        # TransFuser branch (frozen)
        tf_feats = self._run_transfuser(batch)

        # Motion encoding
        emb_action, emb_ego = self.motion_encoder(
            batch["history_trajectory"], batch["ego_status"],
        )
        motion_tokens = torch.cat([emb_action, emb_ego], dim=1)  # (B, 2, D)

        # Conditional features
        cond_feats = self._build_conditional_features(
            tf_feats, gaussian_tokens, motion_tokens,
        )

        # Diffusion decoder training
        diff_out = self.diffusion_decoder.forward_train(
            batch["gt_actions"], cond_feats,
        )

        losses = self.loss_fn(
            noise_pred=diff_out["noise_pred"],
            noise_target=diff_out["noise_target"],
            fused_repr=diff_out["fused_repr"],
        )

        return {
            "losses": losses,
            "noise_pred": diff_out["noise_pred"],
        }

    def forward_phase3(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Phase 3: Joint fine-tuning with all losses.

        Both GaussianFormer3D and decoder are trainable.
        Total loss: L_diff + β·L_rep + λ_ce·L_ce + λ_lov·L_lovász
        """
        # Gaussian branch (trainable)
        gaussian_feats, gaussian_props, occ_pred = self._run_gaussian_branch(batch)

        # Pool → tokens
        gaussian_means = gaussian_props[..., :3]
        gaussian_tokens = self.gaussian_pooling(gaussian_feats, gaussian_means)

        # TransFuser (frozen)
        tf_feats = self._run_transfuser(batch)

        # Motion
        emb_action, emb_ego = self.motion_encoder(
            batch["history_trajectory"], batch["ego_status"],
        )
        motion_tokens = torch.cat([emb_action, emb_ego], dim=1)

        # Conditional features
        cond_feats = self._build_conditional_features(
            tf_feats, gaussian_tokens, motion_tokens,
        )

        # Diffusion training
        diff_out = self.diffusion_decoder.forward_train(
            batch["gt_actions"], cond_feats,
        )

        # Combined loss: occupancy + diffusion + decorrelation
        losses = self.loss_fn(
            occ_pred=occ_pred,
            occ_target=batch["occ_label"],
            noise_pred=diff_out["noise_pred"],
            noise_target=diff_out["noise_target"],
            fused_repr=diff_out["fused_repr"],
        )

        return {
            "losses": losses,
            "occ_pred": occ_pred,
            "noise_pred": diff_out["noise_pred"],
        }

    @torch.no_grad()
    def forward_inference(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Inference: generate trajectory candidates.

        Returns top-K waypoint trajectories for PDMS evaluation.
        """
        self.eval()
        B = batch["images"].shape[0]

        # Gaussian branch
        gaussian_feats, gaussian_props, occ_pred = self._run_gaussian_branch(batch)
        gaussian_means = gaussian_props[..., :3]
        gaussian_tokens = self.gaussian_pooling(gaussian_feats, gaussian_means)

        # TransFuser
        tf_feats = self._run_transfuser(batch)

        # Motion
        emb_action, emb_ego = self.motion_encoder(
            batch["history_trajectory"], batch["ego_status"],
        )
        motion_tokens = torch.cat([emb_action, emb_ego], dim=1)

        # Conditional features
        cond_feats = self._build_conditional_features(
            tf_feats, gaussian_tokens, motion_tokens,
        )

        # Generate candidates
        trajectories = self.diffusion_decoder.forward_inference(cond_feats, B)
        # trajectories: (B, K, T, 2) waypoint candidates

        # Rejection sampling: select top-1 (placeholder — implement PDMS scoring)
        top1 = self._rejection_sampling(trajectories)

        return {
            "trajectories": trajectories,   # (B, K, T, 2) all candidates
            "top1_trajectory": top1,         # (B, T, 2) selected
            "occ_pred": occ_pred,
        }

    def _rejection_sampling(
        self, trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """Select best trajectory from candidates.

        Placeholder: picks the one closest to mean (least extreme).
        Replace with proper kinematic feasibility + scoring.
        """
        # (B, K, T, 2) → mean trajectory
        mean_traj = trajectories.mean(dim=1, keepdim=True)  # (B, 1, T, 2)
        dists = ((trajectories - mean_traj) ** 2).sum(dim=(-1, -2))  # (B, K)
        best_idx = dists.argmin(dim=1)  # (B,)

        B, K, T, D = trajectories.shape
        best = trajectories[torch.arange(B, device=trajectories.device), best_idx]
        return best  # (B, T, 2)

    
    # Utilities
    

    def freeze_gaussian_branch(self):
        """Freeze GaussianFormer3D for Phase 2."""
        for module in [self.img_backbone, self.gaussian_lifter,
                       self.gaussian_encoder, self.gaussian_splatting]:
            for param in module.parameters():
                param.requires_grad = False
        print("[Phase 2] Froze GaussianFormer3D branch.")

    def unfreeze_gaussian_branch(self):
        """Unfreeze GaussianFormer3D for Phase 3."""
        for module in [self.img_backbone, self.gaussian_lifter,
                       self.gaussian_encoder, self.gaussian_splatting]:
            for param in module.parameters():
                param.requires_grad = True
        print("[Phase 3] Unfroze GaussianFormer3D branch.")

    def get_param_groups(self, phase: int = 3) -> list:
        """Get optimizer parameter groups with per-component LR control.

        Args:
            phase: training phase (1, 2, or 3).
        """
        groups = []

        if phase in (1, 3):
            # GaussianFormer3D components
            gauss_params = list(self.img_backbone.parameters()) + \
                           list(self.gaussian_lifter.parameters()) + \
                           list(self.gaussian_encoder.parameters()) + \
                           list(self.gaussian_splatting.parameters())
            gauss_params = [p for p in gauss_params if p.requires_grad]
            if gauss_params:
                groups.append({
                    "params": gauss_params,
                    "lr_scale": 0.1 if phase == 3 else 1.0,
                    "name": "gaussian_branch",
                })

        if phase in (2, 3):
            # Decoder + projections
            decoder_params = list(self.diffusion_decoder.parameters()) + \
                             list(self.gaussian_pooling.parameters()) + \
                             list(self.proj_bev.parameters()) + \
                             list(self.proj_img.parameters()) + \
                             list(self.proj_lidar.parameters()) + \
                             list(self.motion_encoder.parameters())
            decoder_params = [p for p in decoder_params if p.requires_grad]
            if decoder_params:
                groups.append({
                    "params": decoder_params,
                    "lr_scale": 1.0,
                    "name": "decoder_branch",
                })

        return groups

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        components = {
            "img_backbone": self.img_backbone,
            "gaussian_lifter": self.gaussian_lifter,
            "gaussian_encoder": self.gaussian_encoder,
            "gaussian_splatting": self.gaussian_splatting,
            "gaussian_pooling": self.gaussian_pooling,
            "transfuser": self.transfuser,
            "proj_layers": nn.ModuleList([self.proj_bev, self.proj_img, self.proj_lidar]),
            "motion_encoder": self.motion_encoder,
            "diffusion_decoder": self.diffusion_decoder,
        }
        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": n, "trainable": trainable}
            total += n
        counts["TOTAL"] = {"total": total}
        return counts
