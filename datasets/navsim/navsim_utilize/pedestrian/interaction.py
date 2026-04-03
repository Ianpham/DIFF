"""
Interaction-Aware Agent for NavSim
====================================
Stage 1, Days 7-9: Agent Architecture

This agent implements:
    1. Interaction encoder: MLP for ped features + small CNN for risk field
    2. Latent projection: μ, log σ² → z (variational, dim=4)
    3. Trajectory decoder: MLP conditioned on z + ego features → 8 poses

For Stage 1, we keep the decoder as a simple MLP. Diffusion decoder comes in Stage 2/3.
The goal here is to validate that the pipeline works end-to-end and that z
actually correlates with interaction outcomes.

Plugs directly into NavSim's training pipeline:
    - Extends AbstractAgent
    - Provides feature/target builders for caching
    - Compatible with AgentLightningModule
    - Evaluable via run_pdm_score
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from feature_target_builders import (
    PedestrianInteractionFeatureBuilder,
    InteractionTargetBuilder,
)



# Model components



class InteractionEncoder(nn.Module):
    """
    Encodes pedestrian interaction features + spatial risk field into a joint embedding.

    Architecture:
        ped_features [20] → MLP → [64]
        risk_field [1, 32, 32] → Conv2D → flatten → [128]
        ego_features [7] → MLP → [32]
        concat [224] → MLP → [128]
    """

    def __init__(
        self,
        ped_feature_dim: int = 20,
        ego_feature_dim: int = 7,  # vx, vy, ax, ay, speed, driving_cmd[0:2]
        risk_field_size: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Pedestrian feature encoder
        self.ped_encoder = nn.Sequential(
            nn.Linear(ped_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Risk field encoder (small CNN)
        self.risk_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # [32, 4, 4]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),  # [32, 2, 2]
            nn.Flatten(),  # [128]
        )

        # Ego feature encoder
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_feature_dim, 32),
            nn.ReLU(),
        )

        # Joint fusion
        fusion_dim = 64 + 128 + 32  # ped + risk + ego = 224
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        ped_features: Tensor,  # [B, 20]
        risk_field: Tensor,  # [B, 1, 32, 32]
        ego_features: Tensor,  # [B, 7]
    ) -> Tensor:
        """Returns joint embedding [B, hidden_dim]."""
        e_ped = self.ped_encoder(ped_features)  # [B, 64]
        e_risk = self.risk_encoder(risk_field)  # [B, 128]
        e_ego = self.ego_encoder(ego_features)  # [B, 32]
        e_joint = torch.cat([e_ped, e_risk, e_ego], dim=-1)  # [B, 224]
        return self.fusion(e_joint)  # [B, hidden_dim]


class LatentProjection(nn.Module):
    """
    Projects interaction embedding to a variational latent z.
    z ~ N(μ, σ²) during training, z = μ during inference.
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 4):
        super().__init__()
        self.mu_head = nn.Linear(input_dim, latent_dim)
        self.logvar_head = nn.Linear(input_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(
        self, embedding: Tensor, training: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns: (z, mu, logvar)
        z is sampled during training, mu during inference.
        """
        mu = self.mu_head(embedding)  # [B, latent_dim]
        logvar = self.logvar_head(embedding)  # [B, latent_dim]
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)  # numerical stability

        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar


class TrajectoryDecoder(nn.Module):
    """
    Simple MLP trajectory decoder conditioned on latent z and ego features.
    Outputs 8 poses × 3 (x, y, heading) in ego-relative frame.

    For Stage 1, this is a plain MLP. Will be replaced by diffusion decoder in Stage 2/3.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        ego_feature_dim: int = 7,
        hidden_dim: int = 256,
        num_poses: int = 8,
    ):
        super().__init__()
        self.num_poses = num_poses
        input_dim = latent_dim + ego_feature_dim

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_poses * 3),
        )

    def forward(self, z: Tensor, ego_features: Tensor) -> Tensor:
        """
        Returns trajectory [B, num_poses, 3] (x, y, heading).
        """
        x = torch.cat([z, ego_features], dim=-1)
        out = self.decoder(x)
        return out.view(-1, self.num_poses, 3)



# Loss functions



class InteractionAwareLoss(nn.Module):
    """
    Combined loss for interaction-aware trajectory prediction.

    L = L_trajectory + β·L_KL + γ·L_contrastive + δ·L_heuristic

    All auxiliary losses are weighted by interaction_score so non-interactive
    scenes don't dominate the auxiliary signals.
    """

    def __init__(
        self,
        beta_kl: float = 0.01,
        gamma_contrastive: float = 0.1,
        delta_heuristic: float = 0.05,
        num_outcome_classes: int = 6,
    ):
        super().__init__()
        self.beta_kl = beta_kl
        self.gamma_contrastive = gamma_contrastive
        self.delta_heuristic = delta_heuristic

        # Contrastive: simple classification of outcome from z
        self.outcome_classifier = nn.Linear(4, num_outcome_classes)

    def kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """KL(q(z|x) || N(0, I))"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    def forward(
        self,
        pred_trajectory: Tensor,  # [B, 8, 3]
        gt_trajectory: Tensor,  # [B, 8, 3]
        mu: Tensor,  # [B, 4]
        logvar: Tensor,  # [B, 4]
        z: Tensor,  # [B, 4]
        interaction_score: Tensor,  # [B, 1]
        outcome_label: Tensor,  # [B, 1] (long)
    ) -> Dict[str, Tensor]:
        """Compute all loss components."""
        batch_size = pred_trajectory.shape[0]

        # 1. Trajectory loss (L1 + heading)
        pos_loss = F.l1_loss(
            pred_trajectory[:, :, :2], gt_trajectory[:, :, :2], reduction="none"
        ).mean(dim=(1, 2))  # [B]

        # Heading loss (angular distance)
        pred_heading = pred_trajectory[:, :, 2]
        gt_heading = gt_trajectory[:, :, 2]
        heading_diff = torch.atan2(
            torch.sin(pred_heading - gt_heading),
            torch.cos(pred_heading - gt_heading),
        )
        heading_loss = heading_diff.abs().mean(dim=1)  # [B]

        traj_loss = (pos_loss + 0.5 * heading_loss).mean()

        # 2. KL divergence (weighted by interaction_score)
        kl = self.kl_divergence(mu, logvar)  # [B]
        kl_weighted = (kl * interaction_score.squeeze(-1)).mean()

        # 3. Contrastive: outcome classification from z
        outcome_logits = self.outcome_classifier(z)  # [B, num_classes]
        contrastive_loss_raw = F.cross_entropy(
            outcome_logits, outcome_label.squeeze(-1), reduction="none"
        )  # [B]
        contrastive_loss = (
            contrastive_loss_raw * interaction_score.squeeze(-1)
        ).mean()

        # Total loss
        total_loss = (
            traj_loss
            + self.beta_kl * kl_weighted
            + self.gamma_contrastive * contrastive_loss
        )

        return {
            "loss": total_loss,
            "traj_loss": traj_loss.detach(),
            "kl_loss": kl_weighted.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "outcome_accuracy": (
                outcome_logits.argmax(dim=-1) == outcome_label.squeeze(-1)
            )
            .float()
            .mean()
            .detach(),
        }



# Main Agent



class InteractionAwareAgent(AbstractAgent):
    """
    NavSim agent with pedestrian interaction awareness.

    Plugs into NavSim's standard training pipeline. Uses:
        - PedestrianInteractionFeatureBuilder for features
        - InteractionTargetBuilder for pseudo-labels
        - InteractionEncoder + LatentProjection + TrajectoryDecoder for the model
    """

    def __init__(
        self,
        lr: float = 1e-4,
        latent_dim: int = 4,
        hidden_dim: int = 128,
        decoder_hidden_dim: int = 256,
        beta_kl: float = 0.01,
        gamma_contrastive: float = 0.1,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
        map_root: Optional[str] = None,
        risk_field_size: int = 32,
    ):
        super().__init__(trajectory_sampling=trajectory_sampling)

        self.lr = lr
        self.latent_dim = latent_dim
        self._map_root = map_root
        self._risk_field_size = risk_field_size

        # Model components
        self.interaction_encoder = InteractionEncoder(
            ped_feature_dim=20,
            ego_feature_dim=7,
            risk_field_size=risk_field_size,
            hidden_dim=hidden_dim,
        )
        self.latent_projection = LatentProjection(
            input_dim=hidden_dim, latent_dim=latent_dim
        )
        self.trajectory_decoder = TrajectoryDecoder(
            latent_dim=latent_dim,
            ego_feature_dim=7,
            hidden_dim=decoder_hidden_dim,
            num_poses=trajectory_sampling.num_poses,
        )

        # Loss
        self.loss_fn = InteractionAwareLoss(
            beta_kl=beta_kl,
            gamma_contrastive=gamma_contrastive,
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    def name(self) -> str:
        return "InteractionAwareAgent"

    def get_sensor_config(self) -> SensorConfig:
        """No raw sensor data needed — we work from annotations and ego status."""
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        """Nothing to initialize."""
        pass

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [
            PedestrianInteractionFeatureBuilder(
                map_root=self._map_root,
                risk_field_size=self._risk_field_size,
            ),
        ]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            InteractionTargetBuilder(
                num_trajectory_frames=self._trajectory_sampling.num_poses,
            ),
        ]

    def _extract_ego_features(self, features: Dict[str, Tensor]) -> Tensor:
        """
        Extract ego dynamics from the ped_interaction_features tensor.
        The first 7 dims are: vx, vy, ax, ay, speed, driving_cmd[0], driving_cmd[1]
        """
        ped_feat = features["ped_interaction_features"]  # [B, 20]
        return ped_feat[:, :7]  # [B, 7]

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass.

        Input features (from cache):
            ped_interaction_features: [B, 20]
            ped_risk_field: [B, 1, 32, 32]
            has_relevant_pedestrian: [B, 1]

        Returns:
            trajectory: [B, 8, 3]
            z: [B, 4]
            mu: [B, 4]
            logvar: [B, 4]
        """
        ped_features = features["ped_interaction_features"]  # [B, 20]
        risk_field = features["ped_risk_field"]  # [B, 1, H, W]
        ego_features = self._extract_ego_features(features)  # [B, 7]

        # Encode interaction
        embedding = self.interaction_encoder(ped_features, risk_field, ego_features)

        # Project to latent
        z, mu, logvar = self.latent_projection(embedding, training=self.training)

        # Decode trajectory
        trajectory = self.trajectory_decoder(z, ego_features)

        return {
            "trajectory": trajectory,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }

    def compute_loss(
        self,
        features: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        predictions: Dict[str, Tensor],
    ) -> torch.Tensor:
        """Compute training loss."""
        losses = self.loss_fn(
            pred_trajectory=predictions["trajectory"],
            gt_trajectory=targets["trajectory"],
            mu=predictions["mu"],
            logvar=predictions["logvar"],
            z=predictions["z"],
            interaction_score=targets["interaction_score"],
            outcome_label=targets["interaction_outcome_label"],
        )
        return losses["loss"]

    def get_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Inference: compute trajectory from AgentInput.
        This is called by NavSim's evaluation pipeline.
        """
        self.eval()
        features: Dict[str, Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # Add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        return Trajectory(poses, self._trajectory_sampling)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Return callbacks for monitoring z quality during training."""
        return [ZLatentMonitorCallback()]



# Training callback for monitoring z quality



class ZLatentMonitorCallback(pl.Callback):
    """
    Monitors the interaction latent z during training.
    Logs z statistics and checks for mode collapse.
    """

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            features, targets = batch
            with torch.no_grad():
                predictions = pl_module.agent.forward(features)

            z = predictions["z"]  # [B, 4]
            mu = predictions["mu"]  # [B, 4]

            # Log z statistics
            for dim in range(z.shape[1]):
                trainer.logger.log_metrics(
                    {
                        f"z/dim{dim}_mean": z[:, dim].mean().item(),
                        f"z/dim{dim}_std": z[:, dim].std().item(),
                        f"mu/dim{dim}_std": mu[:, dim].std().item(),
                    },
                    step=trainer.global_step,
                )

            # Check for collapse: if all mu are nearly identical, z is useless
            mu_var = mu.var(dim=0)  # variance across batch per dimension
            if (mu_var < 0.01).all():
                print(
                    f"   WARNING: z appears collapsed (all dims var < 0.01). "
                    f"Consider increasing gamma_contrastive or reducing beta_kl."
                )