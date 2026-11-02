"""
CFC-DINO Loss Functions

Three-component loss, each independently switchable:

  L_total = lambda_recon    * L_recon
          + lambda_contrast * L_contrast
          + lambda_warp     * L_warp

L_recon    — MSE reconstruction loss
L_contrast — Teacher-Student cosine inconsistency (BYOL-style), per-step [B,T,H]
L_warp     — Time-warp invariance: sparse-view Student vs full-view Teacher,
             sequence-level (final hidden state only) [B,H]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReconLoss(nn.Module):
    def forward(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, x)


class ContrastiveLoss(nn.Module):
    """
    Per-step cosine similarity loss between Student predictor output
    and Teacher representation.

        L = 1 - mean( cosine_sim(p_student[b,t], z_teacher[b,t]) )

    Teacher gradient is stopped — only Student is updated.
    """

    def forward(
        self,
        p_student: torch.Tensor,  # [B, T, H]  Student predictor output
        z_teacher: torch.Tensor,  # [B, T, H]  Teacher encoder output
    ) -> torch.Tensor:
        p = F.normalize(p_student, dim=-1)
        z = F.normalize(z_teacher.detach(), dim=-1)
        return 1 - (p * z).sum(dim=-1).mean()


class CfcDinoLoss(nn.Module):
    """
    Weighted combination of reconstruction, contrastive, and time-warp losses.
    Set lambda to 0 to fully disable that component (no compute wasted).

    Args:
        lambda_recon:    Weight for reconstruction loss
        lambda_contrast: Weight for trajectory contrastive loss
        lambda_warp:     Weight for time-warp invariance loss (0 = off)
    """

    def __init__(
        self,
        lambda_recon:    float = 1.0,
        lambda_contrast: float = 1.0,
        lambda_warp:     float = 0.5,
    ):
        super().__init__()
        self.lambda_recon    = lambda_recon
        self.lambda_contrast = lambda_contrast
        self.lambda_warp     = lambda_warp
        self.recon_loss      = ReconLoss()
        self.contrast_loss   = ContrastiveLoss()

    def forward(
        self,
        x:         torch.Tensor,           # [B, T, C]  original input
        recon:     Optional[torch.Tensor], # [B, T, C]  reconstruction (None = off)
        p_student: torch.Tensor,           # [B, T, H]  trajectory-view predictor output
        z_teacher: torch.Tensor,           # [B, T, H]  teacher encoder output
        p_warp:    Optional[torch.Tensor] = None,  # [B, H]  sparse-view predictor output
        z_t_final: Optional[torch.Tensor] = None,  # [B, H]  teacher final hidden state
    ):
        zero = torch.tensor(0.0, device=x.device)

        L_recon = (self.recon_loss(x, recon)
                   if (self.lambda_recon > 0 and recon is not None) else zero)

        L_contrast = (self.contrast_loss(p_student, z_teacher)
                      if self.lambda_contrast > 0 else zero)

        # Time-warp: compare sparse-view student final state with teacher final state
        L_warp = (self.contrast_loss(p_warp, z_t_final)
                  if (self.lambda_warp > 0 and p_warp is not None) else zero)

        total = (self.lambda_recon    * L_recon
               + self.lambda_contrast * L_contrast
               + self.lambda_warp     * L_warp)

        return total, {
            "L_recon":    L_recon.item(),
            "L_contrast": L_contrast.item(),
            "L_warp":     L_warp.item(),
            "L_total":    total.item(),
        }
