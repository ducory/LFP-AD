"""
CFC-DINO Anomaly Detector —  Causal Prediction (v3)

Key insight:
  CfC is inherently causal as an RNN — z_s(t) only depends on x_1..x_t.
  Teacher sees the full window bidirectionally (or with future context via
  a lookahead offset k).  At every time step t, we measure whether the
  Student can predict the Teacher's state at t+k from past alone.

  s(t) = 1 - cos( predictor(z_s(t)),  z_t(t+k) )

  Normal: past predicts near-future well   → low divergence at every step.
  Anomaly at step t: Teacher's state shifts → Student can't predict → spike.


Architecture:
  - Teacher encoder:  CfC on full window → z_t [B, T, H]   (EMA, no grad)
  - Student encoder:  CfC on full window → z_s [B, T, H]   (causal by nature)
  - Predictor:        MLP  z_s(t) → p(t)  predicts z_t(t+k)
  - DINO centering:   center vector subtracted from teacher targets

Training loss:
  L = (1/T') Σ_{t=0}^{T-k-1}  [ 1 - cos( p(t), z_t(t+k) - center ) ]

Anomaly score:
  Same formula, per time-step.  Steps near the end where t+k > T use k=0
  (self-prediction) as fallback.
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfc_temporal import CfcTemporalEncoder


# ======================================================================
# Predictor
# ======================================================================

class Predictor(nn.Module):
    """Asymmetric MLP predictor (BYOL-style) on the Student branch."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# Main model
# ======================================================================

class CfcAnomalyDetector(nn.Module):
    """

    Args:
        input_dim:         Number of input sensor channels.
        hidden_size:       CfC hidden state dimension.
        num_layers:        Number of stacked CfC layers.
        backbone_units:    Hidden units inside each CfC cell MLP.
        backbone_layers:   MLP depth inside each CfC cell.
        backbone_dropout:  Dropout rate in the CfC backbone.
        lookahead:         Number of future steps k for the prediction target.
                           Teacher target at step t is z_t(t+k).
        momentum_base:     Initial EMA momentum (start of training).
        momentum_final:    Final EMA momentum (end of training).
        center_momentum:   EMA momentum for the DINO center vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.1,
        lookahead: int = 5,
        momentum_base: float = 0.990,
        momentum_final: float = 0.999,
        center_momentum: float = 0.99,
    ):
        super().__init__()
        self.lookahead = lookahead
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.center_momentum = center_momentum

        enc_kwargs = dict(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            return_sequences=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )

        # Student branch (trainable)
        self.student_encoder = CfcTemporalEncoder(**enc_kwargs)
        self.predictor = Predictor(hidden_size)

        # Teacher branch (EMA copy, frozen)
        self.teacher_encoder = CfcTemporalEncoder(**enc_kwargs)
        self._init_teacher()

        # DINO centering: running mean of teacher outputs
        self.register_buffer("center", torch.zeros(hidden_size))

    # ------------------------------------------------------------------
    # Teacher management
    # ------------------------------------------------------------------

    def _init_teacher(self):
        self.teacher_encoder.load_state_dict(
            self.student_encoder.state_dict()
        )
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        for p_s, p_t in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters(),
        ):
            p_t.data.mul_(momentum).add_((1 - momentum) * p_s.data)

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor):
        """
        Update DINO center from teacher hidden states.

        Args:
            teacher_out: [B, T, H] — full teacher sequence.
        """
        # Use mean over both batch and time
        batch_center = teacher_out.mean(dim=(0, 1))  # [H]
        m = self.center_momentum
        self.center.mul_(m).add_((1 - m) * batch_center)

    # ------------------------------------------------------------------
    # Momentum schedule
    # ------------------------------------------------------------------

    def get_momentum(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return self.momentum_final
        ratio = step / (total_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.momentum_final - (self.momentum_final - self.momentum_base) * cosine


    def _pointwise_divergence(
        self,
        z_student: torch.Tensor,
        z_teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-step causal divergence.

        Student state at t predicts Teacher state at t+k.
        For the last k steps where t+k >= T, we fall back to t+k = T-1
        (predict the final teacher state).

        Args:
            z_student: [B, T, H]  student hidden states (causal)
            z_teacher: [B, T, H]  teacher hidden states (centered)

        Returns:
            divergence: [B, T]  per-step scores
        """
        B, T, H = z_student.shape
        k = self.lookahead

        # Student predictions at every step
        p = self.predictor(z_student)                    # [B, T, H]
        p = F.normalize(p, dim=-1)                       # [B, T, H]

        # Build teacher targets: z_teacher(t+k) for each t
        # For t where t+k < T: use z_teacher[:, t+k, :]
        # For t where t+k >= T: use z_teacher[:, T-1, :]
        indices = torch.arange(T, device=z_student.device)
        target_indices = torch.clamp(indices + k, max=T - 1)  # [T]
        z_target = z_teacher[:, target_indices, :]       # [B, T, H]
        z_target = F.normalize(z_target, dim=-1)         # [B, T, H]

        # Cosine divergence
        cos_sim = (p * z_target).sum(dim=-1)             # [B, T]
        return 1.0 - cos_sim                             # [B, T]

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        step: int = 0,
        total_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        One training iteration.

        Args:
            x:            [B, T, input_dim]
            dt:           [B, T] optional timespans
            step:         current global step
            total_steps:  total steps for schedule

        Returns:
            loss, log_dict
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            z_t = self.teacher_encoder(x, dt)            # [B, T, H]

        # Centered teacher targets
        z_t_centered = z_t - self.center                 # [B, T, H]

        # Student forward (causal by CfC's RNN nature)
        z_s = self.student_encoder(x, dt)                # [B, T, H]

        # Point-wise causal divergence
        div = self._pointwise_divergence(z_s, z_t_centered)  # [B, T]

        # Only compute loss on valid prediction steps (0 .. T-k-1)
        T = x.shape[1]
        k = min(self.lookahead, T - 1)
        if k > 0:
            loss = div[:, :T - k].mean()
        else:
            loss = div.mean()

        # Update teacher & center
        momentum = self.get_momentum(step, total_steps)
        self.update_teacher(momentum)
        self.update_center(z_t)

        log = {
            "L_total": loss.item(),
            "div_mean": div.mean().item(),
            "div_std": div.std().item(),
            "momentum": momentum,
        }
        return loss, log

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns per-step anomaly scores [B, T]."""
        return self.anomaly_score(x, dt)

    @torch.no_grad()
    def anomaly_score(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-step anomaly score [B, T].

        s(t) = 1 - cos( predictor(z_s(t)), z_t(t+k) - center )

        Args:
            x:  [B, T, input_dim]
            dt: [B, T] or None

        Returns:
            scores: [B, T]
        """
        z_t = self.teacher_encoder(x, dt)                # [B, T, H]
        z_t_centered = z_t - self.center                  # [B, T, H]

        z_s = self.student_encoder(x, dt)                 # [B, T, H]

        return self._pointwise_divergence(z_s, z_t_centered)  # [B, T]

    @torch.no_grad()
    def score_components(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Detailed score components for analysis.

        Returns dict with:
            "s_divergence":  [B, T]  per-step causal divergence
            "z_student":     [B, T, H]  student hidden states
            "z_teacher":     [B, T, H]  teacher hidden states (raw, not centered)
        """
        z_t = self.teacher_encoder(x, dt)
        z_s = self.student_encoder(x, dt)
        z_t_centered = z_t - self.center

        div = self._pointwise_divergence(z_s, z_t_centered)

        return {
            "s_divergence": div,
            "z_student": z_s,
            "z_teacher": z_t,
        }