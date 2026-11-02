"""
CFC-DINO: Integrated Spatial-Temporal Model
Combines DINOv2 spatial encoder with CfC temporal encoder for
video understanding with irregular time intervals.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from .dino_encoder import DinoEncoder
from .cfc_temporal import CfcTemporalEncoder


class CfcDino(nn.Module):
    """
    CFC-DINO: DINOv2 + CfC integrated model for sequential visual understanding.

    Architecture:
        Frame sequence [B, T, C, H, W]  +  timestamps [B, T]
                 ↓ (per-frame, shared weights)
        DINOv2 Spatial Encoder  →  [B, T, dino_dim]
                 ↓ (optional projection)
        Linear Projection       →  [B, T, cfc_input_dim]
                 ↓ (with Δt between frames)
        CfC Temporal Encoder    →  [B, hidden_size]
                 ↓
        Task Head               →  [B, num_classes]

    Args:
        num_classes:       Output classes (or 0 to return raw features)
        dino_arch:         DINOv2 backbone ('dinov2_vits14', 'dinov2_vitb14', etc.)
        dino_feature_type: Which DINOv2 features to use ('cls', 'avg', 'cls_avg')
        dino_frozen:       Freeze DINOv2 weights
        dino_pretrained:   Load DINOv2 pretrained weights
        cfc_hidden_size:   CfC hidden state dimension
        cfc_num_layers:    Number of stacked CfC layers
        cfc_backbone_units: Hidden units in CfC backbone MLP
        cfc_backbone_layers: Depth of CfC backbone MLP
        cfc_dropout:       Dropout in CfC backbone
        projection_dim:    If set, projects DINOv2 features before CfC (reduces dim)
        head_dropout:      Dropout before classification head
    """

    def __init__(
        self,
        num_classes: int,
        dino_arch: str = "dinov2_vitb14",
        dino_feature_type: Literal["cls", "avg", "cls_avg"] = "cls",
        dino_frozen: bool = True,
        dino_pretrained: bool = True,
        cfc_hidden_size: int = 256,
        cfc_num_layers: int = 1,
        cfc_backbone_units: int = 256,
        cfc_backbone_layers: int = 1,
        cfc_dropout: float = 0.0,
        projection_dim: Optional[int] = None,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        # 1. DINOv2 Spatial Encoder
        self.dino = DinoEncoder(
            arch=dino_arch,
            feature_type=dino_feature_type,
            frozen=dino_frozen,
            pretrained=dino_pretrained,
        )
        dino_out_dim = self.dino.out_dim

        # 2. Optional projection: reduce DINOv2's large embedding before CfC
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(dino_out_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
            )
            cfc_input_dim = projection_dim
        else:
            self.projection = nn.Identity()
            cfc_input_dim = dino_out_dim

        # 3. CfC Temporal Encoder
        self.cfc = CfcTemporalEncoder(
            input_size=cfc_input_dim,
            hidden_size=cfc_hidden_size,
            num_layers=cfc_num_layers,
            return_sequences=False,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            backbone_dropout=cfc_dropout,
        )

        # 4. Classification Head
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(cfc_hidden_size, num_classes),
            )
        else:
            self.head = nn.Identity()

        self.num_classes = num_classes

    def forward(
        self,
        frames: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frames:     Video frame sequence [B, T, C, H, W]
            timestamps: Absolute timestamps for each frame [B, T].
                        Will be converted to Δt internally.
                        If None, assumes uniform Δt=1.

        Returns:
            logits: [B, num_classes]  (or [B, cfc_hidden_size] if num_classes=0)
        """
        # Compute Δt from absolute timestamps
        if timestamps is not None:
            timespans = self._compute_timespans(timestamps)
        else:
            timespans = None

        # [B, T, C, H, W] -> [B, T, dino_dim]
        spatial_features = self.dino(frames)

        # [B, T, dino_dim] -> [B, T, cfc_input_dim]
        spatial_features = self.projection(spatial_features)

        # [B, T, cfc_input_dim] -> [B, hidden_size]
        temporal_features = self.cfc(spatial_features, timespans)

        # [B, hidden_size] -> [B, num_classes]
        return self.head(temporal_features)

    @staticmethod
    def _compute_timespans(timestamps: torch.Tensor) -> torch.Tensor:
        """
        Convert absolute timestamps to elapsed time between consecutive frames.

        Args:
            timestamps: [B, T] absolute timestamps (seconds or any unit)

        Returns:
            timespans: [B, T] where timespans[:, 0] = 1.0 (first frame baseline),
                               timespans[:, t] = timestamps[:, t] - timestamps[:, t-1]
        """
        B, T = timestamps.shape
        dt = torch.diff(timestamps, dim=1).clamp(min=1e-6)  # [B, T-1]
        # Prepend 1.0 as the "elapsed time" for the first frame
        first = torch.ones(B, 1, device=timestamps.device, dtype=timestamps.dtype)
        return torch.cat([first, dt], dim=1)                 # [B, T]

    def unfreeze_dino(self, num_blocks: int = 2):
        """Fine-tune last N blocks of DINOv2 alongside CfC."""
        self.dino.unfreeze_last_n_blocks(num_blocks)

    def get_trainable_params(self):
        """Returns list of parameter groups with different learning rates."""
        dino_params = [p for p in self.dino.parameters() if p.requires_grad]
        other_params = (
            list(self.projection.parameters())
            + list(self.cfc.parameters())
            + list(self.head.parameters())
        )
        groups = []
        if dino_params:
            groups.append({"params": dino_params, "lr_scale": 0.1})  # smaller LR for backbone
        groups.append({"params": other_params, "lr_scale": 1.0})
        return groups


def build_cfcdino(config: dict) -> CfcDino:
    """
    Build CfcDino from a config dictionary.

    Example config:
        {
            "num_classes": 10,
            "dino_arch": "dinov2_vitb14",
            "dino_frozen": true,
            "cfc_hidden_size": 256,
            "cfc_num_layers": 2,
            "projection_dim": 256
        }
    """
    return CfcDino(**config)
