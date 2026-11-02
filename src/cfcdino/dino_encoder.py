"""
DINOv2 Spatial Feature Extractor
Wraps DINOv2 ViT to extract per-frame visual features from image sequences.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional


class DinoEncoder(nn.Module):
    """
    Wraps a DINOv2 ViT backbone to extract spatial features from individual frames.

    Given a sequence of frames [B, T, C, H, W], processes each frame independently
    with shared weights and returns feature vectors [B, T, embed_dim].

    Args:
        arch: DINOv2 architecture variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        feature_type: Which features to extract:
            - 'cls': CLS token only [B, T, D]
            - 'avg': average of patch tokens [B, T, D]
            - 'cls_avg': concat of cls + avg [B, T, 2*D]
        frozen: Whether to freeze DINOv2 weights (recommended for small datasets)
        pretrained: Load pretrained DINOv2 weights from torch.hub
    """

    EMBED_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(
        self,
        arch: str = "dinov2_vitb14",
        feature_type: Literal["cls", "avg", "cls_avg"] = "cls",
        frozen: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.arch = arch
        self.feature_type = feature_type

        if pretrained:
            self.backbone = torch.hub.load("facebookresearch/dinov2", arch)
        else:
            # Build from local reference (no pretrained weights)
            import sys
            sys.path.insert(0, "reference/dinov2-main")
            from dinov2.models import vision_transformer as vits
            arch_name = arch.replace("dinov2_", "")  # e.g. 'vitb14'
            self.backbone = vits.__dict__[arch_name]()

        self._embed_dim = self.EMBED_DIMS.get(arch, self.backbone.embed_dim)

        if feature_type == "cls_avg":
            self.out_dim = self._embed_dim * 2
        else:
            self.out_dim = self._embed_dim

        if frozen:
            self._freeze()

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze last n transformer blocks for fine-tuning."""
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        # Always unfreeze norm layer
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

    def _extract_features(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract features from a single batch of frames [B, C, H, W] -> [B, D]."""
        out = self.backbone.forward_features(frame)

        if self.feature_type == "cls":
            return out["x_norm_clstoken"]          # [B, D]

        elif self.feature_type == "avg":
            return out["x_norm_patchtokens"].mean(dim=1)  # [B, D]

        elif self.feature_type == "cls_avg":
            cls = out["x_norm_clstoken"]                        # [B, D]
            avg = out["x_norm_patchtokens"].mean(dim=1)         # [B, D]
            return torch.cat([cls, avg], dim=-1)                # [B, 2*D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image sequence [B, T, C, H, W]

        Returns:
            features: [B, T, out_dim]
        """
        B, T, C, H, W = x.shape

        # Merge batch and time dims, process all frames in one pass
        x_flat = x.view(B * T, C, H, W)               # [B*T, C, H, W]
        feats = self._extract_features(x_flat)         # [B*T, D]
        feats = feats.view(B, T, -1)                   # [B, T, D]

        return feats
