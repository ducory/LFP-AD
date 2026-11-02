"""
CfC Temporal Encoder
Processes sequential visual features with irregular time steps using
Closed-form Continuous-time Neural Networks (CfC).
传统 RNN（如 LSTM、GRU）假设每个时间步之间的间隔相同。CfC 的关键区别是：它把实际经过的时间 Δt 作为显式输入，
通过一个闭合形式的连续时间动力学方程来更新隐状态，从而天然地适应不等间距的观测。
"""

import torch
import torch.nn as nn
from typing import Optional


class LeCun(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh(0.666 * x)


ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "lecun": LeCun,
}


class CfcCell(nn.Module):
    """
    Single CfC recurrent cell.

    Implements the closed-form continuous-time dynamics:
        t_interp = sigmoid(t_a * Δt + t_b)
        h_new = ff1 * (1 - t_interp) + t_interp * ff2

    where Δt is the actual elapsed time between observations.
    This allows the cell to model irregular time intervals naturally.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        backbone_units: int = 256,
        backbone_layers: int = 1,
        backbone_activation: str = "lecun",
        backbone_dropout: float = 0.0,
        minimal: bool = False,
        no_gate: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._minimal = minimal
        self._no_gate = no_gate

        # Build backbone MLP: [input + hidden] -> backbone_units
        act_cls = ACTIVATIONS[backbone_activation]
        layers = [
            nn.Linear(input_size + hidden_size, backbone_units),
            act_cls(),
        ]
        for _ in range(1, backbone_layers):
            layers.append(nn.Linear(backbone_units, backbone_units))
            layers.append(act_cls())
            if backbone_dropout > 0:
                layers.append(nn.Dropout(backbone_dropout))
        self.backbone = nn.Sequential(*layers)

        # Output projections
        self.ff1 = nn.Linear(backbone_units, hidden_size)

        if minimal:
            # Minimal variant: direct ODE solution
            self.w_tau = nn.Parameter(torch.zeros(1, hidden_size))
            self.A = nn.Parameter(torch.ones(1, hidden_size))
        else:
            # Full CfC: time-gated interpolation between two states
            self.ff2 = nn.Linear(backbone_units, hidden_size)
            self.time_a = nn.Linear(backbone_units, hidden_size)
            self.time_b = nn.Linear(backbone_units, hidden_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        ts: torch.Tensor,
        return_continuous_params: bool = False,
    ):
        """
        Args:
            x:  Current input  [B, input_size]
            h:  Previous state [B, hidden_size]
            ts: Elapsed time   [B] or [B, 1]
            return_continuous_params: If True, also return (ff1, ff2, t_a, t_b)
                so callers can evaluate h(τ) at arbitrary τ ∈ [0, ts] via:
                    h(τ) = ff1*(1-σ(t_a*τ+t_b)) + ff2*σ(t_a*τ+t_b)

        Returns:
            h_new [B, hidden_size], and optionally (ff1, ff2, t_a, t_b)
        """
        B = x.size(0)
        ts = ts.view(B, 1)

        xh = torch.cat([x, h], dim=-1)
        feat = self.backbone(xh)

        if self._minimal:
            ff1 = self.ff1(feat)
            h_new = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
            if return_continuous_params:
                return h_new, (ff1, None, None, None)
        else:
            ff1 = torch.tanh(self.ff1(feat))
            ff2 = torch.tanh(self.ff2(feat))
            t_a = self.time_a(feat)
            t_b = self.time_b(feat)
            t_interp = torch.sigmoid(t_a * ts + t_b)

            if self._no_gate:
                h_new = ff1 + t_interp * ff2
            else:
                h_new = ff1 * (1.0 - t_interp) + t_interp * ff2

            if return_continuous_params:
                return h_new, (ff1, ff2, t_a, t_b)

        return h_new

    def eval_trajectory(
        self,
        ff1: torch.Tensor,
        ff2: torch.Tensor,
        t_a: torch.Tensor,
        t_b: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the closed-form CfC trajectory at arbitrary time τ.

        Given params computed at observation i, returns h(τ) for τ ∈ [0, Δt_i]:
            h(τ) = ff1·(1 - σ(t_a·τ + t_b)) + ff2·σ(t_a·τ + t_b)

        Args:
            ff1, ff2: [B, hidden_size]
            t_a, t_b: [B, hidden_size]
            tau:      [B, 1]  time offset within the interval

        Returns:
            h_tau: [B, hidden_size]
        """
        gate = torch.sigmoid(t_a * tau + t_b)
        return ff1 * (1.0 - gate) + ff2 * gate


class CfcTemporalEncoder(nn.Module):
    """
    Multi-layer CfC temporal encoder for sequential visual features.

    Processes a sequence of frame features [B, T, D] with associated timestamps,
    supporting irregular time intervals between frames.

    Args:
        input_size:   Feature dimension from DINOv2 encoder
        hidden_size:  CfC hidden state dimension
        num_layers:   Number of stacked CfC layers
        return_sequences: If True, return all hidden states [B, T, hidden_size];
                          If False, return only last state [B, hidden_size]
        backbone_units: Hidden units in CfC backbone MLP
        backbone_layers: Depth of CfC backbone MLP
        backbone_dropout: Dropout rate in backbone
        minimal: Use minimal CfC variant
        inter_dropout: Dropout between CfC layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        return_sequences: bool = False,
        backbone_units: int = 256,
        backbone_layers: int = 1,
        backbone_activation: str = "lecun",
        backbone_dropout: float = 0.0,
        minimal: bool = False,
        inter_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        # Stack of CfC cells; first layer takes dino features, rest take hidden states
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            self.cells.append(
                CfcCell(
                    input_size=in_sz,
                    hidden_size=hidden_size,
                    backbone_units=backbone_units,
                    backbone_layers=backbone_layers,
                    backbone_activation=backbone_activation,
                    backbone_dropout=backbone_dropout,
                    minimal=minimal,
                )
            )

        self.inter_dropout = nn.Dropout(inter_dropout) if inter_dropout > 0 else nn.Identity()

    def forward(
        self,
        features: torch.Tensor,
        timespans: Optional[torch.Tensor] = None,
        collect_continuous_params: bool = False,
    ):
        """
        Args:
            features:   [B, T, input_size]
            timespans:  [B, T] elapsed time per step. Defaults to all-ones.
            collect_continuous_params: If True, also return per-step CfC params
                from the last layer, enabling continuous-time trajectory evaluation.

        Returns:
            hidden: [B, T, hidden_size] or [B, hidden_size]
            continuous_params (only if collect_continuous_params=True):
                list of T tuples (ff1, ff2, t_a, t_b), each [B, hidden_size]
        """
        B, T, _ = features.shape
        device = features.device

        if timespans is None:
            timespans = torch.ones(B, T, device=device)

        h_states = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]

        outputs = []
        cont_params = []  # per-step params from last layer

        for t in range(T):
            x_t = features[:, t, :]
            ts_t = timespans[:, t]

            h_in = x_t
            for layer_idx, cell in enumerate(self.cells):
                is_last = (layer_idx == self.num_layers - 1)
                if collect_continuous_params and is_last:
                    h_new, params = cell(h_in, h_states[layer_idx], ts_t,
                                         return_continuous_params=True)
                    cont_params.append(params)
                else:
                    h_new = cell(h_in, h_states[layer_idx], ts_t)
                h_states[layer_idx] = h_new
                h_in = self.inter_dropout(h_new)

            if self.return_sequences:
                outputs.append(h_in)

        hidden = (torch.stack(outputs, dim=1) if self.return_sequences
                  else h_states[-1])

        if collect_continuous_params:
            return hidden, cont_params
        return hidden
