"""Full-channel raw prediction and the shared valid-position scoring contract."""

import hashlib
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler

from cfcdino.anomaly_detector import CfcAnomalyDetector, Predictor
from cfcdino.cfc_temporal import CfcTemporalEncoder


def future_mse(predictions, observations, k):
    if (predictions.ndim != 3 or predictions.shape != observations.shape
            or not isinstance(k, int) or not 0 <= k < observations.shape[1]):
        raise ValueError('Require matching [batch,time,channels] and 0 <= k < time')
    length = observations.shape[1] - k
    return (predictions[:, :length] - observations[:, k:]).square().mean(dim=-1)


class RawForecaster(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, backbone_units,
                 backbone_layers, backbone_dropout, lookahead):
        super().__init__()
        if not isinstance(lookahead, int) or lookahead < 0:
            raise ValueError('Require a nonnegative integer prediction horizon')
        self.lookahead = lookahead
        self.student_encoder = CfcTemporalEncoder(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
            return_sequences=True, backbone_units=backbone_units,
            backbone_layers=backbone_layers, backbone_dropout=backbone_dropout)
        # Construct common layers in the same RNG order as the latent predictor.
        self.predictor = Predictor(hidden_size)
        self.predictor.net[-1] = nn.Linear(hidden_size, input_dim)

    def predict(self, x, dt=None):
        return self.predictor(self.student_encoder(x, dt))

    def training_step(self, x, dt=None, step=0, total_steps=1):
        residuals = future_mse(self.predict(x, dt), x, self.lookahead)
        loss = residuals.mean()
        return loss, dict(L_total=loss.item(), div_mean=loss.item(), momentum=0.)

    @torch.no_grad()
    def anomaly_score(self, x, dt=None):
        return future_mse(self.predict(x, dt), x, self.lookahead)

    def forward(self, x, dt=None):
        return self.anomaly_score(x, dt)


def make_model(kind, config):
    torch.manual_seed(config['seed'])
    common = dict(input_dim=config['input_c'], hidden_size=config['hidden_size'],
                  num_layers=config['num_layers'], backbone_units=config['backbone_units'],
                  backbone_layers=config['backbone_layers'],
                  backbone_dropout=config['backbone_dropout'], lookahead=config['lookahead'])
    if kind == 'raw':
        return RawForecaster(**common)
    if kind == 'lfp':
        return CfcAnomalyDetector(**common, momentum_base=config['momentum_base'],
                                  momentum_final=config['momentum_final'],
                                  center_momentum=config['center_momentum'])
    raise ValueError('Only lfp and raw are approved')


@torch.no_grad()
def valid_scores(model, x):
    k = model.lookahead
    if not isinstance(k, int) or not 0 <= k < x.shape[1]:
        raise ValueError('Require genuine targets within this window')
    scores = model.anomaly_score(x)
    if scores.ndim != 2 or scores.shape[0] != len(x) or scores.shape[1] < x.shape[1] - k:
        raise ValueError('Model did not return every valid prediction score')
    return scores[:, :x.shape[1] - k]


class EpochOrderSampler(Sampler):
    """A paired permutation stream isolated from model/dropout and loader RNG."""
    def __init__(self, n, seed):
        self.n, self.seed = n, seed
        self.records = []

    def __len__(self):
        return self.n

    def __iter__(self):
        epoch = len(self.records)
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(self.n, generator=generator).numpy().astype('<i8')
        self.records.append(dict(epoch=epoch + 1, count=self.n,
                                 sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                                 first=order[:8].tolist(), last=order[-8:].tolist()))
        return iter(order.tolist())


@torch.no_grad()
def collect_scores(model, data, w, step, reference, batch_size=512, progress=None):
    data = np.asarray(data, dtype=np.float32)
    k = model.lookahead
    if (data.ndim != 2 or w <= 0 or step <= 0 or batch_size <= 0
            or len(data) < w or not 0 <= k < w or not np.isfinite(data).all()):
        raise ValueError('Require finite data, complete windows and valid positive execution sizes')
    if not reference and step != 1:
        raise ValueError('Primary pointwise scoring requires stride one')
    device = next(model.parameters()).device
    tensor = torch.as_tensor(data, device=device)
    starts = np.arange(0, len(data) - w + 1, step, dtype=np.int64)
    width = w - k if reference else 1
    output = np.empty(len(starts) * width, dtype=np.float32)
    model.eval()
    began = last_log = time.perf_counter()
    for offset in range(0, len(starts), batch_size):
        selected = torch.as_tensor(starts[offset:offset + batch_size], device=device)
        windows = tensor[selected[:, None] + torch.arange(w, device=device)[None, :]]
        values = valid_scores(model, windows)
        if not reference:
            values = values[:, -1:]
        output[offset * width:(offset + len(selected)) * width] = values.cpu().numpy().reshape(-1)
        tick = time.perf_counter()
        if progress and (tick - last_log >= 30 or offset + len(selected) == len(starts)):
            progress(dict(windows_done=offset + len(selected), windows_total=len(starts),
                          elapsed_seconds=tick - began))
            last_log = tick
    if not np.isfinite(output).all():
        raise RuntimeError('Nonfinite model scores')
    return output, dict(windows=len(starts), reference_slots=len(output), stride=step,
                        local_origin_first=0 if reference else w-k-1, local_origin_last=w-k-1,
                        first_target=k if reference else w-1,
                        last_target=int(starts[-1] + w - 1),
                        elapsed_seconds=time.perf_counter() - began)
