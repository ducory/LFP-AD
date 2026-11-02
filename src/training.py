"""Original training loop and helpers; legacy scoring is intentionally omitted."""


import time
import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn


from cfcdino.anomaly_detector import CfcAnomalyDetector
from data_factory.data_loader import get_loader_segment


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model, path)
            self.counter = 0

    def _save(self, val_loss, model, path):
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), os.path.join(path, f'{self.dataset}_checkpoint.pth'))
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, base_lr, decay_every=5):
    lr = base_lr * (0.5 ** ((epoch - 1) // decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'Updating learning rate to {lr}')


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.device = torch.device(
            f'cuda:{self.gpu_index}' if torch.cuda.is_available() else 'cpu')

        self._build_model()

    def _build_model(self):
        self.model = CfcAnomalyDetector(
            input_dim=self.input_c,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            backbone_units=self.backbone_units,
            backbone_layers=self.backbone_layers,
            backbone_dropout=self.backbone_dropout,
            lookahead=getattr(self, 'lookahead', 5),
            momentum_base=getattr(self, 'momentum_base', 0.990),
            momentum_final=getattr(self, 'momentum_final', 0.999),
            center_momentum=getattr(self, 'center_momentum', 0.99),
        )
        if self.multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        student_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(student_params, lr=self.lr)

    @property
    def _model(self):
        """Unwrap DataParallel to access model attributes directly."""
        return self.model.module if hasattr(self.model, 'module') else self.model

    def vali(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for input_data, _ in loader:
                x = input_data.float().to(self.device)
                losses.append(self._model.anomaly_score(x).mean().item())
        return np.mean(losses)

    def train(self):
        print("======================TRAIN MODE======================")
        path = self.model_save_path
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=100, verbose=True, dataset_name=self.dataset)

        train_steps = len(self.train_loader)
        total_steps = train_steps * self.num_epochs
        global_step = 0
        time_now = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_time = time.time()
            log_accum = defaultdict(list)
            iter_count = 0

            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1

                x = (batch[0] if not isinstance(batch, dict) else batch["x_global"])
                x = x.float().to(self.device)

                loss, log_dict = self.model.training_step(
                    x, step=global_step, total_steps=total_steps)
                global_step += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0)
                self.optimizer.step()

                for k, v in log_dict.items():
                    log_accum[k].append(v)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            train_loss = np.mean(log_accum["L_total"])
            vali_loss  = self.vali(self.vali_loader)
            momentum   = log_accum["momentum"][-1] if log_accum["momentum"] else 0.0
            elapsed    = time.time() - epoch_time

            print(
                f'Epoch: {epoch+1}/{self.num_epochs} | '
                f'L_train: {train_loss:.4f}  Val: {vali_loss:.6f}  '
                f'momentum: {momentum:.4f}  Cost: {elapsed:.1f}s'
            )

            early_stopping(vali_loss, self.model, path)
            # Always overwrite with latest epoch (val loss is not a reliable
            # proxy for F1 in anomaly detection; test() loads this file)
            torch.save(self.model.state_dict(),
                       os.path.join(path, f'{self.dataset}_checkpoint.pth'))
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr,
                                  decay_every=getattr(self, 'lr_decay_every', 5))

    @staticmethod
    def _minmax(arr, vmin, vmax):
        return (arr - vmin) / (vmax - vmin + 1e-8)

    @staticmethod
    def _combine(recon, contrast, recon_range, contrast_range, alpha):
        """Normalize each component globally, then combine with alpha."""
        c_min, c_max = contrast_range
        s_c = Solver._minmax(contrast, c_min, c_max)
        if recon is not None and recon_range is not None:
            r_min, r_max = recon_range
            s_r = Solver._minmax(recon, r_min, r_max)
            return alpha * s_r + (1 - alpha) * s_c
        return s_c
