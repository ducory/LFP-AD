"""Original paired training setup and final valid-reference evaluation."""
import hashlib
import re
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from training import Solver
from runtime import set_seed
from raw_helpers import EpochOrderSampler, collect_scores, make_model, valid_scores
from scoring import evaluate_case, make_calibration


class PairedSolver(Solver):
    def __init__(self, config, kind):
        self.kind = kind
        self.config = dict(config)
        set_seed(config['seed'])
        super().__init__(config)
        dataset = self.train_loader.dataset
        sampler = EpochOrderSampler(len(dataset), config['seed'])
        self.train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler,
                                       num_workers=0, drop_last=False,
                                       generator=torch.Generator().manual_seed(config['seed']))

    def _build_model(self):
        self.model = make_model(self.kind, self.config).to(self.device)
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(key.encode('utf-8'))
        h.update(str((array.shape, array.dtype)).encode('ascii'))
        h.update(array.tobytes())
    return h.hexdigest()


def parse_curves(log, epochs):
    pattern = (r'Epoch: (\d+)/(\d+) \| L_train: (\S+)  Val: (\S+)  '
               r'momentum: (\S+)  Cost: ([\d.]+)s')
    rows = [dict(epoch=int(a), train_loss=float(c), validation_loss=float(d),
                 momentum=float(e), seconds=float(f)) for a,b,c,d,e,f in re.findall(pattern, log)]
    if ([r['epoch'] for r in rows] != list(range(1, epochs+1))
            or any(not np.isfinite([r['train_loss'],r['validation_loss']]).all() for r in rows)):
        raise ValueError('Frozen training log does not contain all finite epoch curves')
    return rows


def evaluate(solver, directory, progress):
    model = solver.model
    before = tensor_hash(model.state_dict())
    train = solver.train_loader.dataset.train
    test = solver.thre_loader.dataset.test
    labels = np.asarray(solver.thre_loader.dataset.test_labels).reshape(-1)
    w, k = solver.win_size, solver.lookahead
    phases = {}
    refs = {}
    for name,data,step in (('train_scores',train,solver.train_loader.dataset.step),
                           ('test_scores',test,w)):
        refs[name], phases[name] = collect_scores(model, data, w, step, True, progress=progress)
    scores, phases['pointwise_test'] = collect_scores(model, test, w, 1, False, progress=progress)
    targets = np.arange(w-1,len(test),dtype=np.int64)
    calibration = make_calibration(refs['train_scores'], refs['test_scores'], solver.anomaly_ratio)
    calibration.update(protocol='original_windows_valid_positions_train_test_percentile', uses_test_distribution=True)
    case = evaluate_case(scores, targets, labels, k, 'target', calibration)
    case.update(source_indices=targets-k, window_starts=targets-w+1,
                valid_mask=np.arange(len(test)) >= w-1)
    checks = []
    for target in (int(targets[0]),int(targets[len(targets)//2]),int(targets[-1])):
        x = torch.as_tensor(test[target-w+1:target+1], dtype=torch.float32, device=solver.device)[None]
        direct = valid_scores(model,x)[0,-1].item()
        extension = torch.cat([x, torch.full((1,3,x.shape[-1]),100.,device=solver.device)],1)
        future = valid_scores(model,extension)[0,w-k-1].item()
        np.testing.assert_allclose(direct,scores[target-w+1],rtol=1e-5,atol=2e-6)
        np.testing.assert_allclose(future,direct,rtol=1e-6,atol=2e-6)
        if solver.kind == 'raw':
            prediction = model.predict(x)[0,w-k-1]
            oracle = (prediction-x[0,w-1]).square().mean().item()
            np.testing.assert_allclose(oracle,direct,rtol=1e-6,atol=1e-7)
            changed = x.clone()
            changed[:,w-k:] = 123.
            torch.testing.assert_close(model.predict(changed)[0,w-k-1],prediction,rtol=0,atol=0)
        checks.append(dict(target=target,direct=direct,saved=float(scores[target-w+1]),future_extension=future))
    if tensor_hash(model.state_dict()) != before:
        raise RuntimeError('Model changed during scoring')
    np.savez_compressed(directory/'threshold_reference.npz', **refs)
    np.savez_compressed(directory/'pointwise_scores.npz', **{a:b for a,b in case.items() if a != 'summary'})
    return dict(summary=case['summary'],phases=phases,causality_samples=checks,state_unchanged=True,
                n_train_windows=len(solver.train_loader.dataset),train_step=solver.train_loader.dataset.step,
                n_train_observations=len(train),n_test_observations=len(test))
