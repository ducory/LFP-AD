import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def definition_hash(node):
    # Python 3.12 adds empty generic-parameter fields absent from Python 3.9.
    for item in ast.walk(node):
        if hasattr(item, 'type_params') and not item.type_params:
            item._fields = tuple(field for field in item._fields if field != 'type_params')
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class DefinitionTests(unittest.TestCase):
    def test_original_definition_asts(self):
        manifest = json.loads((ROOT / 'provenance/source_manifest.json').read_text(encoding='utf-8'))
        for row in manifest['extracted_definitions']:
            node = ast.parse((ROOT / row['destination']).read_text(encoding='utf-8'))
            for part in row['destination_definition'].split('.'):
                node = next(item for item in node.body if getattr(item, 'name', None) == part)
            checksum = definition_hash(node)
            self.assertEqual(checksum, row['ast_sha256'], row['destination_definition'])


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),
                     'CPU model checks require the existing scientific environment')
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        torch.set_num_threads(2)

    def test_five_saved_checkpoints_load_strictly_and_score_finite_prefixes(self):
        from raw_helpers import make_model, valid_scores
        for dataset in ('SMD', 'MSL', 'SMAP', 'SWAT', 'PSM'):
            config = json.loads((ROOT / 'configs' / (dataset + '.json')).read_text(encoding='utf-8'))
            model = make_model('lfp', config).eval()
            state = self.torch.load(ROOT / 'checkpoints' / (dataset + '_checkpoint.pth'),
                                    map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            x = self.torch.zeros((2, 16, config['input_c']))
            with self.torch.no_grad():
                scores = valid_scores(model, x)
            self.assertEqual(tuple(scores.shape), (2, 16 - config['lookahead']))
            self.assertTrue(self.torch.isfinite(scores).all().item())

    def test_valid_scores_do_not_change_after_future_extension(self):
        from raw_helpers import make_model, valid_scores
        config = json.loads((ROOT / 'configs/MSL.json').read_text(encoding='utf-8'))
        model = make_model('lfp', config).eval()
        model.load_state_dict(self.torch.load(ROOT / 'checkpoints/MSL_checkpoint.pth',
                                              map_location='cpu', weights_only=True))
        self.torch.manual_seed(42)
        x = self.torch.randn(1, 16, config['input_c'])
        longer = self.torch.cat((x, self.torch.full((1, 3, config['input_c']), 100.)), 1)
        with self.torch.no_grad():
            before = valid_scores(model, x)
            after = valid_scores(model, longer)[:, :before.shape[1]]
        self.torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_endpoint_and_reference_collection_contract(self):
        from raw_helpers import collect_scores
        torch, np = self.torch, self.np
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lookahead = 1
                self.anchor = torch.nn.Parameter(torch.tensor(0.))
            def anomaly_score(self, x):
                return x[:, :, 0] + self.anchor * 0
        data = np.arange(8, dtype=np.float32)[:, None]
        scores, meta = collect_scores(ToyModel(), data, 4, 1, False, batch_size=2)
        np.testing.assert_array_equal(scores, np.arange(2, 7, dtype=np.float32))
        self.assertEqual(meta['first_target'], 3)
        refs, _ = collect_scores(ToyModel(), data, 4, 4, True, batch_size=2)
        np.testing.assert_array_equal(refs, np.array([0, 1, 2, 4, 5, 6], dtype=np.float32))

    def test_sampler_has_seed_plus_epoch_permutations(self):
        from raw_helpers import EpochOrderSampler
        sampler = EpochOrderSampler(17, 42)
        for epoch in range(2):
            observed = list(sampler)
            expected = self.torch.randperm(17, generator=self.torch.Generator().manual_seed(42 + epoch)).tolist()
            self.assertEqual(observed, expected)
        self.assertEqual(len(sampler.records), 2)

    def test_calibration_and_strict_threshold_decision(self):
        from scoring import evaluate_case, make_calibration
        np = self.np
        train = np.array([0, 1, 2], dtype=np.float32)
        test = np.array([3], dtype=np.float32)
        calibration = make_calibration(train, test, 50.)
        expected = np.percentile(np.concatenate((train, test)) / (np.float32(3) + 1e-8), 50.)
        self.assertEqual(calibration['threshold'], expected)
        scores = np.array([1.5, 3, 0, 3], dtype=np.float32)
        case = evaluate_case(scores, np.arange(4), np.array([0, 1, 0, 1]), 1, 'target', calibration)
        np.testing.assert_array_equal(case['predictions'], [0, 1, 0, 1])
        self.assertAlmostEqual(case['summary']['metrics']['pa_f_score'], 1., places=6)
        self.assertAlmostEqual(case['summary']['metrics']['Affiliation_F1'], 1., places=6)

    def test_no_prediction_events_do_not_become_fabricated_affiliation_zero(self):
        import warnings
        from scoring import pa_aff_metrics
        np = self.np
        with warnings.catch_warnings(record=True):
            metrics = pa_aff_metrics(np.zeros(4, dtype=int), np.array([0, 1, 0, 1]))
        self.assertTrue(np.isnan(metrics['Affiliation_F1']))


if __name__ == '__main__':
    unittest.main()
