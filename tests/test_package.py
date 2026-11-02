import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RATIOS = {'SMD': 0.6, 'MSL': 0.4, 'SMAP': 1.1, 'SWAT': 0.25, 'PSM': 1.35}


class PackageTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'reproduce.py'), *args],
                              capture_output=True, text=True)

    def api(self):
        path = ROOT / 'reproduce.py'
        self.assertTrue(path.is_file(), 'The portable reproduction entry is missing')
        spec = importlib.util.spec_from_file_location('release_entry', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_help_without_training_imports(self):
        result = self.cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('evaluate', result.stdout)
        self.assertIn('train', result.stdout)

    def test_final_configurations(self):
        for dataset, ratio in RATIOS.items():
            path = ROOT / 'configs' / (dataset + '.json')
            self.assertTrue(path.is_file(), str(path))
            config = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(config['dataset'], dataset)
            self.assertEqual(config['anomaly_ratio'], ratio)
            self.assertEqual(config['seed'], 42)
            self.assertEqual(config['win_size'], 100)
            self.assertFalse(Path(config['data_path']).is_absolute())
            self.assertFalse(Path(config['model_save_path']).is_absolute())

    def test_bundle_manifest(self):
        result = self.cli('verify')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('verified', result.stdout.lower())

    def test_reference_results_are_not_new_measurements(self):
        path = ROOT / 'reference_results.json'
        self.assertTrue(path.is_file(), str(path))
        reference = json.loads(path.read_text(encoding='utf-8'))
        self.assertFalse(reference['regenerated_in_this_package_preparation'])
        self.assertEqual(set(reference['datasets']), set(RATIOS))
        for row in reference['datasets'].values():
            self.assertIn('pa_f_score', row['metrics'])
            self.assertIn('Affiliation_F1', row['metrics'])
            self.assertTrue((ROOT / row['checkpoint']).is_file())

    def test_output_directory_must_be_new(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileExistsError):
                api.create_output(Path(name))

    def test_output_cannot_be_nested_in_input_data(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as name:
            data = Path(name)
            with self.assertRaises(ValueError):
                api.create_output(data / 'new-output', protected=[data])
            self.assertFalse((data / 'new-output').exists())

    def test_data_hash_mismatch_is_not_silently_accepted(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as name:
            data = Path(name)
            (data / 'x.npy').write_bytes(b'wrong-data')
            expected = {'x.npy': hashlib.sha256(b'expected-data').hexdigest()}
            with self.assertRaises(ValueError):
                api.check_data_files(data, expected)

    def test_nonfinite_json_is_explicitly_null(self):
        api = self.api()
        value = api.json_safe({'Affiliation_F1': float('nan'), 'nested': [float('inf')]})
        self.assertEqual(value, {'Affiliation_F1': None, 'nested': [None]})
        json.dumps(value, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
