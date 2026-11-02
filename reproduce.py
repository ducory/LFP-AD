"""Portable entry for the five frozen LFP-AD PA/Aff configurations."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parent
DATASETS = ('SMD', 'MSL', 'SMAP', 'SWAT', 'PSM')
PACKAGES = {'torch': '2.4.0+cu124', 'numpy': '2.0.2', 'pandas': '2.3.3',
            'scikit-learn': '1.6.1', 'Pillow': '11.3.0', 'scipy': '1.13.1'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, 'item'):
        return json_safe(value.item())
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n',
                          encoding='utf-8')


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def verify_bundle():
    files = read_json(ROOT / 'artifact_manifest.json')['files']
    for name, checksum in files.items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents:
            raise ValueError('Invalid artifact manifest path: ' + name)
        if not path.is_file() or digest(path) != checksum:
            raise ValueError('Artifact missing or changed: ' + name)
    return {'status': 'verified', 'files': len(files)}


def check_data_files(directory, expected):
    directory = Path(directory).resolve()
    checked = {}
    for name, checksum in expected.items():
        if Path(name).name != name:
            raise ValueError('Expected a data filename, not a path')
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError('Required prepared data file missing: ' + str(path))
        actual = digest(path)
        if actual != checksum:
            raise ValueError('Prepared data SHA256 differs: ' + name +
                             '. See DATA.md; raw upstream data is not necessarily this merged array.')
        checked[name] = actual
    return checked


def create_output(path, protected=()):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('Output must be a new directory: ' + str(path))
    for source in protected:
        source = Path(source).resolve()
        if path == source or source in path.parents:
            raise ValueError('Output must not be inside protected input: ' + str(source))
    path.mkdir(parents=True, exist_ok=False)
    return path


def runtime_environment():
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if platform.python_version() != '3.9.25' or versions != PACKAGES:
        raise RuntimeError('Use the pinned Conda environment. Found: ' +
                           json.dumps({'python': platform.python_version(), 'packages': versions}))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Full reproduction requires the recorded CUDA environment; CPU smoke tests are separate.')
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError('The recorded computation uses FP32')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'python': platform.python_version(), 'packages': versions,
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
            'threads': 2, 'amp': False, 'tf32': False}


def reference_comparison(dataset, checkpoint, result):
    reference = read_json(ROOT / 'reference_results.json')['datasets'][dataset]
    if digest(checkpoint) != reference['checkpoint_sha256']:
        return {'canonical_checkpoint': False,
                'note': 'Newly trained/custom checkpoint; archived numbers are not its measured results.'}
    summary = result['summary']
    observed = {key: summary['metrics'][key] for key in ('pa_f_score', 'Affiliation_F1')}
    observed['normalized_threshold'] = summary['calibration']['threshold']
    expected = {key: reference['metrics'][key] for key in ('pa_f_score', 'Affiliation_F1')}
    expected['normalized_threshold'] = reference['normalized_threshold']
    differences = {key: float(observed[key]) - expected[key] for key in expected}
    return {'canonical_checkpoint': True, 'observed': observed, 'archived': expected,
            'differences': differences,
            'all_reported_scalars_exactly_equal': all(value == 0 for value in differences.values())}


def run_job(args):
    verify_bundle()
    dataset_info = read_json(ROOT / 'data_manifest.json')[args.dataset]
    data = Path(args.data_root).resolve() / dataset_info['directory']
    data_hashes = check_data_files(data, dataset_info['files'])
    environment = runtime_environment()
    config = read_json(ROOT / 'configs' / (args.dataset + '.json'))
    checkpoint = None
    if args.command == 'evaluate':
        checkpoint = (Path(args.checkpoint).resolve() if args.checkpoint else
                      ROOT / 'checkpoints' / (args.dataset + '_checkpoint.pth'))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    output = create_output(args.output, protected=[data, ROOT / 'src', ROOT / 'checkpoints', ROOT / 'configs'])
    config.update(data_path=str(data), model_save_path=str(output), mode=args.command)
    save_json(output / 'effective_config.json', config)
    save_json(output / 'environment.json', environment)
    save_json(output / 'data_hashes.json', data_hashes)
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from engine import PairedSolver, evaluate, parse_curves, tensor_hash
    status = output / 'status.json'
    save_json(status, {'stage': args.command, 'status': 'running'})
    try:
        log_path = output / ('training.log' if args.command == 'train' else 'evaluation.log')
        with log_path.open('x', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                solver = PairedSolver(config, 'lfp')
                if args.command == 'train':
                    solver.train()
                    checkpoint = output / (args.dataset + '_checkpoint.pth')
                    saved = torch.load(checkpoint, map_location=solver.device, weights_only=True)
                    if tensor_hash(saved) != tensor_hash(solver.model.state_dict()):
                        raise RuntimeError('Saved checkpoint differs from the final model')
                    if not all(torch.isfinite(value).all().item() for value in saved.values()):
                        raise RuntimeError('Nonfinite checkpoint tensor')
                    order = solver.train_loader.sampler.records
                    if len(order) != config['num_epochs']:
                        raise RuntimeError('Not all recorded training epochs completed')
                    save_json(output / 'training_order.json', order)
                    result = {'checkpoint_sha256': digest(checkpoint), 'checkpoint_rule': 'last',
                              'completed_epochs': len(order), 'evaluation_performed': False}
                else:
                    checksum = digest(checkpoint)
                    saved = torch.load(checkpoint, map_location=solver.device, weights_only=True)
                    solver.model.load_state_dict(saved, strict=True)
                    if not all(torch.isfinite(value).all().item() for value in saved.values()):
                        raise RuntimeError('Nonfinite checkpoint tensor')
                    def progress(value):
                        save_json(status, {'stage': 'evaluate', 'status': 'running', **value})
                    result = evaluate(solver, output, progress)
                    if digest(checkpoint) != checksum:
                        raise RuntimeError('Input checkpoint changed during evaluation')
                    result['checkpoint_sha256'] = checksum
                    result['reference_comparison'] = reference_comparison(args.dataset, checkpoint, result)
                    metrics = result['summary']['metrics']
                    if not all(math.isfinite(float(value)) for value in metrics.values()):
                        result['undefined_metric_note'] = (
                            'Undefined metric values are saved as null; predictions and thresholds were not altered.')
        if args.command == 'train':
            curves = parse_curves(log_path.read_text(encoding='utf-8'), config['num_epochs'])
            save_json(output / 'curves.json', {'rows': curves, 'source': 'original printed training log'})
        save_json(output / 'result.json', result)
        save_json(status, {'stage': args.command, 'status': 'complete'})
        return {'status': 'complete', 'command': args.command, 'output': str(output)}
    except BaseException as error:
        save_json(status, {'stage': args.command, 'status': 'failed', 'error': str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='Show the archived reference metrics; does not calculate metrics')
    commands.add_parser('verify', help='Verify package/checkpoint fingerprints without importing training dependencies')
    check = commands.add_parser('check-data', help='Read-only verification of the exact prepared dataset files')
    check.add_argument('--dataset', choices=DATASETS, required=True)
    check.add_argument('--data-root', type=Path, required=True)
    for name in ('train', 'evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--dataset', choices=DATASETS, required=True)
        command.add_argument('--data-root', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'evaluate':
            command.add_argument('--checkpoint', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'verify':
        result = verify_bundle()
    elif args.command == 'list':
        verify_bundle()
        result = read_json(ROOT / 'reference_results.json')
    elif args.command == 'check-data':
        verify_bundle()
        info = read_json(ROOT / 'data_manifest.json')[args.dataset]
        result = {'dataset': args.dataset, 'verified_data_files':
                  check_data_files(args.data_root / info['directory'], info['files'])}
    else:
        result = run_job(args)
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError,
            importlib.metadata.PackageNotFoundError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
