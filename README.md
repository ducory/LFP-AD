# LFP-AD: minimal reproduction package

This repository contains the five frozen LFP-AD models and the training
and scoring code needed for their target-aligned PA-F1/Aff-F1 evaluation.
It is not a reproduction package for every baseline, ablation or historical
R-AUC/VUS column in the manuscript.

**Status:** public code snapshot at [ducory/LFP-AD](https://github.com/ducory/LFP-AD),
published on 2026-11-02 and locally validated for the five fixed LFP-AD PA-F1/Aff-F1
experiments. Both supplied-checkpoint evaluation and training from
scratch followed by evaluation passed in the recorded environment. See
[Validation status](#validation-status) and [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
No raw dataset files are included.

## Contents

- `src/cfcdino/`: unchanged archived model implementation.
- `src/data_factory/`: unchanged archived loaders and preprocessing.
- `src/training.py`, `src/engine.py`, `src/runtime.py`, `src/scoring.py`:
  original definitions extracted verbatim with portable imports.
- `src/raw_helpers.py`: original sampler and valid-position score collector.
- `src/metrics/`: the PA and affiliation implementations actually used.
- `configs/`: five final, fixed configurations with portable paths.
- `checkpoints/`: five exact frozen checkpoints, with SHA256 fingerprints.
- `reference_results.json`: archived fixed operating points and thresholds.
- `data_manifest.json`: expected fingerprints of the prepared dataset files.
- `provenance/`: source fingerprints, definition AST fingerprints and the
  limited configuration-field substitutions.
- `reproduce.py`: explicit preparation, training and evaluation commands.

## Environment

The recorded training environment is Linux, Python 3.9.25, PyTorch 2.4.0+cu124,
NumPy 2.0.2 and an NVIDIA RTX 4090. Supporting package versions were read from
the existing training Conda environment on 2026-11-01. `environment.yml` is a
minimal dependency specification, not a complete OS/driver/Conda lockfile.

```bash
conda env create -f environment.yml
conda activate lfpad-repro
python -B reproduce.py verify
```

The package does not create or modify a Conda environment automatically.
Training and real-data evaluation require the recorded Python/package versions
and CUDA. CPU checks below are smoke tests, not substitutes for CUDA result
reproduction. A different GPU, driver or environment can change numerical results.

## Data

Read [DATA.md](DATA.md) before running. It provides fixed-version prepared-file
links for SMD, MSL, SMAP and PSM, along with the required directory layout and
hash checks. SWaT requires official access; its complete raw-to-prepared
conversion remains an outstanding release item. Place the **prepared files used
in this evaluation** under a separate data root. Do not assume that an arbitrary
raw download has the same concatenation, preprocessing or file bytes. A file-hash
mismatch is an error, not permission to silently change the data protocol.

```text
dataset/
  SMD/SMD_train.npy, SMD_test.npy, SMD_test_label.npy
  MSL/MSL_train.npy, MSL_test.npy, MSL_test_label.npy
  SMAP/SMAP_train.npy, SMAP_test.npy, SMAP_test_label.npy
  PSM/train.csv, test.csv, test_label.csv
  SWaT/swat_train2.csv, swat2.csv
```

```bash
python -B reproduce.py check-data --dataset MSL --data-root /path/to/dataset
```

## Evaluate a supplied frozen checkpoint

The following command performs neural scoring when explicitly executed. This
evaluation entry has been validated on all five datasets in the recorded
environment. Choose a new output directory for each invocation; it must not exist.

```bash
python -B reproduce.py evaluate --dataset MSL --data-root /path/to/dataset --output outputs/MSL/evaluate-001
```

The dataset choices are `SMD`, `MSL`, `SMAP`, `SWAT` and `PSM`. `SWAT` uses the
`SWaT` data-directory spelling. Each invocation loads that dataset's fixed
configuration. Scoring always uses batch size 512 and the archived, dataset-specific
percentile ratio; it does not search ratios or optimize a metric.

Outputs include `pointwise_scores.npz`, `threshold_reference.npz`, `result.json`,
`effective_config.json`, `environment.json`, `data_hashes.json`, a log and status.
The result compares the new measurements with the archived scalars; a difference
is reported rather than replaced by a reference value. Undefined metric values
are represented by JSON `null` without changing predictions or thresholds.

## Train once at a fixed configuration

This is an explicit, potentially long-running training command. It is not
executed by `verify`, `list`, `check-data` or the test suite.

```bash
python -B reproduce.py train --dataset MSL --data-root /path/to/dataset --output outputs/MSL/train-001
python -B reproduce.py evaluate --dataset MSL --data-root /path/to/dataset --checkpoint outputs/MSL/train-001/MSL_checkpoint.pth --output outputs/MSL/retrained-evaluate-001
```

The original loop, Adam update, gradient clipping, learning-rate schedule, EMA,
centering and `seed + epoch` permutation sampler are retained. The saved model
is the last epoch, not a best-validation checkpoint. A custom/newly trained
checkpoint is distinguished from the distributed frozen checkpoint.

## Evaluation contract

- Standardization is fitted on the complete training array by the original loader.
- Existing merged-array organization is retained; sequence/entity boundaries are
  not reconstructed or reset by this package.
- Primary scoring uses stride-one complete windows and one endpoint score per
  window. The first evaluated zero-based target is `W - 1 = 99`.
- Calibration uses pooled training and unlabelled test reference scores. Only
  `W - k` genuine prediction positions per reference window are retained.
- Reference train-window stride is 100 for SMD and 1 for the other four datasets;
  test-reference stride is `W`. Overlapping reference scores are not deduplicated.
- The reference min/max normalization and percentile calculation are unchanged.
  Predictions use **normalized score strictly greater than normalized threshold**.
  The stored raw threshold is provenance only, not a replacement decision rule.
- The normal-reference distribution includes unlabelled test data, so this is
  not an entirely online, test-independent calibration protocol.
- Dataset-specific configurations were selected using test detection performance
  and then fixed. They are not an independent normal-validation selection result.
- Validation loaders are historical: MSL/SMAP/PSM use test inputs; SMD uses a suffix
  of the training array, and SWaT uses its historical training suffix. These are
  not independent holdouts and do not select the saved last-epoch checkpoint.
- Results use seed 42 and do not quantify variation over random initializations.
- This entry calculates PA-F1 and Aff-F1 only. It does not compute or reinterpret
  the manuscript's historical binary-input R-AUC/VUS values.

## Archived reference points

These are the unchanged archived fixed operating points. Subsequent local
checkpoint replays and fresh-training evaluations reproduced them in the recorded
environment. `reference_results.json` remains the original reference record, not
a replacement for newly measured outputs. Units are percent.

| Dataset | PA-F1 | Aff-F1 | Upper-tail percentage |
| --- | ---: | ---: | ---: |
| SMD | 86.986900 | 77.116105 | 0.60 |
| MSL | 94.991342 | 76.099636 | 0.40 |
| SMAP | 94.533363 | 78.054079 | 1.10 |
| SWaT | 97.638378 | 76.488948 | 0.25 |
| PSM | 96.326060 | 82.440479 | 1.35 |

```bash
python -B reproduce.py list
python -B -m unittest discover -s tests -v
```

The tests check source-definition identity, safe output behavior, checkpoints,
synthetic score alignment and metric arithmetic. They do not run training or
evaluate a real benchmark. They create temporary test files only; no real data
files or existing result directories are modified.

## Validation status

The following local validation was completed on 2026-11-01 using the existing
recorded Linux Conda environment, RTX 4090, exact prepared data and seed 42:

- Full CUDA evaluation of all five supplied frozen checkpoints, followed by
  independent CPU verification of saved scores, thresholds, predictions and PA/Aff.
- Training all five fixed models from scratch for their complete epoch counts
  (60 epochs in total), then evaluating each model's own last-epoch checkpoint.
- Exact comparison of final model tensors, full sample-order records, printed
  loss/momentum curves and complete score/reference/time-axis arrays with the
  archived runs, plus independent CPU recomputation of PA-F1/Aff-F1.

Fresh checkpoint files have different ZIP root names from the archived containers,
so their whole-file hashes differ; final tensors and every internal archive
payload were compared exactly. This metadata distinction did not permit changes
to numerical weights or scores. Detailed execution/audit records are retained
separately; they are not bundled with private server paths in this package.

These checks validate the five fixed PA-F1/Aff-F1 workflows under the recorded
conditions. They are not a clean Conda installation, cross-hardware, multi-seed
or all-manuscript-experiment validation. No new environment was created for
these runs, and the environment creation instructions above have not been
independently tested in a fresh environment.

CPU checkpoint/synthetic tests also passed in the existing Linux Conda environment
with CUDA hidden. The older affiliation implementation can emit docstring
escape-sequence warnings; it is retained unchanged rather than being silently
rewritten during packaging. The unit tests and full-data validation are separate
checks; running the unit tests does not start a benchmark or training job.

## Release notes

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the release checklist.
The authors must choose licenses for their code and model weights and finish the
attribution review. No repository-level license has been invented or granted
by preparing this folder. Do not commit restricted data, credentials or private
server paths. This public snapshot does not resolve the outstanding license,
attribution or SWaT preparation items. No tagged release has been created.
