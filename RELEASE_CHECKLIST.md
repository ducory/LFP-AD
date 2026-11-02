# Release checklist

Prepared on 2026-11-01; validation/release status updated on 2026-11-02.
The five fixed PA-F1/Aff-F1 workflows are locally validated, not publicly released.

## Included

- [x] Five fixed model/training configurations with portable paths and evaluated ratios.
- [x] Five frozen checkpoint files and source/data fingerprints.
- [x] Original model, preprocessing, sampler, training, valid-reference scoring and PA/Aff arithmetic.
- [x] Explicit train/evaluate commands; no automatic searches or dataset downloads.
- [x] Minimal Conda dependency specification and reproduction documentation.
- [x] No raw data, historical search archives, SSH configuration or manuscript files.

## Completed local validation

- [x] Complete CPU/runtime verification and retain its external audit report
  (CPU/synthetic checks with CUDA hidden; distinct from the full-data checks below).
- [x] Replay all five supplied frozen checkpoints using the new entry, exact
  prepared inputs and recorded CUDA environment; verify the saved outputs on CPU.
- [x] Train all five fixed models from scratch for all 60 planned epochs in total,
  then evaluate their own last-epoch checkpoints. No additional search was run.
- [x] Independently verify final tensors, sample orders, printed curves, complete
  score/reference/time-axis arrays, fixed thresholds, predictions and PA/Aff.
- [x] Document fixed-revision SMD/MSL/SMAP/PSM prepared downloads with matching
  content fingerprints, directory layout and read-only checks in DATA.md.

## Validation boundaries

- The runs used the existing pinned Conda environment, same hardware, exact data
  and seed 42. A fresh Conda installation, other hardware and multiple seeds
  have not been validated.
- The acceptance covers the five fixed LFP models' PA-F1/Aff-F1 workflows,
  not every manuscript baseline, ablation or historical metric column.
- Locally available SWaT inputs were verified, but official access plus a complete
  validated raw-to-prepared conversion is not yet fully documented.

## Required before public release

- [ ] Finish SWaT's official-access and exact preparation documentation, including
  a validated conversion to both archived CSV inputs. Do not redistribute the data.
- [ ] Review upstream notices/permissions for the historical loader and PA adaptations.
- [ ] Authors choose and approve repository and checkpoint licenses.
- [ ] Authors review the package, publish the chosen repository/version and add
  its actual URL/release tag to the manuscript. No URL is fabricated here.

Until the remaining release checks are completed, describe this as a locally
validated five-model PA-F1/Aff-F1 reproduction package, not as publicly released
code or validation of every experiment in the manuscript.
