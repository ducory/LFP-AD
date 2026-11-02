# Data sources and exact prepared-file contract

No raw or processed benchmark data is redistributed in this package.
`data_manifest.json` lists the exact files consumed by the archived loaders and
their expected SHA256 values. Matching prepared downloads have been identified
for SMD, MSL, SMAP and PSM. SWaT access and preparation are handled separately below.

## Original sources

- SMD: [OmniAnomaly / ServerMachineDataset](https://github.com/NetManAIOps/OmniAnomaly).
- MSL and SMAP: [Telemanom](https://github.com/khundman/telemanom).
- PSM: [eBay RANSynCoders](https://github.com/eBay/RANSynCoders), `data/`.
- SWaT: [SUTD iTrust dataset access](https://www.sutd.edu.sg/itrust/itrust-labs/datasets).
  Request the appropriate dataset and comply with its access/use terms.

These are the original dataset sources, not a claim that their current raw
downloads equal the prepared inputs used here. Retain the original dataset
citations and comply with the applicable owners' terms. The matching prepared
files below are an additional reproduction source, not a replacement attribution.

## Exact prepared downloads: SMD, MSL, SMAP and PSM

The [official Time-Series-Library repository](https://github.com/thuml/Time-Series-Library)
links to the THUML-hosted [prepared-data mirror](https://huggingface.co/datasets/thuml/Time-Series-Library).
Use the fixed data-repository revision
`2b66e59ee19dac8f6f19fb5d4997f289fdfea357`, rather than the moving `main` branch.
Download only the three files listed for each dataset:

| Dataset | Training observations | Test observations | Test labels |
| --- | --- | --- | --- |
| SMD | [SMD_train.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/SMD/SMD_train.npy) | [SMD_test.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/SMD/SMD_test.npy) | [SMD_test_label.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/SMD/SMD_test_label.npy) |
| MSL | [MSL_train.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/MSL/MSL_train.npy) | [MSL_test.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/MSL/MSL_test.npy) | [MSL_test_label.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/MSL/MSL_test_label.npy) |
| SMAP | [SMAP_train.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/SMAP/SMAP_train.npy) | [SMAP_test.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/SMAP/SMAP_test.npy) | [SMAP_test_label.npy](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/SMAP/SMAP_test_label.npy) |
| PSM | [train.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/PSM/train.csv) | [test.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/PSM/test.csv) | [test_label.csv](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/2b66e59ee19dac8f6f19fb5d4997f289fdfea357/PSM/test_label.csv) |

The content fingerprints of all 12 files match `data_manifest.json`. The source
audit compared the mirror's Git LFS SHA256 metadata for 11 files and hashed the
actual HTTP response for the non-LFS `PSM/test_label.csv` in memory. It did not
download all 12 files again or rerun any experiment. This establishes a matching
public source, not which URL was used for the original historical download.

The SMD/MSL/SMAP files are already merged arrays. Do not concatenate, reorder,
filter channels or apply a new preprocessing pipeline before passing these exact
files to the existing loaders. The historical raw-to-merged conversion has not
been reconstructed; it is not needed when using the matching prepared downloads.

### Place and verify the files

Use a separate data root; do not overwrite existing experimental inputs. Preserve
these directory and file names:

```text
<data-root>/
  SMD/SMD_train.npy, SMD_test.npy, SMD_test_label.npy
  MSL/MSL_train.npy, MSL_test.npy, MSL_test_label.npy
  SMAP/SMAP_train.npy, SMAP_test.npy, SMAP_test_label.npy
  PSM/train.csv, test.csv, test_label.csv
```

Run from the package root, replacing `/path/to/dataset` with that data root:

```bash
python -B reproduce.py check-data --dataset SMD --data-root /path/to/dataset
python -B reproduce.py check-data --dataset MSL --data-root /path/to/dataset
python -B reproduce.py check-data --dataset SMAP --data-root /path/to/dataset
python -B reproduce.py check-data --dataset PSM --data-root /path/to/dataset
```

`check-data` only reads and hashes the files. It does not train, score, normalize
or modify them. Each command must exit successfully and list all three files
under `verified_data_files` before proceeding. The original loaders apply their
train-fitted standardization at runtime.

## SWaT: official access and outstanding preparation check

Request SWaT through [SUTD iTrust](https://www.sutd.edu.sg/itrust/itrust-labs/datasets)
and follow its [terms of usage](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/terms-of-usage/).
Those terms prohibit sharing the dataset with others, privately or publicly.
Do not include SWaT data in this repository or redistribute it through another
download service. Availability in a third-party mirror does not replace these terms.

The historical data directory contains `SWaT_Dataset_Normal_v1.xlsx` and
`SWaT_Dataset_Attack_v0.xlsx`. The evaluation consumes these prepared files:

```text
<data-root>/SWaT/swat_train2.csv
<data-root>/SWaT/swat2.csv
```

**Outstanding:** a complete, validated conversion from the officially obtained
XLSX files to both exact CSV inputs is not yet documented. The existing `data.py`
contains commented training-conversion fragments, not a complete executable
training-and-test preparation pipeline. Row selection, timestamp removal,
column order, label encoding and CSV serialization must be verified before
claiming that this conversion reproduces the archived files. Do not invent or
silently substitute a different conversion to bypass the file checks.

Users who already possess the exact authorized prepared inputs can check them:

```bash
python -B reproduce.py check-data --dataset SWAT --data-root /path/to/dataset
```

The CLI spelling is `SWAT`; the data-directory spelling is `SWaT`. A passing
fingerprint check verifies file identity, not data-access permission or a new
raw-to-prepared reproduction. SWaT's access/preparation item remains open.

## Loader expectations

- SMD/MSL/SMAP: `<DATASET>_train.npy`, `<DATASET>_test.npy` are time-by-channel
  arrays; `<DATASET>_test_label.npy` contains aligned binary test labels.
- PSM: `train.csv`, `test.csv`, `test_label.csv` use the original CSV header and
  column layout. The loader drops the first column of each file and applies
  `numpy.nan_to_num` to the observation arrays before standardization.
- SWaT: `swat_train2.csv` and `swat2.csv` contain feature columns followed by one
  final label column. The loader removes the final column from both observation
  matrices and takes test labels from the final column of `swat2.csv`.
- Channel counts in the fixed configurations are 38 (SMD), 55 (MSL), 25 (SMAP),
  51 (SWaT) and 25 (PSM). Exact file fingerprints remain the definitive input
  check; matching shape alone is not enough.
- Train-fitted `StandardScaler`, the historical validation sources and existing
  merged-series boundaries are unchanged. Command/context channels are not
  selectively removed for MSL/SMAP.

Do not rename an arbitrary upstream array to make it look like a prepared file.
If a fingerprint differs, investigate the historical preparation first rather
than silently accepting a different evaluation dataset.
