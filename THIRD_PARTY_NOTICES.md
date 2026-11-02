# Third-party notices and license status

This is a local preparation, not a public licensing decision. There is currently
no repository-level LICENSE for the authors' code or the distributed checkpoints.
The copyright holders must approve the intended licenses before a public release.

## CfC

The encoder implementation follows the Closed-form Continuous-time Neural
Networks architecture. Relevant upstream implementation:
https://github.com/raminmh/CfC (Apache-2.0).
The upstream license is retained in `licenses/CfC-Apache-2.0.txt` for attribution;
this does not assert that the complete LFP-AD project has that license.

## Affiliation metrics

The affiliation metric implementation and its local fast equivalent build on
https://github.com/ahstat/affiliation-metrics-py (MIT).
The upstream license is retained in `licenses/affiliation-MIT.txt`.
The distributed files are the archived project implementation, not a claim that
every file is byte-identical to the current upstream repository.

## Dataset loaders and PA scoring

The historical loaders and PA scoring routines are retained from the experiment
snapshot, with their source fingerprints. The original project cited/reused
TSAD benchmark implementations. The exact upstream revision, notices and
redistribution permissions for these historical adaptations still need an
author/provenance review. Do not treat this preparation as legal clearance.

## Data and weights

The benchmarks remain subject to their owners' terms. No dataset is included.
The included five checkpoints are exact local author-supplied artifacts;
their public redistribution/license must be approved with the code release.
