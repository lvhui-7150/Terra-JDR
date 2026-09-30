# Terra-JDR: Code and Data

This repository contains the code and data for the terrain-aware joint
delivery--relay scheduling framework. The manuscript and its PDF/LaTeX sources
are intentionally not included.

## Repository Structure

```text
.
├── code/
│   ├── core/             Shared models, schedulers, bounds, and verification
│   ├── variants/         Problem-specific solver variants
│   ├── verification/     Independent re-computations
│   └── visualization/    Figure-generation code
├── data/
│   └── processed/        Processed model inputs and DEM
├── results/
│   ├── optimization/     Transport, joint, robust, and bound outputs
│   ├── verification/     Communication and resource verification records
│   └── experiments/      Algorithm-convergence and scenario experiments
├── tools/                Repository maintenance utilities
├── requirements.txt
└── CITATION.cff
```

## Environment

Python 3.11 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Main Code

- `code/core/single_site_model.py`: terrain-aware payload and batching.
- `code/core/transport.py`: flight physics and transport scheduling.
- `code/core/joint_schedule_model.py`: joint delivery--relay scheduling.
- `code/core/communication.py`: LOS and communication audits.
- `code/core/partition_integrated.py`: atomic task blocks and resource peaks.
- `code/core/makespan_global_bound.py`: route-workload lower-bound model.
- `code/core/replay.py`: independent replay and verification.

Problem-specific variants are in `code/variants/`. Independent checks are in
`code/verification/`.

## Reproduce Figures

From the repository root:

```powershell
python code/visualization/make_figures.py
```

Generated files are written to `figures/`. This directory is ignored by Git
because the repository is intended to contain code and data only.

## Reproducibility Scope

- CP-SAT is exact over the supplied candidate sets, not over every continuous
  trajectory and every possible box subset.
- Communication records are deterministic audits under the recorded
  link-budget and sampling parameters.
- The data correspond to the mountain flood case described by the code and
  documentation.

## Repository

`https://github.com/lvhui-7150/Terra-JDR`

## License

No open-source license has been selected yet. Add a `LICENSE` file before
publishing if reuse permissions are intended.
