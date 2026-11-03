# Reproducibility notes

Every number, table and figure in the book is produced by a script in `chapters/` with a
fixed seed. This page explains how to regenerate them and what level of agreement to expect.

## Environment

* Python 3.11-3.13, CPU only, **double precision**, one thread (`torch.set_num_threads(1)`).
* Exact versions used for the book: `requirements-lock.txt`
  (PyTorch 2.15.0, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, pandas 3.0.5).
* Minimum versions known to work: `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt            # or: pip install -r requirements.txt
```

On Linux, `pip install torch` pulls the CUDA build (~2-3 GB); the CPU build is enough:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Regenerating results

```bash
PYTHONPATH=. python chapters/ch07_integral_equations/<script>.py   # one example
python scripts/run_all.py --chapter ch07                         # one chapter
python scripts/run_all.py                                        # everything
python scripts/run_all.py --quick                                # smoke test (minutes)
pytest -q                                                        # automated tests
```

Outputs go to `outputs/<chapter>/<example>/`:

| file | content |
|------|---------|
| `results.json` | errors, hyperparameters, training time, seed, software versions |
| `history_*.csv` | loss components (and error metrics) during training |
| `*.csv` | predictions, tables |
| `figures/*.pdf`, `figures/*.png` | the figures used in the book (PDF) and README (PNG) |

## What agreement to expect

* Same machine and versions: identical results (deterministic kernels, fixed seeds).
* Different CPU / BLAS / PyTorch version: floating-point summation order changes, so
  training trajectories drift slightly. Final errors typically agree to within a factor of
  about 2 (the second significant digit can differ); qualitative conclusions do not change.
* `--quick` mode uses tiny iteration counts and is only a code-path test.

## Seeds

The default seed of each example is hard-coded in the script (`Example(..., default_seed=...)`),
printed at start-up and stored in `results.json`. Use `--seed N` to study seed sensitivity;
several chapters report statistics over seeds explicitly.

## Archiving

Releases are archived on Zenodo (metadata in `.zenodo.json`), so that the code version used
for the book (v2.1.0) remains available under a persistent DOI.
