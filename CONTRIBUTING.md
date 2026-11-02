# Contributing

Corrections, bug reports and improvements are welcome.

* **Errata for the book**: open an issue with the chapter, page/equation number and the
  correction.
* **Code**: please keep the conventions of `pinnode/` (double precision, `Example` for seeds
  and outputs, `--quick` mode, figures through `pinnode.plotting`) and make sure
  `pytest -q` passes. New examples go into the chapter folder they belong to, with an entry
  in that folder's `README.md`.
* **Newer library versions**: if a script fails with a newer PyTorch/NumPy/SciPy release,
  please report the versions (`python -c "import pinnode.utils as u; print(u.environment_info())"`).
