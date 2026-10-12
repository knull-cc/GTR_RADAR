"""Runtime shims so the *unmodified* GTR code runs on current stacks (e.g. Colab, Python 3.12).

- numpy >= 2.0 removed the ``np.Inf`` alias, which ``utils/tools.py::EarlyStopping`` still uses.
  We restore the alias instead of editing the original file.

Import this module before anything from ``utils`` / ``exp`` is instantiated.
"""
import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # noqa: N816  (alias removed in numpy 2.0)
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # noqa: N816
