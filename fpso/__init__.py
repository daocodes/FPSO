"""Regime-conditioned FPSO: a constraint-native metaheuristic whose search
behaviour and risk parameters are conditioned on a causally-estimated latent
market regime.

Package layout:

============= ================================================================
config        Frozen configuration objects and the YAML loader.
data          Panel acquisition/caching, universe selection, moment estimation.
regime        Feature construction, detectors, smoothing, parameter policy.
optimizer     Constraint repair, objective, and the FPSO search itself.
baselines     1/N and minimum variance under identical constraints.
backtest      Rebalance schedule, rolling engine, vectorbt ledger, results.
evaluation    Metrics, bootstrap/HAC inference, and the paper's figures.
experiments   Command-line entry points.
============= ================================================================

Importing this package pins BLAS to a single thread before numpy can load, so
that runs are bit-reproducible. See :mod:`fpso.determinism` for why that is not
optional here — multithreaded reductions made one arm of this study
irreproducible at 5e-14, and the harness's determinism check did not catch it.
"""

# Must precede any import that pulls in numpy: the thread-count variables are
# read when the BLAS library loads, and setting them afterwards has no effect.
from fpso.determinism import pin_blas_threads as _pin_blas_threads

_pin_blas_threads()

__version__ = "1.0.0"
