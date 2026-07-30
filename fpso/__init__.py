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
"""

__version__ = "1.0.0"
