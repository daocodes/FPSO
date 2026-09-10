# Regime Effects Are Turnover Effects

Controlled evaluation of regime-conditioned portfolio optimization on CRSP daily
equities, 2011-2024. Submitted to ICAIF '26.

A large literature reports that conditioning allocation on an inferred market
regime improves performance net of transaction costs. This study finds the effect
is explained by turnover: a persistent HMM posterior produces persistent
allocations, so the conditioned strategy trades less — whether or not the signal
knows anything. Trading less is what pays.

- The signal is injected at all three points where it can act on a mean-variance
  decision: **preferences**, **beliefs** about risk, **forecasts** of return.
- Against a naive baseline all three appear to work. Against an unconditional arm
  whose turnover penalty is *solved by bisection* to trade exactly as much, all
  three go negative.
- 32 arms, 30 matched seeds per arm, 4 transaction-cost levels.

Numbers and evidence: [docs/findings.md](docs/findings.md). Figures in
[paper/figures/](paper/figures), tables in [paper/tables/](paper/tables).

## Quick start

Runs offline on a synthetic panel — no credentials, no network, about a minute.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -m "not integration"
python -m fpso.experiments.build_cache --source synthetic --cache-dir cache
python -m fpso.experiments.run_study --quick --arms static regime_hmm
python -m fpso.experiments.analyze
```

Reproducing the paper's numbers needs WRDS access to CRSP:

```bash
cp .env.example .env          # WRDS_USERNAME / WRDS_PASSWORD
python -m fpso.experiments.build_cache        # one-time CRSP pull -> cache/
python -m fpso.experiments.calibrate_policy   # 2008-2010 holdout, then freeze
python -m fpso.experiments.run_study          # 32 arms x 30 seeds
python -m fpso.experiments.analyze            # -> paper/tables, paper/figures
```

The CRSP pull is the only step that touches the network.

## The arm matrix

Every treatment ships with its own controls, which is what makes this an
experiment rather than a backtest:

- **Treatments** — `regime_hmm` (preferences), `moments_hmm` (regime-weighted
  covariance), `tilt_hmm` (regime-tilted expected returns).
- **Turnover-matched controls** — `static_matched_*`, the decisive test.
- **Falsification** — `*_shuffled`, permuted label sequences preserving state
  marginals and dwell times. If a treatment cannot beat this, it is parameter
  jitter.
- **Ceilings** — `*_oracle`, detector fitted on the full sample. Acausal by
  construction, reported as an upper bound, never claimed.
- **No-timing** — `pinned_calm`, `tilt_constant`, separating parameter *levels*
  from *timing*.
- **Detector swaps** — `regime_gmm`, `regime_volq`. Does a learned detector beat
  volatility terciles?
- **Baselines and ablations** — `equal_weight`, `min_variance`,
  `ablation_no_pso`, `moments_longwindow`.

Arms share one parameter map and differ only in the detector, so comparisons
isolate the signal rather than two different policies. Seeds are matched across
arms, so every comparison is paired.

## Engineering notes

- **Lookahead is machine-checked, not asserted.** `test_lookahead.py` reruns the
  pipeline on a panel whose data after date *T* is replaced with garbage and
  asserts every decision at or before *T* is bit-identical. Parameterized over
  four detectors and three cut dates.
- **Seven of thirty-two arms did not reproduce.** Multithreaded BLAS varies its
  reduction order; across ~200 EM iterations that compounds into a posterior
  differing at ~1e-14. Arms taking `argmax` absorbed it, arms weighting
  continuously by the posterior passed it into `mu`/`Sigma` where the swarm
  amplified it. The determinism guarantee had been passing only because every
  mechanism tested until then quantized the detector's output.
- **Manifests hash the source, not just the git SHA.** `git status --porcelain`
  covers the whole tree, so regenerating a figure stamps a run `-dirty` — which
  happened to 450 of this study's first 719 runs.
- **A control silently degenerated and every test still passed.** The
  falsification arm collapsed to 97% CALM, and because CALM is the low-turnover
  configuration it looked like the best arm in the study. Marginals and switching
  rate are now assertions.
- **Only the support is combinatorial.** With the K names fixed, the weighting
  sub-problem is concave over a convex set. Solving it exactly beats the swarm at
  100% of 10,080 rebalances by a median of 43%, removes 46% of realized turnover,
  and changes the sign of no matched comparison.
- **Costs are per trade, turnover one-way.** Rotating a position generates a sale
  and a purchase, so realised drag is `2 x rate x one-way turnover`. One source of
  truth, pinned by a test against the simulator.
- Point-in-time universe, delisting returns compounded in, expanding (not
  full-sample) standardization, filtered (not smoothed) posteriors, HMM states
  canonicalized by volatility after every refit.

## Layout

```
fpso/
├── config/       frozen dataclasses + YAML loader with `extends:`
├── data/         panel sources (WRDS / Parquet / synthetic), universe, moments
├── regime/       features, HMM + alternatives + controls, smoothing, policy
├── optimizer/    constraint repair, objective, FPSO search, exact weighting
├── baselines/    1/N and minimum variance under identical constraints
├── backtest/     schedule, rolling engine, vectorbt ledger, results + manifest
├── evaluation/   metrics, bootstrap/HAC inference, figures, conviction
├── adaptive/     shrinkage policies with oracle and control ladder
└── experiments/  build_cache · calibrate_policy · match_turnover · run_study ·
                  analyze · verify_reproduction · convergence_probe
configs/          base.yaml, regime_policy.yaml, arms/*.yaml
paper/            camera-ready figures and result tables
tests/            unit (offline) and integration (WRDS)
```

Every result directory carries a `manifest.json` recording config, git SHA,
source hash, and library versions, so a figure traces to the code that made it.

## Tests

```bash
pytest -m "not integration"   # offline, no credentials. CI contract.
pytest -m integration         # hits live WRDS; needs .env
```

The ones carrying research weight: `test_lookahead.py` (causality),
`test_reproducibility.py` (same seed reproduces, *different* seeds actually
differ), `test_constraints.py` (property tests on repair under NaN/inf/all-zeros),
`test_controls.py` (falsification arm must match marginals and switching rate),
`test_regime_moments.py` (weighted estimator degrades to flat when the posterior
is uninformative, so a null stays reachable).

## Data

CRSP daily via WRDS: names ever top-200 by market cap between 2005 and 2024
(443 PERMNOs, 5,033 trading days), plus the value-weighted index for regime
features. The panel starts in 2005 so the 2011 evaluation window opens with a
standardized feature history and a satisfied HMM burn-in.

`cache/`, `results*/` and `.env` are not committed — no licensed data ships with
this repo. Rebuild with `build_cache`.
