# Regime-Conditioned FPSO

Research code for **"Regime-Adaptive Hybrid Swarm Optimization for
Cardinality-Constrained Portfolio Selection"** (ICAIF '26 submission).

**Research question (RQ3).** *Does conditioning a constraint-native
metaheuristic's search behaviour and risk parameters on a causally-estimated
latent market regime improve risk-adjusted out-of-sample performance over a
static parameterization?*

A Gaussian HMM estimates the current market state (calm / turbulent / crisis)
from causal volatility and drawdown features at each monthly rebalance. That
estimate can enter the optimizer at two different points, and the study tests
both:

**Mechanism 1 — regime-conditioned preferences.** The detected regime selects an
FPSO parameter configuration: exploration rate, attraction decay, risk aversion,
turnover penalty, cardinality and position caps.

**Mechanism 2 — regime-conditioned beliefs.** The detector's posterior weights
each historical day by how much its latent state resembles today's, and the
covariance is estimated from that weighted sample. In a crisis, Sigma is built
mostly from past crisis days, so the optimizer *sees* elevated risk and
correlation in its inputs rather than being told to fear them through a penalty
weight.

Mechanism 1 was implemented first and produced a null result; the oracle arm
showed the ceiling was not label quality but the injection point, which is what
motivated mechanism 2. See **[docs/findings.md](docs/findings.md)** for the full
result and its evidence.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline: no WRDS credentials, no network. Verifies the whole pipeline.
pytest -m "not integration"
python -m fpso.experiments.build_cache --source synthetic --cache-dir cache
python -m fpso.experiments.run_study --quick --arms static regime_hmm
python -m fpso.experiments.analyze
```

To reproduce the paper's numbers you need WRDS access to CRSP:

```bash
cp .env.example .env          # then fill in WRDS_USERNAME / WRDS_PASSWORD
python -m fpso.experiments.build_cache            # one-time CRSP pull -> cache/
python -m fpso.experiments.calibrate_policy       # 2008-2010 holdout, then freeze
python -m fpso.experiments.run_study              # 11 arms
python -m fpso.experiments.analyze                # tables -> paper/tables,
                                                  # figures -> paper/figures
```

The CRSP pull is the only step that touches the network. Everything after it
runs from `cache/`, which is what makes the study reproducible and what makes an
11-arm x 30-seed matrix affordable.

---

## What the experiment is

| # | Arm | Detector | What it establishes |
|---|-----|----------|---------------------|
| 1 | `static` | none | The ablation baseline: one parameterization for all states |
| 2 | `regime_hmm` | Gaussian HMM | **Headline** |
| 3 | `regime_gmm` | Gaussian mixture | Does Markov *persistence* matter, or is clustering enough? |
| 4 | `regime_volq` | volatility terciles | Does the *learned* detector beat a transparent rule? |
| 5 | `regime_shuffled` | matched-marginal noise | **Falsification.** Same parameter variation, no signal |
| 6 | `regime_oracle` | HMM on the full sample | Lookahead upper bound. Reported, never claimed |
| 7 | `equal_weight`, `min_variance` | — | Standard external baselines |
| 8 | `pinned_calm`, `pinned_crisis` | constant | Separates parameter *levels* from *timing* |
| 9 | `moments_hmm` | Gaussian HMM | **Mechanism 2**: regime-weighted covariance |
| 10 | `moments_oracle`, `moments_shuffled` | — | Ceiling and falsification for mechanism 2 |
| 11 | `moments_longwindow` | none | Isolates window length from regime weighting |
| 12 | `static_matched_*` | none | **Turnover-matched controls** — the decisive test |

Arms 2-6 share **one** parameter map and differ only in the detector, so the
comparison isolates the regime signal rather than two different policies. Arm 5
is what makes this an experiment: if arm 2 does not beat it, the mechanism is
parameter jitter, and the paper says so.

Three further arms round out the matrix. `ablation_no_pso` (pure firefly) and
`ablation_annual` (the pre-refactor yearly cadence) measure the two changes this
refactor made to the method, rather than asserting them. `regime_hmm_calibrated`
re-runs the headline under the *pre-sample-calibrated* parameter map instead of
the a-priori one — if the conclusion survives that swap, it does not hinge on how
those twelve numbers were chosen.

Seeds are **matched across arms**, so every comparison is paired.

---

## Design decisions a reviewer will ask about

**Causality is enforced structurally and machine-checked.** The backtest engine
is the only component that touches dates; every other component receives a
pre-sliced frame. `tests/unit/test_lookahead.py` runs the pipeline, then reruns
it on a panel whose observations after date *T* have been replaced with garbage,
and asserts that every decision at or before *T* — weights, regime label,
parameters, objective value — is bit-identical. It is parameterized over four
detectors and three cut dates.

**Feature standardization is expanding, not full-sample.** Z-scoring against
full-sample moments would let 2020 leak into a 2011 observation. The lookahead
test covers this.

**HMM states are canonicalized.** Fitted state indices are arbitrary and permute
across refits. States are sorted by mean realized volatility and relabelled
CALM < TURBULENT < CRISIS after every fit, so the regime-to-parameter map means
the same thing in 2013 and in 2019.

**Inference is filtered, not smoothed.** At the last observation of a sequence
the smoothed posterior equals the filtered one, so `predict_proba(X[:t+1])[-1]`
is exactly `P(s_t | x_{1:t})`. Smoothed label paths are used only for figures and
are marked as such in the captions.

**The parameter map never saw the evaluation sample.** The shipped headline uses
the a-priori map in `configs/regime_policy.yaml`: twelve values set from economic
reasoning and fitted to nothing at all. `calibrate_policy.py` provides the
alternative — coordinate search on a 2008-2010 pre-sample holdout spanning the
crisis, the recovery and a calm stretch — and writes
`configs/regime_policy.frozen.yaml` with a timestamp, a git SHA and the full
search trace. It moved 3 of the 12 values. Both are run, as `regime_hmm` and
`regime_hmm_calibrated`, so the sensitivity is reported rather than assumed away.

(The holdout opens in 2008 rather than at the 2005 panel start because the
detector needs roughly three years of warm-up: 252 days of rolling volatility and
drawdown, then 252 observations of expanding standardization, then 252 more to
identify the HMM. Calibrating from 2005 would have spent most of the window in
burn-in, comparing configurations that were in fact identical.)

**Monthly rebalancing.** ~168 decisions over 2011-2024 rather than the 14 an
annual cadence gives. A regime switch that can fire 14 times in fifteen years is
not a regime switch — COVID would be one observation.

**The universe is point-in-time.** Names are ranked by market capitalisation *at
the rebalance date*. The previous rule took the first 50 PERMNOs, which is
approximately "the 50 oldest-listed survivors" — survivorship- and age-biased.
`FirstPermnoUniverse` retains the old rule so the bias can be measured.

**Delistings cost what they cost.** Delisting returns from `crsp.dsedelist` are
compounded into a name's final observation. Filling those days with 0.0 would
make exiting the sample free.

**Repair, not penalty, enforces feasibility.** Every candidate is projected onto
the feasible set before scoring, so the penalty terms are identically zero along
the search path and `Phi(w) == R(w)` in practice. This is asserted in
`tests/unit/test_objective.py` rather than assumed, and the paper describes
repair as the operative constraint mechanism.

**FPSO now contains PSO.** The pre-refactor optimizer carried `velocity` and
`best_weights` on each particle but never read them: the search was pure firefly,
while the paper described a firefly/PSO hybrid. The PSO velocity term is
implemented as a fixed, non-learned blend (`pso_blend`); setting it to 0 recovers
the old behaviour, which is shipped as the `ablation_no_pso` arm.

**Accounting is delegated to vectorbt.** Target weights go through
`Portfolio.from_orders(size_type='targetpercent')`, which simulates the actual
orders. The previous ledger charged `tc_rate * turnover` against the first day of
each window, which ignored drift between rebalances and mis-timed the cost.

---

## Layout

```
fpso/
├── config/       frozen config dataclasses + YAML loader with `extends:`
├── data/         panel sources (WRDS / Parquet / synthetic), universe,
│                 flat and regime-weighted moment estimators
├── regime/       features, HMM + alternatives + controls, smoothing, policy
├── optimizer/    constraint repair, objective, FPSO search
├── baselines/    1/N and minimum variance under identical constraints
├── backtest/     schedule, rolling engine, vectorbt ledger, results + manifest
├── evaluation/   metrics, bootstrap/HAC inference, figure builders
└── experiments/  build_cache · calibrate_policy · match_turnover ·
                  run_study · analyze
configs/          base.yaml, regime_policy.yaml, arms/*.yaml
paper/figures/    F1-F9, PDF (camera-ready) and PNG
paper/tables/     headline / comparison / regime-conditional / cost CSVs
tests/            unit (offline) and integration (WRDS) suites
```

Every result directory carries a `manifest.json` recording the config, git SHA,
Python version and the versions of numpy / pandas / scipy / scikit-learn /
hmmlearn / vectorbt, so a figure can always be traced to the code that made it.

`algorithm/`, `backtester/` and `run_backtest.py` are the pre-refactor modules.
They are superseded by `fpso/` and are left in place only because the working
tree has an unfinished merge; delete them once that is resolved.

---

## Tests

```bash
pytest -m "not integration"   # offline; no network, no credentials. CI contract.
pytest -m integration         # hits live WRDS; needs .env
```

The three that carry research weight:

* **`test_lookahead.py`** — the causality proof described above.
* **`test_reproducibility.py`** — same seed reproduces exactly; *different* seeds
  actually differ (a dropped RNG would silently turn 30 runs into 1); a
  rebalance's random stream does not depend on its position in the schedule.
* **`test_constraints.py`** — property tests on the repair operator: budget, box,
  cardinality, idempotence, and deterministic tie-breaking, under adversarial
  inputs including all-zeros, NaN and infinities.
* **`test_controls.py`** — the falsification control must match the treatment
  arm's label marginals *and* switching rate. The first version of that control
  silently degenerated into a static arm and every other test still passed, so
  the comparison is now an assertion.
* **`test_regime_moments.py`** — the regime-weighted estimator must degrade to
  the flat one when the posterior is uninformative, so a null result stays
  reachable and cannot be manufactured by the plumbing.

---

## Data

CRSP daily stock file via WRDS. The cached panel is the union of names that were
ever top-200 by market capitalisation between 2005 and 2024 (443 PERMNOs,
5,033 trading days), plus CRSP's value-weighted index return as the market proxy
for regime features. The panel starts in 2005 so the 2011 evaluation window opens
with a fully standardized feature history and a satisfied HMM burn-in.

`cache/`, `results/` and `.env` are not committed. Rebuild the cache with
`build_cache`; the manifest in each result directory records which panel
configuration produced it.
