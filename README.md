# Regime Effects Are Turnover Effects

Original Motivation: This paper was supposed to be the introduction of a novel Firefly Portfolio Swarm Optimization for Portfolio Optimization that also incorporated Regime Classification. We determined in the results that Regime Classification actually has its drawbacks and we decided to move the paper in that direction.

Research code for a controlled evaluation of regime-conditioned portfolio
optimization, submitted to ICAIF '26.

Accepted to IOCRF '26

## Result

The state signal was injected at each of the three points where it can act on a
mean-variance decision — the optimizer's **preferences**, its **beliefs** about
risk, and its **forecasts** of return — across 32 arms, 30 matched seeds per arm
and four transaction-cost levels on CRSP daily data, 2011-2024.

| | Preferences | Beliefs | Expected returns |
|---|---|---|---|
| vs. an unconditional baseline (150 bps) | +0.113 | +0.388 | +0.294 |
| vs. a turnover-matched control (50 bps) | **-0.069** | **-0.034** | **-0.047** |

Measured the way this literature measures, all three mechanisms work. None
survives control.

Two further results support the reading:

- Across 27 comparable arms, net-of-cost performance is ordered almost perfectly
  by realized turnover alone (Spearman rho = -0.95, p = 2.1e-14). Gross of costs
  the same correlation is insignificant (rho = +0.25, p = 0.21), so the ordering
  is an economic effect operating through realized trading costs rather than an
  accounting identity.
- Re-running a single arm under a different random seed moves realized Sharpe by
  0.233 and fourteen-year terminal wealth by 1.85x — exceeding four of the five
  effects the study measures. Most of this literature reports one seed.

The methodological contribution is the control itself: for each treatment, an
unconditional arm whose turnover penalty is *solved by bisection* until it trades
exactly as much as the treatment. Same optimizer, same constraints, same seeds,
same realized turnover, no regime model. If the signal contributes anything
beyond the trading volume it induces, the treatment has to beat this arm.

Full numbers and the evidence behind each claim: **[docs/findings.md](docs/findings.md)**.

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

Reproducing the paper's numbers needs WRDS access to CRSP:

```bash
cp .env.example .env          # then fill in WRDS_USERNAME / WRDS_PASSWORD
python -m fpso.experiments.build_cache            # one-time CRSP pull -> cache/
python -m fpso.experiments.calibrate_policy       # 2008-2010 holdout, then freeze
python -m fpso.experiments.run_study              # 32 arms x 30 seeds
python -m fpso.experiments.analyze                # tables -> paper/tables,
                                                  # figures -> paper/figures
```

The CRSP pull is the only step that touches the network. Everything after it runs
from `cache/`, which is what makes the study reproducible and a 32-arm x 30-seed
matrix affordable.

Two further entry points support claims the paper makes about itself:

```bash
python -m fpso.experiments.verify_reproduction --all-arms   # re-execute and diff
python -m fpso.experiments.convergence_probe --dates 6 --restarts 60
```

`verify_reproduction` re-runs an arm and asserts the target weights are
bit-identical to the stored ones. It is not decoration: it found that seven of the
thirty-two arms did not reproduce, which the existing determinism tests had not
caught.

---

## What the experiment is

| # | Arm | Detector | What it establishes |
|---|-----|----------|---------------------|
| 1 | `static` | none | The ablation baseline: one parameterization for all states |
| 2 | `regime_hmm` | Gaussian HMM | Mechanism 1: regime-conditioned preferences |
| 3 | `regime_gmm` | Gaussian mixture | Does Markov *persistence* matter, or is clustering enough? |
| 4 | `regime_volq` | volatility terciles | Does the *learned* detector beat a transparent rule? |
| 5 | `regime_shuffled` | matched-marginal noise | Falsification. Same parameter variation, no signal |
| 6 | `regime_oracle` | HMM on the full sample | Lookahead upper bound. Reported, never claimed |
| 7 | `equal_weight`, `min_variance` | — | Standard external baselines |
| 8 | `pinned_calm`, `pinned_crisis` | constant | Separates parameter *levels* from *timing* |
| 9 | `moments_hmm` | Gaussian HMM | Mechanism 2: regime-weighted covariance |
| 10 | `moments_oracle`, `moments_shuffled` | — | Ceiling and falsification for mechanism 2 |
| 11 | `moments_longwindow` | none | Isolates window length from regime weighting |
| 12 | `static_matched_*` | none | **Turnover-matched controls — the decisive test** |
| 13 | `tilt_hmm` | Gaussian HMM | Mechanism 3: regime-tilted expected returns |
| 14 | `tilt_oracle`, `tilt_shuffled`, `tilt_constant` | — | Ceiling, falsification and no-timing control for mechanism 3 |
| 15 | `tilt_size`, `tilt_net_issuance` | Gaussian HMM | Tilts on *exogenous* scores the return panel cannot express |
| 16 | `warmstart_*`, `lowvol_screen` | — | Search-initialisation and screening variants |

The three mechanisms are the three points at which a state signal can enter a
mean-variance optimizer. Together they exhaust the injection points, which is what
lets the study make a claim about regime conditioning rather than about one
implementation of it.

Arms 2-6 share **one** parameter map and differ only in the detector, so the
comparison isolates the regime signal rather than two different policies. Arm 5 is
what makes this an experiment: if arm 2 does not beat it, the mechanism is
parameter jitter, and the paper says so.

Three further arms round out the matrix. `ablation_no_pso` (pure firefly) and
`ablation_annual` (the pre-refactor yearly cadence) measure the two changes this
refactor made to the method rather than asserting them. `regime_hmm_calibrated`
re-runs the headline under the pre-sample-calibrated parameter map instead of the
a-priori one — if the conclusion survives that swap, it does not hinge on how
those twelve numbers were chosen.

Seeds are **matched across arms**, so every comparison is paired.

---

## Design decisions a reviewer will ask about

**Causality is enforced structurally and machine-checked.** The backtest engine is
the only component that touches dates; every other component receives a pre-sliced
frame. `tests/unit/test_lookahead.py` runs the pipeline, then reruns it on a panel
whose observations after date *T* have been replaced with garbage, and asserts that
every decision at or before *T* — weights, regime label, parameters, objective
value — is bit-identical. It is parameterized over four detectors and three cut
dates.

**Bit-level reproducibility is verified by re-execution, not asserted.**
Multithreaded BLAS chooses its reduction order dynamically, and floating-point
addition is not associative. Across the HMM's ~200 EM iterations per refit that
compounds into a posterior differing at ~1e-14 between runs of identical code.
Whether it mattered depended on how an arm *consumed* the posterior: arms taking
`argmax` were unaffected, because a 1e-14 wobble never flips an argmax, while arms
weighting continuously by the posterior (`moments_*`, `tilt_*`) passed it straight
into `mu`/`Sigma`, where the swarm amplified it. Seven of thirty-two arms did not
reproduce. `fpso/determinism.py` pins BLAS to one thread before numpy loads; the
affected arms were re-run and now reproduce bit-identically.

The general lesson is worth more than the fix: the determinism guarantee had been
holding only because every mechanism tested until then quantized the detector's
output. A reproducibility check that passes can be passing for the wrong reason.

**Manifests record a source hash, not just a git SHA.** `git status --porcelain`
reports the whole working tree, so regenerating a figure stamps a run `-dirty`
without a line of source having changed — which is what happened to 450 of this
study's first 719 runs. `source_hash` digests the package itself, so two runs
sharing it provably executed the same bytes. The thread configuration is recorded
too, since an eight-thread run is not bit-comparable with a one-thread run.

**Transaction costs are quoted per trade, and turnover one-way.** The two do not
multiply directly: rotating a position generates a sale and a purchase, so the
realised drag is `2 x rate x one-way turnover`. `cost_per_unit_turnover()` is the
single source of truth and a test pins the invariant against the simulator.

**Feature standardization is expanding, not full-sample.** Z-scoring against
full-sample moments would let 2020 leak into a 2011 observation. The lookahead test
covers this.

**HMM states are canonicalized.** Fitted state indices are arbitrary and permute
across refits. States are sorted by mean realized volatility and relabelled
CALM < TURBULENT < CRISIS after every fit, so the regime-to-parameter map means the
same thing in 2013 and in 2019.

**Inference is filtered, not smoothed.** At the last observation of a sequence the
smoothed posterior equals the filtered one, so `predict_proba(X[:t+1])[-1]` is
exactly `P(s_t | x_{1:t})`. Smoothed label paths are used only for figures and are
marked as such in the captions.

**The parameter map never saw the evaluation sample.** The shipped headline uses
the a-priori map in `configs/regime_policy.yaml`: twelve values set from economic
reasoning and fitted to nothing at all. `calibrate_policy.py` provides the
alternative — coordinate search on a 2008-2010 pre-sample holdout spanning the
crisis, the recovery and a calm stretch — and writes
`configs/regime_policy.frozen.yaml` with a timestamp, a git SHA and the full search
trace. It moved 3 of the 12 values. Both are run, as `regime_hmm` and
`regime_hmm_calibrated`, so the sensitivity is reported rather than assumed away.

(The holdout opens in 2008 rather than at the 2005 panel start because the detector
needs roughly three years of warm-up: 252 days of rolling volatility and drawdown,
then 252 observations of expanding standardization, then 252 more to identify the
HMM. Calibrating from 2005 would have spent most of the window in burn-in,
comparing configurations that were in fact identical.)

**Monthly rebalancing.** ~168 decisions over 2011-2024 rather than the 14 an annual
cadence gives. A regime switch that can fire 14 times in fifteen years is not a
regime switch — COVID would be one observation.

**The universe is point-in-time.** Names are ranked by market capitalisation *at the
rebalance date*. The previous rule took the first 50 PERMNOs, which is approximately
"the 50 oldest-listed survivors" — survivorship- and age-biased.
`FirstPermnoUniverse` retains the old rule so the bias can be measured.

**Delistings cost what they cost.** Delisting returns from `crsp.dsedelist` are
compounded into a name's final observation. Filling those days with 0.0 would make
exiting the sample free.

**Repair, not penalty, enforces feasibility.** Every candidate is projected onto the
feasible set before scoring, so the penalty terms are identically zero along the
search path and `Phi(w) == R(w)` in practice. This is asserted in
`tests/unit/test_objective.py` rather than assumed.

**Only the support is combinatorial.** Once the K names are fixed, maximizing the
objective over their weights is concave over a convex set, so it has a unique global
maximum reachable in milliseconds. Re-solving the weights exactly on the swarm's own
support raises the objective at 100% of 10,080 rebalances, by a median of 43%. That
removes 46% of realized turnover and does not change the sign of any matched
comparison — the study's conclusion does not depend on the search.

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
│                 flat and regime-weighted moment estimators, tilt scorers
├── regime/       features, HMM + alternatives + controls, smoothing, policy
├── optimizer/    constraint repair, objective, FPSO search, exact weighting
├── baselines/    1/N and minimum variance under identical constraints
├── backtest/     schedule, rolling engine, vectorbt ledger, results + manifest
├── evaluation/   metrics, bootstrap/HAC inference, figure builders, conviction
├── adaptive/     shrinkage policies and the oracle/control ladder
└── experiments/  build_cache · calibrate_policy · match_turnover · run_study ·
                  analyze · verify_reproduction · convergence_probe
configs/          base.yaml, regime_policy.yaml, arms/*.yaml
paper/figures/    camera-ready PDF and PNG
paper/tables/     headline / comparison / regime-conditional / cost CSVs
tests/            unit (offline) and integration (WRDS) suites
```

Every result directory carries a `manifest.json` recording the config, git SHA,
source hash, Python version and the versions of numpy / pandas / scipy /
scikit-learn / hmmlearn / vectorbt, so a figure can always be traced to the code
that made it.

---

## Tests

```bash
pytest -m "not integration"   # offline; no network, no credentials. CI contract.
pytest -m integration         # hits live WRDS; needs .env
```

The ones that carry research weight:

- **`test_lookahead.py`** — the causality proof described above.
- **`test_reproducibility.py`** — same seed reproduces exactly; *different* seeds
  actually differ (a dropped RNG would silently turn 30 runs into 1); a rebalance's
  random stream does not depend on its position in the schedule.
- **`test_constraints.py`** — property tests on the repair operator: budget, box,
  cardinality, idempotence and deterministic tie-breaking, under adversarial inputs
  including all-zeros, NaN and infinities.
- **`test_controls.py`** — the falsification control must match the treatment arm's
  label marginals *and* switching rate. The first version of that control silently
  degenerated into a static arm and every other test still passed, so the comparison
  is now an assertion.
- **`test_regime_moments.py`** — the regime-weighted estimator must degrade to the
  flat one when the posterior is uninformative, so a null result stays reachable and
  cannot be manufactured by the plumbing.

---

## Data

CRSP daily stock file via WRDS. The cached panel is the union of names that were
ever top-200 by market capitalisation between 2005 and 2024 (443 PERMNOs, 5,033
trading days), plus CRSP's value-weighted index return as the market proxy for
regime features. The panel starts in 2005 so the 2011 evaluation window opens with
a fully standardized feature history and a satisfied HMM burn-in.

`cache/`, `results*/` and `.env` are not committed. Rebuild the cache with
`build_cache`; the manifest in each result directory records which panel
configuration produced it.
