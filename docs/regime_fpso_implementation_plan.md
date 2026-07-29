# Implementation Plan — Idea 1: Regime-Conditioned FPSO

**Target:** ICAIF '26 submission
**Claim to establish (RQ3):** *Conditioning a constraint-native metaheuristic's search
behavior and risk parameters on a causally-estimated latent market regime improves
risk-adjusted out-of-sample performance over a static parameterization.*
**Status:** plan only — no code written yet.

---

## 0. Codebase audit — what must change before Idea 1 is even implementable

I read every module in `algorithm/`, `backtester/`, and `run_backtest.py` against
`paper/main.md`. Nine findings, ordered by how hard they block the experiment.

### Blockers (Idea 1 cannot produce a credible result without these)

**B1 — The repo does not currently import.**
[get_performance.py:6-7](../backtester/get_performance.py#L6-L7) imports
`algorithm.firefly` and `algorithm.data_caller`, which no longer exist.
`python -c "import backtester.get_performance"` → `ModuleNotFoundError`.
So `run_backtest.py` is dead at import time. These are leftovers from the pre-refactor
layout; the legacy Monte-Carlo helpers at the bottom of the file are the only consumers
and they re-import `pandas` locally anyway.

**B2 — The backtest rebalances annually, not monthly.**
[run_backtest.py:175-180](../run_backtest.py#L175-L180) loops `train_year → test_year`:
14 rebalances over 2010–2024. The paper says monthly (~30 timesteps,
[main.md:459](../paper/main.md#L459)) and `icaif_fit_ideas.md` assumes monthly.
This is the single most important blocker: **a regime switch that fires 14 times over
15 years is not a regime switch.** COVID would be one observation. Regime conditioning
needs monthly decisions (≈168 rebalances) with a trailing 12-month estimation window,
which also removes the current artifact that the whole of 2020 is traded on parameters
chosen from 2019 data.

**B3 — No data cache; every backtest re-queries WRDS.**
`_run_window` issues two `crsp.dsf` queries per window. The experiment matrix below is
7 arms × 30 seeds × ~168 rebalances. Refetching is infeasible and non-reproducible
(WRDS results can change under you). Need a one-time full-panel pull cached to Parquet,
then all backtests are in-memory slices. This is also what makes the work reproducible
by a reviewer without WRDS credentials.

**B4 — No RNG control anywhere.**
`Particle.__init__` and `apply_hybrid_update` call global `np.random`. There is no seed
parameter in the entire codebase. The paper already claims "backtested ... thirty times
before taking the average" ([main.md:433](../paper/main.md#L433)) — those 30 runs are
currently unreproducible. Every stochastic component must take an injected
`np.random.Generator`, with per-(arm, seed, rebalance) streams derived from a master
`SeedSequence`.

**B5 — Hyperparameters are hardcoded inside the window function.**
[run_backtest.py:113-124](../run_backtest.py#L113-L124) constructs `FPSO(...)` with
literals and then mutates `.K` / `.u` as attributes. There is no parameter object to
*condition on a regime* — Idea 1's entire mechanism has nothing to bind to.

### Methodological issues (fix, or explicitly disclose in the paper)

**B6 — Universe selection is biased.**
`get_sp500_constituents(db, train_end)[:num_assets]` ([run_backtest.py:93](../run_backtest.py#L93))
slices the first 50 PERMNOs. PERMNO is assigned roughly in order of CRSP listing date,
so "first 50" ≈ "50 oldest-listed surviving large caps." That is a survivorship- and
age-biased universe, and a reviewer who reads the released code will see it.
Fix: select top-N by market cap at the rebalance date (the fallback query already
computes `ABS(prc) * shrout` — reuse it), or drop the cap and use the full index.

**B7 — Missing returns are filled with 0.0, so delistings are costless.**
`clean_and_pivot_data` does `.ffill().fillna(0.0)`
([data_processor.py:88](../algorithm/data_processor.py#L88)). A stock that stops
trading mid-window contributes zero return rather than its delisting return. Combined
with B6 this biases returns upward. Fix: use `crsp.dsedelist` delisting returns, or at
minimum drop the name and renormalize, and disclose the choice.

**B8 — The penalty terms in Eq. (2) never bind.**
`repair_portfolio_constraints` is applied to every particle before every fitness
evaluation. Post-repair, `sum(w) == 1`, `w <= u`, and `card(w) <= K` all hold exactly,
so `budget_penalty`, `box_penalty`, and `cardinality_penalty` in
[portfolio_model.py:49-69](../algorithm/portfolio_model.py#L49-L69) evaluate to ~0
always. In effect **Φ(w) ≡ R(w)** and ρ₁, ρ₂, ρ₃ = 100 are decorative. This is
defensible as a *repair-dominant* design (and is arguably better than penalty-only),
but the paper presents the penalized fitness as the operative objective. Either
document repair as the primary constraint mechanism and demote penalties to a
safety net, or add an unrepaired-penalty ablation. Cheap to fix in prose; must not be
left as-is if the code is released.

**B9 — The optimizer has no PSO term.**
`Particle.velocity` and `Particle.best_weights` are written but never read.
`apply_hybrid_update` ([fpso_optimizer.py:96-111](../algorithm/fpso_optimizer.py#L96-L111))
is pure firefly: centroid attraction + random walk. There is no cognitive
(`p_best - x`) or social (`g_best - x`) velocity term. Meanwhile
[main.md:253-256](../paper/main.md#L253-L256) claims FPSO "combin[es] FA's
diversity-preserving centroid attraction **with PSO's personal- and global-best
refinement**." As implemented, the "PSO" in FPSO is not present.

This is out of scope for Idea 1 (it is exactly what Idea 2 refactors), but it is the
most serious paper↔code mismatch in the repo. **Recommendation:** add the PSO velocity
term in Phase 1 as a *fixed, non-learned* blend so the paper's own description is true,
and keep the learned FA/PSO switch as Idea 2 / future work. Cost: ~20 lines. If we
choose not to, the paper text must be corrected to describe a firefly-with-repair
algorithm.

---

## 1. Design principles

The plan is built around four properties. Everything below is in service of these.

1. **Causality is structurally enforced, not assumed.** No component can see data past
   its decision date, and there is an automated test that proves it (§6).
2. **Every result is reproducible from a config file + a seed.** Config, git SHA, and
   package versions are written into every result manifest.
3. **Every axis of the experiment is swappable behind an interface.** Detector,
   optimizer, policy, and schedule are all ABCs. Ablations become config changes, not
   code edits — which is the only way the ablation matrix in §5 is affordable.
4. **The regime mechanism is falsifiable.** Arms 5 and 6 in §5 exist specifically to
   let the experiment say "the regime signal did nothing." A paper that cannot lose
   its own test is not evidence.

---

## 2. Target architecture

Migrate `algorithm/` + `backtester/` into a single `fpso/` package (use `git mv` to
preserve history). The current split does not survive the additions.

```
fpso/
├── config/
│   ├── schema.py        # frozen dataclasses: FPSOParams, RegimeConfig,
│   │                    #   ScheduleConfig, DataConfig, ExperimentConfig
│   └── loader.py        # YAML → schema, with validation + defaults
├── data/
│   ├── source.py        # DataSource(ABC); WRDSDataSource; ParquetPanelSource
│   ├── cache.py         # CachedDataSource decorator (Parquet, content-hashed key)
│   ├── universe.py      # UniverseProvider(ABC); PointInTimeSP500; TopMarketCap
│   └── moments.py       # MomentEstimator(ABC); SampleMoments; (LedoitWolf later)
├── regime/
│   ├── base.py          # RegimeLabel(IntEnum); RegimeAssignment; RegimeDetector(ABC)
│   ├── features.py      # RegimeFeatureBuilder — causal, expanding standardization
│   ├── hmm.py           # GaussianHMMDetector  (headline)
│   ├── mixture.py       # GaussianMixtureDetector  (ablation: no temporal structure)
│   ├── heuristic.py     # VolatilityQuantileDetector (ablation: no learning)
│   ├── controls.py      # ShuffledLabelDetector, OracleDetector (falsification arms)
│   ├── smoothing.py     # MinimumDwellSmoother — causal state machine
│   └── policy.py        # RegimeParameterPolicy: RegimeLabel → FPSOParams
├── optimizer/
│   ├── base.py          # PortfolioOptimizer(ABC); OptimizationResult
│   ├── constraints.py   # ConstraintSet; SimplexBoxCardinalityRepair
│   ├── objective.py     # Objective(ABC); MeanVarianceTurnover; PenalizedObjective
│   └── fpso.py          # FPSO(PortfolioOptimizer) — RNG-injected, params-driven
├── baselines/
│   ├── equal_weight.py  # EqualWeight(PortfolioOptimizer)
│   └── min_variance.py  # MinimumVariance(PortfolioOptimizer)
├── backtest/
│   ├── schedule.py      # RebalanceSchedule(ABC); MonthlySchedule; AnnualSchedule
│   ├── engine.py        # RollingBacktestEngine — owns the no-lookahead contract
│   ├── ledger.py        # turnover, transaction costs, path accumulation
│   └── results.py       # BacktestResult; Parquet/JSON persistence + manifest
├── evaluation/
│   ├── metrics.py       # (from get_performance.py, cleaned)
│   ├── statistics.py    # stationary bootstrap, Ledoit–Wolf Sharpe test, Holm
│   └── figures.py       # regime-shaded SPX, cumulative/drawdown, regime-conditional
└── experiments/
    ├── run_backtest.py     # single arm
    └── run_regime_study.py # full matrix, parallel over (arm, seed)
```

### Key interfaces

```python
# fpso/regime/base.py
class RegimeLabel(IntEnum):
    CALM = 0
    TURBULENT = 1
    CRISIS = 2

@dataclass(frozen=True)
class RegimeAssignment:
    as_of: pd.Timestamp
    label: RegimeLabel
    posterior: np.ndarray        # P(s_t | x_{1:t}), length n_states
    raw_label: RegimeLabel       # pre-smoothing, for diagnostics
    is_burn_in: bool             # True => caller must fall back to static params

class RegimeDetector(ABC):
    @abstractmethod
    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """Fit using ONLY rows with index <= as_of."""

    @abstractmethod
    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        """Filtered state estimate at as_of. Must not touch rows > as_of."""
```

```python
# fpso/optimizer/base.py
class PortfolioOptimizer(ABC):
    @abstractmethod
    def solve(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        w_prev: np.ndarray,
        rng: np.random.Generator,
    ) -> OptimizationResult: ...
```

`FPSOParams` is a frozen dataclass (`beta_0, gamma, alpha, delta, lambda_v, lambda_t,
rho_1..3, K, u, num_particles, max_iter`) with a `.replace(**overrides)` method. The
regime policy is literally `dict[RegimeLabel, dict[str, float]]` applied via `.replace`,
so a regime config is data, not code — and the static arm is the same code path with an
empty override map. **The static baseline and the regime arm differ only in
configuration**, which is what makes the ablation clean.

---

## 3. The regime subsystem — protocol details

This is where the paper's credibility lives, so the details are spelled out.

### 3.1 Features (all causal, computed at rebalance date *t*)

Market proxy = cap-weighted return of the current universe (or CRSP value-weighted
index `crsp.dsi.vwretd` — cleaner, one query, and standard).

| Feature | Definition |
|---|---|
| `rv_21` | 21-day realized volatility, annualized |
| `rv_ratio` | log(`rv_21` / `rv_252`) — vol regime relative to its own trailing norm |
| `ret_63` | 63-day cumulative market return |
| `dd_252` | drawdown from trailing 252-day peak |

**Standardization must be expanding, not full-sample.** The paper currently says
"We normalize via Z-score normalization" ([main.md:308](../paper/main.md#L308)) with no
qualifier — full-sample z-scoring is a lookahead leak and a reviewer will flag it.
`RegimeFeatureBuilder` standardizes each feature with expanding mean/std through *t*
only, and the lookahead test in §6 will catch any regression.

### 3.2 HMM protocol

- **Model:** `hmmlearn.GaussianHMM(n_components=3, covariance_type="diag")`. Diagonal
  covariance with 4 features and ~2500 daily observations is well-conditioned; full
  covariance is an ablation, not the default.
- **Fit window:** all feature rows with index ≤ *t*. Refit every `refit_months` (default
  12) and warm-start from the previous fit's parameters; run inference every month.
  Monthly refitting churns parameters and buys nothing.
- **State identification (critical).** HMM state indices are arbitrary and change across
  refits — without canonicalization the regime→parameter map means a different thing
  each year, which would silently destroy the experiment. After every fit, sort states
  by their mean `rv_21` ascending and relabel → CALM / TURBULENT / CRISIS. Deterministic,
  interpretable, and stable across refits.
- **Filtered inference.** The label at *t* must condition on the past only. Convenient
  fact we exploit: **at the final observation of a sequence, the smoothed posterior
  equals the filtered posterior** (γ_T = α_T, since there is no future evidence). So
  `predict_proba(X[:t+1])[-1]` is exactly P(s_t | x_{1:t}) with no leak. Historical
  labels used *for plotting* are smoothed and will be marked as such in the figure
  caption; labels used *for decisions* are always the last-row filtered estimate.
- **Smoothing.** `MinimumDwellSmoother`: a new regime must persist `min_dwell` (default
  2) consecutive rebalances before the acting label switches. Implemented as a causal
  finite-state machine, so it introduces a deliberate one-month lag rather than a leak.
- **Burn-in.** Regime conditioning is inactive until `burn_in_months` (default 36) of
  feature history exist; before that the static parameterization is used. With features
  starting 2007 and OOS starting 2011, burn-in is satisfied from day one of the
  evaluation period — but the guard is in the code and reported.

### 3.3 Regime → parameter policy

Directionally as in `icaif_fit_ideas.md`:

| Regime | `alpha` | `gamma` | `lambda_v` | `lambda_t` | `K` | `u` |
|---|---|---|---|---|---|---|
| CALM | low (exploit) | low | low | high (trade little) | high | base |
| TURBULENT | high (explore) | high | high | base | base | base |
| CRISIS | high | high | highest | low (let it trade) | low | tight |

**How these numbers get chosen is the reviewer's first question.** Two defensible
options, and we must pick one *before* looking at 2010–2024 results:

- **(A) Pre-sample calibration (recommended).** Grid/coordinate search the per-regime
  overrides on a **2005–2009 holdout** (CRSP covers it; it contains the GFC, so all three
  regimes are represented), then **freeze** and evaluate untouched on 2010–2024. Costs
  ~half a day; buys a clean "parameters were never fit on the evaluation sample" claim.
- **(B) A priori fixed.** Set values from the economic reasoning above with no tuning
  and state that explicitly. Free, weaker, but honest and unimpeachable.

Either way the frozen config file is committed with a timestamp and included in the
artifact, so the freeze is auditable.

---

## 4. Backtest engine contract

`RollingBacktestEngine.run(config, arm, seed) -> BacktestResult`

Per rebalance date *t* in `MonthlySchedule(start, end)`:

1. `universe = universe_provider.at(t)` — point-in-time, no future constituents.
2. `panel = source.slice(universe, t - estimation_window, t)` — closed on the left,
   **exclusive of t+1 onward**.
3. `mu, sigma = moment_estimator.estimate(panel)`.
4. `assignment = detector.infer(features, as_of=t)` → `params = policy.apply(base, assignment)`.
5. `result = optimizer.solve(mu, sigma, w_prev, rng=stream(arm, seed, t))`.
6. Hold to the next rebalance; ledger accrues realized returns and charges
   `tc_rate * one_way_turnover` on the rebalance day.

The engine is the *only* component that touches dates, and it hands every downstream
component a pre-sliced frame. That single choke point is what makes the lookahead test
in §6 both possible and meaningful.

Recorded per rebalance: weights, regime label + posterior, params used, turnover,
objective value, wall-clock solve time (RQ1 asks about NRT tractability — we should
actually measure it), and convergence history.

---

## 5. Experiment matrix

| # | Arm | Detector | Purpose |
|---|---|---|---|
| 1 | `static` | none | Existing result → becomes the ablation baseline |
| 2 | `regime-hmm` | GaussianHMM | **Headline** |
| 3 | `regime-gmm` | GaussianMixture | Does temporal (Markov) structure matter, or is marginal clustering enough? |
| 4 | `regime-volq` | vol tercile rule | Does the *learned* detector beat a transparent rule? |
| 5 | `regime-shuffled` | permuted labels, matched marginals | **Falsification.** Same parameter variation, no signal. If arm 2 ≈ arm 5, the mechanism is parameter jitter, not regime detection. |
| 6 | `regime-oracle` | HMM fit on full sample | Upper bound on regime value. Labeled as lookahead in the paper. |
| 7 | `1/N`, `min-var` | — | Standard external baselines |

× 30 seeds (arms 1–6; arm 7 is deterministic). Seeds are **matched across arms** so
every comparison is paired — this roughly halves the variance of ΔSharpe estimates and
is free.

Arm 5 is the one that makes this a real experiment. Arms 3 and 4 answer "why an HMM?"
before a reviewer asks.

### Statistical protocol

- **Primary:** paired ΔSharpe (arm 2 − arm 1) on matched seeds, with a stationary
  bootstrap CI (Politis–Romano, expected block ≈ 21 days to respect autocorrelation).
- **Secondary:** ΔMaxDrawdown, ΔVol, Δturnover, Δ net-of-cost return at each `tc_rate`
  in {0, 0.5%, 1%, 1.5%} (the existing sweep).
- **Sharpe difference test:** Ledoit–Wolf (2008) HAC-robust test on seed-averaged paths.
- **Multiplicity:** Holm correction across the arm comparisons.
- **Regime-conditional table:** performance of each arm *within* each detected regime.
  This is where the story lives even if the pooled ΔSharpe is modest — and per the ideas
  doc, drawdown/vol is the likely win, so this table should be built to showcase it.

### Figures

- **F1** — Market cumulative return 2010–2024, shaded by detected regime. Sanity plot
  and the paper's most persuasive single image. (Caption states smoothed labels.)
- **F2** — Static vs. Regime-FPSO cumulative return + drawdown, with regime shading.
- **F3** — Regime-conditional Sharpe / MDD bars, static vs. regime.
- **F4** — Estimated transition matrix + dwell-time distribution (shows the HMM learned
  persistent, economically sensible states rather than noise).

---

## 6. Testing strategy

Beyond ordinary unit tests, three tests carry real research weight:

**T1 — The lookahead test (flagship).**
Run the full backtest on the cached panel. Then overwrite every observation strictly
after date *T* with garbage (e.g. `+1e6`), re-run, and assert that every decision —
weights, regime label, params — at dates ≤ *T* is **bit-identical**. Parameterized over
several *T*. This is a machine-checked proof of causality for the entire pipeline, and
it is worth a sentence in the paper.

**T2 — Determinism / reproducibility.** Same config + seed → identical results across
processes. Different seeds → different results (guards against a silently-dropped RNG).

**T3 — Constraint invariants.** Property-style random inputs into
`SimplexBoxCardinalityRepair`: assert `sum(w)==1 ± 1e-12`, `0 <= w <= u`,
`card(w) <= K`, and idempotence (`repair(repair(w)) == repair(w)`) — for adversarial
inputs including all-zeros, NaN/inf, and infeasible bound sets.

Plus: golden-file regression on a small synthetic panel (catches silent numerical drift
during refactors), objective correctness vs. closed form, state-canonicalization
stability under state permutation, smoother causality, and the existing WRDS tests
kept behind the `integration` marker.

CI-ready: `pytest -m "not integration"` must pass with no network and no credentials.

---

## 7. Dependencies

Verified as available for Python 3.13.7 / arm64 (wheels download cleanly):

```
hmmlearn>=0.3.3      # Gaussian HMM
scikit-learn>=1.9    # GaussianMixture, hmmlearn dependency
scipy>=1.18          # stats, hmmlearn dependency
pyarrow>=25.0        # Parquet cache
PyYAML               # configs
```

`requirements.txt` should be pinned (`==`) for reproducibility, with a `requirements-dev.txt`
for pytest/ruff. No `hmmlearn` fallback implementation is needed given wheel availability;
the `RegimeDetector` ABC means one could be swapped in without touching the engine.

---

## 8. Phased schedule

| Phase | Days | Deliverable | Done when |
|---|---|---|---|
| **0 — Unblock** | 0.5 | Fix B1 dead imports; pin deps; one-time full CRSP panel pull → Parquet cache (2005–2024, universe union) | `pytest -m "not integration"` green; backtest runs offline from cache |
| **1 — Core refactor** | 1.0 | `fpso/` package; config dataclasses; RNG injection (B4); optimizer split into objective/constraints/fpso; `PortfolioOptimizer` ABC + baselines; **PSO velocity term (B9)** | T2, T3 pass; static FPSO reproduces current annual numbers within seed noise |
| **2 — Engine** | 1.0 | `MonthlySchedule` (B2); `RollingBacktestEngine`; ledger; results persistence + manifest; universe fix (B6) & delisting handling (B7) | **T1 lookahead test passes**; monthly static baseline runs end-to-end |
| **3 — Regime subsystem** | 1.5 | Features, HMM + canonicalization, smoother, policy, control detectors; F1 regime plot | F1 shows economically sensible regimes (2011, 2015-16, 2018, 2020, 2022 flagged) |
| **4 — Experiments** | 1.0 | Experiment runner; policy calibration on 2005–2009 (option A) then **freeze**; run full matrix (7 arms × 30 seeds) | All arms complete; results + manifests on disk |
| **5 — Analysis** | 1.0 | Statistics module; F2–F4; results tables; regime-conditional table | Paired CIs + Ledoit–Wolf test computed with Holm correction |
| **6 — Paper** | 1.0 | Retitle; rewrite intro/contributions around learning-guided constraint-native optimization; add RQ3; Related Work +regime-switching (Hamilton 1989; Ang & Bekaert 2002); fix B8 prose; move centroid derivation to appendix | Draft complete, `[add source here]` placeholders filled |

Phases 0–2 are prerequisites that would be needed for *any* of the five ideas, so the
time is not idea-specific risk. Phase 3 onward is Idea 1 proper.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| **Regime-FPSO does not beat static on return** | Expected, per the ideas doc. Lead with drawdown/vol (F3 + regime-conditional table are built for this). The paper's claim is *risk-adjusted* performance, stated that way from the start. |
| **Arm 2 ≈ Arm 5 (shuffled control)** | This is a real negative result, and finding it *before* submission is the point of arm 5. Fallback: report it honestly and pivot the contribution toward the regime-conditional analysis + the open-source causal backtest harness, or escalate to Idea 2. |
| **Monthly rebalancing changes all existing numbers** | Unavoidable and correct — the paper already claims monthly. Report annual as a rebalance-frequency ablation; it becomes a supporting result rather than a loss. |
| **Regime labels too noisy** | Three defenses already in the design: min-dwell smoothing, annual refit cadence, monthly (not daily) decisions. Report the dwell distribution (F4) so noise is visible rather than hidden. |
| **Compute** | 7 × 30 × 168 solves. At 30 particles × 60 iterations that is ~35k solves. Parallelize over (arm, seed) with `ProcessPoolExecutor`; cached panel makes each solve pure CPU. Estimate a single arm-seed first and adjust `max_iter` / seed count from the measurement, not from a guess. |
| **Scope creep from B6–B9** | B6/B7 are ~2 hours each and materially affect credibility → do them. B9 is ~20 lines → do it. B8 is a prose fix → do it in Phase 6. |

---

## 10. Open decisions (need your call before Phase 0)

1. **Package restructure into `fpso/`** — recommended, but it touches every file. The
   alternative is adding `regime/` alongside the current layout and leaving the existing
   modules as-is, which is faster now and messier for the artifact release.
2. **Policy calibration: option (A) 2005–2009 pre-sample tuning, or (B) a priori fixed?**
   (A) is stronger and costs ~half a day.
3. **B9 — add the PSO velocity term now?** Recommended: the paper currently describes an
   algorithm the code does not implement.
4. **B6/B7 — fix universe selection and delisting returns, or disclose as limitations?**
   Recommended: fix. They change the headline numbers, so the decision should be made
   before the runs, not after.
