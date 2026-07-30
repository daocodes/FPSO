# Findings

**Study:** Regime-Conditioned FPSO — ICAIF '26
**Evaluation window:** 2011-01 to 2024-12, CRSP daily, monthly rebalancing (168 decisions)
**Matrix:** 20 arms x 30 matched seeds (deterministic baselines at 1 seed), 4 transaction-cost levels
**Status:** experiments complete; paper not yet rewritten around these results

---

## 1. The headline

> **Hold turnover fixed and the regime effect disappears — at both injection
> points, at every transaction-cost level.**
>
> Against plain `static`, regime conditioning looks increasingly good as costs
> rise (+0.113 Sharpe at 1.5% for mechanism 1, +0.388 for mechanism 2). Against a
> `static` arm whose turnover penalty has been *solved* so that it trades exactly
> as much, the effect is **-0.062** and **-0.043** respectively: flat across the
> cost sweep, and slightly negative.
>
> Matching turnover removes **95%** of mechanism 1's cost-conditionality and
> **97%** of mechanism 2's. That is a direct demonstration that turnover was the
> whole channel, not an inference from a correlation.

See [f9_turnover_matched](../paper/figures/f9_turnover_matched.pdf) — the single
clearest figure in the study.

The supporting evidence, from the wider matrix:

* Across 17 comparable arms, net-of-cost performance is ordered almost perfectly
  by **realized turnover alone**: Spearman rho = **-0.94 (p = 1.8e-8)**. Gross of
  costs the same correlation is insignificant (+0.38, p = 0.14), so this is an
  economic effect rather than an accounting identity.
  ([f8_turnover_explains_effect](../paper/figures/f8_turnover_explains_effect.pdf))
* A **flat five-year covariance window with no regime model at all**
  (`moments_longwindow`, +0.435) outperforms regime-weighted covariance driven by
  **perfect future-knowledge labels** (`moments_oracle`, +0.392).
* A **permanently-CALM parameterization** (`pinned_calm`, +0.196) outperforms the
  HMM-switched parameterization (`regime_hmm`, +0.113) — and the HMM-switched arm
  does not beat its own **signal-free control** (`regime_shuffled`, +0.120).

**The methodological message, which is the publishable contribution:** the
regime-switching portfolio literature reports regime effects without
turnover-matched controls. This study shows that in a realistic
cardinality-constrained setting, an uncontrolled regime effect of the size
typically reported is fully accounted for by the turnover the regime mechanism
happens to induce — and provides the matched control that demonstrates it rather
than merely arguing for it.

---

## 2. What was tested

The regime signal can enter the optimizer at two points. Both were built and both
were tested against their own controls.

### Mechanism 1 — regime-conditioned *preferences*

The detected regime selects FPSO's hyperparameters: exploration rate `alpha`,
attraction decay `gamma`, risk aversion `lambda_v`, turnover penalty `lambda_t`,
cardinality `K` and position cap `u`.

**Result: null.** `regime_hmm` vs. `static` is -0.003 Sharpe at 0.5% cost
(95% CI [-0.080, +0.073], p = 0.96).

**Why it fails.** Half those knobs (`alpha`, `gamma`) govern *how FPSO searches*,
and FPSO already converges — tuning them changes the route, not the destination.
The other half change the objective, but they act on `mu` and `Sigma` estimated
from a flat 252-day window that is itself regime-blind. In March 2020 the
mechanism tells the optimizer to fear risk while handing it a covariance matrix
dominated by the preceding eleven calm months. It modulates *preferences* over
*stale beliefs*.

The oracle arm confirms the ceiling is the injection point, not label quality:
`regime_oracle` sees the future and still yields +0.006 (p = 0.88).

### Mechanism 2 — regime-conditioned *beliefs*

Each historical day is weighted by `w_d = sum_k P(s_t = k) P(s_d = k)`, the
posterior probability that it shares today's latent state, and `Sigma` is the
weighted second moment. In a crisis, `Sigma` is built from past crisis days —
higher volatility *and* higher cross-sectional correlation — so the optimizer
sees the risk in its inputs.

**Result: the effect is real but not attributable to regimes.** `moments_hmm`
gains +0.388 at 1.5% cost, roughly 3.4x mechanism 1. But `moments_longwindow` —
the same five-year lookback with **no regime weighting whatsoever** — gains
+0.435. The entire benefit is the longer estimation window, which stabilises
`Sigma`, which stabilises weights, which cuts turnover from 0.39 to 0.15.

Without that control arm the study would have reported a large, entirely
spurious regime effect. It was added only after the oracle probe showed a
suspiciously large turnover reduction.

### The turnover-matched control — the decisive test

The objection to everything above is reasonable: *regime detection is supposed to
change how you trade, so a turnover reduction is the mechanism working.* The arm
that settles it is a `static` FPSO whose turnover penalty `lambda_t` has been
**solved by bisection** until it trades exactly as much as the regime arm does.
Same optimizer, same constraints, same seeds, same turnover — no regime model.

| | Treatment | Matched control | Turnover (treat / control) | Regime effect @0.5% | @1.5% |
|---|---|---|---|---|---|
| Mechanism 1 | `regime_hmm` | `lambda_t` 0.010 -> **0.0254** | 0.3145 / 0.3053 | **-0.069** [-0.144, -0.000] | **-0.062** [-0.148, +0.014] |
| Mechanism 2 | `moments_hmm` | `lambda_t` 0.010 -> **0.0781** | 0.1794 / 0.1695 | **-0.034** [-0.215, +0.145] | **-0.043** [-0.235, +0.149] |

The regime signal does not merely fail to help once turnover is held fixed — it
is **consistently negative** at every cost level tested. For mechanism 1 the
Ledoit-Wolf test puts p = 0.039 at 0.5% cost; the bootstrap interval only just
touches zero. Mechanism 2's effect is smaller and well inside noise.

The strongest evidence is the *shape*, not the level. Measured against plain
`static`, mechanism 1's advantage swings by **+0.175** across the cost sweep and
mechanism 2's by **+0.417** — both flipping from negative to positive as costs
rise. Measured against the turnover-matched control those swings collapse to
**+0.010** and **-0.013**. Closing the turnover channel removes 95% and 97% of
the cost-sensitivity, which is what "turnover was the mechanism" predicts and
what no other explanation does.

**On the legitimacy of the matching.** The control is calibrated to the
treatment's *turnover* — a design covariate read off its trading — and never to
its Sharpe. That is matching on a nuisance variable, as one would match a control
group on age; calibrating on the outcome would be circular. Like the shuffled and
oracle arms it is a control, not a deployable strategy, and the paper should
describe it as one.

---

## 3. Result tables

### Sharpe difference vs. `static`, by transaction cost

| Arm | 0% | 0.5% | 1.0% | 1.5% | Turnover |
|---|---|---|---|---|---|
| `moments_hmm_policy` | -0.057 | +0.116 | +0.286 | **+0.452** | 0.134 |
| `moments_longwindow` *(no regime info)* | -0.047 | +0.117 | +0.278 | **+0.435** | 0.148 |
| `moments_shuffled` *(control)* | -0.016 | +0.132 | +0.278 | +0.420 | 0.171 |
| `moments_oracle` *(lookahead)* | -0.014 | +0.124 | +0.261 | +0.392 | 0.186 |
| `moments_hmm` | -0.029 | +0.113 | +0.253 | +0.388 | 0.179 |
| `equal_weight` | -0.081 | +0.024 | +0.128 | +0.227 | 0.230 |
| `min_variance` | -0.127 | -0.015 | +0.096 | +0.203 | 0.167 |
| `pinned_calm` *(no timing)* | -0.017 | +0.057 | +0.128 | +0.196 | 0.298 |
| `regime_hmm_calibrated` | -0.101 | -0.012 | +0.076 | +0.159 | 0.284 |
| `regime_shuffled` *(control)* | -0.007 | +0.037 | +0.080 | +0.120 | 0.330 |
| `regime_volq` | -0.054 | +0.005 | +0.062 | +0.116 | 0.314 |
| `regime_hmm` **(original headline)** | -0.063 | -0.003 | +0.057 | +0.113 | 0.315 |
| `regime_oracle` *(lookahead)* | -0.044 | +0.006 | +0.055 | +0.101 | 0.323 |
| `regime_gmm` | -0.062 | -0.009 | +0.043 | +0.091 | 0.321 |
| `static` | 0 | 0 | 0 | 0 | 0.391 |
| `ablation_no_pso` | -0.025 | -0.023 | -0.031 | -0.033 | 0.373 |
| `pinned_crisis` *(no timing)* | -0.025 | -0.079 | -0.131 | -0.179 | 0.431 |

The two turnover-matched controls are excluded from this table on purpose: they
are calibrated to a specific treatment arm's turnover, so ranking them against
`static` alongside everything else would double-count the very quantity they were
built to hold fixed. They belong in the paired table above.

Reading the table by turnover column reproduces the ranking almost exactly. That
is the finding.

`ablation_annual` is omitted from the table and from F8: at 14 rebalances vs.
168, turnover *per rebalance* is not comparable. Its lifetime turnover is 10.0
against static's 65.6, and it posts **+0.440** at 1.5% — consistent with, and the
most extreme case of, the same effect.

That +0.440 is the *date-aligned paired* estimate from `comparison_table`.
Comparing the two arms' separately-computed Sharpes gives +0.482, because the
annual arm's return path only begins at its first rebalance in December 2011 and
so covers eleven fewer months. The paired figure is the correct one to quote; the
gap is worth remembering whenever an arm's evaluation window differs from the
baseline's.

### Secondary results

* **Gross of costs, unconditional FPSO takes the top three slots** — 0.989
  (`static_matched_regime_hmm`), 0.980 (`static_matched_moments_hmm`), 0.979
  (`static`). All three are the same optimizer differing only in `lambda_t`; no
  regime arm reaches them. Every mechanism here gives up gross return.
* **The learned detector never beats a transparent rule.** `regime_volq`
  (volatility terciles, nothing fitted) matches `regime_hmm` at every cost level.
* **The PSO term helps slightly.** `ablation_no_pso` is -0.025 to -0.033 across
  all cost levels — consistently negative, never significant. Weak support for
  keeping it; the paper's description of FPSO is now at least true of the code.
* **Pre-sample calibration changes nothing.** `regime_hmm_calibrated` moved 3 of
  12 parameter values and lands within noise of the a-priori map.
* **The mechanism fires rarely.** 83% of post-burn-in rebalances are CALM; about
  7 crisis episodes in 14 years (2016, COVID twice, 2022).

---

## 4. Two bugs the controls caught

Both produced plausible numbers and broke no test. Recording them because each is
a reusable warning, and because the second one is the reason to trust the first.

### The falsification control had silently degenerated

`ShuffledLabelDetector` originally sampled labels from the HMM's fitted
transition matrix raised to the 21st power. The fitted chain is persistent for
CALM (monthly self-transition ~0.88) but not for the stressed states (~0.28), so
random draws almost never produced two consecutive turbulent or crisis months —
and the minimum-dwell smoother, doing exactly its job of rejecting
non-persistent labels, filtered them away. The control collapsed to **97% CALM
with 1.1 switches per run** against the treatment's 83% and 10. It had become a
static arm wearing a control's name, and because the CALM block is the
low-turnover configuration, it *looked like the best arm in the study*.

Fixed by permuting the treatment arm's realized label episodes, which preserves
marginals and dwell times exactly and destroys only the timing. Now 9.8 switches
per run with exactly matched marginals. `tests/unit/test_controls.py` asserts
both properties so it cannot silently regress.

### The mechanism-2 effect was almost entirely a window-length artifact

The regime-weighted estimator needs a five-year lookback because weighting
discards non-matching days. That longer window is itself stabilising. Adding
`moments_longwindow` — flat covariance, same window, no regime model — showed it
accounts for the whole effect and then some.

**The general lesson for the paper:** when a mechanism changes more than one
thing, every changed thing needs its own control arm. Both bugs were invisible in
the performance numbers and obvious the moment the right control existed.

---

## 5. What this means for the paper

RQ3 as originally posed — *does conditioning FPSO's search on a learned market
state improve risk-adjusted performance?* — is answered **no**, at two different
injection points, with controls that rule out the obvious alternative
explanations. That is a defensible ICAIF submission if framed as such:

1. **Lead with the turnover-matched result.** "Regime effects in constrained
   portfolio optimization are turnover effects" is a sharper and more useful
   claim than a marginal positive would have been — and F9 demonstrates it
   rather than arguing it. Add RQ4: *does the measured regime effect survive
   turnover matching?* It does not, at either injection point.
2. **Keep the negative results as the evidence,** not as an appendix. The oracle
   arms are what make the argument airtight: perfect regime knowledge, twice,
   buys nothing.
3. **Offer the harness and the protocol as the contribution.** A causal backtest
   with a machine-checked lookahead proof, matched-seed pairing, permutation
   controls that preserve marginals and dwell times, oracle upper bounds, and a
   turnover-matched control solver — which together caught two silent confounds
   in this very study — is worth releasing on its own. Framing the paper as an
   evaluation protocol whose finding happens to be negative puts the novelty in
   the method, so the negative result does not have to carry the paper alone.
4. **State the scope honestly.** One universe (top-50 US large caps), one
   objective, one metaheuristic, 2011-2024. The claim is about this setting.

Remaining work is prose, not code: retitle, rewrite the contributions around the
turnover finding, add turnover-matched comparison as a methodological
recommendation, and correct the paper-vs-code mismatches already fixed in the
implementation (repair-dominant constraints, the PSO term, monthly cadence,
point-in-time universe, delisting returns).

---

## 6. Reproducing

```bash
python -m fpso.experiments.build_cache      # one-time CRSP pull (needs WRDS)
python -m fpso.experiments.run_study        # all arms
python -m fpso.experiments.analyze --cost 0.01
```

The turnover-matched controls are solved separately, because they depend on a
completed treatment arm:

```bash
python -m fpso.experiments.match_turnover --target-arm regime_hmm
python -m fpso.experiments.match_turnover --target-arm moments_hmm
python -m fpso.experiments.run_study --arms static_matched_regime_hmm static_matched_moments_hmm
```

Each writes an arm config carrying the solved `lambda_t`, the target and achieved
turnover, and a timestamp, so the calibration is auditable.

Tables land in `paper/tables/`, figures F1-F9 in `paper/figures/`. Every result
directory carries a `manifest.json` with the config, git SHA and package
versions. `pytest -m "not integration"` (200 tests) runs offline against a
synthetic panel with no credentials.
