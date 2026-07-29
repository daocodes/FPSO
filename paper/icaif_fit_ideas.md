# Making FPSO Fit ICAIF '26 — Ideas, Ranked by (Scope Fit × Feasibility in 1 Week)

## The honest diagnosis

FPSO as written is a **metaheuristics / operations-research paper**. ICAIF's topic list
never mentions swarm intelligence, evolutionary computation, or metaheuristics — every
methodology bullet is learning-based (RL, LLMs, agents, DFL, meta-learning, UQ).
A reviewer's most likely rejection reason: *"nice optimizer, but where is the AI?"*

The fix is **not** to bolt an unrelated model onto the paper. It is to make one of the
*decisions FPSO already makes internally* into a **learned decision**, and reframe the
paper around that. Three internal decisions are candidates:

1. **When to explore vs. exploit** (the FA↔PSO phase transition — currently a fixed
   diversity threshold).
2. **What the hyperparameters should be** (`beta_0`, `gamma`, `alpha`, `lambda_v`,
   `K`, penalty weights — currently static across the whole backtest).
3. **What inputs to trust** (`mu`, `Sigma` — currently raw historical estimates).

Each idea below attacks one of these. All effort estimates assume the existing
backtest harness works and CRSP data is already processed.

---

## Tier 1 — Recommended (do one of these two, ideally both since they compose)

### Idea 1: Regime-Conditioned FPSO ("classification regime" — your idea, and it's the right one)

**One-liner:** An unsupervised market-regime classifier (HMM or GMM on rolling
return/volatility features) detects the current market state at each rebalance, and
FPSO's search behavior and risk parameters are **conditioned on the detected regime**.

**What you actually build (3–4 days):**
- Fit a 2–3 state Gaussian HMM (`hmmlearn`, ~30 lines) on rolling S&P 500 return +
  realized-vol features. States interpret naturally as *calm / turbulent / crisis*.
  Fit only on trailing data at each rebalance — no lookahead.
- Map each regime to an FPSO configuration. Concretely:
  - **Calm:** low `alpha` (less random walk), high `lambda_t` (low turnover),
    higher `K` — exploit, stay diversified, trade little.
  - **Turbulent:** high `alpha`, high `gamma` (localized attraction), higher
    `lambda_v` — explore more, penalize variance harder.
  - **Crisis:** tighter upper bound `u`, lower `K`, highest `lambda_v` — concentrate
    into defensive names, cut risk.
- Rerun the existing backtest with the regime switch on. Add one ablation row:
  *FPSO (static) vs. Regime-FPSO* — this ablation **is** the paper's new headline result.
  2010–2024 conveniently contains real regime breaks (2015-16, Volmageddon 2018,
  COVID 2020, 2022 rate shock), so the story writes itself.

**Why it's the best fit:**
- Hits **"Financial time series analysis," "Forecasting of financial scenarios,"
  "AI-driven risk management,"** and (with an HMM) **"validation and calibration of
  financial models."**
- Addresses a *real* gap you can defend in Related Work: the metaheuristic portfolio
  literature (Chang 2000 → Cura 2009 → Deng 2012 → Bacanin 2014) is uniformly
  **regime-blind** — one parameterization for all market states. Learning-based
  allocators adapt but can't handle hard constraints; metaheuristics handle constraints
  but don't adapt. Regime-FPSO sits exactly in the intersection, which is exactly the
  positioning your Section "Positioning of This Work" already argues.
- Zero risk to the existing results: static FPSO becomes the ablation baseline, so all
  the work you've already done stays in the paper.

**New paper title flavor:** *"Regime-Adaptive Hybrid Swarm Optimization for
Cardinality-Constrained Portfolio Selection."*

**Risks:** HMM regime labels can be noisy at daily frequency → use monthly rebalance
dates (you already do) and a minimum-dwell smoothing. If regime-FPSO doesn't beat
static FPSO on returns, it will almost certainly beat it on **max drawdown and
volatility** — lead with the risk-adjusted story in that case.

---

### Idea 2: Learned Exploration–Exploitation Controller (bandit/RL over the FA↔PSO switch)

**One-liner:** Replace the hand-tuned diversity-threshold phase transition with a
**contextual bandit / lightweight RL policy** that decides, each iteration, whether the
swarm takes an FA step (explore) or a PSO step (exploit).

**What you actually build (3–4 days):**
- State: 3–4 cheap features you already compute or nearly compute — swarm diversity
  (mean pairwise distance), iterations since last global-best improvement, normalized
  iteration count, current best-fitness slope.
- Actions: {FA update, PSO update} — or three actions if you add "restart worst decile."
- Reward: improvement in global best fitness this iteration.
- Algorithm: **don't** reach for deep RL. A tabular Q-learner over discretized states
  or a LinUCB contextual bandit is ~100 lines, trains in minutes across backtest
  windows, and is fully explainable (you can *plot the learned policy* — great figure:
  "the controller learns to explore early and after regime breaks, exploit late").
- Note: `fpso_optimizer.py` currently runs a single blended update rather than the
  two-phase design the paper describes — you need to implement the explicit FA-step /
  PSO-step separation anyway to match the paper text. This idea makes that mandatory
  refactor into a contribution.

**Why it fits:**
- Directly hits **"Reinforcement learning and sequential decision-making"** and
  **"AI agents and multi-agent systems"** (swarm = agents, controller = meta-agent).
- Turns the paper's currently weakest, most hand-wavy component (the phase-transition
  threshold, flagged in the intro with `[add source here]`) into its most novel one.
  Learned control of metaheuristics ("learning to optimize") is an active gap —
  almost nothing applies it to constrained portfolio search.
- The paper's own conclusion already promises RL integration as future work — you'd be
  delivering it.

**Risks:** RL adds a "did you tune the RL fairly?" review question — preempt with the
ablation (fixed-threshold vs. random-switch vs. learned-switch). Random-switch is the
killer baseline: if learned beats random, the contribution is proven.

---

### Why Tier 1 first: Ideas 1 + 2 compose into one clean narrative

*"FPSO's search behavior is controlled by learned components at two timescales:
a regime classifier adapts the objective/constraints across rebalances (slow), and a
bandit controller adapts exploration–exploitation within each optimization run (fast)."*

That is a genuinely ICAIF-shaped paper: sequential decision-making + time-series
learning + constraint-native optimization. If you only have time for one, **do Idea 1**
— it's lower variance, needs no refactor of the optimizer core, and its ablation story
is the easiest to write.

---

## Tier 2 — Good, feasible, but weaker as the *headline*

### Idea 3: LLM-Seeded Swarm ("agentic fireflies," the defensible version)

**One-liner:** An LLM agent reads a structured summary of the current market window
(sector momentum, vol, macro headlines) and proposes 5–10 **candidate portfolios that
seed part of the initial swarm**; FPSO then refines/repairs them under hard constraints.

- **Fit:** "LLMs for financial workflows," "AI agents." And it inverts your own LLMO
  baseline elegantly: instead of LLM *vs.* swarm, it's LLM *priors* + swarm
  *constraint enforcement* — directly answering the constraint-fidelity criticism you
  level at Hou 2024 in Related Work.
- **Feasibility (2–3 days):** cheap — one LLM call per rebalance (~180 calls per
  backtest), parse tickers/weights, repair with the existing
  `repair_portfolio_constraints`, dirichlet-fill the rest of the swarm. Ablation:
  random seeds vs. LLM seeds vs. LLM-only (your existing LLMO baseline).
- **Why not headline:** reviewers have seen many "we prompted GPT for portfolios"
  papers; as a *seeding* mechanism it's a solid secondary experiment but a risky main
  contribution. Also adds API cost/reproducibility questions (pin the model, log all
  prompts, release them).
- **Caution:** LLM training-data leakage — the model may "know" that e.g. NVDA mooned
  post-2020. Mitigate by only giving it anonymized/recent-window features, or
  acknowledge it explicitly. Reviewers *will* raise this.

### Idea 4: Full "agentic swarm" — heterogeneous firefly agents

Fireflies get roles (explorer / exploiter / contrarian) with different `alpha`/`gamma`,
possibly role reassignment by a controller. This is really Idea 2 wearing a costume —
if you want the multi-agent framing, implement Idea 2 and *describe* the swarm as a
multi-agent system in the writing. Don't build a separate agent framework in a week.

---

## Tier 3 — Fits scope, but skip given one week

- **Idea 5: DFL-style prediction feeding FPSO.** Train a forecaster (LightGBM/LSTM)
  for `mu` and pipe it into FPSO — "predict-then-optimize with a constraint-native
  optimizer" as a rebuttal to end-to-end DFL. Genuinely interesting, but doing the
  forecasting *credibly* (proper CV, no leakage) is its own paper and reviewers will
  judge the forecasting quality. Too much surface area for a week.
- **Idea 6: Uncertainty quantification.** Bootstrap-resample `mu`/`Sigma`, report
  weight stability and return distributions across resamples. Hits "robustness and
  uncertainty quantification" and is only ~1–2 days — **worth doing as a supporting
  subsection regardless of headline choice**, since it also strengthens the currently
  thin validation section. But it can't carry scope-fit alone.
- **Idea 7: Explainability of swarm decisions.** Attribute final weights to search
  trajectory / regime states. Nice figure material, not a contribution.

---

## What addresses the biggest gap in the research space?

The gap your own Related Work section sets up is: **learning-based methods can't handle
hard combinatorial constraints; constraint-native metaheuristics can't learn or adapt.**
The highest-value contribution is therefore *any* mechanism where a learned component
steers a constraint-native optimizer. Ideas 1, 2, and 3 are all instances of this one
thesis at different timescales (rebalance-level, iteration-level, initialization-level).
Pick the thesis as your framing and present whichever mechanisms you finish as its
instantiations — that way even partial results tell one coherent story.

---

## Suggested one-week plan (Idea 1 primary, Idea 2 stretch, Idea 6 filler)

| Day | Work |
|---|---|
| 1 | Fit HMM regime classifier on trailing features; sanity-plot regimes vs. SPX 2010–2024 (this plot goes in the paper). |
| 2 | Wire regime → FPSO parameter map into the backtest loop; pick the 3 regime configs. |
| 3 | Run full backtests: static FPSO, Regime-FPSO, existing baselines. |
| 4 | If ahead: implement two-phase FA/PSO split + bandit controller (Idea 2). If behind: bootstrap UQ runs (Idea 6). |
| 5 | Ablations (regime on/off; if Idea 2 done: fixed vs. random vs. learned switch) + stat tests. |
| 6 | Rewrite intro/related-work/contributions around "learning-guided constraint-native optimization"; new figures; retitle. |
| 7 | Buffer, polish, fill the remaining `[add source here]` placeholders, internal read-through. |

**Paper edits that must accompany whichever idea you pick:**
- Retitle and rewrite the contributions list to lead with the learned component.
- Add regime detection (or RL control) to Related Work — cite HMM regime-switching
  (Hamilton 1989; Ang & Bekaert 2002) or learning-to-optimize literature respectively.
- Move pure-metaheuristic material (centroid derivation details) toward an appendix;
  ICAIF reviewers care about the learning loop and the financial results.
- Keep RQ1/RQ2 but add **RQ3:** *"Does conditioning FPSO's search on a learned market
  state improve risk-adjusted performance over a static parameterization?"*
