"""The paper's figures.

Each public function builds one figure, saves it to ``paper/figures/`` at print
resolution, and returns the path. They take plain pandas objects rather than
result containers so that a figure can be regenerated from saved Parquet without
re-running a backtest.

===== =====================================================================
F1    Market cumulative return with the detected-regime ribbon. The sanity
      plot: if the shaded crisis stretches do not line up with 2011, 2015-16,
      2018, 2020 and 2022, the detector is not finding regimes.
F2    Static vs. regime-conditioned FPSO — cumulative return and drawdown.
F3    Regime-conditional Sharpe and max drawdown, static vs. regime. This is
      where the risk story lives when the pooled difference is modest.
F4    Estimated transition matrix and regime dwell-time distribution: evidence
      that the HMM learned persistent, economically sensible states.
F5    Transaction-cost sensitivity of the headline comparison.
F6    Paired Sharpe differences vs. the static baseline with bootstrap
      intervals. The figure that carries the falsification result: if the
      headline arm and the shuffled control coincide, it is visible at a glance.
F7    Effect size as a function of transaction cost. The study's central
      result is cost-conditional — every mechanism here trades gross return for
      turnover reduction, so its sign flips somewhere in the cost sweep, and a
      single-cost table hides that entirely.
F8    Turnover against net-of-cost effect, one point per arm. The study's
      headline evidence: a single variable the mechanisms were never designed to
      control orders every arm in the matrix.
F9    The same effect measured against an unmatched and a turnover-matched
      control. The apparent effect rises steeply with cost; matched, it is flat
      and slightly negative. The clincher for the turnover explanation.
===== =====================================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fpso.evaluation.metrics import summarize
from fpso.evaluation.style import (
    BASELINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    REGIME_COLORS,
    REGIME_ORDER,
    SEQUENTIAL_HUE,
    SERIES_COLORS,
    SURFACE,
    add_caption,
    as_percent,
    label_line_end,
    pad_right,
    paper_style,
)

DEFAULT_FIGURE_DIR = Path("paper/figures")


def _save(fig, directory: Path, stem: str) -> Path:
    """Write both a PDF (for the camera-ready) and a PNG (for quick review).

    The style context is re-entered here on purpose. Callers save after their
    ``with paper_style()`` block has closed, at which point the print DPI and the
    tight bounding box are no longer in effect — and without the tight box, long
    tick labels and captions are cropped out of the saved file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / f"{stem}.pdf"
    with paper_style():
        fig.savefig(pdf_path)
        fig.savefig(directory / f"{stem}.png")
    plt.close(fig)
    return pdf_path


def _draw_regime_ribbon(ax, regime_labels: pd.Series) -> None:
    """Paint the acting regime as a full-strength ordinal strip.

    Consecutive equal labels are merged into one span so the ribbon reads as
    regime *episodes* rather than as a barcode of monthly decisions.
    """
    labels = regime_labels.dropna()
    if labels.empty:
        return

    span_start = labels.index[0]
    current = labels.iloc[0]
    for timestamp, label in labels.items():
        if label != current:
            ax.axvspan(span_start, timestamp, color=REGIME_COLORS.get(current, INK_MUTED), lw=0)
            span_start, current = timestamp, label
    ax.axvspan(span_start, labels.index[-1], color=REGIME_COLORS.get(current, INK_MUTED), lw=0)

    ax.set_yticks([])
    ax.set_ylabel("Regime", rotation=0, ha="right", va="center", labelpad=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[name], label=name.title())
        for name in REGIME_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=3,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.6,
    )


def plot_market_regimes(
    market_returns: pd.Series,
    regime_labels: pd.Series,
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f1_market_regimes",
) -> Path:
    """F1: market cumulative return above a ribbon of detected regimes."""
    with paper_style():
        fig, (ax_market, ax_ribbon) = plt.subplots(
            2, 1, figsize=(7.0, 3.6), height_ratios=[6, 1], sharex=True,
            gridspec_kw={"hspace": 0.12},
        )

        cumulative = (1.0 + market_returns).cumprod() - 1.0
        ax_market.plot(cumulative.index, cumulative.to_numpy(), color=SERIES_COLORS["static"])
        ax_market.axhline(0.0, color=BASELINE, lw=0.8)
        ax_market.set_ylabel("Cumulative market return")
        ax_market.set_title(
            "Detected market regimes over the evaluation period", loc="left"
        )
        as_percent(ax_market.yaxis)

        _draw_regime_ribbon(ax_ribbon, regime_labels)
        ax_ribbon.set_xlim(cumulative.index[0], cumulative.index[-1])

        add_caption(
            ax_ribbon,
            "Regime labels are the acting (post-smoothing) filtered estimates used for\n"
            "decisions at each rebalance; no future data enters any label.",
            offset_points=-46.0,
        )
    return _save(fig, directory, stem)


def plot_cumulative_and_drawdown(
    paths: dict[str, pd.Series],
    regime_labels: pd.Series | None = None,
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f2_cumulative_drawdown",
    title: str = "Static vs. regime-conditioned FPSO",
) -> Path:
    """F2: cumulative return and underwater curve for each arm."""
    colors = _assign_series_colors(paths)

    with paper_style():
        n_panels = 3 if regime_labels is not None else 2
        height_ratios = [5, 3, 1] if regime_labels is not None else [5, 3]
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(7.0, 5.0), height_ratios=height_ratios,
            sharex=True, gridspec_kw={"hspace": 0.15},
        )
        ax_cumulative, ax_drawdown = axes[0], axes[1]

        for name, path in paths.items():
            equity = (1.0 + path).cumprod()
            cumulative = equity - 1.0
            ax_cumulative.plot(
                cumulative.index, cumulative.to_numpy(), color=colors[name], label=name
            )
            label_line_end(
                ax_cumulative, cumulative.index[-1], cumulative.iloc[-1], name, colors[name]
            )

            underwater = equity / equity.cummax() - 1.0
            ax_drawdown.plot(
                underwater.index, underwater.to_numpy(), color=colors[name], label=name
            )

        ax_cumulative.axhline(0.0, color=BASELINE, lw=0.8)
        ax_cumulative.set_ylabel("Cumulative return")
        ax_cumulative.set_title(title, loc="left")
        ax_cumulative.legend(loc="upper left", ncol=len(paths))
        as_percent(ax_cumulative.yaxis)

        ax_drawdown.axhline(0.0, color=BASELINE, lw=0.8)
        ax_drawdown.set_ylabel("Drawdown")
        as_percent(ax_drawdown.yaxis)

        # Room for the end-of-line labels. Applied to the shared axis before the
        # ribbon copies the limits, so all three panels stay aligned.
        pad_right(ax_cumulative)

        if regime_labels is not None:
            _draw_regime_ribbon(axes[2], regime_labels)
            axes[2].set_xlim(*ax_cumulative.get_xlim())
    return _save(fig, directory, stem)


def plot_regime_conditional_bars(
    metrics_by_arm: dict[str, pd.DataFrame],
    metric_names: tuple[str, str] = ("sharpe", "max_drawdown"),
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f3_regime_conditional",
) -> Path:
    """F3: per-regime performance, one grouped-bar panel per metric.

    Args:
        metrics_by_arm: arm name -> the frame returned by
            :func:`fpso.evaluation.metrics.summarize_by_regime`.
    """
    colors = _assign_series_colors(metrics_by_arm)
    regimes = [
        r for r in REGIME_ORDER
        if any(r in frame.index for frame in metrics_by_arm.values())
    ]
    pretty = {"sharpe": "Sharpe ratio", "max_drawdown": "Maximum drawdown",
              "annual_volatility": "Annualized volatility", "annual_return": "Annualized return"}

    with paper_style():
        fig, axes = plt.subplots(1, len(metric_names), figsize=(7.0, 2.8))
        axes = np.atleast_1d(axes)

        positions = np.arange(len(regimes), dtype=float)
        # Bars are narrower than their slot, leaving a visible surface gap
        # between the paired bars rather than letting them touch.
        slot = 0.8 / max(len(metrics_by_arm), 1)
        width = slot * 0.82

        for ax, metric in zip(axes, metric_names, strict=True):
            for offset, (arm, frame) in enumerate(metrics_by_arm.items()):
                values = [
                    float(frame.loc[r, metric]) if r in frame.index else np.nan
                    for r in regimes
                ]
                x = positions + (offset - (len(metrics_by_arm) - 1) / 2) * slot
                bars = ax.bar(x, values, width=width, color=colors[arm], label=arm)
                _label_bars(ax, bars, values, metric)

            ax.set_xticks(positions)
            ax.set_xticklabels([r.title() for r in regimes])
            ax.axhline(0.0, color=BASELINE, lw=0.8)
            ax.set_title(pretty.get(metric, metric), loc="left", fontsize=10)
            if metric in {"max_drawdown", "annual_volatility", "annual_return"}:
                as_percent(ax.yaxis)

        axes[0].legend(loc="best", ncol=1)
        fig.suptitle(
            "Performance within each detected regime", x=0.0, ha="left",
            fontsize=11, fontweight="bold", color=INK_PRIMARY,
        )
        fig.tight_layout()
    return _save(fig, directory, stem)


def _label_bars(ax, bars, values, metric: str) -> None:
    """Direct-label bar values; the numbers are the point of this figure."""
    for bar, value in zip(bars, values, strict=True):
        if np.isnan(value):
            continue
        is_percentage = metric in {"max_drawdown", "annual_volatility", "annual_return"}
        text = f"{value:.0%}" if is_percentage else f"{value:.2f}"
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3 if value >= 0 else -11),
            textcoords="offset points",
            ha="center", fontsize=7.5, color=INK_SECONDARY,
        )


def plot_transition_and_dwell(
    transition_matrix: np.ndarray,
    dwell_lengths: dict[str, list[int]],
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f4_transition_dwell",
) -> Path:
    """F4: transition-probability heatmap and dwell-time distribution."""
    with paper_style():
        fig, (ax_heat, ax_dwell) = plt.subplots(1, 2, figsize=(7.0, 2.9))

        image = ax_heat.imshow(transition_matrix, cmap=SEQUENTIAL_HUE, vmin=0.0, vmax=1.0)
        labels = [r.title() for r in REGIME_ORDER[: transition_matrix.shape[0]]]
        ax_heat.set_xticks(range(len(labels)), labels)
        ax_heat.set_yticks(range(len(labels)), labels)
        ax_heat.set_xlabel("To")
        ax_heat.set_ylabel("From")
        ax_heat.set_title("Estimated daily transition matrix", loc="left", fontsize=10)
        ax_heat.grid(False)
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                value = transition_matrix[i, j]
                # Ink flips to white on the dark end of the ramp so every cell
                # value stays legible without a second colour scale.
                ax_heat.text(
                    j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if value > 0.55 else INK_PRIMARY,
                )
        fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04).outline.set_visible(False)

        offsets = np.linspace(-0.25, 0.25, max(len(dwell_lengths), 1))
        for offset, (regime, lengths) in zip(offsets, dwell_lengths.items(), strict=True):
            if not lengths:
                continue
            jitter = np.random.default_rng(0).normal(0, 0.03, len(lengths))
            ax_dwell.scatter(
                offset + jitter,
                lengths,
                color=REGIME_COLORS.get(regime, INK_MUTED), s=18, alpha=0.85, label=regime.title(),
                edgecolor="#fcfcfb", linewidth=0.5,
            )
        ax_dwell.set_xticks([])
        ax_dwell.set_ylabel("Episode length (rebalances)")
        ax_dwell.set_title("Regime dwell times", loc="left", fontsize=10)
        ax_dwell.legend(loc="upper right")

        fig.tight_layout()
    return _save(fig, directory, stem)


def plot_cost_sensitivity(
    returns_by_arm_and_cost: dict[str, dict[float, pd.Series]],
    metric: str = "sharpe",
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f5_cost_sensitivity",
) -> Path:
    """F5: how the headline metric degrades as transaction costs rise."""
    colors = _assign_series_colors(returns_by_arm_and_cost)

    with paper_style():
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        for arm, by_cost in returns_by_arm_and_cost.items():
            rates = sorted(by_cost)
            values = [getattr(summarize(by_cost[rate]), metric) for rate in rates]
            ax.plot(rates, values, marker="o", color=colors[arm], label=arm)
            label_line_end(ax, rates[-1], values[-1], arm, colors[arm])

        pad_right(ax, 0.22)
        ax.set_xlabel("One-way transaction cost")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title("Sensitivity to transaction costs", loc="left")
        as_percent(ax.xaxis, decimals=1)
        ax.legend(loc="best")
    return _save(fig, directory, stem)


def plot_effect_sizes(
    comparisons: pd.DataFrame,
    baseline_name: str = "static",
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f6_effect_sizes",
) -> Path:
    """F6: paired Sharpe difference vs. the baseline, with bootstrap intervals.

    A forest plot rather than a bar chart: the quantity being compared is an
    *estimate with uncertainty*, and bars would imply a precision the intervals
    say is not there. Rows are directly labelled, so identity never rests on
    colour, and a filled marker marks an interval that excludes zero — the
    secondary encoding that keeps the significance signal readable in greyscale
    and for a colourblind reader.

    This is the figure that carries the falsification result: if the headline arm
    and the shuffled control sit on top of each other, that is visible at a glance.

    Args:
        comparisons: the frame returned by
            :func:`fpso.experiments.analyze.comparison_table`.
    """
    frame = comparisons.sort_values("delta_sharpe")
    labels = [str(index).replace("_", " ") for index in frame.index]

    with paper_style():
        fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(frame) + 1.6))
        positions = np.arange(len(frame), dtype=float)

        ax.axvline(0.0, color=BASELINE, lw=1.0)
        for y, (name, row) in zip(positions, frame.iterrows(), strict=True):
            significant = row["delta_sharpe_lo"] > 0 or row["delta_sharpe_hi"] < 0
            colour = _effect_colour(str(name))
            ax.plot(
                [row["delta_sharpe_lo"], row["delta_sharpe_hi"]], [y, y],
                color=colour, lw=2.0, solid_capstyle="round",
            )
            ax.plot(
                row["delta_sharpe"], y,
                marker="o", markersize=7,
                color=colour if significant else SURFACE,
                markeredgecolor=colour, markeredgewidth=1.8,
            )

        ax.set_yticks(positions, labels)
        ax.set_xlabel(f"Sharpe difference vs. {baseline_name.replace('_', ' ')}")
        ax.set_title("Paired effect on risk-adjusted return", loc="left")
        ax.grid(axis="y", visible=False)

        add_caption(
            ax,
            "Points are paired differences on matched seeds; bars are 95% stationary-\n"
            "bootstrap intervals. Filled markers denote an interval excluding zero.",
            offset_points=-40.0,
        )
    return _save(fig, directory, stem)


def _effect_colour(arm_name: str) -> str:
    """Colour one forest-plot row by the role that arm plays in the experiment."""
    key = arm_name.lower()
    if "shuffled" in key or "oracle" in key:
        return SERIES_COLORS["control"]
    if "regime" in key:
        return SERIES_COLORS["regime"]
    return SERIES_COLORS["baseline"]


def plot_cost_conditional_effects(
    returns_by_arm_and_cost: dict[str, dict[float, pd.Series]],
    baseline: str,
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f7_cost_conditional_effect",
) -> Path:
    """F7: Sharpe difference vs. the baseline, traced across transaction costs.

    Reporting a single cost level would be misleading here. Every mechanism in
    this study buys turnover reduction at the price of gross return, so each
    curve crosses zero somewhere — and *where* it crosses is the economically
    meaningful quantity, not the value at whichever cost the table happened to
    fix.

    Args:
        returns_by_arm_and_cost: arm name -> {cost rate -> seed-averaged returns}.
            Must include `baseline`, which is subtracted and not drawn.
    """
    control = returns_by_arm_and_cost[baseline]
    treatments = {k: v for k, v in returns_by_arm_and_cost.items() if k != baseline}
    colors = _assign_series_colors(treatments)

    with paper_style():
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.axhline(0.0, color=BASELINE, lw=1.0)

        for arm, by_cost in treatments.items():
            rates = sorted(set(by_cost) & set(control))
            deltas = [
                summarize(by_cost[r]).sharpe - summarize(control[r]).sharpe for r in rates
            ]
            ax.plot(rates, deltas, marker="o", color=colors[arm], label=arm)
            label_line_end(ax, rates[-1], deltas[-1], arm, colors[arm])

        pad_right(ax, 0.30)
        ax.set_xlabel("One-way transaction cost")
        ax.set_ylabel(f"Sharpe difference vs. {baseline}")
        ax.set_title("Where each mechanism starts to pay", loc="left")
        as_percent(ax.xaxis, decimals=1)
        ax.legend(loc="upper left")
        add_caption(
            ax,
            "Above the zero line the mechanism beats the baseline. Every curve rises with\n"
            "cost because each one trades gross return for lower turnover.",
            offset_points=-42.0,
        )
    return _save(fig, directory, stem)


def plot_turnover_matched_comparison(
    panels: dict[str, dict[str, dict[float, pd.Series]]],
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f9_turnover_matched",
) -> Path:
    """F9: each mechanism's effect against an unmatched and a matched control.

    The two curves in each panel answer different questions. Against plain
    `static` the effect climbs steeply with transaction cost — the result a paper
    without turnover controls would report. Against a `static` arm whose turnover
    penalty has been tuned to trade exactly as much, the curve is flat and sits
    slightly below zero: the cost-sensitivity disappears because the channel it
    travelled through has been closed.

    Args:
        panels: mechanism label -> {"treatment"/"unmatched"/"matched" -> cost sweep}.
    """
    with paper_style():
        fig, axes = plt.subplots(
            1, len(panels), figsize=(7.2, 3.2), sharey=True
        )
        axes = np.atleast_1d(axes)

        for ax, (label, series) in zip(axes, panels.items(), strict=True):
            treatment = series["treatment"]
            for control_key, control_label, colour in (
                ("unmatched", "vs. static", SERIES_COLORS["static"]),
                ("matched", "vs. turnover-matched static", SERIES_COLORS["regime"]),
            ):
                control = series[control_key]
                rates = sorted(set(treatment) & set(control))
                deltas = [
                    summarize(treatment[r]).sharpe - summarize(control[r]).sharpe
                    for r in rates
                ]
                ax.plot(rates, deltas, marker="o", color=colour, label=control_label)

            ax.axhline(0.0, color=BASELINE, lw=1.0)
            ax.set_title(label, loc="left", fontsize=10)
            ax.set_xlabel("One-way transaction cost")
            as_percent(ax.xaxis, decimals=1)

        axes[0].set_ylabel("Sharpe difference")
        axes[0].legend(loc="upper left")
        fig.suptitle(
            "Matching turnover removes the effect", x=0.0, ha="left",
            fontsize=11, fontweight="bold", color=INK_PRIMARY,
        )
        fig.tight_layout()
        add_caption(
            axes[0],
            "Blue is what a study without turnover controls would report. Orange holds\n"
            "turnover fixed and the effect vanishes, flat and slightly negative.",
            offset_points=-52.0,
        )
    return _save(fig, directory, stem)


ARM_FAMILIES = (
    ("moments", "Regime-weighted beliefs"),
    ("pinned", "Pinned parameter level"),
    ("regime", "Regime-conditioned preferences"),
)


def plot_turnover_explains_effect(
    turnover: pd.Series,
    effect: pd.Series,
    directory: Path = DEFAULT_FIGURE_DIR,
    stem: str = "f8_turnover_explains_effect",
    cost_label: str = "1.5%",
) -> Path:
    """F8: net-of-cost effect against realized turnover, one point per arm.

    Every arm in the matrix — two injection points for the regime signal, four
    detectors, two falsification controls, a lookahead oracle and three external
    baselines — collapses onto one downward line. The rank correlation is
    reported in the caption because it is the paper's central claim: once
    turnover is accounted for, the regime signal has nothing left to explain.

    Args:
        turnover: mean one-way turnover per rebalance, indexed by arm.
        effect: Sharpe difference vs. the baseline at `cost_label`, same index.
    """
    from scipy import stats

    frame = pd.concat({"turnover": turnover, "effect": effect}, axis=1).dropna()
    rho, p_value = stats.spearmanr(frame["turnover"], frame["effect"])

    with paper_style():
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.axhline(0.0, color=BASELINE, lw=1.0)

        # Least-squares guide line. Drawn thin and behind the marks: it carries no
        # inferential weight, it just makes the direction legible at a glance.
        slope, intercept = np.polyfit(frame["turnover"], frame["effect"], 1)
        span = np.array([frame["turnover"].min(), frame["turnover"].max()])
        ax.plot(span, slope * span + intercept, color=INK_MUTED, lw=1.0, zorder=1)

        for family, label in ARM_FAMILIES + (("", "Baselines and ablations"),):
            members = [
                arm for arm in frame.index
                if (family in str(arm)) if family
            ] if family else [
                arm for arm in frame.index
                if not any(f in str(arm) for f, _ in ARM_FAMILIES)
            ]
            if not members:
                continue
            ax.scatter(
                frame.loc[members, "turnover"], frame.loc[members, "effect"],
                s=46, label=label, zorder=2,
                color=_family_colour(family), edgecolor=SURFACE, linewidth=1.0,
            )

        for arm in frame.index:
            ax.annotate(
                str(arm).replace("_", " "),
                xy=(frame.loc[arm, "turnover"], frame.loc[arm, "effect"]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=6.8, color=INK_SECONDARY,
            )

        pad_right(ax, 0.12)
        ax.set_xlabel("Mean one-way turnover per rebalance")
        ax.set_ylabel(f"Sharpe difference vs. static (at {cost_label} cost)")
        ax.set_title("Turnover orders every arm in the study", loc="left")
        ax.legend(loc="upper right")
        add_caption(
            ax,
            f"Spearman rho = {rho:.2f} (p = {p_value:.1e}, n = {len(frame)}). Gross of costs the\n"
            "same correlation is insignificant, so this is an economic effect rather than "
            "an identity.",
            offset_points=-42.0,
        )
    return _save(fig, directory, stem)


def _family_colour(family: str) -> str:
    """One hue per mechanism family, stable across figures."""
    return {
        "moments": SERIES_COLORS["regime"],
        "pinned": SERIES_COLORS["control"],
        "regime": SERIES_COLORS["static"],
    }.get(family, SERIES_COLORS["baseline"])


def _assign_series_colors(named) -> dict[str, str]:
    """Map arm names onto the categorical palette in a fixed, meaningful order.

    Colour follows the *entity*: an arm keeps its hue across every figure, and
    dropping an arm from a chart never repaints the ones that remain.
    """
    # Ordered most-specific first: "regime_shuffled" must match "shuffled" and
    # take the control colour, not fall through to the headline regime colour.
    role_for = (
        ("shuffled", "control"),
        ("oracle", "control"),
        ("static", "static"),
        ("regime", "regime"),
        ("hmm", "regime"),
    )
    palette_order = ["static", "regime", "baseline", "control"]

    if len(named) > len(palette_order):
        raise ValueError(
            f"{len(named)} series exceeds the {len(palette_order)}-slot categorical "
            "palette. Facet the chart or fold the extras into an 'Other' series "
            "rather than generating new hues, which would break CVD separation."
        )

    assigned: dict[str, str] = {}
    used: set[str] = set()
    for name in named:
        key = str(name).lower()
        role = next((role for token, role in role_for if token in key), None)
        if role is None or role in used:
            role = next(r for r in palette_order if r not in used)
        used.add(role)
        assigned[name] = SERIES_COLORS[role]
    return assigned
