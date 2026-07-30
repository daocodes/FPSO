"""Inference for the arm comparisons.

Three tools, each answering a question a reviewer will actually ask:

* **Is the Sharpe difference real?** Daily returns are autocorrelated and
  heteroskedastic, so a naive t-test on a Sharpe difference is not valid. The
  stationary bootstrap of Politis and Romano (1994) resamples geometric-length
  blocks, preserving short-range dependence, and yields a confidence interval for
  any statistic of the paired paths.
* **Is it robust to the estimator?** :func:`ledoit_wolf_sharpe_test` implements
  the HAC-robust test of Ledoit and Wolf (2008) for the difference of two Sharpe
  ratios on overlapping samples, as an analytic cross-check on the bootstrap.
* **Did we look at too many comparisons?** :func:`holm_adjust` applies the
  Holm step-down correction across the arm comparisons.

Seeds are matched across arms, so all comparisons are paired: the same random
stream drives arm 1 seed 7 and arm 2 seed 7. Pairing removes the seed-level
variance component and roughly halves the width of the interval on the
difference, for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BootstrapInterval:
    """A point estimate with a bootstrap confidence interval and p-value."""

    statistic: float
    lower: float
    upper: float
    p_value: float
    confidence_level: float
    n_resamples: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero."""
        return self.lower > 0.0 or self.upper < 0.0

    def format(self) -> str:
        """Compact rendering for tables: ``0.183 [0.021, 0.344], p=0.031``."""
        return (
            f"{self.statistic:.3f} [{self.lower:.3f}, {self.upper:.3f}], "
            f"p={self.p_value:.3f}"
        )


def annualized_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe of a daily return array (risk-free rate = 0)."""
    r = np.asarray(returns, dtype=float)
    std = r.std(ddof=1)
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * r.mean() / std) if std > 0 else 0.0


class StationaryBootstrap:
    """Politis-Romano stationary bootstrap for paired return paths.

    Block lengths are geometric with mean `expected_block`; 21 days (one trading
    month) is the default because it spans the horizon over which daily equity
    returns show meaningful dependence without destroying the sample size.
    """

    def __init__(
        self,
        n_resamples: int = 5000,
        expected_block: int = 21,
        confidence_level: float = 0.95,
        seed: int = 20260101,
    ):
        self.n_resamples = n_resamples
        self.expected_block = expected_block
        self.confidence_level = confidence_level
        self.rng = np.random.default_rng(seed)

    def paired_sharpe_difference(
        self, treatment: pd.Series, control: pd.Series
    ) -> BootstrapInterval:
        """CI for Sharpe(treatment) - Sharpe(control) on a common date index.

        Both paths are resampled with the *same* block indices, which preserves
        the contemporaneous correlation between the two strategies — the reason a
        paired test is more powerful than two independent ones.
        """
        aligned = pd.concat([treatment, control], axis=1).dropna()
        a = aligned.iloc[:, 0].to_numpy(dtype=float)
        b = aligned.iloc[:, 1].to_numpy(dtype=float)
        observed = annualized_sharpe(a) - annualized_sharpe(b)

        draws = np.empty(self.n_resamples, dtype=float)
        for i in range(self.n_resamples):
            index = self._resample_index(len(a))
            draws[i] = annualized_sharpe(a[index]) - annualized_sharpe(b[index])

        return self._to_interval(observed, draws)

    def paired_statistic(
        self,
        treatment: pd.Series,
        control: pd.Series,
        statistic,
    ) -> BootstrapInterval:
        """CI for ``statistic(treatment) - statistic(control)`` under the same blocks.

        Used for the drawdown and volatility differences reported alongside Sharpe.
        """
        aligned = pd.concat([treatment, control], axis=1).dropna()
        a = aligned.iloc[:, 0].to_numpy(dtype=float)
        b = aligned.iloc[:, 1].to_numpy(dtype=float)
        observed = statistic(a) - statistic(b)

        draws = np.empty(self.n_resamples, dtype=float)
        for i in range(self.n_resamples):
            index = self._resample_index(len(a))
            draws[i] = statistic(a[index]) - statistic(b[index])

        return self._to_interval(observed, draws)

    def _resample_index(self, n: int) -> np.ndarray:
        """Draw one stationary-bootstrap index path of length `n`."""
        p_restart = 1.0 / self.expected_block
        index = np.empty(n, dtype=int)
        index[0] = self.rng.integers(n)
        restarts = self.rng.random(n) < p_restart
        new_starts = self.rng.integers(0, n, size=n)
        for t in range(1, n):
            index[t] = new_starts[t] if restarts[t] else (index[t - 1] + 1) % n
        return index

    def _to_interval(self, observed: float, draws: np.ndarray) -> BootstrapInterval:
        """Percentile interval, plus a two-sided p-value by interval inversion."""
        tail = (1.0 - self.confidence_level) / 2.0
        lower, upper = np.quantile(draws, [tail, 1.0 - tail])
        # Centre the draws on zero to obtain the null distribution of the
        # difference, then count how often it is at least as extreme as observed.
        centred = draws - draws.mean()
        p_value = float(np.mean(np.abs(centred) >= abs(observed)))
        return BootstrapInterval(
            statistic=float(observed),
            lower=float(lower),
            upper=float(upper),
            p_value=p_value,
            confidence_level=self.confidence_level,
            n_resamples=self.n_resamples,
        )


def ledoit_wolf_sharpe_test(
    treatment: pd.Series, control: pd.Series, block_size: int | None = None
) -> tuple[float, float]:
    """HAC-robust test for the difference of two Sharpe ratios (Ledoit-Wolf 2008).

    The delta-method standard error of the Sharpe difference is computed from the
    HAC (Newey-West) covariance of the four moments (mu_a, mu_b, m2_a, m2_b),
    which is what makes the test valid under the autocorrelation and volatility
    clustering present in daily equity returns.

    Note the estimator is method-of-moments throughout, so volatility is the
    uncorrected (population) second moment rather than the ddof=1 sample
    standard deviation used in the reported metrics table. The two differ by a
    factor sqrt(n/(n-1)) — about 0.03% at n = 1500 — so the Sharpe difference
    reported here can sit a hair away from ``metrics_table``'s. Correcting the
    moments would break the delta-method gradient below, so the estimator is left
    as derived and the discrepancy is documented instead.

    Returns:
        The Sharpe difference (annualized) and its two-sided p-value.
    """
    from scipy import stats

    aligned = pd.concat([treatment, control], axis=1).dropna()
    a = aligned.iloc[:, 0].to_numpy(dtype=float)
    b = aligned.iloc[:, 1].to_numpy(dtype=float)
    n = len(a)
    if n < 30:
        raise ValueError(f"Need at least 30 paired observations, got {n}.")

    mu_a, mu_b = a.mean(), b.mean()
    m2_a, m2_b = (a**2).mean(), (b**2).mean()
    sig_a, sig_b = np.sqrt(m2_a - mu_a**2), np.sqrt(m2_b - mu_b**2)
    if sig_a <= 0 or sig_b <= 0:
        raise ValueError("Both return paths must have positive volatility.")

    difference = mu_a / sig_a - mu_b / sig_b

    # Gradient of (mu_a/sig_a - mu_b/sig_b) w.r.t. (mu_a, mu_b, m2_a, m2_b).
    gradient = np.array(
        [
            m2_a / sig_a**3,
            -m2_b / sig_b**3,
            -0.5 * mu_a / sig_a**3,
            0.5 * mu_b / sig_b**3,
        ]
    )
    moments = np.column_stack([a, b, a**2, b**2])
    omega = _newey_west_covariance(moments, block_size or _default_lag(n))
    variance = float(gradient @ omega @ gradient) / n
    if variance <= 0:
        raise ValueError("Non-positive variance estimate; check the input paths.")

    z_statistic = difference / np.sqrt(variance)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_statistic))))
    return float(difference * np.sqrt(TRADING_DAYS_PER_YEAR)), p_value


def _default_lag(n: int) -> int:
    """Newey-West truncation lag, using the usual n^(1/4) rule."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** 0.25)))


def _newey_west_covariance(moments: np.ndarray, lag: int) -> np.ndarray:
    """HAC covariance of the demeaned moment conditions with Bartlett weights."""
    centred = moments - moments.mean(axis=0)
    n = len(centred)
    omega = centred.T @ centred / n
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma = centred[k:].T @ centred[:-k] / n
        omega += weight * (gamma + gamma.T)
    return omega


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment, controlling the family-wise error rate.

    Holm is used rather than Bonferroni because it is uniformly more powerful at
    the same FWER guarantee, and rather than Benjamini-Hochberg because the arm
    comparisons are few and each is individually load-bearing for the claim.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (m - rank) * p))
        adjusted[name] = running_max
    return adjusted


def max_drawdown_statistic(returns: np.ndarray) -> float:
    """Max drawdown as a plain function, for use inside the bootstrap."""
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def volatility_statistic(returns: np.ndarray) -> float:
    """Annualized volatility as a plain function, for use inside the bootstrap."""
    return float(np.std(np.asarray(returns, dtype=float), ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
