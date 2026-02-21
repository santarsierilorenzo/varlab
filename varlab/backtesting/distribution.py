"""
Distribution tests for VaR backtesting.

This module implements statistically rigorous tests used in Value-at-Risk
(VaR) model validation
"""

from __future__ import annotations

from typing import Iterable, Optional, Literal, Dict, Any, Union
from statsmodels.tsa.stattools import bds
from scipy.stats import norm, t, kstest
from dataclasses import dataclass
import pandas as pd
import numpy as np


Dist = Literal["normal", "t"]

@dataclass(frozen=True)
class DistributionTestResult:
    """
    Standardized result container for distribution tests.

    Attributes
    ----------
    test_name : str
        Name of the statistical test.
    statistic : Optional[float]
        Test statistic value (if applicable).
    p_value : Optional[float]
        P-value under the null hypothesis.
    reject : Optional[bool]
        True if H0 rejected at chosen alpha.
    info : Dict[str, Any]
        Additional diagnostic information.
    """
    test_name: str
    statistic: Optional[Union[float, np.ndarray]]
    p_value: Optional[Union[float, np.ndarray]]
    reject: Optional[bool]
    info: Dict[str, Any]


def _pit(
    values: Iterable[float],
    mu: float,
    sigma: float,
    dist: Dist = "normal",
    df: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the Probability Integral Transform (PIT).

    The PIT is defined as:

        U = F_X(X)

    where F_X is the cumulative distribution function of the assumed
    parametric distribution.

    Parameters
    ----------
    values : Iterable[float]
        Observations at which the CDF is evaluated.
    mu : float
        Location parameter of the distribution.
    sigma : float
        Scale parameter of the distribution. Must be strictly positive.
    dist : {"normal", "t"}, default="normal"
        Parametric distribution used to compute the CDF.
    df : float, optional
        Degrees of freedom of the Student's t distribution.
        Required if dist="t".

    Returns
    -------
    np.ndarray
        Array of PIT values in the interval [0, 1].

    Raises
    ------
    ValueError
        If sigma <= 0 or if df is missing/invalid when dist="t".
    NotImplementedError
        If the specified distribution is not supported.

    Notes
    -----
    For correctly specified predictive distributions, PIT values
    should be i.i.d. Uniform(0, 1).
    """
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be strictly positive")

    x: np.ndarray = np.asarray(values, dtype=float)
    d: str = dist.strip().lower()

    if d == "normal":
        return norm.cdf(x, loc=mu, scale=sigma)

    if d == "t":
        if df is None or df <= 0:
            raise ValueError(
                "df must be provided and > 0 for t distribution"
            )
        return t.cdf(x, df=df, loc=mu, scale=sigma)

    raise NotImplementedError(f"{dist} not implemented")


def _randomized_pit(
    history: Iterable[float],
    x: float,
) -> float:
    """
    Compute the randomized PIT for an empirical discrete distribution.

    The randomized PIT is defined as:

        U = F(x^-) + V * P(X = x)

    where:
        - F(x^-) is the left-limit of the empirical CDF,
        - P(X = x) is the empirical probability mass at x,
        - V ~ Uniform(0, 1).

    Parameters
    ----------
    history : Iterable[float]
        Historical observations defining the empirical distribution.
        NaN values are ignored.
    x : float
        Observation at which the randomized PIT is evaluated.

    Returns
    -------
    float
        Randomized PIT value in the interval [0, 1].

    Raises
    ------
    ValueError
        If history is empty after removing NaN values.
    """
    hist: np.ndarray = np.asarray(history, dtype=float)
    hist = hist[~np.isnan(hist)]

    n: int = hist.size
    if n == 0:
        raise ValueError("history must not be empty")

    less: int = int(np.sum(hist < x))
    equal: int = int(np.sum(hist == x))

    v: float = float(np.random.uniform(0.0, 1.0))

    return (less + v * equal) / n


def _rolling_pit(
    values: pd.Series,
    case: Literal["continuous", "discrete"] = "continuous",
    window: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """
    Compute rolling PIT over a time series.

    Parameters
    ----------
    values : pd.Series
        Time series of observations.
    case : {"continuous", "discrete"}
        Type of distribution assumption.
    window : int
        Rolling window size.
    ddof : int
        Delta degrees of freedom for rolling std.

    Returns
    -------
    pd.Series
        Rolling PIT values aligned with input index.
    """
    if case not in {"continuous", "discrete"}:
        raise ValueError("case must be either continuous or discrete")

    if window < 1:
        raise ValueError("window must be >= 1")

    values = values.astype(float)

    if case == "continuous":
        mu = (
            values
            .rolling(window)
            .mean()
            .shift(1)
        )
        
        sigma = (
            values
            .rolling(window)
            .std(ddof=ddof)
            .shift(1)
        )

        pit = norm.cdf(values, loc=mu, scale=sigma)

        return pit

    # Discrete case
    u = [np.nan] * len(values)

    for t in range(window, len(values)):
        hist = values.iloc[t - window:t].values
        x = values.iloc[t]
        u[t] = _randomized_pit(hist, x)

    return pd.Series(u, index=values.index)


def _expanding_pit(
    values: pd.Series,
    case: Literal["continuous", "discrete"] = "continuous",
    min_periods: int = 30,
    ddof: int = 1,
) -> pd.Series:
    """
    Compute expanding-window PIT over a time series.

    Parameters
    ----------
    values : pd.Series
        Time series of observations.
    case : {"continuous", "discrete"}
        Distribution assumption.
    min_periods : int
        Minimum number of past observations required to compute PIT.
    ddof : int
        Delta degrees of freedom for standard deviation estimation.

    Returns
    -------
    pd.Series
        Expanding PIT values aligned with input index.
        First `min_periods` values are NaN.
    """
    if case not in {"continuous", "discrete"}:
        raise ValueError("case must be either continuous or discrete")

    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    values = values.astype(float)

    u = [np.nan] * len(values)

    for t in range(min_periods, len(values)):
        hist = values.iloc[:t]

        if case == "continuous":
            mu = hist.mean()
            sigma = hist.std(ddof=ddof)

            if sigma <= 0 or np.isnan(sigma):
                continue

            u[t] = norm.cdf(values.iloc[t], loc=mu, scale=sigma)

        else:
            u[t] = _randomized_pit(hist.values, values.iloc[t])

    return pd.Series(u, index=values.index)


def _kolmogorov_test(
    pit: pd.Series,
    alpha: float = 0.05,
) -> DistributionTestResult:
    """
    Perform a Kolmogorov-Smirnov test for uniformity on PIT values.

    The function tests whether the provided PIT series follows
    a Uniform(0, 1) distribution.

    Parameters
    ----------
    pit : pd.Series
        Series of PIT values.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    DistributionTestResult
        Object containing:
        - test statistic
        - p-value
        - rejection decision
        - effective sample size

    Notes
    -----
    Under the null hypothesis:

        H0: U_t ~ Uniform(0, 1)

    This test evaluates the marginal uniformity condition only.
    KS assumes independence; results may be size-distorted when PIT is
    constructed via overlapping windows.
    """
    u = pd.Series(pit).dropna().astype(float)

    if len(u) < 20:
        raise ValueError("Sample too small for KS test")

    if (u < 0).any() or (u > 1).any():
        raise ValueError("PIT values outside [0, 1]")

    stat, pval = kstest(u.values, "uniform")

    return DistributionTestResult(
        test_name="Kolmogorov-Smirnov Uniform Test",
        statistic=float(stat),
        p_value=float(pval),
        reject=bool(pval < alpha),
        info={
            "sample_size": int(len(u)),
            "alpha": alpha,
        },
    )


def _independence_test(
    pit: pd.Series,
    max_dim: int = 4,
    alpha: float = 0.05,
) -> DistributionTestResult:
    """
    Perform BDS test for i.i.d. hypothesis on PIT values.

    H0: PIT series is i.i.d.
    """
    u = pd.Series(pit).dropna().astype(float)

    if len(u) < 100:
        raise ValueError("Sample too small for reliable BDS test")

    epsilon = 0.5 * np.std(u.values, ddof=1)

    stats, pvals = bds(
        u.values,
        max_dim=max_dim,
        epsilon=epsilon,
    )

    reject = bool(np.any(pvals < alpha))

    return DistributionTestResult(
        test_name="BDS Independence Test",
        statistic=stats,
        p_value=pvals,
        reject=reject,
        info={
            "sample_size": int(len(u)),
            "alpha": alpha,
            "max_dim": max_dim,
        },
    )


def pit_diagnostics(
    values: pd.Series,
    case: Literal["continuous", "discrete"] = "continuous",
    window_type: Literal["expanding", "rolling"] = "rolling",
    window: int = 60,
    min_periods: int = 60,
    ddof: int = 1,
    max_dim: int = 4,
    alpha: float = 0.05,
) -> Dict[str, DistributionTestResult]:
    """
    Run full PIT diagnostics: marginal uniformity and independence.

    Returns
    -------
    dict
        {
            "uniformity": KS result,
            "independence": BDS result
        }
    """
    if window_type == "rolling":
        pit = _rolling_pit(values, case, window, ddof)
    else:
        pit = _expanding_pit(values, case, min_periods, ddof)

    pit = pit.dropna()

    ks_res = _kolmogorov_test(pit, alpha=alpha)
    bds_res = _independence_test(
        pit,
        max_dim=max_dim,
        alpha=alpha,
    )

    return {
        "uniformity": ks_res,
        "independence": bds_res,
    }
