"""
Distribution tests for VaR backtesting.

This module implements statistically rigorous tests used in
Value-at-Risk (VaR) model validation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Literal, Dict, Any, Union
from statsmodels.tsa.stattools import bds
from scipy.stats import norm, t, kstest, chi2
from dataclasses import dataclass
import pandas as pd
import numpy as np


Dist = Literal["normal", "t"]


@dataclass(frozen=True)
class DistributionTestResult:
    """
    Standardized result container for distribution tests.
    """
    test_name: str
    statistic: Optional[Union[float, np.ndarray]]
    p_value: Optional[Union[float, np.ndarray]]
    reject: Optional[bool]
    info: Dict[str, Any]


def _pit_checks(
    case: Literal["continuous", "discrete"],
    distribution: Dist,
    df: Optional[int],
) -> None:
    if case not in {"continuous", "discrete"}:
        raise ValueError("case must be either 'continuous' or 'discrete'")

    if distribution not in {"normal", "t"}:
        raise ValueError("distribution must be 'normal' or 't'")

    if distribution == "t":
        if df is None or df <= 2:
            raise ValueError("df must be provided and > 2 for t distribution")


def _randomized_pit(
    history: Iterable[float],
    x: float,
) -> float:
    """
    Randomized PIT for empirical discrete distribution.

    U = F(x^-) + V * P(X = x),  V ~ Uniform(0,1)
    """
    hist = np.asarray(history, dtype=float)
    hist = hist[~np.isnan(hist)]

    n = hist.size
    if n == 0:
        raise ValueError("history must not be empty")

    less = np.sum(hist < x)
    equal = np.sum(hist == x)

    v = np.random.uniform(0.0, 1.0)

    return float((less + v * equal) / n)


def _rolling_pit(
    values: pd.Series,
    case: Literal["continuous", "discrete"] = "continuous",
    distribution: Dist = "normal",
    df: Optional[int] = None,
    window: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """
    Rolling PIT consistent with mean-zero parametric VaR.

    Assumes:
        X_t = sigma_t * Z_t
        Z_t ~ D(0,1)
    """
    _pit_checks(case, distribution, df)

    if window < 1:
        raise ValueError("window must be >= 1")

    values = values.astype(float)

    if case == "continuous":

        sigma = values.rolling(window).std(ddof=ddof).shift(1)

        pit = pd.Series(np.nan, index=values.index)

        valid = (~sigma.isna()) & (sigma > 0)

        if not valid.any():
            return pit

        z = values[valid] / sigma[valid]

        if distribution == "normal":
            pit[valid] = norm.cdf(z)
        else:
            pit[valid] = t.cdf(z, df=df)

        return pit

    # Discrete case
    u = [np.nan] * len(values)

    for t_idx in range(window, len(values)):
        hist = values.iloc[t_idx - window:t_idx].values
        x = values.iloc[t_idx]
        u[t_idx] = _randomized_pit(hist, x)

    return pd.Series(u, index=values.index)


def _expanding_pit(
    values: pd.Series,
    case: Literal["continuous", "discrete"] = "continuous",
    distribution: Dist = "normal",
    df: Optional[int] = None,
    min_periods: int = 30,
    ddof: int = 1,
) -> pd.Series:
    """
    Expanding-window PIT consistent with mean-zero parametric VaR.
    """
    _pit_checks(case, distribution, df)

    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    values = values.astype(float)
    u = [np.nan] * len(values)

    for t_idx in range(min_periods, len(values)):
        hist = values.iloc[:t_idx]

        if case == "continuous":
            sigma = hist.std(ddof=ddof)

            if sigma <= 0 or np.isnan(sigma):
                continue

            z = values.iloc[t_idx] / sigma

            if distribution == "normal":
                u[t_idx] = norm.cdf(z)
            else:
                u[t_idx] = t.cdf(z, df=df)

        else:
            u[t_idx] = _randomized_pit(hist.values, values.iloc[t_idx])

    return pd.Series(u, index=values.index)


def _kolmogorov_test(
    pit: pd.Series,
    alpha: float = 0.05,
) -> DistributionTestResult:
    """
    Kolmogorov-Smirnov test for PIT uniformity.

    H0: U_t ~ Uniform(0,1)

    NOTE:
        Assumes independence. Size may be distorted under rolling PIT.
    """
    u = pd.Series(pit).dropna().astype(float)

    if len(u) < 20:
        raise ValueError("Sample too small for KS test")

    if (u < 0).any() or (u > 1).any():
        raise ValueError("PIT values outside [0,1]")

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
    BDS test for i.i.d. hypothesis on PIT values.

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


from scipy.stats import norm, chi2


def _berkowitz_test(
    pit: pd.Series,
    alpha: float = 0.05,
    eps: float = 1e-12,
) -> DistributionTestResult:
    """
    Berkowitz (2001) likelihood ratio test for conditional density forecasts.

    The test proceeds as follows:

        1. Apply the probit transform:
               z_t = Phi^{-1}(u_t)

        2. Estimate the Gaussian AR(1) model:
               z_t = c + rho z_{t-1} + epsilon_t,
               epsilon_t ~ N(0, sigma^2)

        3. Perform a likelihood ratio (LR) test against the restricted model:
               z_t ~ i.i.d. N(0,1)

    Null hypothesis:
        H0: c = 0, rho = 0, sigma^2 = 1

    Under H0, the LR statistic is asymptotically chi-square with 3 degrees
    of freedom.
    """
    # Prepare PIT series
    u = pd.Series(pit).dropna().astype(float)

    if len(u) < 50:
        raise ValueError("Sample too small for Berkowitz test")

    # Probit transform requires values strictly inside (0,1).
    # We clip to avoid infinite values in norm.ppf.
    if (u <= 0).any() or (u >= 1).any():
        u = np.clip(u.values, eps, 1 - eps)
    else:
        u = u.values

    # Step 1: Probit transform
    # Under H0: z_t ~ i.i.d. N(0,1)
    z = norm.ppf(u)

    # Restricted log-likelihood (H0: i.i.d. N(0,1))
    # l_res = sum log phi(z_t)
    l_res = np.sum(norm.logpdf(z))

    # Unrestricted model: Gaussian AR(1)
    #
    # Since the model is linear and Gaussian:
    #   - The MLE of (c, rho) coincides with OLS estimates.
    #   - The MLE of sigma^2 is the mean of squared residuals
    #     (i.e., SSR / (T-1)).
    #
    # We use the conditional likelihood (conditioning on z_1).
    y = z[1:]
    X = np.column_stack([np.ones(len(y)), z[:-1]])

    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    c_hat = beta_hat[0]
    rho_hat = beta_hat[1]

    resid = y - X @ beta_hat

    # MLE of variance (NOT the unbiased estimator)
    sigma2_hat = np.mean(resid ** 2)

    T = len(z)

    # Unrestricted log-likelihood (conditional)
    #
    # l_unres = -(T-1)/2 * [ log(2π) + log(sigma^2_hat) + 1 ]
    #
    # The "+1" term comes from substituting the MLE of sigma^2
    # into the Gaussian likelihood.
    ll_unres = -((T - 1) / 2) * (
        np.log(2 * np.pi)
        + np.log(sigma2_hat)
        + 1
    )

    # Likelihood ratio statistic
    #
    # LR = 2 (l_unres - l_res)
    #
    # Theoretically LR >= 0 (nested models), but small negative values
    # may arise from floating-point precision.
    lr_stat = 2 * (ll_unres - l_res)
    lr_stat = max(lr_stat, 0.0)

    # Survival function is numerically more stable than 1 - CDF
    pval = chi2.sf(lr_stat, df=3)

    return DistributionTestResult(
        test_name="Berkowitz LR Test",
        statistic=float(lr_stat),
        p_value=float(pval),
        reject=bool(pval < alpha),
        info={
            "sample_size": int(len(z)),
            "alpha": alpha,
            "c_hat": float(c_hat),
            "rho_hat": float(rho_hat),
            "sigma2_hat": float(sigma2_hat),
            "ll_res": float(l_res),
            "ll_unres": float(ll_unres),
            "df": 3,
        },
    )


def pit_diagnostics(
    values: pd.Series,
    case: Literal["continuous", "discrete"] = "continuous",
    distribution: Dist = "normal",
    df: Optional[int] = None,
    window_type: Literal["rolling", "expanding"] = "rolling",
    window: int = 60,
    min_periods: int = 60,
    ddof: int = 1,
    max_dim: int = 4,
    alpha: float = 0.05,
    eps: float = 1e-12,
) -> Dict[str, DistributionTestResult]:
    """
    Run PIT diagnostics: marginal uniformity and independence.
    """
    if window_type == "rolling":
        pit = _rolling_pit(
            values,
            case=case,
            distribution=distribution,
            df=df,
            window=window,
            ddof=ddof,
        )
    else:
        pit = _expanding_pit(
            values,
            case=case,
            distribution=distribution,
            df=df,
            min_periods=min_periods,
            ddof=ddof,
        )

    pit = pit.dropna()

    ks_res = _kolmogorov_test(pit, alpha=alpha)
    bds_res = _independence_test(pit, max_dim=max_dim, alpha=alpha)
    berkowitz_res = _berkowitz_test(pit, alpha=alpha, eps=eps)

    return {
        "uniformity": ks_res,
        "independence": bds_res,
        "berkowitz": berkowitz_res,
    }
