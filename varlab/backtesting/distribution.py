"""
Distribution tests for VaR backtesting.

This module implements statistically rigorous tests used in
Value-at-Risk (VaR) model validation.
"""

from __future__ import annotations

from typing import Optional, Literal, Dict, Any, Union
from .pit import rolling_pit, expanding_pit
from scipy.stats import norm, t, kstest, chi2
from statsmodels.tsa.stattools import bds
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
    l_res = np.sum(norm.logpdf(z[1:]))

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
    l_unres = -((T - 1) / 2) * (
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
    lr_stat = 2 * (l_unres - l_res)
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
            "l_res": float(l_res),
            "l_unres": float(l_unres),
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
        pit = rolling_pit(
            values,
            case=case,
            distribution=distribution,
            df=df,
            window=window,
            ddof=ddof,
        )
    else:
        pit = expanding_pit(
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
