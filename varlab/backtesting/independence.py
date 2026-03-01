"""
Indepedence tests for VaR backtesting.

This module implements statistically rigorous tests used in
Value-at-Risk (VaR) model validation.
"""

from __future__ import annotations

from typing import Optional, Literal, Dict, Any, Union, Sequence, Iterable
from .pit import rolling_pit, expanding_pit
from statsmodels.tsa.stattools import acf
from scipy.stats import chi2, norm
from dataclasses import dataclass
import pandas as pd
import numpy as np


Dist = Literal["normal", "t"]


@dataclass(frozen=True)
class IndependenceTestResult:
    """
    Standardized result container for tests.
    """
    test_name: str
    statistic: Optional[Union[float, np.ndarray]]
    p_value: Optional[Union[float, np.ndarray]]
    reject: Optional[bool]
    info: Dict[str, Any]


def christoffersen_test(
    exceedances: Sequence,
    alpha: float = 0.05,
    eps: float = 1e-12,
) -> IndependenceTestResult:
    """
    Christoffersen (1998) Exceedance Independence Test.

    This test evaluates whether VaR exceedances occur independently
    over time.

    Let I_t be the exceedance indicator:

        I_t = 1 if loss_t > VaR_t
        I_t = 0 otherwise

    Under correct model specification, the sequence {I_t} should be
    independent over time and follow a Bernoulli distribution.

    The test models the exceedance process as a first-order
    two-state Markov chain and performs a likelihood ratio (LR)
    test of:

        H0: π0 == π1   (independence)
        H1: π0 != π1   (first-order dependence)

    where:
        π0 = P(I_t = 1 | I_{t-1} = 0)
        π1 = P(I_t = 1 | I_{t-1} = 1)

    Under H0, the LR statistic is asymptotically chi-square with
    1 degree of freedom.
    """

    # Convert exceedances to binary numpy array
    x = np.asarray(exceedances, dtype="int64")

    if len(x) < 2:
        raise ValueError("At least two observations required.")

    if not set(np.unique(x)).issubset({0, 1}):
        raise ValueError("Exceedances must be binary (0/1).")

    # Split into previous and current states
    # Transitions are defined from t-1 to t
    prev = x[:-1]
    curr = x[1:]

    # Transition counts for the two-state Markov chain
    #
    # n00 : 0 -> 0
    # n01 : 0 -> 1
    # n10 : 1 -> 0
    # n11 : 1 -> 1
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))

    # Conditional MLEs under the alternative (Markov dependence)
    #
    # π0_hat = P(I_t = 1 | I_{t-1} = 0)
    # π1_hat = P(I_t = 1 | I_{t-1} = 1)
    #
    # Clipping avoids log(0) in likelihood evaluation.
    denom_0 = n00 + n01
    denom_1 = n10 + n11

    if denom_0 == 0:
        # No transitions starting from state 0.
        # Under this extreme case, likelihood contribution
        # from state 0 transitions is zero.
        m_hat_0 = eps
    else:
        m_hat_0 = np.clip(n01 / denom_0, eps, 1.0 - eps)

    if denom_1 == 0:
        # No transitions starting from state 1.
        m_hat_1 = eps
    else:
        m_hat_1 = np.clip(n11 / denom_1, eps, 1.0 - eps)

    # MLE under the null (independence)
    #
    # Under H0, the exceedance probability is constant:
    #
    #     p = P(I_t = 1)
    #
    # Estimated using transition counts:
    #
    #     p_hat = (n01 + n11) / N
    #
    # where N = total number of transitions = T - 1.
    N = n00 + n01 + n10 + n11

    if N == 0:
        raise ValueError("Insufficient transitions for test.")

    p_hat = np.clip((n01 + n11) / N, eps, 1.0 - eps)

    # Log-likelihood under H0 (independent Bernoulli process)
    l0 = (
        (n01 + n11) * np.log(p_hat)
        + (n00 + n10) * np.log(1 - p_hat)
    )

    # Log-likelihood under H1 (first-order Markov chain)
    l1 = (
        n00 * np.log(1 - m_hat_0)
        + n01 * np.log(m_hat_0)
        + n10 * np.log(1 - m_hat_1)
        + n11 * np.log(m_hat_1)
    )

    # Likelihood ratio statistic
    #
    # LR = 2 (l1 - l0)
    #
    # Theoretically LR >= 0 (nested models). Small negative values
    # may arise due to floating-point precision.
    lr = max(0.0, 2 * (l1 - l0))

    # Under H0: LR ~ chi-square(1), =^.^=
    pval = chi2.sf(lr, df=1)

    return IndependenceTestResult(
        test_name="Christoffersen Exceedance Independence Test",
        statistic=float(lr),
        p_value=float(pval),
        reject=bool(pval < alpha),
        info={
            "sample_size": int(len(x)),
            "alpha": alpha,
            "m_hat_0": float(m_hat_0),
            "m_hat_1": float(m_hat_1),
            "l_H0": float(l0),
            "l_H1": float(l1),
            "df": 1,
        },
    )


def loss_quantile_independence_test(
    values: Iterable,
    case: Literal["continuous", "discrete"] = "continuous",
    distribution: Dist = "normal",
    df: Optional[int] = None,
    window_type: Literal["rolling", "expanding"] = "rolling",
    window: int = 60,
    min_periods: int = 60,
    max_lag: int = 5,
    alpha: float = 0.05,
    eps: float = 1e-12,
    n_sim: int = 5_000,
    seed: Optional[int] = 0,
    ddof: int = 1,
    mean: Literal["zero", "sample"] = "zero",
) -> IndependenceTestResult:
    """
    Loss-Quantile Independence Test based on maximum autocorrelation.

    This test evaluates the serial independence of Probability Integral
    Transform (PIT) values associated with loss quantiles.

    Procedure
    ---------
    1. Compute PIT values using either rolling or expanding window.
    2. Apply the probit transform:

           z_t = Φ^{-1}(u_t)

       Under correct model specification:
           z_t ~ i.i.d. N(0,1)

    3. Compute sample autocorrelations up to lag `max_lag`.
    4. Define the test statistic as:

           T_obs = max |acf_k|,  k = 1, ..., max_lag

    5. Approximate the null distribution of T_obs via Monte Carlo
       simulation using i.i.d. Gaussian samples of equal length.

    Null Hypothesis
    ---------------
        H0: PIT values are independent over time.

    Alternative Hypothesis
    ----------------------
        H1: Serial dependence exists in PIT values.

    The p-value is computed as the proportion of simulated statistics
    exceeding the observed statistic.

    Parameters
    ----------
    values : Iterable
        Time series of returns or losses.
    case : {"continuous", "discrete"}, default="continuous"
        Specifies whether the underlying distribution is continuous
        (parametric PIT) or discrete (randomized PIT).
    distribution : {"normal", "t"}, default="normal"
        Distribution assumed for parametric PIT.
    df : Optional[int], default=None
        Degrees of freedom for Student-t distribution.
    window_type : {"rolling", "expanding"}, default="rolling"
        Type of window used to compute PIT.
    window : int, default=60
        Rolling window size.
    min_periods : int, default=60
        Minimum observations for expanding window.
    max_lag : int, default=5
        Maximum lag considered in autocorrelation computation.
    alpha : float, default=0.05
        Significance level for rejection decision.
    eps : float, default=1e-12
        Numerical clipping level for PIT values.
    n_sim : int, default=20000
        Number of Monte Carlo simulations for null distribution.
    seed : Optional[int], default=0
        Random seed for reproducibility.
    ddof : int, default=1
        Delta degrees of freedom used in rolling/expanding volatility.

    Returns
    -------
    IndependenceTestResult
        Structured test result containing:
            - test statistic
            - p-value
            - rejection decision
            - diagnostic information

    Notes
    -----
    - The test is simulation-based and computationally intensive.
    - Results may be sensitive to `max_lag` and `n_sim`.
    - Assumes correct PIT construction under the null.
    """
    values = pd.Series(values).astype(float)

    if window_type == "rolling":
        pit = rolling_pit(
            values,
            case=case,
            distribution=distribution,
            df=df,
            window=window,
            ddof=ddof,
            mean=mean,
        )
    else:
        pit = expanding_pit(
            values,
            case=case,
            distribution=distribution,
            df=df,
            min_periods=min_periods,
            ddof=ddof,
            mean=mean,
        )

    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")

    if n_sim < 1:
        raise ValueError("n_sim must be >= 1")

    u = pd.Series(pit).dropna().astype(float).to_numpy()

    if u.size < 50:
        raise ValueError("Sample too small for loss-quantile independence")

    if not np.all(np.isfinite(u)):
        raise ValueError("Non-finite PIT values")

    u = np.clip(u, eps, 1.0 - eps)
    z = norm.ppf(u)

    if z.size <= max_lag + 1:
        raise ValueError("Sample too small for requested max_lag")

    # statsmodels returns acf for lags 0..max_lag; drop lag 0
    acf_obs = acf(z, nlags=max_lag, fft=True)[1:]
    t_obs = float(np.max(np.abs(acf_obs)))

    rng = np.random.default_rng(seed)
    n = z.size

    t_sim = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        z0 = rng.standard_normal(n)
        acf0 = acf(z0, nlags=max_lag, fft=True)[1:]
        t_sim[i] = float(np.max(np.abs(acf0)))

    pval = float(np.mean(t_sim >= t_obs))

    return IndependenceTestResult(
        test_name="Loss-Quantile Independence Test",
        statistic=t_obs,
        p_value=pval,
        reject=bool(pval < alpha),
        info={
            "sample_size": int(n),
            "alpha": float(alpha),
            "max_lag": int(max_lag),
            "n_sim": int(n_sim),
            "acf": acf_obs,
        },
    )
