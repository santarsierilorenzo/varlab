"""
Unconditional coverage tests for VaR backtesting.

This module implements statistically rigorous tests used in Value-at-Risk
(VaR) model validation
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any
from .independence import christoffersen_test
from scipy.stats import binom, chi2
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CoverageTestResult:
    """
    Standardized result container for coverage tests.

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
    statistic: Optional[float]
    p_value: Optional[float]
    reject: Optional[bool]
    info: Dict[str, Any]


def _validate_exceedances(
    exceedances: Sequence[int],
) -> np.ndarray:
    """
    Validate and convert exceedances to numpy array.

    Returns
    -------
    np.ndarray
        Binary array of shape (T,).
    """
    if len(exceedances) == 0:
        raise ValueError("exceedances cannot be empty.")

    arr = np.asarray(exceedances, dtype=int)

    unique_vals = np.unique(arr)
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError("exceedances must be binary (0/1).")

    return arr


def _validate_probability(
    value: float,
    name: str
) -> None:
    """Validate probability in (0, 1)."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1).")


def exact_binomial_coverage_test(
    exceedances: Sequence[int],
    confidence_level: float,
    alpha: Optional[float] = None,
) -> CoverageTestResult:
    """
    Exact two-sided binomial coverage test.

    Tests the null hypothesis:

        H0 : q* = q

    where q = 1 - confidence_level.

    The total number of exceedances follows Binomial(T, q)
    under H0.

    Parameters
    ----------
    exceedances : Sequence[int]
        Binary exceedance indicators.
    confidence_level : float
        VaR confidence level.
    alpha : Optional[float]
        Significance level. If provided, reject decision returned.

    Returns
    -------
    CoverageTestResult
    """
    arr = _validate_exceedances(exceedances)
    _validate_probability(confidence_level, "confidence_level")

    if alpha is not None:
        _validate_probability(alpha, "alpha")

    t: int = len(arr)
    x: int = int(np.sum(arr))
    q: float = 1.0 - confidence_level

    p_value: float = 2.0 * min(
        binom.cdf(x, t, q),
        1.0 - binom.cdf(x - 1, t, q)
    )
    p_value = min(p_value, 1.0)

    reject: Optional[bool] = None
    if alpha is not None:
        reject = p_value < alpha

    return CoverageTestResult(
        test_name="Exact Binomial Coverage",
        statistic=float(x),
        p_value=float(p_value),
        reject=reject,
        info={
            "sample_size": t,
            "observed_violations": x,
            "expected_violations": t * q,
            "theoretical_probability": q,
        },
    )


def kupiec_pof_test(
    exceedances: Sequence[int],
    confidence_level: float,
    alpha: Optional[float] = None,
) -> CoverageTestResult:
    """
    Kupiec (1995) likelihood ratio coverage test.

    LR_POF = -2 log [ L(q) / L(q_hat) ]

    Under H0:

        LR_POF ~ chi-square(1)

    Parameters
    ----------
    exceedances : Sequence[int]
        Binary exceedance indicators.
    confidence_level : float
        VaR confidence level.
    alpha : Optional[float]
        Significance level.

    Returns
    -------
    CoverageTestResult
    """
    arr = _validate_exceedances(exceedances)
    _validate_probability(confidence_level, "confidence_level")

    if alpha is not None:
        _validate_probability(alpha, "alpha")

    t: int = len(arr)
    x: int = int(np.sum(arr))

    q: float = 1.0 - confidence_level
    q_hat: float = x / t

    # Handle edge cases explicitly (no crashes)
    if x == 0:
        log_num = t * np.log(1.0 - q)
        log_den = 0.0
    elif x == t:
        log_num = t * np.log(q)
        log_den = 0.0
    else:
        log_num = (
            (t - x) * np.log(1.0 - q) +
            x * np.log(q)
        )
        log_den = (
            (t - x) * np.log(1.0 - q_hat) +
            x * np.log(q_hat)
        )

    lr_stat: float = -2.0 * (log_num - log_den)
    p_value: float = chi2.sf(lr_stat, df=1)

    reject: Optional[bool] = None
    if alpha is not None:
        reject = p_value < alpha

    return CoverageTestResult(
        test_name="Kupiec POF",
        statistic=float(lr_stat),
        p_value=float(p_value),
        reject=reject,
        info={
            "sample_size": t,
            "observed_violations": x,
            "expected_violations": t * q,
            "theoretical_probability": q,
            "estimated_probability": q_hat,
        },
    )


def basel_traffic_light_test(
    exceedances: Sequence[int],
    confidence_level: float,
) -> CoverageTestResult:
    """
    Basel Traffic Light classification for one-day 99% VaR.

    This is a regulatory rule defined by Basel II/III and is valid
    exclusively for:
        - 1-day horizon
        - 99% confidence level
        - 250-observation backtesting window

    The function is intentionally non-parametric to prevent misuse.

    Parameters
    ----------
    exceedances : Sequence[int]
        Binary exceedance indicators (1 if loss > VaR, else 0).
    confidence_level : float
        VaR confidence level. Must be exactly 0.99.

    Returns
    -------
    CoverageTestResult
    """
    if confidence_level != 0.99:
        raise ValueError(
            "Basel Traffic Light is defined only for 99% VaR."
        )

    arr = _validate_exceedances(exceedances)

    window: int = 250
    if len(arr) < window:
        raise ValueError(
            "At least 250 observations are required for "
            "Basel Traffic Light test."
        )

    last_window = arr[-window:]
    violations: int = int(np.sum(last_window))

    yellow_multiplier_map = {
        5: 3.4,
        6: 3.5,
        7: 3.65,
        8: 3.75,
        9: 3.85,
    }

    if violations <= 4:
        zone = "green"
        multiplier = 3.0
    elif violations <= 9:
        zone = "yellow"
        multiplier = yellow_multiplier_map[violations]
    else:
        zone = "red"
        multiplier = 4.0

    return CoverageTestResult(
        test_name="Basel Traffic Light",
        statistic=float(violations),
        p_value=None,
        reject=None,
        info={
            "window": window,
            "confidence_level": confidence_level,
            "violations": violations,
            "zone": zone,
            "multiplier": multiplier,
        },
    )


def christoffersen_conditional_coverage_test(
    exceedances: Sequence[int],
    confidence_level: float,
    alpha: Optional[float] = None,
) -> CoverageTestResult:
    """
    Christoffersen (1998) Conditional Coverage Test.

    This test jointly evaluates:

        1. Unconditional coverage (Kupiec POF test)
        2. Independence of exceedances (Markov test)

    The test statistic is:

        LR_cc = LR_uc + LR_ind

    Under the null hypothesis:

        H0: Correct violation frequency AND independence

    The statistic is asymptotically chi-square with 2 degrees of freedom.
    """

    # Unconditional coverage (Kupiec)
    kupiec_res = kupiec_pof_test(
        exceedances,
        confidence_level,
        alpha=None,
    )

    lr_uc = kupiec_res.statistic

    # Independence component
    ind_res = christoffersen_test(
        exceedances,
        alpha=0.05,  # not used for decision here
    )

    lr_ind = ind_res.statistic

    # Joint statistic
    lr_cc = lr_uc + lr_ind

    p_value = chi2.sf(lr_cc, df=2)

    reject: Optional[bool] = None
    if alpha is not None:
        reject = p_value < alpha

    return CoverageTestResult(
        test_name="Christoffersen Conditional Coverage",
        statistic=float(lr_cc),
        p_value=float(p_value),
        reject=reject,
        info={
            "sample_size": kupiec_res.info["sample_size"],
            "confidence_level": confidence_level,
            "lr_uc": float(lr_uc),
            "lr_ind": float(lr_ind),
            "df": 2,
        },
    )
