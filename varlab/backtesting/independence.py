"""
Indepedence tests for VaR backtesting.

This module implements statistically rigorous tests used in
Value-at-Risk (VaR) model validation.
"""

from __future__ import annotations

from typing import Optional, Literal, Dict, Any, Union, Sequence
from statsmodels.tsa.stattools import bds
from scipy.stats import chi2
from dataclasses import dataclass
import numpy as np


Dist = Literal["normal", "t"]


@dataclass(frozen=True)
class IndependenceTestResult:
    """
    Standardized result container for distribution tests.
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
    m_hat_0 = np.clip(n01 / (n00 + n01), eps, 1 - eps)
    m_hat_1 = np.clip(n11 / (n10 + n11), eps, 1 - eps)

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
    p_hat = np.clip((n01 + n11) / N, eps, 1 - eps)

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

