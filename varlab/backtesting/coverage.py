from scipy.stats import binom, chi2
from typing import Sequence, Tuple
import numpy as np


def standard_coverage_test(
    exceedances: Sequence[bool],
    confidence_level: float,
    significance_level: float = 0.05,
) -> Tuple[bool, Tuple[int, int, int]]:
    """
    Perform the exact binomial coverage test for VaR backtesting.

    Statistical framework
    ---------------------
    This function evaluates the unconditional coverage of a VaR model
    using the exact binomial test.

    Let I_t be the indicator of a VaR violation:

        I_t = 1 if loss_t > VaR_{1-q}
              0 otherwise

    Under the null hypothesis

        H0 : q* = q

    where q = 1 - confidence_level, the total number of exceedances

        X = sum_{t=1}^T I_t

    follows a Binomial(T, q) distribution.

    The acceptance region is constructed using binomial quantiles,
    allocating half of the significance level to each tail.

    Assumptions
    -----------
    - Exceedances are IID Bernoulli under H0.
    - The VaR model is fixed and not estimated within the test.
    - This is an unconditional coverage test.

    Returns
    -------
    Tuple[bool, Tuple[int, int, int]]
        reject : bool
            True if H0 is rejected.
        (lower_bound, upper_bound, observed_violations)
            Acceptance bounds and observed violation count.
    """
    sample_size: int = len(exceedances)
    violation_probability: float = 1.0 - confidence_level
    observed_violations: int = int(np.sum(exceedances))

    alpha_half: float = significance_level / 2.0

    lower_bound: int = int(
        binom.ppf(alpha_half, sample_size, violation_probability)
    )

    upper_bound: int = int(
        binom.ppf(
            1.0 - alpha_half,
            sample_size,
            violation_probability,
        )
    )

    reject: bool = (
        observed_violations < lower_bound
        or observed_violations > upper_bound
    )

    return reject, (lower_bound, upper_bound, observed_violations)


def kupiec_pf(
    exceedances: Sequence[int],
    confidence_level: float,
) -> Tuple[float, float]:
    """
    Perform the Kupiec Proportion of Failures (PF) coverage test for VaR
    backtesting.

    Statistical framework
    ---------------------
    This function evaluates the unconditional coverage of a VaR model
    using a likelihood ratio (LR) test, as proposed by Kupiec (1995).

    Let I_t be the indicator of a VaR violation:

        I_t = 1 if loss_t > VaR_{1-q}
            0 otherwise

    Under the null hypothesis

        H0 : q* = q

    where q = 1 - confidence_level denotes the theoretical probability of
    exceedance implied by the VaR model, the total number of exceedances

        X = sum_{t=1}^T I_t

    follows a Binomial(T, q) distribution.

    The test is based on the likelihood ratio statistic

        LR_POF = -2 log [ L(q) / L(q_hat) ]

    where q_hat = X / T is the maximum likelihood estimator of q.

    Under the null hypothesis, the test statistic is asymptotically
    distributed as

        LR_POF ~ chi-square(1).

    The null hypothesis is rejected if the statistic lies in the right tail
    of the chi-square distribution at the chosen significance level.

    Assumptions
    -----------
    - Exceedances are IID Bernoulli under H0.
    - The VaR model is fixed and not estimated within the test.
    - This is an unconditional coverage test.

    Returns
    -------
    Tuple[float, float]
        lr_stat : float
            Likelihood ratio test statistic.
        p_value : float
            Right-tail p-value under the chi-square(1) distribution.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1).")

    if len(exceedances) == 0:
        raise ValueError("exceedances cannot be empty.")

    exceedances_arr = np.asarray(exceedances, dtype=int)

    if not set(np.unique(exceedances_arr)).issubset({0, 1}):
        raise ValueError("exceedances must be binary (0/1).")

    t: int = len(exceedances_arr)
    x: int = int(np.sum(exceedances_arr))

    q: float = 1.0 - confidence_level
    q_hat: float = x / t

    if q_hat == 0.0 or q_hat == 1.0:
        raise ValueError(
            "Kupiec test undefined when all observations "
            "are violations or none."
        )

    log_num = (
        (t - x) * np.log(1.0 - q) +
        x * np.log(q)
    )

    log_den = (
        (t - x) * np.log(1.0 - q_hat) +
        x * np.log(q_hat)
    )

    lr_stat: float = -2.0 * (log_num - log_den)
    p_value: float = 1.0 - chi2.cdf(lr_stat, df=1)

    return lr_stat, p_value
