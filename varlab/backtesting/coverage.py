from typing import Sequence, Tuple
from scipy.stats import binom
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
