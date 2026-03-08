"""
VaR backtesting statistical tests.

Includes:
    - Coverage tests
    - Independence tests
    - Distribution diagnostics (PIT-based)
"""

from .coverage import (
    exact_binomial_coverage,
    kupiec_pof,
    basel_traffic_light,
    christoffersen_conditional_coverage,
    CoverageTestResult,
)

from .independence import (
    christoffersen_independence,
    loss_quantile_independence,
    IndependenceTestResult,
)

from .distribution import (
    pit_diagnostics,
    DistributionTestResult,
)

__all__ = [
    # Coverage
    "exact_binomial_coverage",
    "kupiec_pof",
    "basel_traffic_light",
    "christoffersen_conditional_coverage",
    "CoverageTestResult",
    # Independence
    "christoffersen_independence",
    "loss_quantile_independence",
    "IndependenceTestResult",
    # Distribution
    "pit_diagnostics",
    "DistributionTestResult",
]
