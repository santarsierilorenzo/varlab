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
    exact_binomial_coverage_test,
    kupiec_pof_test,
    basel_traffic_light_test,
    christoffersen_conditional_coverage_test,
    CoverageTestResult,
)

from .independence import (
    christoffersen_independence,
    loss_quantile_independence,
    christoffersen_test,
    loss_quantile_independence_test,
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
    "exact_binomial_coverage_test",
    "kupiec_pof_test",
    "basel_traffic_light_test",
    "christoffersen_conditional_coverage_test",
    "CoverageTestResult",
    # Independence
    "christoffersen_independence",
    "loss_quantile_independence",
    "christoffersen_test",
    "loss_quantile_independence_test",
    "IndependenceTestResult",
    # Distribution
    "pit_diagnostics",
    "DistributionTestResult",
]
