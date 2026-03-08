from __future__ import annotations

"""
High-level runner for VaR backtesting diagnostics.
"""
from . import (
    coverage,
    distribution as distribution_tests,
    independence,
    es_validation,
)
from typing import Any, Dict, Optional, Sequence, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

# Backward-compatible module alias used by tests/integration code.
distribution = distribution_tests


class WindowType(str, Enum):
    """Supported PIT window schemes."""
    ROLLING = "rolling"
    EXPANDING = "expanding"


class PitCase(str, Enum):
    """Supported PIT transformation cases."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class TestCategory(str, Enum):
    """Backtest categories."""
    COVERAGE = "coverage"
    DISTRIBUTION = "distribution"
    INDEPENDENCE = "independence"
    EXPECTED_SHORTFALL = "expected_shortfall"


def _to_builtin(x: Any) -> Any:
    """Convert numpy scalars to Python builtins."""

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, np.bool_):
        return bool(x)

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        return float(x)

    return x


def _ensure_series(
    returns: pd.Series | Sequence[float]
) -> pd.Series:
    """Ensure returns are a pandas Series."""

    if isinstance(returns, pd.Series):
        return returns

    arr = np.asarray(returns)

    if arr.ndim != 1:
        raise ValueError("returns must be 1-dimensional.")

    return pd.Series(arr)


@dataclass(frozen=True)
class BacktestResult:
    """Standardized single-test result."""

    test_name: str
    statistic: Any
    p_value: Any
    reject: Optional[bool]
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary."""

        return {
            "test_name": self.test_name,
            "statistic": _to_builtin(self.statistic),
            "p_value": _to_builtin(self.p_value),
            "reject": self.reject,
            "info": {
                k: _to_builtin(v)
                for k, v in self.info.items()
            },
        }

    @property
    def outcome(self) -> str:
        """Return PASS or FAIL."""

        outcome = self.info.get("outcome")

        if outcome not in {"PASS", "FAIL"}:
            raise ValueError(
                f"Missing outcome for test '{self.test_name}'."
            )

        return outcome


@dataclass(frozen=True)
class DiagnosticRunResult:
    """Aggregated result of a diagnostic run."""

    results: Dict[TestCategory, Dict[str, BacktestResult]]
    confidence_level: float
    sample_size: int
    test_date: str

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        """Return category -> test -> PASS/FAIL."""

        return {
            category.value: {
                name: result.outcome
                for name, result in tests.items()
            }
            for category, tests in self.results.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return full JSON result."""

        return {
            "meta": {
                "confidence_level": self.confidence_level,
                "sample_size": self.sample_size,
                "test_date": self.test_date,
            },
            "results": {
                category.value: {
                    name: res.to_dict()
                    for name, res in tests.items()
                }
                for category, tests in self.results.items()
            },
        }

    def _overall(self) -> str:
        outcomes = [
            r.outcome
            for tests in self.results.values()
            for r in tests.values()
        ]

        return "FAIL" if "FAIL" in outcomes else "PASS"

    def _format_name(self, name: str) -> str:
        return name.replace("_", " ").title()

    def report(self) -> str:
        """Return ASCII report."""

        width = 66
        col_test = 52

        conf = f"{self.confidence_level * 100:.2f}%"

        lines = [
            "=" * width,
            "VALUE AT RISK BACKTEST REPORT".center(width),
            "=" * width,
            (
                f"Confidence level: {conf} | "
                f"Sample size: {self.sample_size} | "
                f"Date: {self.test_date}"
            ),
            f"Overall result: {self._overall()}",
            "",
        ]

        snapshot = self.snapshot()

        for category, tests in snapshot.items():

            passed = sum(x == "PASS" for x in tests.values())
            total = len(tests)

            lines.append(f"{category.upper()} ({passed}/{total})")
            lines.append("-" * width)

            for name, result in tests.items():

                label = self._format_name(name)

                lines.append(
                    f"{label:<{col_test}}{result:>8}"
                )

            lines.append("")  # Blank line between categories

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()

    def __repr__(self) -> str:
        return self.report()


def _result_from_object(obj: Any) -> BacktestResult:
    """Convert internal test object to BacktestResult."""

    return BacktestResult(
        test_name=obj.test_name,
        statistic=obj.statistic,
        p_value=obj.p_value,
        reject=obj.reject,
        info=getattr(obj, "info", {}),
    )


def run(
    returns: pd.Series | Sequence[float],
    exceedances: Sequence[bool],
    confidence: float,
    window_type: WindowType | str,
    pit_case: PitCase | str,
    alpha: float = 0.05,
    n_sim: int = 5000,
    window: int = 120,
    min_periods: Optional[int] = None,
    distribution: Literal["normal", "t"] = "normal",
    df: Optional[int] = None,
    mean: Literal["zero", "sample"] = "zero",
    ddof: int = 1,
    max_dim: int = 4,
    pit_eps: float = 1e-12,
    loss_max_lag: int = 5,
    loss_eps: float = 1e-12,
    loss_seed: Optional[int] = 0,
    christoffersen_eps: float = 1e-12,
    run_basel_if_applicable: bool = True,
    var_forecast: Sequence[float] | float | None = None,
    es_forecast: Sequence[float] | float | None = None,
    es_tail_alpha: Optional[float] = None,
    es_test_epsilon: float = 0.05,
    es_test_alternative: Literal[
        "greater",
        "less",
        "two-sided",
    ] = "greater",
    test_types: Sequence[TestCategory] | None = None,
) -> DiagnosticRunResult:
    """
    Run VaR backtesting diagnostics.
    """

    if isinstance(window_type, str):
        window_type = WindowType(window_type)

    if isinstance(pit_case, str):
        pit_case = PitCase(pit_case)

    if test_types is None:
        test_types = (
            TestCategory.COVERAGE,
            TestCategory.DISTRIBUTION,
            TestCategory.INDEPENDENCE,
        )

    returns = _ensure_series(returns)

    if len(returns) != len(exceedances):
        raise ValueError(
            "returns and exceedances must have same length."
        )

    results: Dict[TestCategory, Dict[str, BacktestResult]] = {}

    if TestCategory.COVERAGE in test_types:

        cov = {

            "exact_binomial": coverage.exact_binomial_coverage(
                exceedances,
                confidence=confidence,
                alpha=alpha,
            ),

            "kupiec_pof": coverage.kupiec_pof(
                exceedances,
                confidence=confidence,
                alpha=alpha,
            ),

            "christoffersen_conditional":
                coverage.christoffersen_conditional_coverage(
                    exceedances,
                    confidence=confidence,
                    alpha=alpha,
                ),
        }

        if run_basel_if_applicable and np.isclose(confidence, 0.99):

            cov["basel_traffic_light"] = coverage.basel_traffic_light(
                exceedances,
                confidence=confidence,
            )

        results[TestCategory.COVERAGE] = {
            k: _result_from_object(v)
            for k, v in cov.items()
        }

    # Keep rolling/expanding behavior consistent unless caller overrides.
    min_periods = window if min_periods is None else min_periods

    if TestCategory.DISTRIBUTION in test_types:

        pit = distribution_tests.pit_diagnostics(
            returns,
            case=pit_case.value,
            distribution=distribution,
            df=df,
            window_type=window_type.value,
            window=window,
            min_periods=min_periods,
            ddof=ddof,
            max_dim=max_dim,
            alpha=alpha,
            eps=pit_eps,
            mean=mean,
        )

        results[TestCategory.DISTRIBUTION] = {
            name: _result_from_object(obj)
            for name, obj in pit.items()
        }

    if TestCategory.INDEPENDENCE in test_types:

        ind = {

            "christoffersen": independence.christoffersen_independence(
                exceedances,
                alpha=alpha,
                eps=christoffersen_eps,
            ),

            "loss_quantile":
                independence.loss_quantile_independence(
                    returns,
                    case=pit_case.value,
                    distribution=distribution,
                    df=df,
                    window_type=window_type.value,
                    window=window,
                    min_periods=min_periods,
                    max_lag=loss_max_lag,
                    eps=loss_eps,
                    n_sim=n_sim,
                    seed=loss_seed,
                    ddof=ddof,
                    mean=mean,
                    alpha=alpha,
                ),
        }

        results[TestCategory.INDEPENDENCE] = {
            k: _result_from_object(v)
            for k, v in ind.items()
        }

    if TestCategory.EXPECTED_SHORTFALL in test_types:
        if var_forecast is None or es_forecast is None:
            raise ValueError(
                "var_forecast and es_forecast are required when "
                "running expected shortfall diagnostics."
            )

        tail_alpha = (
            1.0 - confidence
            if es_tail_alpha is None
            else es_tail_alpha
        )

        es_results = {
            "mcneil_frey": es_validation.mcneil_frey(
                returns=returns,
                var=var_forecast,
                es=es_forecast,
                epsilon=es_test_epsilon,
                alternative=es_test_alternative,
            ),
            "acerbi_szekely": es_validation.acerbi_szekely(
                returns=returns,
                var=var_forecast,
                es=es_forecast,
                alpha=tail_alpha,
                epsilon=es_test_epsilon,
                alternative=es_test_alternative,
            ),
        }

        results[TestCategory.EXPECTED_SHORTFALL] = {
            k: _result_from_object(v)
            for k, v in es_results.items()
        }

    return DiagnosticRunResult(
        results=results,
        confidence_level=confidence,
        sample_size=len(returns),
        test_date=datetime.utcnow().strftime("%Y-%m-%d"),
    )
