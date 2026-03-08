from __future__ import annotations

"""
High-level runner for VaR backtesting diagnostics.
"""
from . import coverage, distribution, independence
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np


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
    reject: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary."""

        return {
            "test_name": self.test_name,
            "statistic": _to_builtin(self.statistic),
            "p_value": _to_builtin(self.p_value),
            "reject": bool(self.reject),
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
        reject=bool(obj.reject),
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
    run_basel_if_applicable: bool = True,
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

    if TestCategory.DISTRIBUTION in test_types:

        pit = distribution.pit_diagnostics(
            returns,
            case=pit_case.value,
            window_type=window_type.value,
            window=window,
            alpha=alpha,
        )

        results[TestCategory.DISTRIBUTION] = {
            name: _result_from_object(obj)
            for name, obj in pit.items()
        }

    if TestCategory.INDEPENDENCE in test_types:

        ind = {

            "christoffersen": independence.christoffersen_independence(
                exceedances
            ),

            "loss_quantile":
                independence.loss_quantile_independence(
                    returns,
                    case=pit_case.value,
                    window_type=window_type.value,
                    window=window,
                    n_sim=n_sim,
                ),
        }

        results[TestCategory.INDEPENDENCE] = {
            k: _result_from_object(v)
            for k, v in ind.items()
        }

    return DiagnosticRunResult(
        results=results,
        confidence_level=confidence,
        sample_size=len(returns),
        test_date=datetime.utcnow().strftime("%Y-%m-%d"),
    )
