"""
High-level runner for VaR backtesting diagnostics.

Provides a single entry-point that aggregates coverage,
independence and distribution tests into a unified,
JSON-serializable output format.
"""

from typing import Any, Dict, Iterable, Literal, Sequence
import numpy as np

from .backtesting import coverage, distribution, independence


WindowType = Literal["rolling", "expanding"]
PitCase = Literal["discrete", "continuous"]


def _validate_inputs(
    returns: Sequence[Any],
    exceedances: Sequence[bool],
    confidence: float,
    window: int,
) -> None:
    """Validate basic input constraints."""
    if len(returns) != len(exceedances):
        raise ValueError(
            "returns and exceedances must have same length. "
            f"Got {len(returns)} vs {len(exceedances)}."
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")
    if window <= 0:
        raise ValueError("window must be positive.")
    if len(returns) < 2:
        raise ValueError("Need at least 2 observations.")


def _to_builtin(x: Any) -> Any:
    """Convert numpy types into Python builtins."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def _serialize_result(obj: Any) -> Dict[str, Any]:
    """Standardize test result objects into dictionaries."""
    return {
        "test_name": obj.test_name,
        "statistic": _to_builtin(obj.statistic),
        "p_value": _to_builtin(obj.p_value),
        "reject": _to_builtin(obj.reject),
        "info": {
            k: _to_builtin(v)
            for k, v in getattr(obj, "info", {}).items()
        },
    }


def run(
    returns: Iterable[Any],
    exceedances: Iterable[bool],
    confidence: float,
    window_type: WindowType,
    pit_case: PitCase,
    alpha: float = 0.05,
    n_sim: int = 5000,
    window: int = 120,
    run_basel_if_applicable: bool = True,
) -> Dict[str, Any]:
    """
    Run a suite of VaR backtests and diagnostics.

    Returns a flat dictionary of standardized test results.
    """

    _validate_inputs(
        returns=returns,
        exceedances=exceedances,
        confidence=confidence,
        window=window,
    )

    results: Dict[str, Any] = {}

    # Coverage tests
    results["coverage_exact_binomial"] = _serialize_result(
        coverage.exact_binomial_coverage_test(
            exceedances,
            confidence=confidence,
            alpha=alpha,
        )
    )

    results["coverage_kupiec_pof"] = _serialize_result(
        coverage.kupiec_pof_test(
            exceedances,
            confidence=confidence,
            alpha=alpha,
        )
    )

    results["coverage_christoffersen_conditional"] = (
        _serialize_result(
            coverage.christoffersen_conditional_coverage_test(
                exceedances,
                confidence=confidence,
                alpha=alpha,
            )
        )
    )

    if run_basel_if_applicable and confidence == 0.99:
        results["coverage_basel_traffic_light"] = (
            _serialize_result(
                coverage.basel_traffic_light_test(
                    exceedances,
                    confidence=confidence,
                )
            )
        )

    # Distribution diagnostics (flattened)
    pit_results = distribution.pit_diagnostics(
        returns,
        case=pit_case,
        window_type=window_type,
        window=window,
        alpha=alpha,
    )

    for name, obj in pit_results.items():
        results[f"distribution_{name}"] = _serialize_result(obj)

    # Independence tests
    results["independence_christoffersen"] = _serialize_result(
        independence.christoffersen_test(
            exceedances
        )
    )

    results["independence_loss_quantile"] = _serialize_result(
        independence.loss_quantile_independence_test(
            returns,
            case=pit_case,
            window_type=window_type,
            window=window,
            n_sim=n_sim,
        )
    )

    return results
