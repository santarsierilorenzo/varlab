from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass(frozen=True)
class EsTestResult:
    """
    Standardized container for Expected Shortfall backtest results.
    """
    test_name: str
    statistic: Optional[float]
    p_value: Optional[float]
    reject: Optional[bool]
    info: Dict[str, Any]


def mcneil_frey_test(
    returns: np.ndarray,
    var: Union[np.ndarray, float],
    es: float,
    alpha: Optional[float] = None,
    alternative: str = "greater",
) -> EsTestResult:
    """
    Perform McNeil-Frey (2000) Expected Shortfall backtest.

    H0: E[L - ES | L > VaR] = 0

    Parameters
    ----------
    alternative : {"greater", "less", "two-sided"}
        Type of alternative hypothesis.
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(
            "alternative must be 'greater', 'less' or 'two-sided'."
        )

    returns = np.asarray(returns, dtype=float)
    losses: np.ndarray = -returns

    if np.isscalar(var):
        var_arr: np.ndarray = np.full_like(losses, float(var))
    else:
        var_arr = np.asarray(var, dtype=float)
        if var_arr.shape != losses.shape:
            raise ValueError("var must match returns shape.")

    exceed_mask: np.ndarray = losses > var_arr

    if not np.any(exceed_mask):
        raise ValueError("No VaR exceedances found.")

    excess_losses: np.ndarray = losses[exceed_mask] - es
    tail_mean = np.mean(losses[exceed_mask])

    if excess_losses.size < 2:
        raise ValueError(
            "Not enough exceedances for statistical inference."
        )

    t_stat, p_value = stats.ttest_1samp(
        excess_losses,
        popmean=0.0,
        alternative=alternative,
    )

    reject: Optional[bool] = None
    if alpha is not None:
        reject = p_value < alpha

    return EsTestResult(
        test_name="McNeil-Frey (2000)",
        statistic=float(t_stat),
        p_value=float(p_value),
        reject=reject,
        info={
            "sample_size": int(losses.size),
            "observed_violations": int(excess_losses.size),
            "alternative": alternative,
            "tail_mean": float(tail_mean)
        },
    )
