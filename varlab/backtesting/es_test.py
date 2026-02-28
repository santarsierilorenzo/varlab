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

    Null hypothesis
    ---------------
    H0: E[L_t - ES | L_t > VaR_t] = 0

    The test evaluates whether the Expected Shortfall forecast
    systematically underestimates or overestimates realized losses
    beyond the VaR threshold.

    IMPORTANT
    ---------
    VaR must be an out-of-sample forecast:

        VaR_t = VaR_{t | F_{t-1}}

    It must be computed using only information available at time t-1.
    Using in-sample estimates invalidates the statistical interpretation
    of the test.

    The ES input can be either:

    - A constant value (e.g. regulatory or target ES), or
    - An out-of-sample forecast ES_t consistent with VaR_t.

    If ES is dynamic, it must also satisfy:

        ES_t = ES_{t | F_{t-1}}

    Parameters
    ----------
    returns : np.ndarray
        Realized returns.
    var : np.ndarray | float
        Forecasted VaR in loss space.
    es : float
        Constant Expected Shortfall forecast in loss space.
    alpha : Optional[float]
        Significance level for rejection decision.
    alternative : {"greater", "less", "two-sided"}
        Type of alternative hypothesis:
            "greater" -> ES underestimation test (standard case)
            "less" -> ES overestimation test
            "two-sided" -> any systematic bias
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
