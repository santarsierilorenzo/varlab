from typing import Optional, Dict, Any, Union, Tuple
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


def _validate_alternative(alternative: str) -> None:
    """
    Validate alternative hypothesis specification.
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError(
            "alternative must be 'greater', 'less' or 'two-sided'."
        )


def _prepare_losses(returns: np.ndarray) -> np.ndarray:
    """
    Convert returns to loss space.
    """
    returns = np.asarray(returns, dtype=float)
    return -returns


def _broadcast_to_losses(
    losses: np.ndarray,
    value: Union[np.ndarray, float],
    name: str,
) -> np.ndarray:
    """
    Broadcast scalar input or validate array to match losses shape.
    """
    if np.isscalar(value):
        return np.full_like(losses, float(value))

    arr = np.asarray(value, dtype=float)

    if arr.shape != losses.shape:
        raise ValueError(f"{name} must match returns shape.")

    return arr


def _ttest_mean_zero(
    sample: np.ndarray,
    alternative: str,
) -> Tuple[float, float]:
    """
    Perform one-sample t-test against zero mean.
    """
    if sample.size < 2:
        raise ValueError(
            "Not enough observations for statistical inference."
        )

    t_stat, p_value = stats.ttest_1samp(
        sample,
        popmean=0.0,
        alternative=alternative,
    )

    return float(t_stat), float(p_value)


def _decision(
    p_value: float,
    significance_level: Optional[float],
) -> Optional[bool]:
    """
    Return rejection decision if significance level is provided.
    """
    if significance_level is None:
        return None

    return p_value < significance_level


def mcneil_frey_test(
    returns: np.ndarray,
    var: Union[np.ndarray, float],
    es: Union[np.ndarray, float],
    alpha: Optional[float] = None,
    alternative: str = "greater",
) -> EsTestResult:
    """
    Perform the McNeil-Frey (2000) Expected Shortfall backtest.

    Null hypothesis
    ---------------
    H0:
        E[L_t - ES_t | L_t > VaR_t] = 0

    The test evaluates whether the Expected Shortfall forecast
    correctly captures the average magnitude of losses conditional
    on VaR exceedances.

    IMPORTANT
    ---------
    VaR must be an out-of-sample forecast:

        VaR_t = VaR_{t | F_{t-1}}

    It must be computed using only information available at time t-1.
    Using in-sample estimates invalidates the statistical interpretation.

    ES can be either:

    - A dynamic out-of-sample ES forecast:
          ES_t = ES_{t | F_{t-1}}

    - A constant ES target (e.g. ES targeting strategy).

    In the latter case, the test evaluates whether realized tail
    losses are consistent with the fixed ES risk budget.

    Parameters
    ----------
    returns : np.ndarray
        Realized portfolio returns.
    var : np.ndarray | float
        Forecasted VaR in loss space (positive values).
    es : np.ndarray | float
        Forecasted ES or constant ES target (positive values).
    alpha : Optional[float]
        Significance level for rejection decision.
    alternative : {"greater", "less", "two-sided"}
        "greater"  -> ES underestimation
        "less"     -> ES overestimation
        "two-sided"-> any systematic bias
    """
    _validate_alternative(alternative)

    losses: np.ndarray = _prepare_losses(returns)
    var_arr: np.ndarray = _broadcast_to_losses(losses, var, "var")
    es_arr: np.ndarray = _broadcast_to_losses(losses, es, "es")

    exceed_mask: np.ndarray = losses > var_arr

    if not np.any(exceed_mask):
        raise ValueError("No VaR exceedances found.")

    excess_losses: np.ndarray = (
        losses[exceed_mask] - es_arr[exceed_mask]
    )

    tail_mean: float = float(np.mean(losses[exceed_mask]))

    t_stat, p_value = _ttest_mean_zero(
        excess_losses,
        alternative,
    )

    return EsTestResult(
        test_name="McNeil-Frey (2000)",
        statistic=t_stat,
        p_value=p_value,
        reject=_decision(p_value, alpha),
        info={
            "sample_size": int(losses.size),
            "observed_violations": int(excess_losses.size),
            "alternative": alternative,
            "tail_mean": tail_mean,
        },
    )


def acerbi_szekely_test(
    returns: np.ndarray,
    var: Union[np.ndarray, float],
    es: Union[np.ndarray, float],
    alpha: float,
    significance_level: Optional[float] = None,
    alternative: str = "greater",
) -> EsTestResult:
    """
    Perform the Acerbi-Szekely (2014) Expected Shortfall backtest.

    Null hypothesis
    ---------------
    H0:
        E[Z_t] = 0

    where

        Z_t =
        (1/alpha) * (L_t - VaR_t) * 1_{L_t >= VaR_t}
        + VaR_t - ES_t

    This test evaluates the joint correctness of VaR and ES
    forecasts.

    IMPORTANT
    ---------
    VaR and ES must be genuine out-of-sample forecasts:

        VaR_t = VaR_{t | F_{t-1}}
        ES_t  = ES_{t | F_{t-1}}

    A constant ES value may be supplied (e.g. ES targeting). In that
    case, the test evaluates consistency of realized losses with a
    fixed ES risk target rather than with a fully dynamic joint
    VaR-ES forecast.

    Parameters
    ----------
    returns : np.ndarray
        Realized portfolio returns.
    var : np.ndarray | float
        Forecasted VaR in loss space.
    es : np.ndarray | float
        Forecasted ES or constant ES target.
    alpha : float
        Tail probability (e.g. 0.025).
    significance_level : Optional[float]
        Significance level for rejection decision.
    alternative : {"greater", "less", "two-sided"}
        "greater"  -> ES underestimation
        "less"     -> ES overestimation
        "two-sided"-> any systematic bias
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    _validate_alternative(alternative)

    losses: np.ndarray = _prepare_losses(returns)
    var_arr: np.ndarray = _broadcast_to_losses(losses, var, "var")
    es_arr: np.ndarray = _broadcast_to_losses(losses, es, "es")

    indicator: np.ndarray = (losses >= var_arr).astype(float)

    z_t: np.ndarray = (
        (losses - var_arr) * indicator / alpha
        + var_arr
        - es_arr
    )

    t_stat, p_value = _ttest_mean_zero(
        z_t,
        alternative,
    )

    return EsTestResult(
        test_name="Acerbi-Szekely (2014)",
        statistic=t_stat,
        p_value=p_value,
        reject=_decision(p_value, significance_level),
        info={
            "sample_size": int(losses.size),
            "alpha": float(alpha),
            "alternative": alternative,
            "mean_z": float(np.mean(z_t)),
            "std_z": float(np.std(z_t, ddof=1)),
        },
    )
