from typing import Iterable, Optional, Literal
import numpy as np
from .base import (
    estimate_sigma,
    weighted_sorted_dist,
    tail_quantile,
    time_scaling,
)

ArrayLike = Iterable[float]
Method = Literal["empirical", "parametric"]

def var(
    returns: ArrayLike,
    n_days: int = 1,
    confidence: float = 0.99,
    method: Method = "empirical",
    weights: Optional[ArrayLike] = None,
    distribution: str = "normal",
    df: Optional[int] = None,
    lamb: Optional[float] = None,
) -> float:
    """
    Compute Value at Risk (VaR) for a return series or portfolio.

    Supports empirical and parametric methods with normal or t-Student
    distributions. Returns a positive VaR.

    Notes
    -----
    The function is scale-agnostic with respect to the input returns:

    - If `returns` represent PnL values, the VaR is expressed in monetary units
    - If `returns` represent simple returns, the VaR is expressed in percentage
    terms (assuming unit notional).
    """
    losses = -np.asarray(returns, dtype=float)

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    if method == "empirical":
        return _empirical_var(
            losses=losses,
            gamma=confidence,
            n_days=n_days,
            lamb=lamb,
        )

    if method == "parametric":
        return _parametric_var(
            losses=losses,
            gamma=confidence,
            n_days=n_days,
            weights=weights,
            distribution=distribution,
            df=df,
        )

    raise ValueError(f"Unsupported VaR method: {method}.")


def _empirical_var(
    losses: np.ndarray,
    gamma: float,
    n_days: int,
    lamb: Optional[float],
) -> float:
    """
    Empirical (historical) VaR.
    """
    if losses.ndim != 1:
        raise ValueError("Empirical VaR requires 1D returns.")

    if lamb is None:
        # NOTE: The formal definition of VaR requires finding the infimum of
        # the set of losses where the cumulative distribution exceeds gamma.
        # In practical terms: always choose the greater value when estimating
        # the quantile =^.^=
        q = np.quantile(losses, gamma, method="higher")
        var_value = q

    else:
        sorted_pnl, _, cum_w = weighted_sorted_dist(losses, lamb)
        idx = int(np.searchsorted(cum_w, gamma, side="right"))
        idx = min(idx, len(sorted_pnl) - 1)
        var_value = float(sorted_pnl[idx])

    return time_scaling(var_value, n_days)


def _parametric_var(
    losses: np.ndarray,
    gamma: float,
    n_days: int,
    distribution: str,
    df: Optional[int],
    weights: Optional[Iterable[float]] = None,
    mean: Literal["zero", "sample"] = "zero",
) -> float:
    """
    Parametric Value-at-Risk under i.i.d. assumption.

    The model assumes:

        L_t = mu + sigma * Z_t

    where Z_t follows a standardized distribution (normal or Student-t).

    If mean="zero", mu is assumed equal to 0.
    If mean="sample", mu is estimated from the sample.

    Multi-day VaR is computed as:

        VaR_N = N * mu + sqrt(N) sgima *  z_gamma

    where z_gamma is the gamma-quantile of the standardized distribution.
    """
    sigma = estimate_sigma(
        values=losses,
        weights=weights,
    )

    mu = np.mean(losses) if mean == "sample" else 0.0

    z = tail_quantile(
        gamma=gamma,
        distribution=distribution,
        df=df,
    )

    if distribution == "t":
        if df is None:
            raise ValueError(
                "df must be provided for Student-t distribution."
            )
        # Standardize Student-t so that Var(Z) = 1.
        # The standard t has variance df / (df - 2),
        # so we rescale the quantile accordingly.
        z *= np.sqrt((df - 2) / df)

    var_value = mu * n_days + time_scaling(z * sigma, n_days)

    return float(var_value)
