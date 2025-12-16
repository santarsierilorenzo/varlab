from typing import Iterable, Optional
from scipy.stats import norm, t
import numpy as np


def var(
    returns: Iterable[float],
    n_days: int = 1,
    confidence: float = 0.99,
    method: str = "empirical",
    weights: Optional[Iterable[float]] = None,
    distribution: str = "normal",
    df: Optional[int] = None,
    lamb: Optional[float] = None
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
    returns_arr = np.asarray(returns, dtype=float)

    if returns_arr.ndim not in (1, 2):
        raise ValueError("returns must be 1D or 2D.")

    if method == "empirical":
        if returns_arr.ndim != 1:
            raise ValueError("Empirical VaR requires 1D returns.")
        return _empirical_var(
            returns_arr,
            n_days,
            confidence,
            lamb
        )

    if method == "parametric":
        return _parametric_var(
            returns_arr,
            n_days,
            confidence,
            weights,
            distribution,
            df,
        )

    raise ValueError(f"Unsupported VaR method: {method}.")


def _empirical_var(
    returns: np.ndarray,
    n_days: int,
    confidence: float,
    lamb: Optional[float] = None
) -> float:
    """
    Historical VaR using empirical quantiles with square-root-of-time scaling.

    ---
    Note on observation weighting:
    This function implements the weighting formula proposed by John C. Hull in
    the book "Options, Futures, and Other Derivatives". 

    If a lambda parameter is provided, the function applies exponential
    weighting  to the returns. The weights assigned to past observations
    decline at a rate controlled by lambda: a higher lambda (closer to 1)
    implies a slower decline, resulting in a "longer" memory of past volatility
    """
    if np.min(returns) >= 0.0:
        raise ValueError(
            "Historical VaR undefined: no negative returns."
        )
    if lamb != None:
        n = len(returns)
        w = (lam**(np.arange(n, 1, -1)) * (1 - lam)) / (1 - lam**n)
        returns = returns * w

    gamma = 1.0 - confidence
    lam = 0.90

    # NOTE: The formal definition of VaR requires finding the infimum of the
    # set of losses where the cumulative distribution exceeds gamma.
    # In practical terms: always choose the greater value when estimating the
    # quantile =^.^=
    q = np.quantile(returns, gamma, method="higher")

    return -q * np.sqrt(n_days)


def _parametric_var(
    returns: np.ndarray,
    n_days: int,
    confidence: float,
    weights: Optional[Iterable[float]],
    distribution: str,
    df: Optional[int],
) -> float:
    """
    Parametric VaR assuming i.i.d. returns and zero mean.
    """
    sigma = _estimate_sigma(returns, weights)

    gamma = 1.0 - confidence
    q = _quantile(gamma, distribution, df)

    return -q * sigma * np.sqrt(n_days)


def _estimate_sigma(
    returns: np.ndarray,
    weights: Optional[Iterable[float]],
) -> float:
    """
    Estimate volatility for single assets or portfolios.
    """
    if returns.ndim == 1:
        if returns.size < 2:
            raise ValueError("Insufficient sample size.")
        sigma = returns.std(ddof=1)
    else:
        if weights is None:
            raise ValueError("Weights required for portfolio VaR.")
        weights_arr = np.asarray(weights, dtype=float)
        sigma = _portfolio_volatility(returns, weights_arr)

    if sigma <= 0.0:
        raise ValueError("Zero or negative volatility.")

    return sigma


def _quantile(
    gamma: float,
    distribution: str,
    df: Optional[int],
) -> float:
    """
    Return left-tail quantile for the selected distribution.
    """
    if distribution == "normal":
        return norm.ppf(gamma)

    if distribution == "tstudent":
        if df is None:
            raise ValueError("df must be provided for t-student.")
        if df <= 2:
            raise ValueError("df must be > 2.")
        return t.ppf(gamma, df=df)

    raise ValueError(f"Unsupported distribution: {distribution}.")


def _portfolio_volatility(
    returns: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute portfolio volatility using the variance–covariance matrix.
    """
    if returns.shape[1] != weights.shape[0]:
        raise ValueError("Weights dimension mismatch.")

    cov = np.cov(returns, rowvar=False)
    var = weights.T @ cov @ weights

    return var ** 0.5


def var_targeting(
    target_var: float,
    portfolio_var: float,
    weights: Iterable[float],
) -> np.ndarray:
    """
    Rescale portfolio weights to match a target VaR.

    Notes
    -----
    The economic interpretation of the output depends on the scale of the
    inputs:

    - If VaR is expressed in unit terms (VaR per unit notional), the output
    represents rescaled portfolio weights.

    - If VaR is expressed in monetary units, the output represents rescaled
    position sizes.
    """

    if portfolio_var <= 0.0:
        raise ValueError("portfolio_var must be positive.")

    return np.asarray(weights, dtype=float) * target_var / portfolio_var
