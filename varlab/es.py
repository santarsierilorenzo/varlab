from typing import Iterable
import numpy as np


ArrayLike = Iterable[float]


def scale_positions(
    positions: ArrayLike,
    risk_current: float,
    risk_target: float,
) -> np.ndarray:
    """
    Rescale positions to match a target risk level.

    Notes
    -----
    The economic interpretation of the output depends on the scale of the
    input risk measures:

    - If the risk measure is expressed in unit terms (e.g. VaR or ES per unit
      notional), the output represents rescaled portfolio weights.
      
    - If the risk measure is expressed in monetary units, the output represents
      rescaled position sizes.

    The function is agnostic with respect to the specific risk measure used,
    provided it is homogeneous of degree one.
    """
    if risk_current <= 0.0:
        raise ValueError("risk_current must be positive.")

    if risk_target <= 0.0:
        raise ValueError("risk_target must be positive.")

    scale = risk_target / risk_current
    return np.asarray(positions, dtype=float) * scale
