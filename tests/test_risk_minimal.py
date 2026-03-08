import numpy as np
import pandas as pd

from varlab import es, rolling_var, var
from varlab.backtesting import diagnostic


def test_empirical_var_matches_higher_quantile():
    returns = np.array([-0.03, -0.01, 0.00, 0.02, -0.04], dtype=float)
    expected = np.quantile(-returns, 0.99, method="higher")

    result = var(returns, method="empirical", confidence=0.99)

    assert np.isclose(result, expected)


def test_empirical_es_is_tail_mean():
    returns = np.array([-0.03, -0.01, 0.00, 0.02, -0.04], dtype=float)
    losses = -returns
    q = np.quantile(losses, 0.80, method="higher")
    expected = losses[losses >= q].mean()

    result = es(returns, method="empirical", confidence=0.80)

    assert np.isclose(result, expected)


def test_rolling_var_output_shape_and_nans():
    returns = pd.Series([0.01, -0.02, 0.005, -0.01, 0.003], dtype=float)

    out = rolling_var(
        returns,
        window=3,
        method="empirical",
        confidence=0.95,
    )

    assert len(out) == len(returns)
    assert out.iloc[0:2].isna().all()
    assert out.iloc[2:].notna().all()


def test_diagnostic_run_coverage_only_smoke():
    returns = pd.Series(np.random.default_rng(42).normal(0.0, 0.01, size=250))
    exceedances = returns < -0.02

    res = diagnostic.run(
        returns=returns,
        exceedances=exceedances,
        confidence=0.99,
        window_type="rolling",
        pit_case="discrete",
        test_types=(diagnostic.TestCategory.COVERAGE,),
    )

    snapshot = res.snapshot()

    assert "coverage" in snapshot
    assert "exact_binomial" in snapshot["coverage"]
    assert "VALUE AT RISK BACKTEST REPORT" in res.report()
