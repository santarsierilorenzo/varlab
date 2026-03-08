import numpy as np
import pandas as pd
import importlib

from varlab import es, rolling_var, var
from varlab.base import estimate_sigma
from varlab.backtesting import diagnostic

var_module = importlib.import_module("varlab.var")
es_module = importlib.import_module("varlab.es")


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


def test_diagnostic_run_preserves_tri_state_reject_for_basel():
    returns = pd.Series(np.random.default_rng(7).normal(0.0, 0.01, size=250))
    exceedances = returns < -0.03

    res = diagnostic.run(
        returns=returns,
        exceedances=exceedances,
        confidence=0.99,
        window_type="rolling",
        pit_case="discrete",
        test_types=(diagnostic.TestCategory.COVERAGE,),
    )

    basel = res.to_dict()["results"]["coverage"]["basel_traffic_light"]
    assert basel["reject"] is None


def test_diagnostic_run_passes_alpha_to_christoffersen_independence(monkeypatch):
    captured = {"alpha": None}

    def fake_christoffersen(exceedances, alpha=0.05, eps=1e-12):
        captured["alpha"] = alpha
        return diagnostic.independence.IndependenceTestResult(
            test_name="fake",
            statistic=0.0,
            p_value=1.0,
            reject=False,
            info={"outcome": "PASS"},
        )

    monkeypatch.setattr(
        diagnostic.independence,
        "christoffersen_independence",
        fake_christoffersen,
    )

    returns = pd.Series(np.random.default_rng(0).normal(0.0, 0.01, size=200))
    exceedances = returns < -0.05

    diagnostic.run(
        returns=returns,
        exceedances=exceedances,
        confidence=0.99,
        window_type="rolling",
        pit_case="discrete",
        alpha=0.01,
        test_types=(diagnostic.TestCategory.INDEPENDENCE,),
    )

    assert captured["alpha"] == 0.01


def test_estimate_sigma_multivariate_respects_ddof():
    returns = np.array(
        [
            [0.01, 0.02],
            [0.00, 0.03],
            [0.02, 0.01],
            [0.01, 0.00],
        ],
        dtype=float,
    )
    weights = np.array([0.6, 0.4], dtype=float)

    sigma_ddof0 = estimate_sigma(returns, weights=weights, ddof=0)
    sigma_ddof1 = estimate_sigma(returns, weights=weights, ddof=1)

    assert not np.isclose(sigma_ddof0, sigma_ddof1)


def test_student_df_estimation_uses_fit_mean_flag(monkeypatch):
    observed = {"var_fit_mean": None, "es_fit_mean": None}

    def fake_estimate_student_df_var(returns, fit_mean=False):
        observed["var_fit_mean"] = fit_mean
        return 8.0

    def fake_estimate_student_df_es(returns, fit_mean=False):
        observed["es_fit_mean"] = fit_mean
        return 8.0

    monkeypatch.setattr(
        var_module,
        "estimate_student_df",
        fake_estimate_student_df_var,
    )
    monkeypatch.setattr(
        es_module,
        "estimate_student_df",
        fake_estimate_student_df_es,
    )

    returns = np.random.default_rng(1).normal(0.001, 0.01, size=300)

    var(
        returns,
        method="parametric",
        distribution="t",
        df=None,
        mean="sample",
    )
    es(
        returns,
        method="parametric",
        distribution="t",
        df=None,
        mean="sample",
    )

    assert observed["var_fit_mean"] is True
    assert observed["es_fit_mean"] is True
