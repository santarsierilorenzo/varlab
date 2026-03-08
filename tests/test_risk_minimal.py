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


def test_diagnostic_run_propagates_statistical_parameters(monkeypatch):
    captured = {
        "distribution": None,
        "independence": None,
        "christoffersen_eps": None,
    }

    def fake_pit_diagnostics(
        values,
        case="continuous",
        distribution="normal",
        df=None,
        window_type="rolling",
        window=60,
        min_periods=60,
        ddof=1,
        max_dim=4,
        alpha=0.05,
        eps=1e-12,
        mean="zero",
    ):
        captured["distribution"] = {
            "case": case,
            "distribution": distribution,
            "df": df,
            "window_type": window_type,
            "window": window,
            "min_periods": min_periods,
            "ddof": ddof,
            "max_dim": max_dim,
            "alpha": alpha,
            "eps": eps,
            "mean": mean,
        }
        return {
            "uniformity": diagnostic.distribution.DistributionTestResult(
                test_name="u",
                statistic=0.0,
                p_value=1.0,
                reject=False,
                info={"outcome": "PASS"},
            ),
            "independence": diagnostic.distribution.DistributionTestResult(
                test_name="i",
                statistic=0.0,
                p_value=1.0,
                reject=False,
                info={"outcome": "PASS"},
            ),
            "berkowitz": diagnostic.distribution.DistributionTestResult(
                test_name="b",
                statistic=0.0,
                p_value=1.0,
                reject=False,
                info={"outcome": "PASS"},
            ),
        }

    def fake_christoffersen(exceedances, alpha=0.05, eps=1e-12):
        captured["christoffersen_eps"] = eps
        return diagnostic.independence.IndependenceTestResult(
            test_name="c",
            statistic=0.0,
            p_value=1.0,
            reject=False,
            info={"outcome": "PASS"},
        )

    def fake_loss_quantile_independence(
        values,
        case="continuous",
        distribution="normal",
        df=None,
        window_type="rolling",
        window=60,
        min_periods=60,
        max_lag=5,
        alpha=0.05,
        eps=1e-12,
        n_sim=5000,
        seed=0,
        ddof=1,
        mean="zero",
    ):
        captured["independence"] = {
            "case": case,
            "distribution": distribution,
            "df": df,
            "window_type": window_type,
            "window": window,
            "min_periods": min_periods,
            "max_lag": max_lag,
            "alpha": alpha,
            "eps": eps,
            "n_sim": n_sim,
            "seed": seed,
            "ddof": ddof,
            "mean": mean,
        }
        return diagnostic.independence.IndependenceTestResult(
            test_name="lq",
            statistic=0.0,
            p_value=1.0,
            reject=False,
            info={"outcome": "PASS"},
        )

    monkeypatch.setattr(
        diagnostic.distribution,
        "pit_diagnostics",
        fake_pit_diagnostics,
    )
    monkeypatch.setattr(
        diagnostic.independence,
        "christoffersen_independence",
        fake_christoffersen,
    )
    monkeypatch.setattr(
        diagnostic.independence,
        "loss_quantile_independence",
        fake_loss_quantile_independence,
    )

    returns = pd.Series(np.random.default_rng(3).normal(0.0, 0.01, size=220))
    exceedances = returns < -0.03

    diagnostic.run(
        returns=returns,
        exceedances=exceedances,
        confidence=0.99,
        window_type="expanding",
        pit_case="continuous",
        alpha=0.01,
        n_sim=1234,
        window=80,
        min_periods=75,
        distribution="t",
        df=9,
        mean="sample",
        ddof=0,
        max_dim=3,
        pit_eps=1e-9,
        loss_max_lag=7,
        loss_eps=1e-8,
        loss_seed=42,
        christoffersen_eps=1e-7,
        test_types=(
            diagnostic.TestCategory.DISTRIBUTION,
            diagnostic.TestCategory.INDEPENDENCE,
        ),
    )

    assert captured["distribution"] == {
        "case": "continuous",
        "distribution": "t",
        "df": 9,
        "window_type": "expanding",
        "window": 80,
        "min_periods": 75,
        "ddof": 0,
        "max_dim": 3,
        "alpha": 0.01,
        "eps": 1e-9,
        "mean": "sample",
    }
    assert captured["independence"] == {
        "case": "continuous",
        "distribution": "t",
        "df": 9,
        "window_type": "expanding",
        "window": 80,
        "min_periods": 75,
        "max_lag": 7,
        "alpha": 0.01,
        "eps": 1e-8,
        "n_sim": 1234,
        "seed": 42,
        "ddof": 0,
        "mean": "sample",
    }
    assert captured["christoffersen_eps"] == 1e-7


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
