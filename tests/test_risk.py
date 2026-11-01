"""Each risk model and backtest is checked against a known theoretical answer."""

import numpy as np
import pytest
from scipy import stats

from quantkit import risk

RNG = np.random.default_rng(42)


def test_student_t_simulation_preserves_covariance():
    cov = np.array([[0.04, 0.018], [0.018, 0.09]]) / 252
    x = risk.simulate_student_t(cov, nu=5, n=400_000, rng=RNG)
    np.testing.assert_allclose(np.cov(x, rowvar=False), cov, rtol=0.05)


def test_monte_carlo_normal_converges_to_closed_form():
    cov = np.array([[1.0, 0.5], [0.5, 2.0]]) * 1e-4
    w = np.array([0.6, 0.4])
    sims = risk.simulate_gaussian(cov, 1_000_000, RNG) @ w
    mc_var, mc_es = risk.var_es(sims, 0.99)
    var, es = risk.normal_var_es(np.sqrt(w @ cov @ w), 0.99)
    assert mc_var == pytest.approx(var, rel=0.01)
    assert mc_es == pytest.approx(es, rel=0.01)


@pytest.mark.parametrize("nu", [3.5, 5.0, 10.0])
def test_student_t_closed_form_matches_simulation(nu):
    sigma = 0.02
    x = stats.t.rvs(nu, size=2_000_000, random_state=RNG) * sigma * np.sqrt((nu - 2) / nu)
    var, es = risk.student_t_var_es(sigma, nu, 0.975)
    s_var, s_es = risk.var_es(x, 0.975)
    assert var == pytest.approx(s_var, rel=0.01)
    assert es == pytest.approx(s_es, rel=0.02)


def test_evt_recovers_true_tail_quantile_of_fat_tailed_data():
    nu = 4
    z = stats.t.rvs(nu, size=20_000, random_state=RNG)
    var, es = risk.evt_var_es(z, 0.995, tail_fraction=0.05)
    true_var = -stats.t.ppf(0.005, nu)
    assert var == pytest.approx(true_var, rel=0.08)
    assert es > var


def test_kupiec_does_not_reject_exact_rate_and_rejects_double_rate():
    hits = np.zeros(1000, bool)
    hits[::100] = True  # exactly 1%
    assert risk.kupiec_pof(hits, 0.01)[1] == pytest.approx(1.0)
    hits2 = np.zeros(1000, bool)
    hits2[::50] = True  # 2%
    assert risk.kupiec_pof(hits2, 0.01)[1] < 0.01


def test_christoffersen_rejects_clustered_exceptions():
    hits = np.zeros(1000, bool)
    hits[100:105] = hits[600:605] = True  # 1% overall, but in two bursts
    assert risk.kupiec_pof(hits, 0.01)[1] > 0.5
    assert risk.christoffersen_ind(hits)[1] < 0.001
    assert risk.conditional_coverage(hits, 0.01)[1] < 0.001


def test_acerbi_szekely_z2_centered_when_es_is_right_negative_when_understated():
    n, p, sigma = 250_000, 0.025, 0.01
    r = RNG.standard_normal(n) * sigma
    var, es = risk.normal_var_es(sigma, 1 - p)
    ok = risk.acerbi_szekely_z2(r, np.full(n, var), np.full(n, es), p)
    bad = risk.acerbi_szekely_z2(r, np.full(n, var * 0.8), np.full(n, es * 0.8), p)
    assert abs(ok) < 0.05
    assert bad < -0.5


def test_quantile_loss_is_minimized_by_the_true_quantile():
    r = RNG.standard_normal(200_000) * 0.01
    true = -stats.norm.ppf(0.01) * 0.01
    losses = [risk.quantile_loss(r, np.full(len(r), true * k), 0.01) for k in (0.8, 1.0, 1.2)]
    assert losses[1] < losses[0] and losses[1] < losses[2]


@pytest.mark.parametrize("n,zone", [(0, "verde"), (4, "verde"), (5, "amarilla"), (9, "amarilla"), (10, "roja")])
def test_basel_traffic_light_boundaries(n, zone):
    assert risk.basel_zone(n) == zone


def test_garch_recovers_simulated_parameters():
    omega, alpha, beta, nu = 2e-6, 0.08, 0.90, 6.0
    n = 6000
    r, var = np.empty(n), omega / (1 - alpha - beta)
    z = stats.t.rvs(nu, size=n, random_state=RNG) * np.sqrt((nu - 2) / nu)
    for t in range(n):
        r[t] = np.sqrt(var) * z[t]
        var = omega + alpha * r[t] ** 2 + beta * var
    p = risk.fit_garch_t(r)
    assert p["alpha"] == pytest.approx(alpha, abs=0.03)
    assert p["beta"] == pytest.approx(beta, abs=0.04)
    assert p["nu"] == pytest.approx(nu, abs=2.0)
