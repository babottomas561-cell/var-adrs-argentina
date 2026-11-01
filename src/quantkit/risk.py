"""Value-at-Risk / Expected Shortfall models and the statistical backtests used to judge them.

Conventions: returns are simple daily returns (losses negative); VaR and ES are reported as
positive loss numbers; ``alpha`` is the confidence level (0.99 -> tail probability 1%).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.special import xlogy

# ----------------------------------------------------------------------------- volatility

def ewma_cov(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics exponentially weighted covariance (zero mean), most recent row weighted most."""
    n = len(returns)
    w = (1 - lam) * lam ** np.arange(n - 1, -1, -1)
    w /= w.sum()
    return (returns * w[:, None]).T @ returns


def ewma_vol_path(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Ex-ante EWMA volatility for each day, using only data up to the previous day."""
    var = np.empty_like(returns)
    var[0] = returns[:20].var()
    for t in range(1, len(returns)):
        var[t] = lam * var[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return np.sqrt(var)


def fit_garch_t(returns: np.ndarray) -> dict[str, float]:
    """GARCH(1,1) with Student-t innovations, fitted by MLE (returns scaled to percent for
    numerical stability). Returns omega (in return units^2), alpha, beta and nu."""
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    p = res.params
    return {"omega": p["omega"] / 1e4, "alpha": p["alpha[1]"], "beta": p["beta[1]"], "nu": p["nu"]}


def garch_vol_path(returns: np.ndarray, params: dict[str, float]) -> tuple[np.ndarray, float]:
    """Conditional vol for each day (using data up to the previous day) and tomorrow's forecast."""
    o, a, b = params["omega"], params["alpha"], params["beta"]
    var = np.empty(len(returns) + 1)
    var[0] = returns.var()
    for t in range(len(returns)):
        var[t + 1] = o + a * returns[t] ** 2 + b * var[t]
    return np.sqrt(var[:-1]), float(np.sqrt(var[-1]))


# ----------------------------------------------------------------------------- simulation

def simulate_gaussian(cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    chol = np.linalg.cholesky(cov)
    return rng.standard_normal((n, cov.shape[0])) @ chol.T


def simulate_student_t(cov: np.ndarray, nu: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Multivariate Student-t draws whose covariance equals ``cov`` (requires nu > 2).

    A normal vector divided by sqrt(chi2/nu) shares one shock to the scale across all
    assets, so in a bad draw everything falls together: tail dependence that a Gaussian
    model cannot produce.
    """
    scale = cov * (nu - 2) / nu
    z = simulate_gaussian(scale, n, rng)
    w = rng.chisquare(nu, size=(n, 1)) / nu
    return z / np.sqrt(w)


def fit_t_dof(x: np.ndarray, lo: float = 2.5, hi: float = 30.0) -> float:
    """Degrees of freedom of a Student-t fitted by MLE to standardized returns."""
    nu, _, _ = stats.t.fit(x, floc=0.0)
    return float(np.clip(nu, lo, hi))


# ----------------------------------------------------------------------------- risk measures

def var_es(pnl: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """VaR and Expected Shortfall of a sample, as positive loss numbers."""
    q = np.quantile(pnl, 1 - alpha)
    return -q, -pnl[pnl <= q].mean()


def normal_var_es(sigma: float, alpha: float) -> tuple[float, float]:
    p = 1 - alpha
    z = stats.norm.ppf(p)
    return -z * sigma, sigma * stats.norm.pdf(z) / p


def student_t_var_es(sigma: float, nu: float, alpha: float) -> tuple[float, float]:
    """VaR/ES of a Student-t scaled to standard deviation ``sigma``."""
    p = 1 - alpha
    s = sigma * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(p, nu)
    es = s * stats.t.pdf(q, nu) / p * (nu + q ** 2) / (nu - 1)
    return -q * s, es


def evt_var_es(z: np.ndarray, alpha: float, tail_fraction: float = 0.10) -> tuple[float, float]:
    """Peaks-over-threshold: fit a Generalized Pareto to the worst ``tail_fraction`` of
    standardized losses and read VaR/ES off the fitted tail (McNeil & Frey, 2000).

    Below the threshold we trust the data; beyond it, where there are only a handful of
    observations, the GPD extrapolates the shape of the tail.
    """
    losses = -z
    u = np.quantile(losses, 1 - tail_fraction)
    excess = losses[losses > u] - u
    xi, _, beta = stats.genpareto.fit(excess, floc=0.0)
    xi = min(xi, 0.9)  # ES is infinite for xi >= 1
    p = 1 - alpha
    ratio = p / tail_fraction
    var = u + beta / xi * (ratio ** (-xi) - 1) if abs(xi) > 1e-6 else u - beta * np.log(ratio)
    es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
    return var, es


# ----------------------------------------------------------------------------- backtests

def kupiec_pof(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. H0: exception rate equals p. Returns (LR, p-value)."""
    n, x = len(hits), int(hits.sum())
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen independence test. H0: exceptions do not cluster. Returns (LR, p-value)."""
    h = hits.astype(int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def conditional_coverage(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Christoffersen joint test of correct frequency AND independence (chi2 with 2 dof)."""
    lr = kupiec_pof(hits, p)[0] + christoffersen_ind(hits)[0]
    return lr, 1 - stats.chi2.cdf(lr, 2)


def basel_zone(exceptions_250: int) -> str:
    """Basel traffic light for a 99% VaR over the last 250 days."""
    if exceptions_250 <= 4:
        return "verde"
    if exceptions_250 <= 9:
        return "amarilla"
    return "roja"


def acerbi_szekely_z2(returns: np.ndarray, var: np.ndarray, es: np.ndarray, p: float) -> float:
    """Acerbi & Szekely (2014) Z2 test of Expected Shortfall. E[Z2] = 0 if ES is right;
    negative values mean ES understated the losses. At p = 2.5% the 5% critical value is
    about -0.70 (their Table 1), largely independent of the return distribution."""
    hits = returns < -var
    return float(np.sum(returns * hits / es) / (len(returns) * p) + 1)


def quantile_loss(returns: np.ndarray, var: np.ndarray, p: float) -> float:
    """Average tick (pinball) loss of the VaR forecast: the scoring rule for which the true
    quantile is optimal, so it can rank models (lower is better)."""
    q = -var
    hit = (returns < q).astype(float)
    return float(np.mean((p - hit) * (returns - q)))
