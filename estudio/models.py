"""Rolling, strictly out-of-sample VaR/ES forecasts for every model in the study.

For each day t, every model sees only days t-window .. t-1 and forecasts day t's loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantkit.risk import (
    evt_var_es,
    ewma_cov,
    ewma_vol_path,
    fit_garch_t,
    fit_t_dof,
    garch_vol_path,
    simulate_gaussian,
    simulate_student_t,
    student_t_var_es,
    var_es,
)

MODELS = {
    "hs": "Simulación histórica",
    "mc_norm": "MC normal, cov. muestral",
    "mc_norm_ewma": "MC normal, cov. EWMA",
    "mc_t_ewma": "MC t-Student, cov. EWMA",
    "fhs": "Simulación histórica filtrada",
    "garch_t": "GARCH(1,1)-t",
    "evt": "EVT filtrado (GARCH + GPD)",
}

LEVELS = {"var99": 0.99, "var975": 0.975}


def _measures(sample: np.ndarray) -> dict[str, float]:
    v99, _ = var_es(sample, 0.99)
    v975, e975 = var_es(sample, 0.975)
    return {"var99": v99, "var975": v975, "es975": e975}


def rolling_forecasts(rets: pd.DataFrame, w: np.ndarray, window: int = 500, lam: float = 0.94,
                      n_sims: int = 5_000, refit_every: int = 21, seed: int = 7,
                      models: list[str] | None = None) -> pd.DataFrame:
    """DataFrame indexed by forecast date with the realized portfolio return ('ret') and,
    for each model m, columns m_var99, m_var975 and m_es975 (positive losses)."""
    models = models or list(MODELS)
    rng = np.random.default_rng(seed)
    x = rets.to_numpy()
    port = x @ w
    rows = []
    nu = garch = evt_q = None
    for t in range(window, len(x)):
        win = x[t - window:t]
        pw = win @ w
        row = {"date": rets.index[t], "ret": port[t]}
        refit = (t - window) % refit_every == 0

        if "hs" in models:
            row |= {f"hs_{k}": v for k, v in _measures(pw).items()}
        if "mc_norm" in models:
            sims = simulate_gaussian(np.cov(win, rowvar=False).reshape(len(w), len(w)), n_sims, rng) @ w
            row |= {f"mc_norm_{k}": v for k, v in _measures(sims).items()}

        cov_e = ewma_cov(win, lam)
        vol = ewma_vol_path(pw, lam)
        vol_next = np.sqrt(lam * vol[-1] ** 2 + (1 - lam) * pw[-1] ** 2)
        if "mc_norm_ewma" in models:
            sims = simulate_gaussian(cov_e, n_sims, rng) @ w
            row |= {f"mc_norm_ewma_{k}": v for k, v in _measures(sims).items()}
        if "mc_t_ewma" in models:
            if refit or nu is None:  # tail thickness of vol-standardized returns, refit monthly
                nu = fit_t_dof(pw / vol)
            sims = simulate_student_t(cov_e, nu, n_sims, rng) @ w
            row |= {f"mc_t_ewma_{k}": v for k, v in _measures(sims).items()}
            row["nu"] = nu
        if "fhs" in models:  # empirical shape of standardized returns x tomorrow's vol
            row |= {f"fhs_{k}": v for k, v in _measures(pw / vol * vol_next).items()}

        if "garch_t" in models or "evt" in models:
            if refit or garch is None:
                new = fit_garch_t(pw)
                # A jump-dominated window can make MLE collapse to constant variance with
                # extreme tails (alpha + beta ~ 0); keep the last sensible fit instead.
                if garch is None or new["alpha"] + new["beta"] >= 0.5:
                    garch = new
                else:
                    row["garch_fallback"] = 1
            sig_path, sig_next = garch_vol_path(pw, garch)
            if "garch_t" in models:
                v99, _ = student_t_var_es(sig_next, garch["nu"], 0.99)
                v975, e975 = student_t_var_es(sig_next, garch["nu"], 0.975)
                row |= {"garch_t_var99": v99, "garch_t_var975": v975, "garch_t_es975": e975}
            if "evt" in models:
                if refit or evt_q is None:  # GPD tail of GARCH residuals, refit monthly
                    z = pw / sig_path
                    q99, _ = evt_var_es(z, 0.99)
                    q975, e975 = evt_var_es(z, 0.975)
                    evt_q = (q99, q975, e975)
                row |= {"evt_var99": evt_q[0] * sig_next, "evt_var975": evt_q[1] * sig_next,
                        "evt_es975": evt_q[2] * sig_next}
            row["garch_persistence"] = garch["alpha"] + garch["beta"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def fhs_paths(pw: np.ndarray, horizon: int, n_paths: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Multi-day filtered historical simulation (Barone-Adesi et al., 1999): draw standardized
    residuals and let EWMA volatility react to each simulated day, so a bad first day raises
    the risk of the next ones. Returns cumulative simple returns over the horizon."""
    vol = ewma_vol_path(pw, lam)
    z = pw / vol
    var = np.full(n_paths, lam * vol[-1] ** 2 + (1 - lam) * pw[-1] ** 2)
    log_cum = np.zeros(n_paths)
    for _ in range(horizon):
        # A -18 sigma residual (PASO) times a crisis vol can exceed -100%: a long position
        # cannot lose more than everything, so cap the daily loss.
        r = np.maximum(np.sqrt(var) * rng.choice(z, n_paths), -0.99)
        log_cum += np.log1p(r)
        var = lam * var + (1 - lam) * r ** 2
    return np.expm1(log_cum)
