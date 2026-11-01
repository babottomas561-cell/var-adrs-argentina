"""¿Qué modelo de VaR sobrevive al mercado argentino? Backtest de 7 modelos, 2009-2026.

Reproduce todas las figuras y tablas del informe (README.md en la raiz del repo):
    uv run python estudio/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).parent))

from escenarios import escenarios_hipoteticos  # noqa: E402
from models import LEVELS, MODELS, fhs_paths, rolling_forecasts  # noqa: E402

from quantkit import style  # noqa: E402
from quantkit.data import load_prices  # noqa: E402
from quantkit.metrics import TRADING_DAYS  # noqa: E402
from quantkit.risk import (  # noqa: E402
    acerbi_szekely_z2,
    basel_zone,
    christoffersen_ind,
    conditional_coverage,
    ewma_cov,
    ewma_vol_path,
    fit_t_dof,
    kupiec_pof,
    normal_var_es,
    quantile_loss,
    simulate_student_t,
    var_es,
)

HERE = Path(__file__).parent
FIG = HERE / "figures"

# Fixed before the first run (see METODOLOGIA.md).
TICKERS = ["GGAL", "BMA", "BBAR", "YPF", "TGS", "EDN", "TEO", "CRESY"]
W = np.full(len(TICKERS), 1 / len(TICKERS))
WINDOW, LAMBDA, N_SIMS = 500, 0.94, 5_000
Z2_CRITICAL = -0.70        # Acerbi-Szekely 5% critical value at p = 2.5%
N_PATHS, BLOCK = 10_000, 20
SEED = 7
PASO = pd.Timestamp("2019-08-12")
SHOW = ["mc_norm", "mc_t_ewma", "evt"]  # 3-series cap for charts where marks overlap
COLORS = {"hs": style.YELLOW, "mc_norm": style.ORANGE, "mc_norm_ewma": style.MAGENTA,
          "mc_t_ewma": style.BLUE, "fhs": style.VIOLET, "garch_t": style.GREEN, "evt": style.AQUA}
PASS_BG, FAIL_BG, NEUTRAL_BG = "#e3f3e3", "#f8e0df", "#f0efec"


# ============================================================================ analysis

def evaluate(fc: pd.DataFrame) -> pd.DataFrame:
    """Every backtest for every model, one row per model."""
    rows = {}
    r = fc["ret"].to_numpy()
    for m, name in MODELS.items():
        if f"{m}_var99" not in fc:
            continue
        v99, v975, e975 = (fc[f"{m}_{k}"].to_numpy() for k in ["var99", "var975", "es975"])
        hits = r < -v99
        roll = pd.Series(hits.astype(int), index=fc.index).rolling(250).sum().dropna()
        rows[name] = {
            "key": m,
            "Excepciones": int(hits.sum()),
            "Esperadas": round(len(r) * 0.01),
            "Tasa 99%": hits.mean(),
            "Tasa 97,5%": (r < -v975).mean(),
            "Kupiec p": kupiec_pof(hits, 0.01)[1],
            "Christoffersen p": christoffersen_ind(hits)[1],
            "Cob. condicional p": conditional_coverage(hits, 0.01)[1],
            "Días en zona roja": (roll >= 10).mean(),
            "Z2 (ES 97,5%)": acerbi_szekely_z2(r, v975, e975, 0.025),
            "Pérdida cuantílica": quantile_loss(r, v99, 0.01) * 1e4,
            "VaR 99% medio": v99.mean(),
        }
    out = pd.DataFrame(rows).T
    out["Ranking"] = out["Pérdida cuantílica"].astype(float).rank().astype(int)
    return out


def stylized_facts(rets: pd.DataFrame) -> pd.DataFrame:
    port = rets @ W
    cols = {**{t: rets[t] for t in TICKERS}, "Cartera": port}
    rows = {}
    for name, x in cols.items():
        jb = stats.jarque_bera(x)
        rows[name] = {"Vol. anual": x.std() * np.sqrt(TRADING_DAYS), "Asimetría": stats.skew(x),
                      "Curtosis exceso": stats.kurtosis(x), "Peor día": x.min(),
                      "Jarque-Bera p": jb.pvalue}
    return pd.DataFrame(rows).T


def avg_pairwise_corr(rets: pd.DataFrame, window: int = 63) -> pd.Series:
    out = {}
    x = rets.to_numpy()
    iu = np.triu_indices(x.shape[1], 1)
    for t in range(window, len(x) + 1):
        c = np.corrcoef(x[t - window:t], rowvar=False)
        out[rets.index[t - 1]] = c[iu].mean()
    return pd.Series(out)


def mc_convergence(rets: pd.DataFrame, reps: int = 200) -> pd.DataFrame:
    """Simulation noise of the MC t-Student VaR on the last day, as a function of N."""
    rng = np.random.default_rng(SEED)
    win = rets.to_numpy()[-WINDOW:]
    pw = win @ W
    nu = fit_t_dof(pw / ewma_vol_path(pw, LAMBDA))
    cov = ewma_cov(win, LAMBDA)
    rows = []
    for n in [500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000]:
        est = [var_es(simulate_student_t(cov, nu, n, rng) @ W, 0.99)[0] for _ in range(reps)]
        rows.append({"N": n, "VaR medio": np.mean(est), "Error MC (desvío)": np.std(est)})
    return pd.DataFrame(rows).set_index("N")


def ten_day_backtest(rets: pd.DataFrame, fc: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Non-overlapping 10-day windows: square-root-of-time scaling vs simulated paths."""
    rng = np.random.default_rng(SEED)
    x = rets.to_numpy()
    port = x @ W
    res = {k: [] for k in ["Normal EWMA x raíz(10)", "FHS x raíz(10)", "FHS trayectorias (10 días)"]}
    realized = []
    for t in range(WINDOW, len(x) - horizon, horizon):
        pw = x[t - WINDOW:t] @ W
        date = rets.index[t]
        realized.append(np.prod(1 + port[t:t + horizon]) - 1)
        sig = np.sqrt(W @ ewma_cov(x[t - WINDOW:t], LAMBDA) @ W)
        res["Normal EWMA x raíz(10)"].append(normal_var_es(sig, 0.99)[0] * np.sqrt(horizon))
        res["FHS x raíz(10)"].append(fc.loc[date, "fhs_var99"] * np.sqrt(horizon))
        res["FHS trayectorias (10 días)"].append(var_es(fhs_paths(pw, horizon, 4_000, LAMBDA, rng), 0.99)[0])
    realized = np.array(realized)
    rows = {}
    for k, v in res.items():
        hits = realized < -np.array(v)
        rows[k] = {"Ventanas": len(hits), "Excepciones": int(hits.sum()), "Esperadas": round(len(hits) * 0.01, 1),
                   "Tasa": hits.mean(), "Kupiec p": kupiec_pof(hits, 0.01)[1], "VaR 10d medio": np.mean(v)}
    return pd.DataFrame(rows).T


def robustness(rets: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    variants = {
        "Caso base": dict(),
        "Ventana 250": dict(window=250),
        "Ventana 1000": dict(window=1000),
        "λ = 0,97": dict(lam=0.97),
    }
    out = {}
    for name, kw in variants.items():
        fc = rolling_forecasts(rets, W, **({"window": WINDOW, "lam": LAMBDA, "n_sims": N_SIMS} | kw))
        ev = evaluate(fc)
        out[name] = ev["Tasa 99%"]
        out[name + " CC"] = ev["Cob. condicional p"]
    fc = rolling_forecasts(spy, np.ones(1), window=WINDOW, lam=LAMBDA, n_sims=N_SIMS)
    ev = evaluate(fc)
    out["Control: S&P 500"] = ev["Tasa 99%"]
    out["Control: S&P 500 CC"] = ev["Cob. condicional p"]
    return pd.DataFrame(out)


def block_bootstrap_paths(x: np.ndarray, horizon: int, n_paths: int, block: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Resample whole blocks of historical days: keeps fat tails, joint crashes and
    short-run volatility clustering that i.i.d. draws destroy."""
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(x) - block, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon]
    return x[idx]


# ============================================================================ figures

def fig_fan(rets: pd.DataFrame) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    x = rets.to_numpy()
    boot = block_bootstrap_paths(x, TRADING_DAYS, N_PATHS, BLOCK, rng) @ W
    gauss = rng.multivariate_normal(x.mean(0), np.cov(x, rowvar=False), size=(N_PATHS, TRADING_DAYS)) @ W
    stats_ = {}
    for name, p in [("boot", boot), ("norm", gauss)]:
        eq = np.cumprod(1 + p, axis=1)
        mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
        v, e = var_es(eq[:, -1] - 1, 0.95)
        stats_[name] = {"VaR 95% 1 año": v, "ES 95% 1 año": e, "P(caída > 40%)": (mdd < -0.4).mean(),
                        "Mediana": np.median(eq[:, -1] - 1)}
        if name == "boot":
            equity = eq
    fig, ax = plt.subplots(figsize=(10, 5.6))
    days = np.arange(1, TRADING_DAYS + 1)
    pct = np.percentile(equity, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(days, pct[0], pct[4], color=style.BLUE, alpha=0.14, lw=0, label="Percentiles 5-95")
    ax.fill_between(days, pct[1], pct[3], color=style.BLUE, alpha=0.30, lw=0, label="Percentiles 25-75")
    for i in np.random.default_rng(1).choice(N_PATHS, 40, replace=False):
        ax.plot(days, equity[i], color=style.BLUE, lw=0.5, alpha=0.25)
    ax.plot(days, pct[2], color=style.INK, lw=2, label="Mediana")
    ax.axhline(1, color=style.BASELINE, lw=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v - 1:+.0%}"))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.set_yticks([0.4, 0.6, 0.8, 1, 1.5, 2, 3])
    ax.set_xlim(1, days[-1])
    ax.set_xlabel("Ruedas hacia adelante")
    ax.set_ylabel("Retorno de la cartera (escala log)")
    b = stats_["boot"]
    ax.set_title("10.000 años simulados para una cartera de ADRs argentinos", pad=26)
    style.subtitle(ax, f"Bootstrap por bloques 2007-2026. A 1 año: VaR 95% {b['VaR 95% 1 año']:.0%}, "
                       f"ES {b['ES 95% 1 año']:.0%}, P(caída > 40% en algún momento) {b['P(caída > 40%)']:.0%}")
    ax.legend(loc="upper left")
    style.source(fig, "Cartera: GGAL, BMA, BBAR, YPF, TGS, EDN, TEO, CRESY en partes iguales, en USD, "
                      "rebalanceo diario. Fuente: Yahoo Finance.")
    style.save(fig, FIG / "01_abanico.png")
    return {f"{k} ({'bootstrap' if n == 'boot' else 'normal'})": v for n, d in stats_.items() for k, v in d.items()}


def fig_stylized(rets: pd.DataFrame, corr: pd.Series) -> dict[str, float]:
    port = (rets @ W).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    # 1. QQ plot vs normal
    ax = axes[0]
    z = np.sort((port - port.mean()) / port.std())
    theo = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    ax.scatter(theo, z, s=4, color=style.BLUE, lw=0)
    ax.plot([-4.5, 4.5], [-4.5, 4.5], color=style.INK_2, lw=1, ls="--")
    ax.set_xlabel("Cuantil teórico normal")
    ax.set_ylabel("Cuantil observado (desvíos)")
    ax.set_title("1. Colas gruesas", fontsize=12.5)
    ax.annotate("PASO 2019", xy=(theo[0], z[0]), xytext=(10, 4), textcoords="offset points",
                fontsize=9, color=style.INK_2)
    # 2. ACF of returns vs |returns|
    ax = axes[1]
    lags = np.arange(1, 61)
    ax.bar(lags, acf(port, nlags=60)[1:], color=style.MUTED, width=0.8, label="Retornos")
    ax.plot(lags, acf(np.abs(port), nlags=60)[1:], color=style.BLUE, lw=2, label="|Retornos|")
    band = 1.96 / np.sqrt(len(port))
    ax.axhspan(-band, band, color=style.GRID, alpha=0.6, lw=0)
    ax.set_xlabel("Rezago (ruedas)")
    ax.set_ylabel("Autocorrelación")
    ax.legend(loc="upper right")
    ax.set_title("2. La volatilidad viene en racimos", fontsize=12.5)
    # 3. Average pairwise correlation vs volatility regime
    ax = axes[2]
    vol = pd.Series(port, index=rets.index).rolling(63).std().reindex(corr.index)
    hi = vol > vol.quantile(0.8)
    ax.plot(corr, color=style.BLUE, lw=1.2)
    ax.fill_between(corr.index, 0, 1, where=hi.to_numpy(), color=style.RED, alpha=0.10, lw=0,
                    transform=ax.get_xaxis_transform(), label="20% de ruedas más volátiles")
    ax.set_ylabel("Correlación promedio entre ADRs (63d)")
    ax.legend(loc="lower right")
    c_hi, c_lo = corr[hi].mean(), corr[~hi].mean()
    ax.set_title("3. La diversificación desaparece en crisis", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.text(0.01, 0.99, "Tres hechos que rompen el modelo normal", fontsize=15, fontweight="bold", va="top")
    fig.text(0.01, 0.92, f"Curtosis en exceso de la cartera: {stats.kurtosis(port):.0f} (normal = 0). "
                         f"Correlación entre ADRs: {c_hi:.2f} en alta volatilidad vs {c_lo:.2f} en calma",
             fontsize=10.5, color=style.INK_2, ha="left", va="top")
    style.save(fig, FIG / "02_hechos_estilizados.png")
    return {"corr_alta_vol": c_hi, "corr_calma": c_lo}


def fig_tails(rets: pd.DataFrame) -> None:
    port = (rets @ W).to_numpy()
    mu, sd = port.mean(), port.std()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bins = np.linspace(port.min() - 0.01, port.max() + 0.01, 120)
    ax.hist(port, bins=bins, color=style.BLUE, histtype="stepfilled", alpha=0.25, density=True)
    ax.hist(port, bins=bins, color=style.BLUE, histtype="step", lw=1.6, density=True, label="Retornos reales")
    xs = np.linspace(bins[0], bins[-1], 600)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=style.ORANGE, lw=1.8, label="Normal con igual media y volatilidad")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 60)
    paso = rets.loc[PASO].to_numpy() @ W
    sig = (paso - mu) / sd
    ax.annotate(f"PASO 2019: {paso:.0%} en un día\n= {abs(sig):.0f} desvíos estándar", xy=(paso, 2e-3),
                xytext=(paso + 0.06, 0.05), fontsize=9.5, color=style.INK_2,
                arrowprops=dict(arrowstyle="->", color=style.INK_2, lw=1))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Retorno diario de la cartera")
    ax.set_ylabel("Densidad (escala log)")
    ax.set_title("Las colas argentinas no son normales", pad=26)
    log10_p = stats.norm.logsf(abs(sig)) / np.log(10)
    style.subtitle(ax, f"Bajo una normal, un día como el 12/08/2019 tendría probabilidad de 1 en "
                       f"10^{-log10_p:.0f}. Pasó una vez en {len(port) / TRADING_DAYS:.0f} años")
    ax.legend(loc="upper right")
    style.save(fig, FIG / "03_colas.png")


def fig_backtest(fc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, m in zip(axes, ["mc_norm", "mc_t_ewma"]):
        hits = fc["ret"] < -fc[f"{m}_var99"]
        ax.scatter(fc.index, fc["ret"], s=2, color=style.MUTED, alpha=0.5, lw=0)
        ax.plot(fc.index, -fc[f"{m}_var99"], color=COLORS[m], lw=1.4, label=f"{MODELS[m]}: VaR 99%")
        ax.scatter(fc.index[hits], fc["ret"][hits], s=14, color=style.RED, zorder=3,
                   label=f"Excepciones: {hits.sum()} (esperadas {round(len(fc) * 0.01)})")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        ax.set_ylim(-0.25, 0.20)
        ax.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor=style.SURFACE,
                  edgecolor=style.GRID, framealpha=0.95)
        ax.scatter([PASO], [-0.243], marker="v", s=60, color=style.RED, zorder=4, clip_on=False)
        ax.annotate(f"PASO 2019: {fc.loc[PASO, 'ret']:.0%} (fuera de escala)", xy=(PASO, -0.243),
                    xytext=(-12, 2), textcoords="offset points", ha="right", va="center",
                    fontsize=9.5, color=style.INK_2)
    axes[0].set_title("VaR 99% a 1 día, re-estimado cada día con las 500 ruedas previas", pad=26)
    style.subtitle(axes[0], "El normal estático falla en racimos; con colas gruesas y volatilidad adaptativa "
                            "acierta la frecuencia, pero ningún modelo anticipa un salto político")
    style.source(fig, "Puntos grises: retornos diarios reales. Rojo: días en que la pérdida superó el VaR pronosticado. "
                      "Eje recortado en -25%.")
    style.save(fig, FIG / "04_backtest_var.png")


def fig_scorecard(ev: pd.DataFrame) -> None:
    cols = [
        ("Tasa 99%", "Frecuencia\n(Kupiec)", lambda r: f"{r['Tasa 99%']:.2%}", lambda r: r["Kupiec p"] > 0.05),
        ("Christoffersen p", "Sin racimos\n(Christoffersen)", lambda r: f"p = {r['Christoffersen p']:.2f}",
         lambda r: r["Christoffersen p"] > 0.05),
        ("Cob. condicional p", "Cobertura\ncondicional", lambda r: f"p = {r['Cob. condicional p']:.2f}",
         lambda r: r["Cob. condicional p"] > 0.05),
        ("Z2 (ES 97,5%)", "Expected Shortfall\n(Acerbi-Székely)", lambda r: f"Z2 = {r['Z2 (ES 97,5%)']:.2f}",
         lambda r: r["Z2 (ES 97,5%)"] > Z2_CRITICAL),
        ("Días en zona roja", "Basilea: días\nen zona roja", lambda r: f"{r['Días en zona roja']:.1%}", None),
        ("Ranking", "Pérdida cuantílica\n(1 = mejor)", lambda r: f"#{int(r['Ranking'])}", None),
    ]
    n_r, n_c = len(ev), len(cols)
    fig, ax = plt.subplots(figsize=(13, 0.62 * n_r + 2.2))
    ax.set_xlim(0, n_c)
    ax.set_ylim(n_r, -1)
    ax.axis("off")
    for j, (_, head, _, _) in enumerate(cols):
        ax.text(j + 0.5, -0.45, head, ha="center", va="center", fontsize=10, color=style.INK_2, fontweight="bold")
    for i, (name, r) in enumerate(ev.iterrows()):
        ax.text(-0.05, i + 0.5, name, ha="right", va="center", fontsize=10.5, color=style.INK)
        for j, (_, _, fmt, ok) in enumerate(cols):
            passed = None if ok is None else bool(ok(r))
            bg = NEUTRAL_BG if passed is None else (PASS_BG if passed else FAIL_BG)
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.08), 0.92, 0.84, color=bg, lw=0))
            label = fmt(r) if passed is None else f"{fmt(r)}  {'aprueba' if passed else 'rechaza'}"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=9.5, color=style.INK,
                    fontweight="bold" if passed else "normal")
    passes = ev.apply(lambda r: sum(bool(c[3](r)) for c in cols if c[3] is not None), axis=1)
    n_tests = sum(c[3] is not None for c in cols)
    winners = list(passes[passes == n_tests].index)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.84)
    n_days = f"{ev['Esperadas'].iloc[0] * 100:,.0f}".replace(",", ".")
    fig.text(0.01, 0.97, f"Boletín de calificaciones: 7 modelos de VaR, {n_days} días fuera de muestra",
             fontsize=14, fontweight="bold", va="top")
    verdict = (f"Aprueban los {n_tests} tests: {', '.join(winners)}." if winners
               else f"Ningún modelo aprueba los {n_tests} tests.")
    fig.text(0.01, 0.915, f"Verde: aprueba al 5%. {verdict} VaR 99% y ES 97,5% a 1 día, cartera de 8 ADRs.",
             fontsize=10.5, color=style.INK_2, va="top")
    style.save(fig, FIG / "05_boletin.png")


def fig_basel(fc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhspan(0, 4.5, color="#0ca30c", alpha=0.08, lw=0)
    ax.axhspan(4.5, 9.5, color="#fab219", alpha=0.12, lw=0)
    ax.axhspan(9.5, 25, color="#d03b3b", alpha=0.08, lw=0)
    for y, lab in [(2.25, "Zona verde\n(0-4)"), (7, "Zona amarilla\n(5-9)"), (12, "Zona roja (10+):\nmás capital\nexigido")]:
        ax.text(1.01, y, lab, transform=ax.get_yaxis_transform(), fontsize=9.5, color=style.INK_2, ha="left", va="center")
    for m in SHOW:
        hits = (fc["ret"] < -fc[f"{m}_var99"]).astype(int).rolling(250).sum()
        ax.plot(hits, color=COLORS[m], lw=1.8, label=MODELS[m])
    ax.set_ylim(0, 25)
    ax.set_ylabel("Excepciones del VaR 99% en las últimas 250 ruedas")
    ax.legend(loc="upper right", frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID)
    ax.set_title("El semáforo de Basilea, día por día", pad=26)
    style.subtitle(ax, "Con el modelo normal, un banco con esta cartera habría pasado años en zona amarilla y roja")
    style.save(fig, FIG / "06_semaforo_basilea.png")


def fig_events(fc: pd.DataFrame) -> pd.DataFrame:
    worst = fc["ret"].nsmallest(6).index
    tbl = pd.DataFrame({MODELS[m]: fc.loc[worst, f"{m}_var99"] for m in MODELS})
    tbl.insert(0, "Pérdida real", -fc.loc[worst, "ret"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ylabels = [d.strftime("%d/%m/%Y") for d in worst]
    y = np.arange(len(worst))
    ax.scatter(-fc.loc[worst, "ret"], y, marker="X", s=90, color=style.INK, zorder=4, label="Pérdida real")
    for k, m in enumerate(SHOW):
        ax.scatter(fc.loc[worst, f"{m}_var99"], y + (k - 1) * 0.18, s=50, color=COLORS[m], zorder=3,
                   label=f"VaR 99% {MODELS[m]}")
    for yi, d in zip(y, worst):
        ax.plot([0, -fc.loc[d, "ret"]], [yi, yi], color=style.GRID, lw=1, zorder=1)
    ax.set_yticks(y, ylabels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Pérdida en un día")
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor=style.SURFACE, edgecolor=style.GRID)
    ax.set_title("Los 6 peores días: lo que cada modelo esperaba la noche anterior", pad=26)
    style.subtitle(ax, "El VaR mide el riesgo de un día normal-malo, no el del peor día: para eso están el ES y los "
                       "escenarios de estrés")
    style.save(fig, FIG / "07_peores_dias.png")
    return tbl


def fig_mc_error(conv: pd.DataFrame, model_spread: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    rel = conv["Error MC (desvío)"] / conv["VaR medio"]
    ax.plot(conv.index, rel, color=style.BLUE, marker="o", lw=2, label="Error de simulación (desvío del VaR estimado)")
    ax.axhline(model_spread, color=style.ORANGE, lw=2, ls="--", label="Desacuerdo entre los 7 modelos el mismo día")
    ax.axvline(N_SIMS, color=style.INK_2, lw=1, ls=":")
    ax.annotate(f"N usado: {N_SIMS:,}".replace(",", "."), xy=(N_SIMS, 0.6), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=9.5, color=style.INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.set_xlabel("Número de simulaciones")
    ax.set_ylabel("Error relativo del VaR 99%")
    ax.legend(loc="lower left")
    ax.set_title("Más simulaciones no arreglan un mal modelo", pad=26)
    style.subtitle(ax, f"Con {N_SIMS:,} simulaciones el ruido de Monte Carlo es ~{rel.loc[N_SIMS]:.1%} del VaR; "
                       f"elegir mal el modelo cambia el VaR ~{model_spread:.0%}".replace(",", "."))
    style.save(fig, FIG / "08_error_montecarlo.png")


def fig_robustness(rob: pd.DataFrame) -> None:
    rate_cols = [c for c in rob.columns if not c.endswith(" CC")]
    vals = rob[rate_cols].astype(float).to_numpy()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("div", style.DIVERGING[::-1])
    norm = plt.matplotlib.colors.TwoSlopeNorm(vcenter=0.01, vmin=0.0, vmax=0.025)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(rate_cols)), rate_cols)
    ax.set_yticks(range(len(rob)), rob.index)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, model in enumerate(rob.index):
        for j, c in enumerate(rate_cols):
            ok = rob.loc[model, c + " CC"] > 0.05
            v = vals[i, j]
            ax.text(j, i, f"{v:.2%}{'  ✓' if ok else ''}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v - 0.01) > 0.009 else style.INK, fontweight="bold" if ok else "normal")
    ax.set_title("¿Las conclusiones dependen de los supuestos?", pad=26)
    style.subtitle(ax, "Tasa de excepciones del VaR 99% (objetivo 1%). Rojo: subestima el riesgo; azul: lo "
                       "sobreestima. ✓ = aprueba cobertura condicional al 5%")
    style.save(fig, FIG / "09_robustez.png")


def fig_ten_day(td: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    rates = td["Tasa"].astype(float)
    colors = [style.ORANGE, style.VIOLET, style.BLUE]
    bars = ax.barh(td.index, rates, color=colors, height=0.5)
    ax.axvline(0.01, color=style.INK, lw=1.2, ls="--")
    for b, (_, r) in zip(bars, td.iterrows()):
        ax.text(b.get_width() + 0.0005, b.get_y() + b.get_height() / 2,
                f"{int(r['Excepciones'])} de {int(r['Ventanas'])} ventanas   Kupiec p = {float(r['Kupiec p']):.2f}",
                va="center", fontsize=9.5, color=style.INK_2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
    ax.set_xlim(0, max(rates) * 2.4 + 0.002)
    ax.set_title("VaR a 10 días (horizonte de Basilea): ¿alcanza con multiplicar por raíz de 10?", pad=26)
    style.subtitle(ax, "Ventanas de 10 días no superpuestas. Objetivo: 1% de excepciones. Pocas ventanas: "
                       "baja potencia estadística")
    style.save(fig, FIG / "10_horizonte_10_dias.png")


def stress_table(rets: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame | None:
    try:
        scen = escenarios_hipoteticos()
    except NotImplementedError:
        print("escenarios.py: escenarios hipotéticos sin definir (TODO), se omite la sección de estrés")
        return None
    last = fc.iloc[-1]
    rows = {name: {"Pérdida de la cartera": -sum(W[i] * shocks[t] for i, t in enumerate(TICKERS))}
            for name, shocks in scen.items()}
    rows["Repetición PASO 2019 (histórico)"] = {"Pérdida de la cartera": -(rets.loc[PASO].to_numpy() @ W)}
    out = pd.DataFrame(rows).T
    out["x VaR 99% actual (EVT)"] = out["Pérdida de la cartera"] / last["evt_var99"]
    out["x ES 97,5% actual (EVT)"] = out["Pérdida de la cartera"] / last["evt_es975"]
    return out


# ============================================================================ main

def pct(df: pd.DataFrame, cols: list[str], digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda v: f"{float(v):.{digits}%}")
    return out


def main() -> None:
    style.apply()
    prices = load_prices(TICKERS)
    rets = prices.pct_change().dropna()
    spy = load_prices(["SPY"], start=str(rets.index[0].date())).pct_change().dropna()
    print(f"Muestra: {rets.index[0].date()} a {rets.index[-1].date()} ({len(rets)} ruedas)")

    facts = stylized_facts(rets)
    corr = avg_pairwise_corr(rets)
    fan = fig_fan(rets)
    corr_stats = fig_stylized(rets, corr)
    fig_tails(rets)

    fc = rolling_forecasts(rets, W, window=WINDOW, lam=LAMBDA, n_sims=N_SIMS)
    ev = evaluate(fc)
    fig_backtest(fc)
    fig_scorecard(ev)
    fig_basel(fc)
    events = fig_events(fc)

    conv = mc_convergence(rets)
    last = fc.iloc[-1]
    v_today = np.array([last[f"{m}_var99"] for m in MODELS])
    spread = (v_today.max() - v_today.min()) / np.median(v_today)
    fig_mc_error(conv, spread)

    td = ten_day_backtest(rets, fc)
    fig_ten_day(td)
    rob = robustness(rets, spy)
    fig_robustness(rob)
    stress = stress_table(rets, fc)

    fallbacks = int(fc.get("garch_fallback", pd.Series(dtype=float)).fillna(0).sum())
    ev_out = ev.drop(columns=["key"])
    ev_out = pct(ev_out, ["Tasa 99%", "Tasa 97,5%", "Días en zona roja", "VaR 99% medio"])
    for c in ["Kupiec p", "Christoffersen p", "Cob. condicional p", "Z2 (ES 97,5%)", "Pérdida cuantílica"]:
        ev_out[c] = ev_out[c].map(lambda v: f"{float(v):.3f}")
    facts_out = facts.copy()
    facts_out["Vol. anual"] = facts_out["Vol. anual"].map(lambda v: f"{v:.0%}")
    facts_out["Peor día"] = facts_out["Peor día"].map(lambda v: f"{v:.1%}")
    for c in ["Asimetría", "Curtosis exceso"]:
        facts_out[c] = facts_out[c].map(lambda v: f"{v:.1f}")
    facts_out["Jarque-Bera p"] = facts_out["Jarque-Bera p"].map(lambda v: f"{v:.0e}")
    rob_out = rob[[c for c in rob.columns if not c.endswith(" CC")]].map(lambda v: f"{float(v):.2%}")
    td_out = pct(td, ["Tasa", "VaR 10d medio"])
    td_out["Kupiec p"] = td_out["Kupiec p"].map(lambda v: f"{float(v):.3f}")

    sections = [
        f"# Resultados (generado por estudio/run.py)\n\nMuestra: {rets.index[0].date()} a {rets.index[-1].date()}. "
        f"Backtest: {fc.index[0].date()} a {fc.index[-1].date()} ({len(fc)} ruedas). "
        f"Re-ajustes GARCH descartados por degenerados: {fallbacks}.",
        f"## Hechos estilizados\n\n{facts_out.to_markdown()}\n\nCorrelación promedio entre ADRs: "
        f"{corr_stats['corr_alta_vol']:.2f} en el 20% de ruedas más volátiles vs {corr_stats['corr_calma']:.2f} resto.",
        f"## Backtest 1 día\n\n{ev_out.to_markdown()}",
        f"## Peores 6 días: pérdida real vs VaR 99% de la noche anterior\n\n{events.map(lambda v: f'{v:.1%}').to_markdown()}",
        f"## Error de Monte Carlo (MC t-Student, último día)\n\n{conv.map(lambda v: f'{v:.4f}').to_markdown()}\n\n"
        f"Desacuerdo entre modelos el último día: {spread:.1%} del VaR mediano.",
        f"## Horizonte 10 días\n\n{td_out.to_markdown()}",
        f"## Robustez (tasa de excepciones VaR 99%)\n\n{rob_out.to_markdown()}",
        f"## Horizonte 1 año (abanico)\n\n" + "\n".join(f"- {k}: {v:.1%}" for k, v in fan.items()),
    ]
    if stress is not None:
        sections.append(f"## Escenarios de estrés\n\n{stress.round(2).to_markdown()}")
    (HERE / "results.md").write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
