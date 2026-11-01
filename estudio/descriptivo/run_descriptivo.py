"""Capítulo de análisis descriptivo del estudio de VaR sobre ADRs argentinos.

Describe cada variable del estudio (precios, retornos, cartera, S&P 500 de control,
volatilidad y VaR estimado) y deja preparados los tests que necesita el modelo
econométrico posterior (estacionariedad, autocorrelación y efectos ARCH).

    uv run python estudio/descriptivo/run_descriptivo.py

Genera las figuras en estudio/descriptivo/figuras/, las tablas en estudio/descriptivo/tablas/
(CSV) y estudio/descriptivo/tablas.md, que es la fuente de los números de ANALISIS_DESCRIPTIVO.md.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import MODELS, rolling_forecasts  # noqa: E402
from run import COLORS, LAMBDA, N_SIMS, SEED, TICKERS, W, WINDOW  # noqa: E402

from quantkit import style  # noqa: E402
from quantkit.data import load_prices  # noqa: E402
from quantkit.metrics import TRADING_DAYS, cagr, max_drawdown  # noqa: E402
from quantkit.risk import ewma_vol_path  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "figuras"
TAB = HERE / "tablas"

SECTOR = {"GGAL": "Bancos", "BMA": "Bancos", "BBAR": "Bancos", "YPF": "Petróleo y gas",
          "TGS": "Petróleo y gas", "EDN": "Electricidad", "TEO": "Telecomunicaciones",
          "CRESY": "Agro e inmobiliario"}
NAMES = {"GGAL": "Grupo Fin. Galicia", "BMA": "Banco Macro", "BBAR": "BBVA Argentina",
         "YPF": "YPF", "TGS": "Transp. Gas del Sur", "EDN": "Edenor", "TEO": "Telecom Argentina",
         "CRESY": "Cresud"}
SECTOR_COLORS = {"Bancos": style.BLUE, "Petróleo y gas": style.ORANGE, "Electricidad": style.AQUA,
                 "Telecomunicaciones": style.YELLOW, "Agro e inmobiliario": style.MAGENTA}
# Los dos eventos que pide el estudio van en rojo; el resto, en gris, como contexto.
EVENTS = [
    ("2008-10-21", "1", "Estatización de las AFJP / crisis global", False),
    ("2019-08-12", "2", "PASO 2019", True),
    ("2020-03-09", "3", "COVID-19", False),
    ("2023-11-20", "4", "Balotaje 2023", False),
    ("2025-09-08", "5", "Elección bonaerense 2025", True),
    ("2025-10-27", "6", "Legislativas 2025", False),
]
PERIODS = [("2007-04-27", "2015-12-09", "Fernández de Kirchner"), ("2015-12-10", "2019-12-09", "Macri"),
           ("2019-12-10", "2023-12-09", "Fernández"), ("2023-12-10", "2026-12-31", "Milei")]
FREQ_EDGES = [-np.inf, -0.10, -0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, np.inf]
SOURCE = "Fuente: Yahoo Finance, precios de cierre ajustados en USD, congelados en data/. Elaboración propia."


# ============================================================================ formato

def num(v: float, d: int = 2) -> str:
    """Número en formato argentino: coma decimal, punto de miles."""
    if pd.isna(v):
        return "–"
    s = f"{v:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float, d: int = 2) -> str:
    return "–" if pd.isna(v) else num(100 * v, d) + "%"


def pval(v: float) -> str:
    return "< 0,001" if v < 0.001 else num(v, 3)


def interval_label(lo: float, hi: float) -> str:
    if np.isinf(lo):
        return f"menor a {pct(hi, 0)}"
    if np.isinf(hi):
        return f"{pct(lo, 0)} o más"
    return f"[{pct(lo, 0)}, {pct(hi, 0)})"


def mark_events(ax: plt.Axes, numbers: bool = True) -> None:
    for d, n, _, key in EVENTS:
        x = pd.Timestamp(d)
        ax.axvline(x, color=style.RED if key else style.MUTED, lw=0.9 if key else 0.6,
                   ls="-" if key else "--", alpha=0.8, zorder=1)
        if numbers:
            dx = {"5": -3, "6": 3}.get(n, 0)  # las dos elecciones de 2025 están a siete semanas
            ax.annotate(n, xy=(x, 1), xycoords=("data", "axes fraction"), xytext=(dx, 1),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                        color=style.RED if key else style.INK_2)


EVENTS_NOTE = "\nEventos: " + "; ".join(f"({n}) {lab} {pd.Timestamp(d):%d/%m/%Y}" + ("\n" if n == "3" else "")
                                       for d, n, lab, _ in EVENTS).replace("\n; ", ";\n")


# ============================================================================ datos

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precios ajustados de los 8 ADRs y del SPY en un panel con el calendario de NYSE.

    Se filtra por días en que operó el SPY y se completa hacia adelante: así un feriado de
    un solo listado no borra el retorno del día siguiente en todo el panel. El caché del SPY
    empieza una rueda después que el de los ADRs (26/04/2007); esa primera fila se conserva
    para que los retornos de los ADRs sean exactamente los de estudio/run.py."""
    adr = load_prices(TICKERS)
    spy = load_prices(["SPY"], start="2007-04-27")
    p = adr.join(spy, how="outer")
    p = p[p["SPY"].notna() | (p.index < spy.index[0])].ffill()
    return p, p.pct_change(fill_method=None).iloc[1:]


def series_of_interest(rets: pd.DataFrame) -> dict[str, pd.Series]:
    port = (rets[TICKERS] @ W).rename("Cartera")
    return {**{t: rets[t] for t in TICKERS}, "Cartera": port, "S&P 500 (SPY)": rets["SPY"].dropna()}


# ============================================================================ tablas

def describe_daily(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, x in series.items():
        jb = stats.jarque_bera(x)
        q = x.quantile([0.01, 0.05, 0.95, 0.99])
        rows[name] = {"N": len(x), "Media": x.mean(), "Mediana": x.median(), "Desvío": x.std(),
                      "Mín.": x.min(), "Máx.": x.max(), "P1": q[0.01], "P5": q[0.05], "P95": q[0.95],
                      "P99": q[0.99], "Asimetría": stats.skew(x), "Curtosis exceso": stats.kurtosis(x),
                      "Jarque-Bera": jb.statistic, "JB p": jb.pvalue}
    return pd.DataFrame(rows).T


def describe_annual(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, x in series.items():
        rows[name] = {"Media aritmética anual": x.mean() * TRADING_DAYS,
                      "Media log anual": np.log1p(x).mean() * TRADING_DAYS,
                      "Crecimiento anual compuesto": cagr(x),
                      "Volatilidad anual": x.std() * np.sqrt(TRADING_DAYS),
                      "Máxima caída acumulada": max_drawdown(x)}
    return pd.DataFrame(rows).T


def frequency_table(x: pd.Series) -> pd.DataFrame:
    cut = pd.cut(x, FREQ_EDGES, right=False)
    counts = cut.value_counts(sort=False)
    mu, sd = x.mean(), x.std()
    normal = np.diff(stats.norm.cdf(FREQ_EDGES, mu, sd))
    out = pd.DataFrame({"Intervalo": [interval_label(i.left, i.right) for i in counts.index],
                        "Frec. absoluta": counts.to_numpy(),
                        "Frec. relativa": counts.to_numpy() / len(x)})
    out["Frec. acumulada"] = out["Frec. absoluta"].cumsum()
    out["Frec. rel. acumulada"] = out["Frec. relativa"].cumsum()
    out["Esperada si fuera normal"] = normal
    return out


def frequency_by_asset(series: dict[str, pd.Series]) -> pd.DataFrame:
    edges = [-np.inf, -0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, np.inf]
    labels = [interval_label(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    rows = {}
    for name, x in series.items():
        rel = pd.cut(x, edges, right=False).value_counts(sort=False).to_numpy() / len(x)
        z = (x - x.mean()) / x.std()
        rows[name] = dict(zip(labels, rel)) | {"Días a más de 3 desvíos": (z.abs() > 3).mean()}
    out = pd.DataFrame(rows).T
    return out


def tail_counts(series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, x in series.items():
        z = (x - x.mean()) / x.std()
        rows[name] = {f"Más de {k} desvíos": int((z.abs() > k).sum()) for k in (3, 4, 5)}
    out = pd.DataFrame(rows).T.astype(float)
    n = len(next(iter(series.values())))
    out.loc["Esperado si fuera normal"] = [2 * stats.norm.sf(k) * n for k in (3, 4, 5)]
    return out


def by_period(port: pd.Series) -> pd.DataFrame:
    rows = {}
    for a, b, name in PERIODS:
        x = port.loc[a:b]
        rows[f"{name} ({pd.Timestamp(a).year}-{min(pd.Timestamp(b).year, port.index[-1].year)})"] = {
            "N": len(x), "Media anual": x.mean() * TRADING_DAYS, "Volatilidad anual": x.std() * np.sqrt(TRADING_DAYS),
            "Peor día": x.min(), "Mejor día": x.max(), "Curtosis exceso": stats.kurtosis(x)}
    return pd.DataFrame(rows).T


def sector_table(rets: pd.DataFrame) -> pd.DataFrame:
    """Peso y contribución al riesgo (descomposición de Euler de la varianza) por sector."""
    cov = rets[TICKERS].cov().to_numpy()
    rc = W * (cov @ W) / (W @ cov @ W)
    df = pd.DataFrame({"Sector": [SECTOR[t] for t in TICKERS], "Peso": W, "Contribución al riesgo": rc},
                      index=TICKERS)
    return df.groupby("Sector", sort=False)[["Peso", "Contribución al riesgo"]].sum()


def stationarity(prices: pd.DataFrame, series: dict[str, pd.Series], port_index: pd.Series) -> pd.DataFrame:
    """ADF (H0: raíz unitaria) y KPSS (H0: estacionaria) para log-precios y retornos."""
    levels = {**{t: np.log(prices[t]) for t in TICKERS}, "Cartera": np.log(port_index),
              "S&P 500 (SPY)": np.log(prices["SPY"])}
    rows = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS avisa cuando el p-valor queda fuera de su tabla
        for name in series:
            lp, r = levels[name].dropna(), series[name]  # el SPY tiene una rueda menos
            adf_p = adfuller(lp, regression="ct", autolag="AIC")
            adf_r = adfuller(r, regression="c", autolag="AIC")
            kpss_p = kpss(lp, regression="ct", nlags="auto")
            kpss_r = kpss(r, regression="c", nlags="auto")
            rows[name] = {"ADF log-precio": adf_p[0], "ADF log-precio p": adf_p[1],
                          "KPSS log-precio": kpss_p[0], "KPSS log-precio p": kpss_p[1],
                          "ADF retorno": adf_r[0], "ADF retorno p": adf_r[1],
                          "KPSS retorno": kpss_r[0], "KPSS retorno p": kpss_r[1]}
    return pd.DataFrame(rows).T


def dependence(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Ljung-Box en retornos y retornos al cuadrado, y ARCH-LM de Engle."""
    rows = {}
    for name, x in series.items():
        lb_r = acorr_ljungbox(x, lags=[10], return_df=True).iloc[0]
        lb_r2 = acorr_ljungbox(x ** 2, lags=[10], return_df=True).iloc[0]
        lb_abs = acorr_ljungbox(x.abs(), lags=[10], return_df=True).iloc[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # aviso de API de statsmodels 0.16
            lm, lm_p, _, _ = het_arch(x - x.mean(), nlags=10)
        rows[name] = {"ACF(1) retorno": acf(x, nlags=1)[1], "Q(10) retorno": lb_r["lb_stat"],
                      "Q(10) p": lb_r["lb_pvalue"], "ACF(1) retorno²": acf(x ** 2, nlags=1)[1],
                      "Q(10) retorno²": lb_r2["lb_stat"], "Q²(10) p": lb_r2["lb_pvalue"],
                      "ACF(1) valor abs.": acf(x.abs(), nlags=1)[1], "Q(10) valor abs.": lb_abs["lb_stat"],
                      "ARCH-LM(10)": lm, "ARCH-LM p": lm_p}
    return pd.DataFrame(rows).T


def var_summary(fc: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for m, name in MODELS.items():
        v = fc[f"{m}_var99"]
        hits = fc["ret"] < -v
        rows[name] = {"Media": v.mean(), "Mediana": v.median(), "Desvío": v.std(), "Mín.": v.min(),
                      "Máx.": v.max(), "Coef. de variación": v.std() / v.mean(), "Excepciones": int(hits.sum())}
    return pd.DataFrame(rows).T


def exceptions_by_year(fc: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for m in ["mc_norm", "mc_t_ewma", "evt"]:
        hits = (fc["ret"] < -fc[f"{m}_var99"]).astype(int)
        out[MODELS[m]] = hits.groupby(fc.index.year).sum()
    df = pd.DataFrame(out)
    df.insert(0, "Ruedas", fc["ret"].groupby(fc.index.year).size())
    df.insert(1, "Esperadas (1%)", df["Ruedas"] * 0.01)
    df.index.name = "Año"
    return df


# ============================================================================ figuras

def grid(n: int = 9, figsize: tuple[float, float] = (14, 10.5), **kw) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(3, 3, figsize=figsize, **kw)
    return fig, axes.ravel()[:n]


def header(fig: plt.Figure, title: str, sub: str, top: float = 0.90) -> None:
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.text(0.01, 0.995, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.01, 0.995 - (1 - top) * 0.5, sub, fontsize=10.5, color=style.INK_2, va="top")


def fig_prices(prices: pd.DataFrame, port_index: pd.Series) -> None:
    base = prices / prices.bfill().iloc[0] * 100
    spy = base["SPY"]
    fig, axes = grid(sharex=True)
    for ax, t in zip(axes, TICKERS + ["Cartera"]):
        y = port_index / port_index.iloc[0] * 100 if t == "Cartera" else base[t]
        ax.plot(spy, color=style.BASELINE, lw=1.1, label="S&P 500 (SPY)")
        ax.plot(y, color=style.INK if t == "Cartera" else SECTOR_COLORS[SECTOR[t]], lw=1.3, label=t)
        ax.axhline(100, color=style.BASELINE, lw=0.8)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mtick.NullFormatter())
        mark_events(ax)
        label = "Cartera equiponderada" if t == "Cartera" else f"{t} · {NAMES[t]}"
        ax.set_title(f"{label}\n", fontsize=11.5, pad=4)
        ax.text(0.02, 0.04, f"Final: {y.iloc[-1]:,.0f}".replace(",", "."), transform=ax.transAxes,
                fontsize=9, color=style.INK_2)
    axes[0].legend(loc="upper right", fontsize=8.5)
    header(fig, "Precios en base 100 (26/04/2007 = 100), escala logarítmica",
           "En gris, el S&P 500. Color por sector." + EVENTS_NOTE, top=0.885)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d01_precios_base100.png")


def fig_returns(series: dict[str, pd.Series]) -> None:
    fig, axes = grid(sharex=True, sharey=True)
    for ax, t in zip(axes, TICKERS + ["Cartera"]):
        x = series[t]
        ax.plot(x, color=style.INK if t == "Cartera" else SECTOR_COLORS[SECTOR[t]], lw=0.45)
        ax.axhline(0, color=style.BASELINE, lw=0.8)
        mark_events(ax)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        ax.set_title(f"{'Cartera equiponderada' if t == 'Cartera' else t}\n", fontsize=11.5, pad=4)
        i = x.idxmin()
        ax.annotate(f"{x.min():.0%}", xy=(i, x.min()), xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=style.INK_2, va="center")
    header(fig, "Retornos diarios simples en USD, 2007-2026",
           "La volatilidad no es constante: se agrupa en crisis y elecciones." + EVENTS_NOTE, top=0.885)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d02_retornos.png")


def fig_hist(series: dict[str, pd.Series], tfits: dict[str, tuple]) -> None:
    fig, axes = grid()
    lim = 0.12
    for ax, t in zip(axes, TICKERS + ["Cartera"]):
        x = series[t]
        bins = np.linspace(-lim, lim, 81)
        ax.hist(x.clip(-lim, lim), bins=bins, density=True, color=style.BLUE, alpha=0.25, lw=0)
        ax.hist(x.clip(-lim, lim), bins=bins, density=True, histtype="step", color=style.BLUE, lw=1)
        xs = np.linspace(-lim, lim, 500)
        ax.plot(xs, stats.norm.pdf(xs, x.mean(), x.std()), color=style.ORANGE, lw=1.6, label="Normal")
        nu, loc, sc = tfits[t]
        ax.plot(xs, stats.t.pdf(xs, nu, loc, sc), color=style.INK, lw=1.3, ls="--", label="t-Student")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        ax.set_title(f"{'Cartera' if t == 'Cartera' else t}   (ν = {num(nu, 1)})", fontsize=11.5)
        ax.set_yticks([])
    axes[0].legend(loc="upper left", fontsize=8.5)
    header(fig, "Histogramas de los retornos diarios con la normal y la t-Student superpuestas",
           "Ambas ajustadas por máxima verosimilitud a toda la muestra. Pico más alto y colas más largas que la normal. "
           "Los días fuera de ±12% se acumulan en los extremos.", top=0.91)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d03_histogramas.png")


def fig_qq(series: dict[str, pd.Series], tfits: dict[str, tuple]) -> None:
    fig, axes = grid()
    for ax, t in zip(axes, TICKERS + ["Cartera"]):
        x = series[t].to_numpy()
        z = np.sort((x - x.mean()) / x.std())
        pr = (np.arange(1, len(z) + 1) - 0.5) / len(z)
        theo = stats.norm.ppf(pr)
        nu, loc, sc = tfits[t]
        tq = (stats.t.ppf(pr, nu, loc, sc) - x.mean()) / x.std()
        ax.scatter(theo, z, s=3, color=style.BLUE, lw=0, label="Observado")
        ax.plot(theo, tq, color=style.INK, lw=1.2, ls="--", label="t-Student ajustada")
        ax.plot([-4, 4], [-4, 4], color=style.ORANGE, lw=1.2, label="Normal")
        ax.set_ylim(max(z.min(), -16) - 0.5, max(z.max(), 4) + 0.5)
        ax.set_title("Cartera" if t == "Cartera" else t, fontsize=11.5)
        if z.min() < -16:
            ax.annotate(f"mín. = {z.min():.0f} desvíos (fuera de escala)", xy=(0.02, 0.03),
                        xycoords="axes fraction", fontsize=8.5, color=style.INK_2)
    axes[0].legend(loc="upper left", fontsize=8.5)
    for ax in axes[6:]:
        ax.set_xlabel("Cuantil teórico normal")
    for ax in axes[::3]:
        ax.set_ylabel("Cuantil observado (desvíos)")
    header(fig, "Gráficos QQ contra la normal",
           "Si los retornos fueran normales, los puntos seguirían la diagonal naranja. Se curvan en las dos colas "
           "y la t-Student (línea punteada) las sigue mucho mejor.", top=0.91)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d04_qq.png")


def fig_freq(freq: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    y = np.arange(len(freq))
    ax.bar(y - 0.2, freq["Frec. relativa"], width=0.4, color=style.BLUE, label="Observada")
    ax.bar(y + 0.2, freq["Esperada si fuera normal"], width=0.4, color=style.ORANGE, label="Esperada si fuera normal")
    ax.set_xticks(y, freq["Intervalo"], rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Frecuencia relativa")
    ax.legend(loc="upper left")
    ax.set_title("Frecuencia relativa por intervalo", fontsize=12.5)
    ax = axes[1]
    ax.step(y, freq["Frec. rel. acumulada"], where="mid", color=style.BLUE, lw=2, label="Observada")
    ax.step(y, freq["Esperada si fuera normal"].cumsum(), where="mid", color=style.ORANGE, lw=1.6, ls="--",
            label="Normal")
    ax.set_xticks(y, freq["Intervalo"], rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Frecuencia relativa acumulada")
    ax.legend(loc="upper left")
    ax.set_title("Ojiva (distribución acumulada)", fontsize=12.5)
    n = freq["Frec. absoluta"].sum()
    calm = freq["Frec. relativa"].iloc[[6, 7]].sum(), freq["Esperada si fuera normal"].iloc[[6, 7]].sum()
    tails = freq["Frec. absoluta"].iloc[[0, -1]].sum(), freq["Esperada si fuera normal"].iloc[[0, -1]].sum() * n
    header(fig, "Distribución de frecuencias del retorno diario de la cartera",
           f"Frente a una normal con igual media y desvío: más días tranquilos (entre −1% y +1%: {pct(calm[0], 0)} vs "
           f"{pct(calm[1], 0)}), menos días intermedios (2% a 5%) y muchos más días fuera de ±10% "
           f"({tails[0]:.0f} vs {num(tails[1], 1)} esperados).", top=0.86)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d05_frecuencias_cartera.png")


def fig_box(series: dict[str, pd.Series]) -> None:
    names = list(series)
    data = [series[n].to_numpy() for n in names]
    colors = [SECTOR_COLORS[SECTOR[n]] if n in SECTOR else (style.INK if n == "Cartera" else style.MUTED) for n in names]
    labels = [n.replace(" (SPY)", "\n(SPY)") for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, zoom in zip(axes, [False, True]):
        bp = ax.boxplot(data, tick_labels=labels, widths=0.6, patch_artist=True, whis=1.5,
                        medianprops=dict(color=style.INK, lw=1.4),
                        flierprops=dict(marker="o", ms=2, alpha=0.35, mec="none", mfc=style.INK_2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set(facecolor=c, alpha=0.35, edgecolor=c)
        ax.axhline(0, color=style.BASELINE, lw=0.8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", visible=False)
        if zoom:
            ax.set_ylim(-0.08, 0.08)
            ax.set_title("Zoom en ±8%: la caja (50% central de los días)", fontsize=12.5)
        else:
            ax.set_title("Escala completa: los atípicos", fontsize=12.5)
    out = {}
    for n in names:
        q1, q3 = series[n].quantile([0.25, 0.75])
        out[n] = ((series[n] < q1 - 1.5 * (q3 - q1)) | (series[n] > q3 + 1.5 * (q3 - q1))).mean()
    header(fig, "Diagramas de caja de los retornos diarios",
           "La caja (50% central de los días) es angosta, pero entre "
           f"{pct(min(out.values()), 1)} y {pct(max(out.values()), 1)} de los días quedan fuera de los bigotes: "
           "con una normal serían 0,7%.", top=0.86)
    style.source(fig, SOURCE + " Bigotes: 1,5 veces el rango intercuartílico.")
    style.save(fig, FIG / "d06_boxplots.png")


def fig_bars(daily: pd.DataFrame, annual: pd.DataFrame) -> None:
    metrics = [("Volatilidad anual", annual["Volatilidad anual"], lambda v: pct(v, 0)),
               ("Peor día", daily["Mín."], lambda v: pct(v, 0)),
               ("Curtosis en exceso (normal = 0)", daily["Curtosis exceso"], lambda v: num(v, 0))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (title, s, fmt) in zip(axes, metrics):
        s = s.astype(float).sort_values(ascending=title != "Peor día")
        colors = [SECTOR_COLORS[SECTOR[n]] if n in SECTOR else (style.INK if n == "Cartera" else style.MUTED)
                  for n in s.index]
        bars = ax.barh(s.index, s.to_numpy(), color=colors, height=0.65)
        for b, v in zip(bars, s):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {fmt(v)} " if v >= 0 else f"{fmt(v)} ",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9, color=style.INK_2)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=12.5)
        if "Curtosis" not in title:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo * 1.25 if lo < 0 else lo, hi * 1.18 if hi > 0 else hi)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SECTOR_COLORS.values()] + \
              [plt.Rectangle((0, 0), 1, 1, color=style.INK), plt.Rectangle((0, 0), 1, 1, color=style.MUTED)]
    fig.legend(handles, list(SECTOR_COLORS) + ["Cartera", "S&P 500"], loc="upper left", ncol=7,
               bbox_to_anchor=(0.005, 0.885), handlelength=1.2)
    header(fig, "Riesgo por activo: volatilidad, peor día y curtosis",
           "Cada ADR argentino es entre 2,4 y 3,2 veces más volátil que el S&P 500. La cartera diversifica la "
           "volatilidad pero no las colas: su curtosis supera a la de cualquier ADR individual.", top=0.80)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d07_barras_riesgo.png")


def fig_sectors(sec: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    colors = [SECTOR_COLORS[s] for s in sec.index]
    ax.pie(sec["Peso"], labels=[f"{s}\n{pct(w, 1)}" for s, w in sec["Peso"].items()], colors=colors,
           startangle=90, counterclock=False, wedgeprops=dict(width=0.45, edgecolor=style.SURFACE, lw=2),
           textprops=dict(fontsize=10, color=style.INK))
    ax.set_title("Composición de la cartera por sector (pesos)", fontsize=12.5)
    ax = axes[1]
    y = np.arange(len(sec))
    ax.barh(y - 0.2, sec["Peso"], height=0.4, color=colors, alpha=0.45, label="Peso en la cartera")
    ax.barh(y + 0.2, sec["Contribución al riesgo"], height=0.4, color=colors, label="Contribución a la varianza")
    for yi, (w, rc) in enumerate(zip(sec["Peso"], sec["Contribución al riesgo"])):
        ax.text(w, yi - 0.2, f" {pct(w, 1)}", va="center", fontsize=9, color=style.INK_2)
        ax.text(rc, yi + 0.2, f" {pct(rc, 1)}", va="center", fontsize=9, color=style.INK)
    ax.set_yticks(y, sec.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlim(0, sec.max().max() * 1.2)
    ax.legend(loc="lower right")
    ax.set_title("Peso vs contribución al riesgo (descomposición de Euler)", fontsize=12.5)
    header(fig, "¿De dónde viene el riesgo de la cartera?",
           "Pesos iguales por activo no son pesos iguales por sector, y tampoco riesgo igual: los bancos son 37,5% "
           "del capital y aportan más que eso a la varianza.", top=0.86)
    style.source(fig, SOURCE + " Contribución de Euler: w_i (Σw)_i / w'Σw, con covarianza de toda la muestra.")
    style.save(fig, FIG / "d08_sectores.png")


def fig_corr(corr: pd.DataFrame) -> None:
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", style.SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=0, vmax=1)
    labels = [c.replace(" (SPY)", "") for c in corr.columns]
    ax.set_xticks(range(len(corr)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), labels)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 else style.INK)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    off = corr.loc[TICKERS, TICKERS].to_numpy()[np.triu_indices(len(TICKERS), 1)]
    ax.set_title("Correlaciones de Pearson entre retornos diarios", pad=26)
    style.subtitle(ax, f"Entre ADRs: promedio {num(off.mean(), 2)} (rango {num(off.min(), 2)} a {num(off.max(), 2)}). "
                       f"Con el S&P 500: {num(corr.loc[TICKERS, 'S&P 500 (SPY)'].mean(), 2)} en promedio")
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d09_correlaciones.png")


def fig_rolling_vol(rets: pd.DataFrame, port: pd.Series) -> None:
    ann = np.sqrt(TRADING_DAYS)
    roll = rets[TICKERS].rolling(63).std() * ann
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(port.rolling(63).std() * ann, color=style.INK, lw=1.4, label="Cartera, desvío móvil 63 ruedas")
    ax.plot(pd.Series(ewma_vol_path(port.to_numpy(), LAMBDA) * ann, index=port.index), color=style.BLUE, lw=0.9,
            alpha=0.85, label=f"Cartera, EWMA λ = {num(LAMBDA, 2)} (la del estudio)")
    ax.plot(rets["SPY"].rolling(63).std() * ann, color=style.MUTED, lw=1.2, label="S&P 500, desvío móvil 63 ruedas")
    mark_events(ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Volatilidad anualizada")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Volatilidad de la cartera vs S&P 500", fontsize=12.5)
    ax = axes[1]
    ax.fill_between(roll.index, roll.min(axis=1), roll.max(axis=1), color=style.BLUE, alpha=0.18, lw=0,
                    label="Rango entre los 8 ADRs")
    ax.plot(roll.median(axis=1), color=style.BLUE, lw=1.3, label="Mediana de los 8 ADRs")
    ax.plot(port.rolling(63).std() * ann, color=style.INK, lw=1.0, label="Cartera")
    mark_events(ax, numbers=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Volatilidad anualizada (63 ruedas)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Volatilidad móvil por activo", fontsize=12.5)
    header(fig, "Volatilidad móvil: regímenes de calma y de crisis",
           "La volatilidad cambia de nivel y persiste: es el hecho que justifica un modelo GARCH." + EVENTS_NOTE,
           top=0.87)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d10_volatilidad_movil.png")


def fig_acf(series: dict[str, pd.Series]) -> None:
    lags = np.arange(1, 41)
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6), sharex=True)
    for row, name in enumerate(["Cartera", "S&P 500 (SPY)"]):
        x = series[name].to_numpy()
        band = 1.96 / np.sqrt(len(x))
        for col, (lab, y) in enumerate([("Retornos", x), ("Retornos al cuadrado", x ** 2),
                                        ("Retornos en valor absoluto", np.abs(x))]):
            ax = axes[row, col]
            a = acf(y, nlags=40)[1:]
            ax.bar(lags, a, color=style.INK if name == "Cartera" else style.MUTED, width=0.7)
            ax.axhspan(-band, band, color=style.BLUE, alpha=0.12, lw=0)
            ax.axhline(0, color=style.BASELINE, lw=0.8)
            ax.set_title(f"{name.replace(' (SPY)', '')}: {lab.lower()}", fontsize=12)
            ax.set_ylim(-0.15, 0.45)
            if row == 1:
                ax.set_xlabel("Rezago (ruedas)")
            if col == 0:
                ax.set_ylabel("Autocorrelación")
    header(fig, "Funciones de autocorrelación: los retornos casi no tienen memoria, su magnitud sí",
           "Banda celeste: ±1,96/√N (95% bajo ruido blanco). En la cartera, el −47% de las PASO domina los retornos "
           "al cuadrado y aplana su ACF; el valor absoluto, menos sensible a un solo día, muestra la persistencia.",
           top=0.88)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d11_acf.png")


def fig_var(fc: pd.DataFrame) -> None:
    keys = list(MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    bp = ax.boxplot([fc[f"{m}_var99"] for m in keys], orientation="horizontal", tick_labels=[MODELS[m] for m in keys],
                    widths=0.6, patch_artist=True, medianprops=dict(color=style.INK, lw=1.4),
                    flierprops=dict(marker="o", ms=2, alpha=0.35, mec="none", mfc=style.INK_2))
    for patch, m in zip(bp["boxes"], keys):
        patch.set(facecolor=COLORS[m], alpha=0.4, edgecolor=COLORS[m])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("VaR 99% a 1 día (pérdida)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Distribución del VaR 99% pronosticado", fontsize=12.5)
    ax = axes[1]
    for m in ["mc_norm", "mc_t_ewma", "evt"]:
        ax.plot(fc[f"{m}_var99"], color=COLORS[m], lw=1.0, label=MODELS[m])
    mark_events(ax)
    ax.set_xlim(fc.index[0], fc.index[-1] + pd.Timedelta(days=60))  # el evento 1 es anterior al backtest
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylabel("VaR 99% a 1 día")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Tres modelos a lo largo del tiempo", fontsize=12.5)
    header(fig, "La variable que se evalúa: el VaR 99% estimado por cada modelo, 2009-2026",
           "Los modelos con volatilidad adaptativa (EWMA, GARCH) mueven mucho el VaR; los de ventana fija (normal "
           "muestral, histórica) lo mueven poco y tarde.", top=0.86)
    style.source(fig, "Pronósticos fuera de muestra con las 500 ruedas previas (estudio/models.py). " + SOURCE)
    style.save(fig, FIG / "d12_var_estimado.png")


def fig_periods(per: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, col, fmt in [(axes[0], "Volatilidad anual", lambda v: pct(v, 0)),
                         (axes[1], "Curtosis exceso", lambda v: num(v, 0))]:
        s = per[col].astype(float)
        bars = ax.barh(s.index, s, color=style.BLUE, height=0.6)
        for b, v in zip(bars, s):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {fmt(v)}", va="center", fontsize=9.5,
                    color=style.INK_2)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, s.max() * 1.2)
        if col.startswith("Vol"):
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        ax.set_title(col.replace("exceso", "en exceso"), fontsize=12.5)
    header(fig, "La cartera por gobierno: el riesgo cambia de régimen",
           "Mandatos presidenciales como partición del calendario, no como hipótesis causal.", top=0.84)
    style.source(fig, SOURCE)
    style.save(fig, FIG / "d13_por_periodo.png")


# ============================================================================ main

def md(df: pd.DataFrame, fmt: dict[str, callable], index: str = "") -> str:
    out = df.copy().astype(object)
    for c in out.columns:
        f = fmt.get(c, fmt.get("*"))
        if f is not None:
            out[c] = [f(v) for v in df[c]]
    out.index.name = index or df.index.name
    return out.to_markdown(disable_numparse=True)


def main() -> None:
    style.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    prices, rets = load()
    series = series_of_interest(rets)
    port = series["Cartera"]
    port_index = pd.concat([pd.Series([1.0], index=[prices.index[0]]), (1 + port).cumprod()])
    print(f"Muestra de retornos: {rets.index[0].date()} a {rets.index[-1].date()} ({len(rets)} ruedas)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfits = {n: stats.t.fit(x) for n, x in series.items()}

    daily = describe_daily(series)
    annual = describe_annual(series)
    freq = frequency_table(port)
    freq_assets = frequency_by_asset(series)
    tails = tail_counts(series)
    per = by_period(port)
    sec = sector_table(rets)
    corr = pd.DataFrame(series).corr()
    stat = stationarity(prices, series, port_index)
    dep = dependence(series)
    t_tbl = pd.DataFrame({n: {"ν (grados de libertad)": f[0], "Escala": f[2]} for n, f in tfits.items()}).T

    print("Calculando el VaR rodante de los 7 modelos (~15 s)...")
    fc = rolling_forecasts(rets[TICKERS], W, window=WINDOW, lam=LAMBDA, n_sims=N_SIMS, seed=SEED)
    vs = var_summary(fc)
    exc = exceptions_by_year(fc)
    ewma = pd.Series(ewma_vol_path(port.to_numpy(), LAMBDA), index=port.index).iloc[20:]
    vol_tbl = describe_daily({"Vol. EWMA diaria de la cartera": ewma})[
        ["N", "Media", "Mediana", "Desvío", "Mín.", "Máx.", "P5", "P95"]]

    fig_prices(prices, port_index)
    fig_returns(series)
    fig_hist(series, tfits)
    fig_qq(series, tfits)
    fig_freq(freq)
    fig_box(series)
    fig_bars(daily, annual)
    fig_sectors(sec)
    fig_corr(corr)
    fig_rolling_vol(rets, port)
    fig_acf(series)
    fig_var(fc)
    fig_periods(per)

    for name, df in {"descriptiva_diaria": daily, "descriptiva_anual": annual, "frecuencias_cartera": freq,
                     "frecuencias_por_activo": freq_assets, "colas": tails, "por_periodo": per,
                     "sectores": sec, "correlaciones": corr, "estacionariedad": stat, "dependencia": dep,
                     "t_student": t_tbl, "var_estimado": vs, "excepciones_por_anio": exc,
                     "volatilidad_ewma": vol_tbl}.items():
        df.to_csv(TAB / f"{name}.csv")

    p2 = lambda v: pct(v, 2)  # noqa: E731
    sections = [
        f"# Tablas del análisis descriptivo (generado por run_descriptivo.py)\n\n"
        f"Retornos diarios simples en USD, {rets.index[0]:%d/%m/%Y} a {rets.index[-1]:%d/%m/%Y}, "
        f"{len(rets)} ruedas. VaR rodante: {fc.index[0]:%d/%m/%Y} a {fc.index[-1]:%d/%m/%Y} ({len(fc)} ruedas).",
        "## T1. Estadística descriptiva de los retornos diarios\n\n" + md(
            daily, {"N": lambda v: num(v, 0), "Asimetría": lambda v: num(v, 2), "Curtosis exceso": lambda v: num(v, 1),
                    "Jarque-Bera": lambda v: num(v, 0), "JB p": pval, "*": p2}),
        "## T2. Retornos y riesgo anualizados\n\n" + md(annual, {"*": lambda v: pct(v, 1)}),
        "## T3. Distribución de frecuencias del retorno diario de la cartera\n\n" + md(
            freq.set_index("Intervalo"), {"Frec. absoluta": lambda v: num(v, 0), "Frec. acumulada": lambda v: num(v, 0),
                                          "*": p2}),
        "## T4. Frecuencia relativa por intervalo y por activo\n\n" + md(freq_assets, {"*": lambda v: pct(v, 1)}),
        "## T5. Días extremos: observados vs esperados bajo normalidad\n\n" + md(tails, {"*": lambda v: num(v, 0) if v >= 1 or v == 0 else num(v, 3)}),
        "## T6. Parámetros de la t-Student ajustada\n\n" + md(t_tbl, {"Escala": p2, "*": lambda v: num(v, 2)}),
        "## T7. Composición y contribución al riesgo por sector\n\n" + md(sec, {"*": lambda v: pct(v, 1)}),
        "## T8. Matriz de correlaciones\n\n" + md(corr, {"*": lambda v: num(v, 2)}),
        "## T9. La cartera por período de gobierno\n\n" + md(
            per, {"N": lambda v: num(v, 0), "Curtosis exceso": lambda v: num(v, 1), "*": lambda v: pct(v, 1)}),
        "## T10. Volatilidad EWMA de la cartera (diaria)\n\n" + md(vol_tbl, {"N": lambda v: num(v, 0), "*": p2}),
        "## T11. VaR 99% a 1 día estimado por cada modelo\n\n" + md(
            vs, {"Excepciones": lambda v: num(v, 0), "Coef. de variación": lambda v: num(v, 2), "*": p2}),
        "## T12. Excepciones del VaR 99% por año\n\n" + md(
            exc, {"Esperadas (1%)": lambda v: num(v, 1), "*": lambda v: num(v, 0)}),
        "## T13. Estacionariedad: ADF y KPSS\n\nADF: H0 raíz unitaria (con tendencia en log-precios, con constante "
        "en retornos). KPSS: H0 estacionariedad. El p-valor de KPSS está acotado entre 0,01 y 0,10 por la tabla de "
        "valores críticos.\n\n" + md(stat, {**{c: pval for c in stat.columns if c.endswith(" p")},
                                           "*": lambda v: num(v, 2)}),
        "## T14. Autocorrelación y efectos ARCH\n\n" + md(
            dep, {"Q(10) p": pval, "Q²(10) p": pval, "ARCH-LM p": pval, "ACF(1) retorno": lambda v: num(v, 3),
                  "ACF(1) retorno²": lambda v: num(v, 3), "ACF(1) valor abs.": lambda v: num(v, 3),
                  "*": lambda v: num(v, 1)}),
    ]
    (HERE / "tablas.md").write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))
    print(f"\nFiguras en {FIG.relative_to(HERE.parents[1])}, tablas en {TAB.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
