"""Proyecto 2 - Arbitraje ADR vs local: ¿se puede capturar la dispersión del CCL implícito?

Reproduce todas las figuras y tablas del README de esta carpeta:
    uv run python projects/02_arbitraje_ccl/run.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from quantkit import style
from quantkit.argentina import ADRS, ccl_deviation, implied_ccl, load_listings, market_ccl
from quantkit.metrics import format_table, summary_table, yearly_returns
from quantkit.pairs import positions, rolling_zscore

HERE = Path(__file__).parent
FIG = HERE / "figures"

# Fixed before the first run (see METHODOLOGY.md).
START = "2003-01-01"      # after the 2001-02 corralito, when BYMA was closed for days at a time
LOOKBACK = 60             # days for the rolling mean/std of each stock's CCL deviation
ENTRY, EXIT, STOP = 2.0, 0.5, 5.0
LAG = 1                   # trade at the close AFTER the signal day (never on the signal prices)
COSTS = {                 # bps per side on traded notional: (local leg in BYMA, ADR leg in NY)
    "Institucional": (15, 3),
    "Minorista": (60, 10),
}
BASE = "Institucional"
CEPO = [("2011-10-31", "2015-12-16"), ("2019-09-01", "2025-04-14")]  # FX control periods
SPLIT = "2016-01-01"      # sub-period check: does the edge persist in the last decade?


def strategy(dev: pd.DataFrame, local_bps: float, adr_bps: float, lag: int = LAG,
             lookback: int = LOOKBACK, entry: float = ENTRY, exit: float = EXIT) -> tuple[pd.Series, pd.DataFrame]:
    """Equal capital per stock with an open signal is not needed: each stock gets 1/N of
    capital (N = stocks quoted that day) and is either flat or in a +/-1 spread position.

    Long spread on stock i = long local share (in USD at the market CCL) + short ADR.
    Its log P&L is exactly the change in dev_i, so the trade earns when dev_i reverts.
    """
    pnl = {}
    pos_all = {}
    for name in dev.columns:
        d = dev[name].dropna()
        if len(d) < lookback + 50:
            continue
        z = rolling_zscore(d, lookback)
        pos = positions(z, entry, exit, STOP)
        held = pos.shift(1 + lag).fillna(0)          # signal at t, filled at close t+lag
        gross = held * d.diff().fillna(0)
        turnover = held.diff().abs().fillna(held.abs())
        pnl[name] = gross - turnover * (local_bps + adr_bps) / 1e4
        pos_all[name] = held
    pnl = pd.DataFrame(pnl)
    n_quoted = dev.notna().sum(axis=1).reindex(pnl.index).clip(lower=1)
    return pnl.sum(axis=1) / n_quoted, pd.DataFrame(pos_all)


# ----------------------------------------------------------------------------- figures

def shade_cepo(ax: plt.Axes, label: bool = True) -> None:
    for i, (a, b) in enumerate(CEPO):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=style.YELLOW, alpha=0.10, lw=0,
                   label="Cepo cambiario" if (label and i == 0) else None)


def fig_ccl(ccl: pd.DataFrame, mkt: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for c in ccl.columns:
        ax.plot(ccl[c], color=style.MUTED, lw=0.6, alpha=0.6)
    ax.plot(mkt, color=style.BLUE, lw=2, label="CCL de mercado (mediana)")
    ax.plot([], [], color=style.MUTED, lw=1, label="CCL implícito de cada acción (13)")
    shade_cepo(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("Pesos por dólar (escala log)")
    ax.legend(loc="upper left")
    last = mkt.dropna()
    ax.set_title("26 años de dólar CCL, implícito en 13 acciones argentinas", pad=26)
    style.subtitle(ax, f"CCL_i = precio local x ratio / precio ADR. De \\${last.iloc[0]:.2f} en 2003 a "
                       f"\\${last.iloc[-1]:,.0f} en oct-2026: las 13 series viajan juntas (ley de un solo precio)")
    style.source(fig, "Precios de cierre sin ajustar, BYMA y NYSE. Fuente: Yahoo Finance. Corregida la escala de YPFD.BA (x10).")
    style.save(fig, FIG / "01_ccl_implicito.png")


def fig_dispersion(dev: pd.DataFrame) -> None:
    disp = dev.abs().median(axis=1).dropna().rolling(21).mean()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    shade_cepo(ax)
    ax.plot(disp, color=style.BLUE, lw=1.4, label="Desvío absoluto mediano vs CCL de mercado (media 21d)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=1))
    ax.legend(loc="upper right")
    in_cepo = pd.Series(False, index=disp.index)
    for a, b in CEPO:
        in_cepo |= (disp.index >= a) & (disp.index <= b)
    med_in, med_out = disp[in_cepo].median(), disp[~in_cepo].median()
    early, late = disp.loc[:"2015-12-31"].median(), disp.loc[SPLIT:].median()
    ax.set_title("La materia prima del arbitraje se agotó", pad=26)
    style.subtitle(ax, f"Desvío mediano {early:.2%} en 2003-2015 vs {late:.2%} desde 2016. Contra la intuición, "
                       f"no fue mayor con cepo ({med_in:.2%} vs {med_out:.2%} sin cepo)")
    style.save(fig, FIG / "02_dispersion.png")


def fig_equity(curves: dict[str, pd.Series], stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    colors = [style.AQUA, style.BLUE, style.ORANGE]
    shown = {k: v for k, v in curves.items() if "mismo cierre" not in k}
    for (name, r), c in zip(shown.items(), colors):
        ax.plot(r.cumsum(), color=c, lw=1.8, label=name)
    ax.axhline(0, color=style.BASELINE, lw=1)
    shade_cepo(ax, label=False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Retorno acumulado (log, sin reinversión)")
    ax.legend(loc="upper left")
    s = stats.loc[f"Neto {BASE.lower()}"]
    post = summary_table({"r": curves[f"Neto {BASE.lower()}"].loc[SPLIT:]}).loc["r"]
    ax.set_title("Arbitraje de CCL entre acciones, fuera de muestra 2003-2026", pad=26)
    style.subtitle(ax, f"Costo {BASE.lower()}: Sharpe {s['Sharpe']:.2f} (t = {s['t-stat']:.1f}) en todo el período, "
                       f"pero {post['Sharpe']:.2f} desde {SPLIT[:4]}. Para un minorista, nunca existió")
    style.source(fig, f"Señal: z-score del desvío de cada CCL vs la mediana, ventana {LOOKBACK}d. Entrada |z|>{ENTRY:g}, "
                      f"salida |z|<{EXIT:g}, stop |z|>{STOP:g}. Ejecución al cierre del día siguiente. Sombreado: cepo.")
    style.save(fig, FIG / "03_equity.png")


def fig_breakeven(costs: np.ndarray, sharpes: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(costs, sharpes, color=style.BLUE, marker="o", ms=5)
    ax.axhline(0, color=style.BASELINE, lw=1)
    for name, (loc, adr) in COSTS.items():
        tot = loc + adr
        ax.axvline(tot, color=style.INK_2, ls="--", lw=1)
        ax.annotate(f"{name}\n({tot} bps)", xy=(tot, 1), xycoords=("data", "axes fraction"), xytext=(4, -6),
                    textcoords="offset points", fontsize=9.5, color=style.INK_2, va="top")
    be = np.interp(0, sharpes[::-1], costs[::-1]) if min(sharpes) < 0 < max(sharpes) else np.nan
    ax.set_xlabel("Costo total por lado, ambas patas (bps)")
    ax.set_ylabel("Sharpe neto")
    ax.set_title("¿Hasta qué costo sobrevive el arbitraje?", pad=26)
    style.subtitle(ax, f"Punto de equilibrio ~{be:.0f} bps por lado. Por encima, la fricción se come toda la ventaja"
                   if np.isfinite(be) else "Sharpe neto según el costo de ejecución")
    style.save(fig, FIG / "04_costo_equilibrio.png")


def fig_robustness(rows: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [style.BLUE if v >= 0 else style.RED for v in rows["Sharpe"]]
    bars = ax.barh(rows.index, rows["Sharpe"], color=colors, height=0.55)
    for b, v in zip(bars, rows["Sharpe"]):
        ax.text(b.get_width() + (0.03 if v >= 0 else -0.03), b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color=style.INK_2)
    ax.axvline(0, color=style.BASELINE, lw=1)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Sharpe neto (costo institucional)")
    ax.set_title("Pruebas de estrés: ¿la ventaja es real o un artefacto?", pad=26)
    style.subtitle(ax, "Rezago de ejecución, horario de cierre BYMA vs NYSE, período, ventana y umbrales")
    style.save(fig, FIG / "05_robustez.png")


def fig_yearly(net: pd.Series) -> None:
    yr = yearly_returns(net)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(yr.index.astype(str), yr.to_numpy(), color=[style.BLUE if v >= 0 else style.RED for v in yr], width=0.7)
    ax.axhline(0, color=style.BASELINE, lw=1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Retorno neto por año calendario", pad=26)
    style.subtitle(ax, f"Positivo en {(yr > 0).sum()} de {len(yr)} años (costo {BASE.lower()})")
    style.save(fig, FIG / "06_por_anio.png")


def main() -> None:
    style.apply()
    usd, ars = load_listings()
    ccl = implied_ccl(usd, ars).loc[START:]
    mkt = market_ccl(ccl)
    dev = ccl_deviation(ccl)
    # A >25% gap vs the market CCL is a bad print or a ratio change, not an opportunity.
    bad = dev.abs() > 0.25
    dev = dev.mask(bad)
    print(f"Masked {int(bad.sum().sum())} of {int(ccl.notna().sum().sum())} stock-days as bad prints")

    loc_b, adr_b = COSTS[BASE]
    net, pos = strategy(dev, loc_b, adr_b)
    gross, _ = strategy(dev, 0, 0)
    retail, _ = strategy(dev, *COSTS["Minorista"])
    same_bar, _ = strategy(dev, loc_b, adr_b, lag=0)
    curves = {"Ejecución en el mismo cierre (no operable)": same_bar, "Bruto": gross,
              f"Neto {BASE.lower()}": net, "Neto minorista": retail}
    stats = summary_table(curves)

    cost_grid = np.arange(0, 125, 10)
    sharpes = [summary_table({"r": strategy(dev, c, 0)[0]}).loc["r", "Sharpe"] for c in cost_grid]

    # Robustness: each row changes one assumption vs the base case.
    edt = net.index.map(lambda d: pd.Timestamp(d).tz_localize("America/New_York").dst() != pd.Timedelta(0))
    edt = pd.Series(np.asarray(edt, dtype=bool), index=net.index)
    rob = {
        "Caso base": net,
        "Solo 2003-2015": net.loc[:"2015-12-31"],
        "Solo 2016-2026": net.loc[SPLIT:],
        "Rezago 2 días": strategy(dev, loc_b, adr_b, lag=2)[0],
        "Solo meses EDT (cierres sincronizados)": net[edt],
        "Solo meses EST (BYMA cierra 1h antes)": net[~edt],
        "Ventana 30 días": strategy(dev, loc_b, adr_b, lookback=30)[0],
        "Ventana 120 días": strategy(dev, loc_b, adr_b, lookback=120)[0],
        "Entrada |z|>1.5": strategy(dev, loc_b, adr_b, entry=1.5)[0],
        "Entrada |z|>2.5": strategy(dev, loc_b, adr_b, entry=2.5)[0],
    }
    rob_tbl = summary_table(rob)

    fig_ccl(ccl, mkt)
    fig_dispersion(dev)
    fig_equity(curves, stats)
    fig_breakeven(cost_grid, sharpes)
    fig_robustness(rob_tbl)
    fig_yearly(net)

    trades = int((pos.diff().abs() > 0).sum().sum() // 2)
    by_stock = pd.Series({c: (pos[c].diff().abs() > 0).sum() // 2 for c in pos.columns}).sort_values(ascending=False)
    results = (f"## Performance ({net.index[0].date()} a {net.index[-1].date()})\n\n{format_table(stats)}\n\n"
               f"Operaciones (ida y vuelta): {trades}. Por accion:\n\n{by_stock.to_frame('trades').T.to_markdown()}\n\n"
               f"## Robustez (costo {BASE.lower()})\n\n{format_table(rob_tbl)}\n\n"
               f"## Sharpe neto vs costo total por lado\n\n"
               f"{pd.Series(sharpes, index=cost_grid).round(2).to_frame('Sharpe').T.to_markdown()}\n\n"
               f"## Retornos anuales netos\n\n{yearly_returns(net).map(lambda v: f'{v:.1%}').to_markdown()}\n")
    (HERE / "results.md").write_text(results)
    print(results)


if __name__ == "__main__":
    main()
