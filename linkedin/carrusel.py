"""Carrusel de LinkedIn (PDF, 1080x1350 por página) a partir de las figuras del estudio.

    uv run python estudio/run.py        # primero, para tener las figuras
    uv run python linkedin/carrusel.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from quantkit import style

HERE = Path(__file__).parent
FIG = HERE.parent / "estudio" / "figures"
OUT = HERE / "carrusel.pdf"

AUTOR = ""  # nombre que aparece en la portada y al pie de cada página
W, H = 8, 10  # pulgadas; a 135 dpi = 1080 x 1350 (4:5, formato vertical de LinkedIn)
M = 0.07      # margen lateral (fracción del ancho)
KICKER = "VaR · MERCADO ARGENTINO"

style.apply()
plt.rcParams["savefig.bbox"] = "standard"  # every page exactly 1080x1350, never cropped to content


def page(n: int, total: int) -> plt.Figure:
    fig = plt.figure(figsize=(W, H))
    fig.text(M, 0.955, KICKER, fontsize=11, color=style.BLUE, fontweight="bold")
    fig.text(1 - M, 0.955, f"{n}/{total}", fontsize=11, color=style.MUTED, ha="right")
    if AUTOR:
        fig.text(M, 0.03, AUTOR, fontsize=10, color=style.MUTED)
    if n < total:
        fig.text(1 - M, 0.03, "Deslizá →", fontsize=10, color=style.MUTED, ha="right")
    return fig


def title(fig: plt.Figure, text: str, y: float = 0.90, size: int = 30, width: int = 26) -> float:
    """Draw a wrapped title; return the y where the next block can start."""
    lines = textwrap.wrap(text, width)
    fig.text(M, y, "\n".join(lines), fontsize=size, fontweight="bold", va="top", linespacing=1.15)
    return y - len(lines) * size * 1.3 / (72 * H) - 0.03


def body(fig: plt.Figure, text: str, y: float, size: int = 17, width: int = 46, color: str = style.INK_2,
         x: float = M) -> float:
    lines = textwrap.wrap(text, width)
    fig.text(x, y, "\n".join(lines), fontsize=size, color=color, va="top", linespacing=1.35)
    return y - len(lines) * size * 1.35 / (72 * H) - 0.025


def image(fig: plt.Figure, name: str, top: float, bottom: float = 0.07) -> None:
    img = mpimg.imread(FIG / name)
    h_px, w_px = img.shape[:2]
    width = 1 - 2 * M
    height = width * (h_px / w_px) * W / H
    height = min(height, top - bottom)
    ax = fig.add_axes((M, top - height, width, height))
    ax.imshow(img)
    ax.axis("off")


def big_number(fig: plt.Figure, value: str, label: str, x: float, y: float, color: str) -> None:
    fig.text(x, y, value, fontsize=54, fontweight="bold", color=color, va="top")
    fig.text(x, y - 0.085, "\n".join(textwrap.wrap(label, 20)), fontsize=14, color=style.INK_2, va="top",
             linespacing=1.3)


def scorecard(fig: plt.Figure, top: float) -> None:
    rows = [
        ("MC normal, cov. muestral", 63, False),
        ("MC normal, cov. EWMA", 62, False),
        ("Simulación histórica", 56, False),
        ("EVT filtrado (GARCH + GPD)", 52, True),
        ("Simulación histórica filtrada", 51, True),
        ("MC t-Student, cov. EWMA", 45, True),
        ("GARCH(1,1)-t", 45, True),
    ]
    ax = fig.add_axes((M, 0.12, 1 - 2 * M, top - 0.12))
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows) + 0.6, -0.8)
    ax.axis("off")
    ax.text(0.0, -0.4, "Modelo", fontsize=12, color=style.MUTED)
    ax.text(0.66, -0.4, "Excepciones", fontsize=12, color=style.MUTED, ha="center")
    ax.text(0.92, -0.4, "Veredicto", fontsize=12, color=style.MUTED, ha="center")
    for i, (name, exc, ok) in enumerate(rows):
        y = i + 0.5
        bg = "#e3f3e3" if ok else "#f8e0df"
        ax.add_patch(FancyBboxPatch((0, y - 0.42), 1, 0.84, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=bg, ec="none", mutation_aspect=0.3))
        ax.text(0.02, y, name, fontsize=15, va="center", fontweight="bold" if i == 0 else "normal")
        ax.text(0.66, y, f"{exc}", fontsize=17, va="center", ha="center", fontweight="bold")
        ax.text(0.92, y, "Aprueba" if ok else "Falla", fontsize=15, va="center", ha="center",
                color=style.GREEN if ok else style.RED, fontweight="bold")
    ax.text(0.0, len(rows) + 0.35, "Esperadas: 44 en 4.410 días. Aprueba = Kupiec, Christoffersen, cobertura "
            "condicional y ES (Z2)", fontsize=11, color=style.MUTED, va="top")


def main() -> None:
    total = 10
    with PdfPages(OUT) as pdf:
        def save(fig: plt.Figure) -> None:
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)

        # 1. Portada
        fig = page(1, total)
        y = title(fig, "Medí el riesgo de una cartera argentina con 7 modelos de VaR", size=34, width=22)
        y = body(fig, "El modelo de manual falla. 17 años de datos reales, 4.410 días fuera de muestra.", y,
                 size=19, width=40)
        image(fig, "01_abanico.png", top=y - 0.01)
        save(fig)

        # 2. La pregunta
        fig = page(2, total)
        y = title(fig, "La pregunta")
        y = body(fig, "Un banco o un fondo mide el riesgo de esta cartera con el Monte Carlo de manual: "
                      "retornos normales y correlaciones fijas. ¿Está midiendo bien?", y)
        y = body(fig, "Cartera: GGAL, BMA, BBAR, YPF, TGS, EDN, TEO y CRESY en partes iguales, en dólares.", y,
                 color=style.INK)
        big_number(fig, "VaR 99%", "La pérdida diaria que solo debería superarse 1 de cada 100 días",
                   M, y - 0.05, style.BLUE)
        big_number(fig, "1%", "Si el modelo es bueno, el 1% de los días la pérdida lo supera. Ni más, ni menos.",
                   0.55, y - 0.05, style.ORANGE)
        save(fig)

        # 3. Colas
        fig = page(3, total)
        y = title(fig, "El problema: las colas argentinas no son normales")
        y = body(fig, "PASO 2019: la cartera cayó 47% en un día, 18 desvíos estándar. Bajo una normal "
                      "eso pasa una vez en 10^70 días. Pasó una vez en 19 años.", y)
        image(fig, "03_colas.png", top=y)
        save(fig)

        # 4. Método
        fig = page(4, total)
        y = title(fig, "El método")
        steps = [
            ("1", "Cada día re-estimo los 7 modelos con las 500 ruedas anteriores. Nada del futuro."),
            ("2", "Comparo la pérdida real con el VaR que el modelo dio la noche anterior."),
            ("3", "4 tests estadísticos: frecuencia, agrupamiento, ambos, y el del Expected Shortfall."),
            ("4", "Lo intento romper: otras ventanas, otro λ y el S&P 500 como control."),
        ]
        for num, text in steps:
            fig.text(M, y, num, fontsize=40, fontweight="bold", color=style.BLUE, va="top")
            y = min(body(fig, text, y - 0.008, size=18, width=38, x=M + 0.09), y - 0.07) - 0.035
        save(fig)

        # 5. Resultado
        fig = page(5, total)
        y = title(fig, "4 de 7 modelos aprueban. El de manual, no.", size=28, width=28)
        scorecard(fig, top=y)
        save(fig)

        # 6. Backtest
        fig = page(6, total)
        y = title(fig, "La normal falla en racimos")
        y = body(fig, "63 excepciones contra 44 esperadas, concentradas en crisis. Un banco con este modelo "
                      "habría pasado el 8% de los días en zona roja de Basilea.", y)
        image(fig, "04_backtest_var.png", top=y)
        save(fig)

        # 7. Peores días
        fig = page(7, total)
        y = title(fig, "Lo que ningún modelo ve")
        y = body(fig, "En COVID los modelos adaptativos ya habían subido el VaR. En las PASO y en septiembre "
                      "2025, ninguno: fueron saltos políticos desde la calma. Para eso están el ES y los "
                      "escenarios de estrés.", y)
        image(fig, "07_peores_dias.png", top=y)
        save(fig)

        # 8. Monte Carlo
        fig = page(8, total)
        y = title(fig, "Más simulaciones no arreglan un mal modelo")
        big_number(fig, "4,5%", "Ruido de Monte Carlo con 5.000 simulaciones", M, y, style.BLUE)
        big_number(fig, "43%", "Diferencia de VaR entre modelos el mismo día", 0.55, y, style.ORANGE)
        image(fig, "08_error_montecarlo.png", top=y - 0.2)
        save(fig)

        # 9. Lecciones
        fig = page(9, total)
        y = title(fig, "3 lecciones")
        lessons = [
            "Hacen falta colas gruesas Y volatilidad que reaccione. Una sola de las dos no alcanza.",
            "El riesgo de modelo es 10 veces el ruido de simulación. Elegí bien antes de simular más.",
            "El VaR mide un día malo, no el peor día. En Argentina, el peor día es político.",
        ]
        for i, text in enumerate(lessons, 1):
            fig.text(M, y, f"{i}", fontsize=40, fontweight="bold", color=style.BLUE, va="top")
            y = min(body(fig, text, y - 0.008, size=19, width=36, color=style.INK, x=M + 0.09), y - 0.07) - 0.04
        save(fig)

        # 10. Cierre
        fig = page(10, total)
        y = title(fig, "Código, datos y tests abiertos", y=0.70, size=34, width=20)
        y = body(fig, "Informe completo, 16 tests unitarios y un comando para reproducir cada número.", y,
                 size=19, width=40)
        body(fig, "Link al repo en los comentarios.", y - 0.03, size=19, color=style.BLUE)
        save(fig)
    print(f"Carrusel: {OUT}")


if __name__ == "__main__":
    main()
