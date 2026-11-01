# Metodología

El objetivo no es encontrar el modelo con el mejor número en el papel, sino el que **menos se rompe** cuando intentamos romperlo.

## 1. Pregunta antes que código

El estudio abre con una hipótesis falsable en una oración y el test que la rechazaría:

> H0: el VaR 99% por Monte Carlo normal con covarianza muestral tiene 1% de excepciones, sin agrupamiento. Se rechaza si la cobertura condicional de Christoffersen da p < 5%.

## 2. Decisiones fijadas antes de correr

Fijadas en `estudio/run.py` antes de la primera corrida sobre datos argentinos:

| Parámetro | Valor | Por qué |
|---|---|---|
| Cartera | 8 ADRs en partes iguales | Liquidez y cobertura sectorial; sin optimizar |
| Ventana de estimación | 500 ruedas (~2 años) | Estándar de la industria; incluye al menos un episodio de estrés |
| λ EWMA | 0,94 | RiskMetrics (1996) |
| Simulaciones | 5.000 | Ver sección de error de Monte Carlo |
| Re-ajuste GARCH / ν / GPD | Cada 21 ruedas | Costo computacional; la vol se actualiza igual cada día |
| Fracción de cola EVT | 10% | McNeil & Frey (2000) |
| Semilla | 7 | Reproducibilidad |
| Valor crítico Z2 | −0,70 | Acerbi & Székely (2014), p = 2,5% |

Cambios posteriores a ver resultados se declaran en el informe (sección *Limitaciones*): la simulación histórica filtrada, la regla de respaldo de GARCH y el recorte de −99% en las trayectorias.

## 3. Higiene de datos

- Precios ajustados de Yahoo Finance, **congelados a CSV** en la primera descarga (`data/`), para que los números publicados no cambien si el proveedor revisa la historia.
- La descarga falla si algún activo tiene menos de 250 precios: nunca se congela una descarga parcial.
- Sesgos nombrados, no escondidos: supervivencia (activos que hoy cotizan) y selección (quién eligió la cartera).

## 4. Fuera de muestra o no pasó

- Cada pronóstico del día *t* usa solo las ruedas *t − 500* a *t − 1*.
- Los modelos se comparan solo sobre el período en que todos tienen pronóstico (2009-2026).
- El horizonte de 10 días se evalúa en ventanas **no superpuestas**, para que los tests no cuenten la misma pérdida varias veces.

## 5. Validación con tests, no con gráficos

- Frecuencia: Kupiec. Independencia: Christoffersen. Ambas: cobertura condicional.
- Expected Shortfall: Z2 de Acerbi-Székely.
- Regulatorio: semáforo de Basilea sobre 250 ruedas.
- Ranking: pérdida cuantílica, que es la *scoring rule* para la que el verdadero cuantil es óptimo.
- Cada modelo y cada test está verificado contra una respuesta teórica conocida (`tests/test_risk.py`).

## 6. Pruebas de estrés del propio estudio

- **Supuestos:** ventana 250 y 1.000, λ = 0,97.
- **Control:** los mismos modelos sobre el S&P 500, para separar lo que es propio de Argentina.
- **Ruido de simulación vs riesgo de modelo:** error de Monte Carlo para N entre 500 y 100.000 comparado con la dispersión entre modelos.
- **Eventos:** qué pronosticaba cada modelo la noche anterior a los 6 peores días.

## 7. Reportar lo que falla

El informe termina con *Limitaciones y lo que no funcionó*. Un resultado sin sus modos de falla es marketing, no investigación.

## 8. Un comando para reproducir

```bash
uv run python estudio/run.py
```
