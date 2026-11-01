# Post de LinkedIn

Adjuntar `carrusel.pdf` como documento. El link al repo va en el primer comentario (LinkedIn muestra menos los posts con links en el cuerpo).

---

El 12 de agosto de 2019, una cartera de ADRs argentinos cayó 47% en un día.

Bajo una distribución normal, eso pasa una vez cada 10^70 días. Pasó una vez en 19 años.

Quise saber si los modelos de riesgo que se usan en la industria están preparados para el mercado argentino. Así que armé un backtest de 7 modelos de Value-at-Risk sobre GGAL, BMA, BBAR, YPF, TGS, EDN, TEO y CRESY, con datos reales de 2007 a 2026.

El método:
→ Cada día re-estimé los 7 modelos usando solo las 500 ruedas previas
→ Comparé la pérdida real con el VaR 99% que cada modelo dio la noche anterior
→ 4.410 días fuera de muestra, 4 tests estadísticos (Kupiec, Christoffersen, cobertura condicional y Acerbi-Székely para el Expected Shortfall)
→ Después intenté romper las conclusiones: otras ventanas, otros parámetros y el S&P 500 como control

Lo que encontré:

1️⃣ El Monte Carlo de manual (retornos normales) falla. 63 excepciones contra 44 esperadas, concentradas en crisis. Un banco con ese modelo habría pasado el 8% de los días en zona roja de Basilea.

2️⃣ Hacen falta dos cosas a la vez: colas gruesas y volatilidad que reaccione. Con una sola no alcanza. Los 4 modelos que aprueban (Monte Carlo t-Student, simulación histórica filtrada, GARCH-t y EVT) tienen las dos.

3️⃣ Más simulaciones no arreglan un mal modelo. Con 5.000 simulaciones el ruido es 4,5% del VaR. La diferencia entre modelos el mismo día es 43%.

4️⃣ Ningún modelo vio venir las PASO ni el 8 de septiembre de 2025. Fueron saltos políticos desde la calma. El VaR mide un día malo, no el peor día: para eso están el Expected Shortfall y los escenarios de estrés.

Todo el código, los datos y los 16 tests unitarios están abiertos. Un comando reproduce cada número del informe, incluidas las limitaciones y lo que no funcionó.

¿Qué modelo de riesgo usan en su mesa para activos argentinos?

#RiskManagement #QuantFinance #ValueAtRisk #Python #MercadoArgentino

---

**Primer comentario:**

Repo con el informe completo, metodología y código: [LINK AL REPO]
