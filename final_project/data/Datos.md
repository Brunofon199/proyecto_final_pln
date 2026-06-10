# Descripción de los datos

Este proyecto utiliza dos conjuntos de datos de análisis de sentimiento financiero: `Sentences_AllAgree.csv` y `FinTextSen.csv`. Ambos se almacenan en formato CSV y comparten el mismo esquema básico:

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `sentences` | texto | Oración, titular, mensaje o microblog financiero a clasificar. |
| `labels` | entero | Etiqueta de sentimiento codificada como `0 = negative`, `1 = neutral`, `2 = positive`. |

La tarea común es clasificación de sentimiento en tres clases. Esto permite comparar modelos ajustados al dominio financiero, como FinBERT, contra modelos generales o modelos instruccionales evaluados mediante estrategias de prompting.

## Relación con el artículo base

El artículo base, *Is Domain Adaptation Worth Your Investment? Comparing BERT and FinBERT on Financial Tasks* (Peng et al., 2021), estudia si la adaptación de dominio en modelos tipo BERT aporta beneficios medibles en tareas financieras. Para sentimiento financiero, el paper reporta experimentos con Financial PhraseBank y FinTextSen, entre otros conjuntos. La idea central relevante para este proyecto es que continuar el preentrenamiento desde BERT con datos financieros resultó más efectivo que entrenar un modelo financiero desde cero con vocabulario especializado.

Los dos archivos usados aquí cubren dos escenarios complementarios de sentimiento financiero:

- `Sentences_AllAgree.csv`: oraciones financieras relativamente formales, tomadas de Financial PhraseBank.
- `FinTextSen.csv`: mensajes breves de microblogs financieros, más cercanos al lenguaje de redes sociales e inversión minorista.

Esta combinación es útil porque permite evaluar si las conclusiones sobre adaptación de dominio se sostienen tanto en texto financiero formal como en texto financiero corto, ruidoso y contextual.

## `Sentences_AllAgree.csv`

`Sentences_AllAgree.csv` corresponde a un subconjunto de Financial PhraseBank. Financial PhraseBank contiene oraciones de noticias financieras anotadas con sentimiento positivo, negativo o neutral por anotadores con conocimiento del dominio financiero. El archivo utilizado contiene únicamente ejemplos con acuerdo alto entre anotadores, por lo que representa una versión más confiable para evaluación.

Ejemplo de registro:

```csv
sentences,labels
"For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m .",2
```

Resumen del archivo:

| Métrica                          |  Valor |
| -------------------------------- | -----: |
| Registros                        |  2,264 |
| Columnas                         |      2 |
| Textos vacíos                    |      0 |
| Etiquetas vacías                 |      0 |
| Promedio de palabras por texto   |  22.44 |
| Máximo de palabras por texto     |     81 |
| Promedio de caracteres por texto | 121.96 |
| Oraciones duplicadas             |      5 |

Distribución de clases:

| Etiqueta | Sentimiento | Registros | Porcentaje |
| ---: | --- | ---: | ---: |
| 0 | Negativo | 303 | 13.38% |
| 1 | Neutral | 1,391 | 61.44% |
| 2 | Positivo | 570 | 25.18% |

Observaciones:

- El conjunto está fuertemente cargado hacia la clase neutral.
- Los textos son más largos y gramaticalmente estructurados que los de microblogs.
- Es un buen conjunto para evaluar comprensión de noticias financieras, resultados empresariales y cambios en indicadores corporativos.
- Al no contener textos vacíos, puede usarse directamente después de convertir `labels` a entero.

## `FinTextSen.csv`

`FinTextSen.csv` proviene de FinTextSen, asociado con SemEval 2017 Task 5 y versiones posteriores usadas para sentimiento financiero en microblogs. El artículo base describe FinTextSen como un conjunto de mensajes de Twitter y StockTwits relacionados con instrumentos financieros. Originalmente el sentimiento podía representarse como una puntuación continua; en la versión usada para este proyecto se trabaja con una codificación discreta de tres clases para mantener comparabilidad con Financial PhraseBank.

Ejemplo de registro:

```csv
sentences,labels
watching for bounce tomorrow,2
out $NFLX -.35,0
```

Resumen del archivo original:

| Métrica | Valor |
| --- | ---: |
| Registros | 1,700 |
| Columnas | 2 |
| Textos vacíos | 20 |
| Etiquetas vacías | 0 |
| Promedio de palabras por texto | 6.12 |
| Máximo de palabras por texto | 25 |
| Promedio de caracteres por texto | 34.61 |
| Filas con `sentences` duplicado | 448 |
| Filas completas duplicadas | 439 |

Distribución de clases en el archivo original:

| Etiqueta | Sentimiento | Registros | Porcentaje |
| ---: | --- | ---: | ---: |
| 0 | Negativo | 581 | 34.18% |
| 1 | Neutral | 27 | 1.59% |
| 2 | Positivo | 1,092 | 64.24% |

Después de eliminar textos vacíos, quedan 1,680 registros:

| Etiqueta | Sentimiento | Registros |
| ---: | --- | ---: |
| 0 | Negativo | 578 |
| 1 | Neutral | 16 |
| 2 | Positivo | 1,086 |

Observaciones:

- El conjunto es muy corto: la mayoría de los mensajes son frases telegráficas propias de microblogs financieros.
- La clase neutral está severamente subrepresentada, especialmente después de limpiar textos vacíos.
- Hay muchos duplicados o mensajes repetidos, lo cual puede afectar métricas si se hacen particiones aleatorias sin control.
- Contiene cashtags, símbolos bursátiles, abreviaturas, fragmentos sin contexto y expresiones propias de trading, por lo que es un buen caso para probar robustez de modelos frente a lenguaje financiero informal.
- Antes de inferencia o entrenamiento debe aplicarse una limpieza mínima: eliminar filas con `sentences` vacío, remover espacios extremos y convertir `labels` a entero.

## Comparación entre datasets

| Característica | `Sentences_AllAgree.csv` | `FinTextSen.csv` |
| --- | --- | --- |
| Dominio textual | Noticias/oraciones financieras | Microblogs financieros |
| Estilo | Formal, descriptivo | Breve, informal, ruidoso |
| Registros originales | 2,264 | 1,700 |
| Registros recomendados tras limpieza | 2,264 | 1,680 |
| Longitud promedio | 22.44 palabras | 6.12 palabras |
| Clase dominante | Neutral | Positiva |
| Principal riesgo experimental | Desbalance hacia neutral | Desbalance extremo y textos vacíos |

---

## Implicaciones para el proyecto

Estos datos permiten plantear una reproducción parcial del artículo base mediante evaluación de modelos de sentimiento financiero, y además habilitan una extensión metodológica alineada con modelos de lenguaje actuales. En lugar de preguntar únicamente si conviene entrenar un FinBERT desde cero o continuar BERT con datos financieros, el proyecto puede formular una pregunta más actual:

**¿Sigue siendo rentable hacer adaptación de dominio cuando existen modelos instruccionales capaces de resolver la tarea mediante prompting zero-shot o few-shot?**

Para responderla, los datasets pueden usarse en tres bloques experimentales:

1. Evaluar un modelo financiero adaptado, por ejemplo FinBERT, sobre ambos datasets.
2. Evaluar un modelo general o instruccional mediante prompting zero-shot.
3. Evaluar el mismo modelo con few-shot prompting, usando ejemplos representativos de cada clase.

Las métricas recomendadas son `macro F1` y `micro F1`. `macro F1` es especialmente importante porque ambos datasets están desbalanceados; en `FinTextSen.csv`, la clase neutral es tan pequeña que una métrica agregada puede ocultar fallas relevantes.

## Decisiones de preprocesamiento recomendadas

Para mantener consistencia experimental:

- Usar el mapeo `0 = negative`, `1 = neutral`, `2 = positive`.
- Eliminar filas sin texto antes de ejecutar modelos.
- No eliminar duplicados en la reproducción inicial, para respetar el archivo recibido.
- Reportar un experimento adicional sin duplicados si se desea analizar sensibilidad.
- Usar las mismas métricas para todos los modelos.
- Mantener separados los resultados de `Sentences_AllAgree.csv` y `FinTextSen.csv`, ya que representan condiciones lingüísticas distintas.
