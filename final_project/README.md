<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,20&height=240&section=header&text=Análisis%20de%20Sentimiento%20Financiero&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Comparación%20de%20Modelos%20Base%20y%20Modelos%20Generativos%20(LLMs)&descAlignY=58&descSize=15" width="100%"/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=19&duration=2800&pause=900&color=1F8BFF&center=true&vCenter=true&random=false&width=980&lines=An%C3%A1lisis+de+Sentimiento+en+Textos+Financieros;Comparaci%C3%B3n+BERT%2C+FinBERT%2C+Llama+y+Gemma;Prompting+Zero-Shot+y+Few-Shot)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFB000?style=for-the-badge&logo=huggingface&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-f55036?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

---

## Objetivo

Responder a la pregunta: **¿Sigue siendo rentable hacer adaptación de dominio cuando existen modelos instruccionales capaces de resolver la tarea mediante prompting zero-shot o few-shot?**

Para lograrlo, se evalúa y compara el rendimiento de modelos clásicos preentrenados y adaptados al dominio financiero (como **BERT** y **FinBERT**) frente a Modelos de Lenguaje Grande (LLMs) instruccionales contemporáneos (**Llama 3.1 8B**, **Llama 3.2 1B**, y **Gemma 2B**) bajo estrategias de inferencia *zero-shot* y *few-shot* para la tarea de clasificación de sentimiento en textos e inversiones financieras.

---

## Datos

El análisis se lleva a cabo sobre dos conjuntos de datos que abarcan distintos aspectos del lenguaje financiero:

1. **Sentences_AllAgree.csv (Financial PhraseBank):**
   - Contiene **2,264** ejemplos de texto formal (noticias financieras).
   - Datos etiquetados en `0 = negative`, `1 = neutral`, `2 = positive`.
   - Orientado a probar la comprensión de descripciones empresariales y datos del mercado con alta coincidencia entre anotadores.

2. **FinTextSen.csv:**
   - Consiste en **1,680** ejemplos (después de lipieza básica).
   - Textos provenientes de microblogs (Twitter, StockTwits), lo cual significa lenguaje más breve, ruidoso e informal ("cashtags", acrónimos).
   - El conjunto representa un desafío distinto al estar fuertemente desbalanceado y repleto de jerga de inversionistas (trading).

---

## Arquitectura

El flujo experimental se ha estructurado de manera iterativa para contrastar el rendimiento relativo de distintas ramas tecnológicas.

```mermaid
flowchart TD
    A([Datasets:
PhraseBank & FinTextSen]) --> B[Preprocesamiento]
    
    B --> C[Modelos Base
Adaptación de Dominio]
    B --> D[Modelos Generativos
Prompting]
    
    C --> B1(BERT Base)
    C --> F1(FinBERT)
    
    D --> L1(Llama 3.1 8B
vía Groq)
    D --> L2(Llama 3.2 1B
Instruct)
    D --> G1(Gemma 2B
Zero-shot / Few-shot)
    
    B1 --> E(Evaluación de Métricas
F1-Score, Inferencia)
    F1 --> E
    L1 --> E
    L2 --> E
    G1 --> E
    
    E --> R([Resultados y Comparación])
```

- **Clasificadores Tradicionales:** Extracción de características y inferencia usando arquitecturas basadas en Transformers (BERT/FinBERT) configuradas para clasificación secuencial.
- **Modelos Generativos:** Inferencia usando APIs (Groq) o inferencia local en HuggingFace con prompts específicos para orientar las salidas de texto a etiquetas discretas.

---

## Estructura del proyecto

```text
final_project/
├── data/
│   ├── Datos.md                      # Documentación completa de los datasets
│   ├── FinTextSen.csv                # Datos crudos de microblogs
│   └── Sentences_AllAgree.csv        # Datos formales de PhraseBank
├── src/
│   ├── experimentos_groq_llama31_8b.ipynb       # Pruebas con Llama 3.1 8B (usando API de Groq)
│   ├── Proyecto_FInal_bert.ipynb                # Inferencia y evaluación tradicional con BERT
│   ├── ProyectoFinal_finbert.ipynb              # Inferencia y evaluación dominial con FinBERT
│   ├── ProyectoFinal_llama32_1b_instruc.ipynb   # Evaluación con Llama 3.2 1B instructivo
│   ├── ProyectoFinalPLN_Gemma_fewshot.ipynb     # Estrategias Few-Shot mediante Gemma
│   └── ProyectoFinalPLN_Gemma_zeroshot.ipynb    # Inferencia Zero-Shot mediante Gemma
└── README.md                         # Este documento
```

--- 

## Instrucciones para ejecutar el código 
El presente trabajo fue realizado a base de notebooks, ls cuales pueden ser ejecutados de forma individual. **Unicamente** en los realizados con la API de Groq, se recomienda utilizar una API personal para este ámbito. A continuación se muestra el desarrollo de las demás secciones de este proyecto final.

---

## Base de datos

Las bases de datos usadas no requieren gestores de bases de datos relacionales ni bases NoSQL para su lectura de este proyecto; se distribuyen directamente en archivos CSV dentro del directorio `data/`. Estas colecciones proporcionan el ecosistema para inferencia directa.

Ambos conjuntos están codificados con etiquetas discretas (`0`, `1`, `2`) y no presentan dependencias externas para su consulta. Todo su procesamiento (limpieza de NA, balanceo, eliminación de textos extremadamente vacíos) se realiza in-memory en cada cuaderno de experimentación mediante Pandas.

---

# Comparación de métricas entre el baseline del decano y nuestra propuesta

## Tabla 1: FinBERT vs BERT

| Métrica de Evaluación | Artículo FinBert Base | Réplica de resultados FinBert Base | Bert Artículo | Réplica de resultados BERT |
|------------------------|-----------------------|-----------------------------------|---------------|----------------------------|
| Micro F1              | 96.86                 | 97.17                             | 96.60         | 63.03                      |
| Macro F1              | 95.61                 | 96.25                             | 95.15         | 56.30                      |

---

## Tabla 2: Comparación entre Bert, Llama y Gemma

| Métrica de Evaluación | Bert Base | Llama 3.2-1B-Instruct | Gemma-4-12B-It |
|------------------------|-----------|-----------------------|----------------|
| Micro F1 - ZS         | 63.03     | 57.12                 | 93.83          |
| Macro F1 - ZS         | 56.30     | 38.78                 | 93.82          |
| Micro F1 - FS         | 53.66     | 39.91                 | 98.02          |
| Macro F1 - FS         | 44.92     | 36.27                 | 98.01          |

# Conclusiones principales 
* FinBERT sigue siendo una referencia fuerte para análisis de sentimiento financiero.
Este punto nos demostró que un modelo preentrenado corretamente, sigue produciendo mejores resultados que un modelo base. Pero como vimos con los resultados de Gemma 12B, un modelo con una myor número de parámetros logra clasificar de mejor manera las pruebas de análisis de sentimientos en finanzas.

* Eficiencia Computacional

Se logró una reducción drástica en el preprocesamiento manual y tiempos de entrenamiento gracias a la extracción automática de características.

* Líneas de Investigación Futura

Optimización de hiperparámetros y aumento de datos.
Evaluación con modelos de lenguaje con inclinación financiera, o realizar un proceso de *fine-tuning* a un modelo que funciona correctamente, para lograr obtener mejores resultados.



## Librerías necesarias

* **[Pandas](https://pandas.pydata.org/)** — Análisis y manipulación general de datos.
* **[Scikit-learn](https://scikit-learn.org/)** — Cálculo de métricas de desempeño (F1-score, accuracy, matrices de confusión).
* **[Transformers (Hugging Face)](https://huggingface.co/docs/transformers/index)** — Carga y uso de modelos BERT, FinBERT, Llama y Gemma.
* **[PyTorch](https://pytorch.org/)** — Backend para la ejecución y ajuste de los modelos en CPU/GPU.
* **[Groq API](https://groq.com/)** — Inferencia veloz en la nube para experimentar con el modelo Llama 3.1 8B de grandes proporciones.
* **[Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)** — Para la visualización de resultados e histogramas.

---

## Conclusiones

[1] Peng, B., Chersoni, E., Hsu, Y.-Y., & Huang, C.-R. (2021). Is Domain Adaptation Worth Your Investment? Comparing BERT and FinBERT on Financial Tasks. Actas y repositorios de investigación en Procesamiento de Lenguaje Natural Financiero.
[2] Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4), 782-796. (Fuente original del dataset Financial PhraseBank).
[3] Cortis, K., Davis, B., McDermott, J., Handschuh, S., & Manandhar, S. (2017). SemEval-2017 Task 5: Fine-Grained Sentiment Analysis on Financial Microblogs and News. Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). (Fuente base para FinTextSen).
[4] Touvron, H., et al. (2023). Llama: Open and Efficient Foundation Language Models. Meta AI Research.
[5] Gemma Team, Google DeepMind. (2024). Gemma: Open Models Based on Gemini Research and Technology.
[6] Hugging Face. (2025). Transformers Documentation. Recuperado de la plataforma oficial de Hugging Face.
[7] Documentación técnica interna del proyecto (Datos.md y README.md). Repositorio: Análisis de Sentimiento Financiero.

---

## Equipo

* **Fonseca González Bruno**
* **Pichardo Ahuatzi Mariano Josué**

<div align="center">


</div>
