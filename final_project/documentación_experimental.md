# 3. Documentación Experimental

El diseño experimental de este proyecto se estructuró con el fin de evaluar rigurosamente la viabilidad económica y técnica de la adaptación de dominio en el ámbito financiero, contrastando el paradigma de ajuste fino tradicional frente al aprendizaje en contexto (In-Context Learning) mediante Large Language Models (LLMs).

---

## 3.1 Preparación de los Datos y Pipeline de Ingeniería de Software

El análisis se cimentó sobre dos corpus textuales con propiedades estadísticas y estructuras lingüísticas marcadamente divergentes:

- **Sentences_AllAgree.csv (Financial PhraseBank):** Un corpus de 2,264 noticias financieras de carácter formal con una longitud promedio de 22.44 palabras. Presenta un desbalance hacia la clase neutral (61.44%).
- **FinTextSen.csv:** Datos procedentes de microblogs bursátiles (Twitter/StockTwits) caracterizados por un lenguaje informal, ruidoso y con presencia de cashtags. Tras la remoción de registros vacíos y limpieza final, el dataset se consolidó en 1,662 instancias con una longitud promedio de 6.12 palabras y un desbalance extremo hacia las clases positiva y negativa.

El flujo de preprocesamiento se automatizó mediante la biblioteca Pandas implementando el siguiente pipeline in-memory:

```python
import pandas as pd

# 1. Eliminación de valores nulos y cadenas vacías
df = df.dropna(subset=['text']).drop_duplicates(keep='first')

# 2. Mapeo estricto de etiquetas a valores discretos enteros
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df['label'] = df['sentiment'].map(label_mapping)
```

## 3.2 Ejecución de la Reproducción (Modelos de Adaptación de Dominio)

La réplica experimental del estudio de Peng et al. (2021) se centró en evaluar si el costo de preentrenamiento continuo se justifica en tareas de clasificación financiera. Las arquitecturas empleadas fueron **google-bert/bert-base-uncased** y **yiyanghkust/finbert**.

Para su ejecución a través de los cuadernos `Proyecto_Final_bert.ipynb` y `ProyectoFinal_finbert.ipynb`:

- Se inicializó el entorno descargando los pesos desde el Hugging Face Hub utilizando la biblioteca **transformers** en un backend de PyTorch.  
- Se aplicó un proceso de tokenización con truncamiento y alineación de vectores (padding) limitando la longitud máxima de las secuencias (`max_length=128` para PhraseBank y `max_length=64` para FinTextSen).  
- Los tensores resultantes se alimentaron a las capas de clasificación secuencial, aplicando una función **Softmax** sobre los logits de salida para mapear las probabilidades y un operador **Argmax** para realizar la asignación categórica final.

---

## 3.3 Ejecución de la Propuesta de Mejora (Modelos Generativos y Prompting)

La propuesta de innovación consistió en eludir el paradigma clásico de extracción de características y ajuste fino, migrando hacia problemas de generación de texto restringida mediante ingeniería de prompts. El flujo de trabajo se dividió operativamente de la siguiente manera:

- **Inferencia Local:** Implementada en los cuadernos `ProyectoFinal_llama32_1b_instruc.ipynb` y los correspondientes a la familia Gemma utilizando recursos de aceleración de hardware unificado Apple Silicon (M-series, 24GB RAM) y GPUs locales mediante PyTorch.  
- **Inferencia Remota en la Nube:** Ejecutada en `experimentos_groq_llama31_8b.ipynb` consumiendo la API de Groq para aprovechar la velocidad de procesamiento de su arquitectura de hardware LPU.  

Se aplicaron estrategias **Zero-Shot** y **Few-Shot** inyectando instrucciones de rol precisas para forzar al modelo a responder exclusivamente con una de las tres etiquetas válidas y evitar alucinaciones operativas:

```plaintext
You're an expert financial sentiment analyst, and you make no mistakes on your choices. 
Also you only answer with positive, negative or neutral.
Sentence: {sentence}
Classify the sentiment of this sentence as Positive, Negative or Neutral, CHOOSE just one.
```
## 3.4 Hiperparámetros y Configuraciones de los Modelos

Para garantizar la reproducibilidad científica absoluta de los experimentos, se parametrizaron los entornos bajo los siguientes criterios estrictos:

| Parámetro / Configuración | Modelos Base (BERT / FinBERT) | Modelos Generativos (Llama / Gemma) |
|---------------------------|-------------------------------|-------------------------------------|
| Temperatura (Temperature) | N/A (Clasificación Directa)   | ≈0.0 (Greedy Decoding para asegurar determinismo) |
| Límite de Tokens de Salida| N/A                           | max_new_tokens entre 5 y 10 tokens  |
| Longitud de Contexto      | 64 - 128 tokens               | Ventanas nativas extendidas (128k - 256k tokens) |
| Mecanismo de Evaluación   | Inferencia directa sobre logits de capa lineal | Inferencia instruccional restringida por prompt |


```mermaid
flowchart TD
    A[Modelos Base: BERT / FinBERT] --> B[Temperatura: N/A]
    A --> C[Límite de Tokens: N/A]
    A --> D[Contexto: 64-128 tokens]
    A --> E[Mecanismo: Logits lineales]

    F[Modelos Generativos: Llama / Gemma] --> G[Temperatura: ≈0.0 Greedy Decoding]
    F --> H[Límite de Tokens: 5-10]
    F --> I[Contexto: 128k-256k tokens]
    F --> J[Mecanismo: Prompt restringido]

    K[Métricas de Evaluación] --> L[Micro F1-Score]
    K --> M[Macro F1-Score]
    K --> N[Matrices de Confusión]


---

## 3.5 Evaluación de los Resultados y Métricas de Rendimiento

Debido al desbalance severo de clases identificado en el análisis exploratorio (particularmente la ausencia crítica de registros neutrales en el entorno informal de redes sociales), la métrica de **exactitud (Accuracy)** fue descartada por inducir a sesgos de evaluación. Se implementaron métricas analíticas a través de la biblioteca **Scikit-Learn**:

- **Micro F1-Score:** Evalúa las contribuciones globales de falsos positivos y verdaderos positivos a nivel de instancias de datos individuales, comportándose de manera equivalente a la exactitud global en problemas multiclase.  
- **Macro F1-Score:** Calcula la media aritmética no ponderada del F1-Score de cada clase independiente:



$$
Macro\ F1 = \frac{1}{N} \sum_{i=1}^{N} (F1)_i
$$



Esta métrica actúa como el penalizador principal de nuestro estudio, castigando severamente a los modelos que fallan al predecir clases minoritarias o subrepresentadas.

Las desviaciones y la direccionalidad de los errores de predicción se mapearon y visualizaron mediante la generación automatizada de **matrices de confusión** desarrolladas con Matplotlib y Seaborn.

