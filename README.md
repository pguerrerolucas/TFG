# Sistema de Recomendación de Cursos basado en Deep Reinforcement Learning

## Descripción

Este proyecto implementa un sistema de recomendación de cursos educativos utilizando técnicas avanzadas de Aprendizaje por Refuerzo Profundo (Deep Reinforcement Learning). El sistema está diseñado para aprender las preferencias de los usuarios en el tiempo y recomendar cursos que maximicen la satisfacción a largo plazo, considerando tanto la relevancia como la diversidad de las recomendaciones.

## Modelos Implementados

Se han implementado y comparado tres algoritmos de Deep Reinforcement Learning:

- **Deep Q-Network (DQN)**: Modelo basado en valor que aprende a predecir la calidad de recomendar cada curso posible.
- **Deep Deterministic Policy Gradient (DDPG)**: Algoritmo actor-crítico para aprendizaje en espacios de acción continuos.
- **Advantage Actor-Critic (A2C)**: Enfoque híbrido que combina estimación de valor y aprendizaje de política.

## Estructura del Proyecto

```
.
├── Data/                     # Directorio de datos
│   ├── Kaggle/               # Datos originales de Coursera
│   └── Processed/            # Datos preprocesados para entrenamiento
├── models/                   # Implementaciones de modelos DRL
│   ├── a2c_model.py         # Implementación del modelo A2C
│   ├── ddpg_model.py        # Implementación del modelo DDPG
│   ├── dqn_model.py         # Implementación del modelo DQN
│   ├── utils.py             # Funciones auxiliares para los modelos
│   └── trained/             # Modelos entrenados
├── results/                  # Resultados y visualizaciones
├── preprocess_coursera_data.py  # Script para preprocesar datos
├── train_recommendation_models.py  # Entrenamiento de modelos
├── recommend_courses.py     # Sistema de recomendación interactivo
├── visualize_results.py     # Visualización de resultados
└── README.md                # Este archivo
```


## Requisitos

- Python 3.12
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Uso

### Preprocesamiento de Datos

```bash
python preprocess_coursera_data.py
```

### Entrenamiento de Modelos

```bash
python train_recommendation_models.py
```

### Generación de Recomendaciones

```bash
python recommend_courses.py
```

### Visualización de Resultados

```bash
python visualize_results.py
```

## Resultados

Los experimentos muestran que el modelo A2C ofrece el mejor rendimiento en términos de recompensa acumulada, diversidad y relevancia de las recomendaciones. Los resultados detallados están disponibles en el directorio `results/`.

## Conclusiones

Este proyecto demuestra la efectividad de los algoritmos de Deep Reinforcement Learning en el contexto de sistemas de recomendación educativa. Los modelos implementados son capaces de aprender políticas de recomendación que balancean efectivamente la relevancia y la diversidad, mejorando la experiencia del usuario en plataformas de aprendizaje online.
