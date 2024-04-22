# KNN y KMeans Image Tagger

Este proyecto contiene dos módulos principales, `KNN.py` y `KMeans.py`, que implementan los algoritmos de aprendizaje automático KNN (k-Nearest Neighbors) y KMeans (K-Means), respectivamente. Estos algoritmos están diseñados para etiquetar imágenes según su similitud y agruparlas en clústeres, lo que es útil para tareas de análisis de imágenes y clasificación.

## Contenido del Proyecto

El proyecto consta de los siguientes archivos:

- `KNN.py`: Implementación del algoritmo KNN para clasificación de imágenes.
- `KMeans.py`: Implementación del algoritmo KMeans para agrupación de imágenes.
- `utils.py`: Módulo de utilidades que proporciona funciones auxiliares para la visualización y manipulación de datos.

## KNN (k-Nearest Neighbors)

El archivo `KNN.py` contiene la implementación del algoritmo KNN, que clasifica un punto de datos basándose en la mayoría de las clases de sus k vecinos más cercanos en el espacio de características. Las principales características de este módulo son:

- **Inicialización del Conjunto de Entrenamiento**: El método `_init_train` inicializa el conjunto de datos de entrenamiento para su procesamiento.
- **Cálculo de Vecinos Más Cercanos**: El método `get_k_neighbours` calcula los k vecinos más cercanos para cada punto de datos de prueba.
- **Votación Máxima**: El método `get_class` determina la clase de cada punto de datos de prueba basándose en la votación máxima de sus vecinos más cercanos.

## KMeans (K-Means)

El archivo `KMeans.py` implementa el algoritmo KMeans, que agrupa un conjunto de puntos de datos en k clústeres basándose en la similitud entre ellos. Las características principales de este módulo incluyen:

- **Inicialización de Centroides**: El método `_init_centroids` inicializa los centroides para la agrupación de datos.
- **Asignación de Etiquetas**: El método `get_labels` asigna etiquetas a cada punto de datos basándose en la cercanía a los centroides.
- **Convergencia del Algoritmo**: El método `converges` comprueba si el algoritmo ha convergido, es decir, si los centroides han dejado de cambiar significativamente.
- **Actualización de Centroides**: El método `get_centroids` recalcula las posiciones de los centroides basándose en los puntos de datos asignados a cada clúster.

