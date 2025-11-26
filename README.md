# Segmentación de Imágenes con Normalized Cuts (NCuts)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositorio contiene una implementación **desde cero** del algoritmo de segmentación de imágenes **Normalized Cuts**, basado en el paper fundacional de *Shi y Malik (2000)*.

Este proyecto fue desarrollado como entrega final para la materia de **Matemáticas para Inteligencia Artificial**, con el objetivo de aplicar conceptos de Teoría de Grafos, Álgebra Lineal (Eigenvalores y Eigenvectores) y Lógica Algorítmica en un problema de Visión por Computadora.

## Descripción del Proyecto

El algoritmo modela una imagen como un grafo ponderado $G=(V, E)$, donde cada píxel es un nodo y las aristas representan la similitud entre ellos. El objetivo es particionar el grafo en segmentos perceptualmente significativos minimizando el criterio de *Normalized Cut*, resolviendo un problema de **eigenvalores generalizados**:

$$(D - W)y = \lambda D y$$

### Características Principales
* **Implementación Propia:** No se utilizan librerías de segmentación de caja negra (como `skimage.segmentation` o `networkx` para clustering). Toda la lógica de construcción de matrices y partición es propia.
* **Matrices Dispersas:** Uso eficiente de `scipy.sparse` para manejar la matriz de afinidad $W$ (que puede ser muy grande).
* **Modularidad:** Código estructurado en clases y módulos reutilizables.
* **Visualización:** Herramientas para visualizar la imagen original, el eigenvector de Fiedler y la segmentación resultante.

## Instalación

Sigue estos pasos para configurar el entorno:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/MpIA_Codigo_TuApellidoTuNombre.git](https://github.com/TU_USUARIO/MpIA_Codigo_TuApellidoTuNombre.git)
   cd MpIA_Codigo_TuApellidoTuNombre

2. **Crear y activar un entorno virtual (recomendado):**

# Windows
python -m venv env
.\env\Scripts\activate

# macOS/Linux
python3 -m venv env
source env/bin/activate

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 🛠️ Uso
Para ejecutar la segmentación en una imagen de prueba, utiliza el script main.py desde la terminal.

**Ejecución básica**
```bash
python main.py --image data/ejemplo.jpg
```

**Parámetros opcionales**
Puedes ajustar la sensibilidad del algoritmo modificando los parámetros de la matriz de afinidad:

```bash--r```: Radio de vecindad (píxeles conectados).
```bash--sigma_i```: Varianza para la intensidad de color.
```bash--sigma_x```: Varianza para la distancia espacial.

```bash
python main.py --image data/paisaje.jpg --r 5 --sigma_i 10 --sigma_x 4
```

## 📂 Estructura del Proyecto

```bash
MpIA_Codigo_ApellidoNombre/
├── data/               # Imágenes de prueba
├── src/                # Código fuente del algoritmo
│   ├── __init__.py
│   ├── ncut.py         # Lógica principal (Clase NCutSegmenter)
│   └── utils.py        # Funciones de carga y visualización
├── notebooks/          # Experimentos y análisis (Jupyter Lab)
├── main.py             # Punto de entrada (CLI)
├── requirements.txt    # Lista de dependencias
└── README.md           # Documentación
```

## 📚 Referencias
- Paper Original: Shi, J., & Malik, J. (2000). Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8), 888-905.
