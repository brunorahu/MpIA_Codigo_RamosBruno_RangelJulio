import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_resize_image(path, target_size=(100, 100)):
    """
    Carga una imagen, la convierte a RGB y la redimensiona para
    que el cálculo de eigenvalores sea viable en tiempo razonable.
    
    Argumentos:
        path (str): Ruta al archivo de imagen.
        target_size (tuple): (ancho, alto) nuevos. El paper usa aprox 100x120.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la imagen en: {path}")
    
    # Cargar en BGR (OpenCV default)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo decodificar la imagen: {path}")

    # Convertir a RGB (para Matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar (Importante para rendimiento)
    # Imágenes grandes colapsarán la RAM.
    img_resized = cv2.resize(img_rgb, target_size)
    
    print(f"Imagen cargada y redimensionada a {img_resized.shape}")
    return img_resized

def plot_results(original, eigenvector, mask):
    """
    Genera una figura comparativa para el reporte.
    
    Argumentos:
        original: Imagen original RGB.
        eigenvector: El Fiedler Vector (valores reales continuos).
        mask: La segmentación binaria final.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Imagen Original
    axes[0].imshow(original)
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')
    
    # 2. Eigenvector (Heatmap)
    # Esto es vital para mostrar la "Solución Relajada" (valores continuos)
    im = axes[1].imshow(eigenvector, cmap='jet')
    axes[1].set_title("Fiedler Eigenvector (Continuo)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Segmentación (Superpuesta)
    # Crear una imagen donde el segmento A se ve normal y el B oscurecido
    seg_overlay = original.copy()
    seg_overlay[~mask] = (seg_overlay[~mask] * 0.3).astype(np.uint8) # Oscurecer fondo
    
    axes[2].imshow(seg_overlay)
    axes[2].set_title("Segmentación (Normalized Cut)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()