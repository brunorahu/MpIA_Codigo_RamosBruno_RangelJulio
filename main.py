import argparse
import sys
from src.ncut import NCutSegmenter
from src.utils import load_and_resize_image, plot_results

def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Implementación de Normalized Cuts (Shi & Malik)")
    
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen de entrada')
    parser.add_argument('--width', type=int, default=100, help='Ancho para redimensionar (default: 100)')
    parser.add_argument('--height', type=int, default=100, help='Alto para redimensionar (default: 100)')
    
    # Hiperparámetros del algoritmo (Valores sugeridos por el paper/experiencia)
    parser.add_argument('--r', type=int, default=5, help='Radio de vecindad (r)')
    parser.add_argument('--sigma_i', type=float, default=10.0, help='Sigma para intensidad de color')
    parser.add_argument('--sigma_x', type=float, default=4.0, help='Sigma para distancia espacial')
    
    args = parser.parse_args()

    try:
        # 1. Cargar Datos
        img = load_and_resize_image(args.image, target_size=(args.width, args.height))
        
        # 2. Inicializar Segmentador
        print(f"Iniciando NCut con r={args.r}, sigma_I={args.sigma_i}, sigma_X={args.sigma_x}")
        segmenter = NCutSegmenter(radius=args.r, sigma_I=args.sigma_i, sigma_X=args.sigma_x)
        
        # 3. Ejecutar Algoritmo
        # Obtenemos el eigenvector continuo y la máscara discreta
        eigenvec, mask = segmenter.segment(img)
        
        # 4. Visualizar
        print("Mostrando resultados...")
        plot_results(img, eigenvec, mask)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()