import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
import time

class NCutSegmenter:
    """
    Implementación del algoritmo Normalized Cuts (Shi & Malik, 2000).
    Segmentación de imágenes basada en partición de grafos espectrales.
    """
    
    def __init__(self, radius=5, sigma_I=10.0, sigma_X=4.0):
        """
        Args:
            radius (int): Radio r para la vecindad de conexión de píxeles.
            sigma_I (float): Parámetro de escala para la similitud de intensidad/color.
            sigma_X (float): Parámetro de escala para la proximidad espacial.
        """
        self.radius = radius
        self.sigma_I = sigma_I
        self.sigma_X = sigma_X
        
    def _get_features(self, img):
        """
        Normaliza la imagen y aplana los canales para facilitar cálculos vectoriales.
        """
        # Asegurar float y rango [0, 1]
        img = img.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        N = h * w
        
        # Si es escala de grises, añadir dimensión de canal
        if len(img.shape) == 2:
            img = img.reshape((h, w, 1))
            
        # Aplanar: Array de (N, canales)
        # F(i) vector de características [cite: 335]
        flat_img = img.reshape((N, -1))
        return flat_img, h, w, N

    def _compute_weights(self, img):
        """
        Construye la matriz de afinidad W dispersa (sparse).
        Calcula w_ij = e^(-||F(i)-F(j)||^2 / sigma_I^2) * e^(-||X(i)-X(j)||^2 / sigma_X^2)
        Referencia: Ecuación (11) del paper.
        """
        features, h, w, N = self._get_features(img)
        
        # Listas para construir la matriz dispersa formato COO (Coordinate list)
        rows = []
        cols = []
        data = []
        
        # Coordenadas espaciales (X, Y) para calcular distancias espaciales vectorizadas
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1) # (N, 2)
        
        print(f"Construyendo grafo para imagen {h}x{w} (N={N} nodos)...")
        start_time = time.time()

        # Iteramos sobre la ventana de vecindad definida por el radio r
        # En lugar de iterar por cada píxel (lento), iteramos por "desplazamiento" (rápido)
        r = int(self.radius)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                # 1. Ignorar el píxel consigo mismo y fuera del radio circular
                if dx == 0 and dy == 0:
                    continue
                if dx**2 + dy**2 > r**2: # Restricción circular ||X(i)-X(j)|| < r 
                    continue
                
                # 2. Identificar vecinos válidos (que no se salgan de la imagen)
                # Desplazamos la imagen 'dy' y 'dx' para alinear píxel i con j
                # Indices originales (i)
                idx_i = np.arange(N).reshape(h, w)
                
                # Recorte para simular el desplazamiento
                # Ejemplo: si dx=1, comparamos img[:, :-1] con img[:, 1:]
                i_slice = (slice(max(0, -dy), min(h, h - dy)), 
                           slice(max(0, -dx), min(w, w - dx)))
                j_slice = (slice(max(0, dy), min(h, h + dy)), 
                           slice(max(0, dx), min(w, w + dx)))
                
                # Obtener índices aplanados de los pares (i, j)
                indices_i = idx_i[i_slice].ravel()
                indices_j = idx_i[j_slice].ravel()
                
                if len(indices_i) == 0:
                    continue

                # 3. Calcular Distancia de Color/Intensidad ||F(i) - F(j)||^2
                diff_feat = features[indices_i] - features[indices_j]
                dist_f_sq = np.sum(diff_feat**2, axis=1)
                
                # 4. Calcular Distancia Espacial ||X(i) - X(j)||^2
                # Como iteramos por (dx, dy) fijo, la distancia es constante para este loop
                dist_x_sq = dx**2 + dy**2
                
                # 5. Calcular Pesos Combinados (Eq. 11)
                # w = exp(-dist_F/s_I) * exp(-dist_X/s_X)
                weights = np.exp(-dist_f_sq / self.sigma_I**2) * np.exp(-dist_x_sq / self.sigma_X**2)
                
                # 6. Almacenar aristas
                # Notar que el grafo es no dirigido, W debe ser simétrica.
                # Al iterar dx, dy negativos y positivos, cubrimos i->j y j->i.
                rows.append(indices_i)
                cols.append(indices_j)
                data.append(weights)
        
        # Concatenar todos los datos
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        
        # Crear matriz dispersa CSR (Compressed Sparse Row) para eficiencia aritmética
        W = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        
        print(f"Grafo construido en {time.time()-start_time:.2f}s. Aristas: {W.nnz}")
        return W

    def segment(self, img, num_cuts=1):
        """
        Ejecuta el pipeline completo de segmentación.
        
        Args:
            img: Imagen de entrada (H, W) o (H, W, C).
            num_cuts: Número de cortes (eigenvectores) a calcular.
            
        Returns:
            eigenvec: El segundo eigenvector más pequeño (Fiedler vector).
            mask: La máscara de segmentación binaria.
        """
        # 1. Construir matriz de pesos W
        W = self._compute_weights(img)
        N = W.shape[0]
        
        # 2. Construir matriz de Grados D (Diagonal)
        # d_i = sum_j w_ij [cite: 133]
        d = np.array(W.sum(axis=1)).flatten()
        D = sparse.diags(d)
        
        # 3. Construir el Laplaciano Normalizado
        # Resolver (D - W)y = lambda D y es equivalente a resolver:
        # D^(-1/2) (D - W) D^(-1/2) z = lambda z  
        # donde z = D^(1/2) y
        
        # Calcular D^(-1/2) con cuidado de división por cero
        with np.errstate(divide='ignore'):
            d_inv_sqrt = 1.0 / np.sqrt(d)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0 # Manejar nodos aislados
        
        D_inv_sqrt = sparse.diags(d_inv_sqrt)
        
        # L = D - W
        # L_norm = D_inv_sqrt @ (D - W) @ D_inv_sqrt
        # Simplificando: L_norm = I - D_inv_sqrt @ W @ D_inv_sqrt
        I = sparse.identity(N, dtype=np.float32)
        L_norm = I - D_inv_sqrt @ W @ D_inv_sqrt
        
        print("Calculando eigenvalores (esto puede tardar)...")
        # 4. Resolver Eigenvalores
        # Buscamos los k+1 eigenvalores más pequeños ('SM' = Smallest Magnitude)
        # El primero es 0 (trivial), el segundo es el Fiedler vector [cite: 178]
        # Usamos eigsh de scipy que usa el algoritmo Lanczos [cite: 229]
        eigenvals, eigenvecs = eigsh(L_norm, k=num_cuts+1, which='SM')
        
        # 5. Extraer el segundo eigenvector (Fiedler vector)
        # eigenvecs[:, 0] es el trivial (constante), eigenvecs[:, 1] es el de corte
        fiedler_z = eigenvecs[:, 1]
        
        # 6. Transformar de vuelta z -> y
        # y = D^(-1/2) z [cite: 175]
        fiedler_y = d_inv_sqrt * fiedler_z
        
        # 7. Particionar
        # Usamos la mediana como punto de corte para binarizar [cite: 244]
        threshold = np.median(fiedler_y)
        mask = fiedler_y > threshold
        
        # Redimensionar a forma de imagen original
        h, w = img.shape[:2]
        return fiedler_y.reshape(h, w), mask.reshape(h, w)