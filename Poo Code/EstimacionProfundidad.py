import open3d as o3d
import numpy as np

class EstimacionProfundidad:
    """
    Clase para estimar la profundidad de baches en una nube de puntos filtrada.
    """
    def __init__(self,pcd,superficie_estimada):
        """
        Inicializa la clase de estimación de profundidad.
        """
        self.superficie_estimada = superficie_estimada
        self.pcd = pcd

    def estimar_profundidad_bache(self):
        """
        Estima la profundidad del bache en una nube de puntos filtrada.

        Parameters:
            pcd (open3d.geometry.PointCloud): Nube de puntos filtrada.

        Returns:
            float: La profundidad estimada del bache.
            np.array: Coordenadas del punto más profundo.
        """
        if self.pcd is None:
            raise ValueError("La nube de puntos no ha sido proporcionada.")

        puntos = np.asarray(self.pcd.points)
        idx_punto_mas_profundo = np.argmin(puntos[:, 2])
        punto_mas_profundo = puntos[idx_punto_mas_profundo]
        profundidad_agujero = self.superficie_estimada - punto_mas_profundo[2]

        return profundidad_agujero, punto_mas_profundo

