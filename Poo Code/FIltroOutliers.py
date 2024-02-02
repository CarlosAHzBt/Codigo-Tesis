import open3d as o3d
import numpy as np

class FiltroOutliers:
    """
    Clase para eliminar outliers de una nube de puntos utilizando Open3D.
    """

    def __init__(self, nb_neighbors=20, std_ratio=0.5):
        """
        Inicializa el FiltroOutliers con los parámetros para la eliminación de outliers.

        Parameters:
            nb_neighbors (int): Número de vecinos a considerar para cada punto.
            std_ratio (float): Desviación estándar ratio para identificar y eliminar outliers.
        """
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def eliminar_outliers(self, pcd):
        """
        Elimina outliers de una nube de puntos usando el método de eliminación estadística.

        Parameters:
            pcd (open3d.geometry.PointCloud): Nube de puntos de la cual eliminar outliers.

        Returns:
            open3d.geometry.PointCloud: Nube de puntos filtrada sin outliers.
        """
        # Aplica el filtro estadístico para eliminar outliers
        pcd_filtrado, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                           std_ratio=self.std_ratio)
        return pcd_filtrado
