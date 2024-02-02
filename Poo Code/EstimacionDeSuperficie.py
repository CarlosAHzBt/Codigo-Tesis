#Clase para obtener altura de captura sacando la media de puntos de la nube de puntos PLY
import open3d as o3d
import numpy as np
from scipy import stats


class EstimacionDeSuperficie:
    def __init__(self, archivo_ply):
        self.archivo_ply = archivo_ply

    def cargar_nube_puntos(self):
        # Cargar archivo PLY y retornar nube de puntos
        nube_puntos = o3d.io.read_point_cloud(self.archivo_ply)
        return nube_puntos

    def estimar_superficie(self):
        nube_puntos = self.cargar_nube_puntos()

        # Obtener las coordenadas de los puntos
        puntos = np.asarray(nube_puntos.points)

        # Verificar si la nube de puntos está vacía
        if puntos.size == 0:
            raise ValueError("La nube de puntos está vacía.")

        # Calcular la moda de las alturas en el eje Z
        puntos = np.asarray(nube_puntos.points)
        superficie_estimada = np.median(puntos[:, 2])
        return superficie_estimada



 