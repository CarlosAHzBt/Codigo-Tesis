#Clase que aplica Ransac a las nubes de puntos que si tienen coordenadas en el ROI para nivel el terreno 
import numpy as np
import open3d as o3d

class RANSAC:
    """
    Clase RANSAC que realiza la detección y nivelación del plano del terreno en una nube de puntos preprocesada.

    Attributes:
        distancia_thresh (float): Umbral de distancia para el algoritmo RANSAC.
    """

    def __init__(self):
        """
        Inicializa la clase RANSAC con el umbral de distancia para el algoritmo RANSAC.

        Parameters:
            distancia_thresh (float): Umbral de distancia para el algoritmo RANSAC.
        """
        self.distancia_thresh = 0.05
    def cargar_ply(self,nombre_archivo):
        pcd = o3d.io.read_point_cloud(nombre_archivo)
        return pcd
    def filtrar_puntos(pcd, z_min, z_max):
        puntos = np.asarray(pcd.points)
        filtrados = puntos[(puntos[:, 2] > z_min) & (puntos[:, 2] < z_max)]
        pcd_filtrado = o3d.geometry.PointCloud()
        pcd_filtrado.points = o3d.utility.Vector3dVector(filtrados)
        return pcd_filtrado 
    
    def segmentar_terreno(self, pcd):
        """
        Segmenta el terreno de una nube de puntos utilizando el algoritmo RANSAC.

        Parameters:
            pcd (open3d.geometry.PointCloud): Nube de puntos preprocesada para segmentar el terreno.

        Returns:
            open3d.geometry.PointCloud: Nube de puntos correspondiente al terreno segmentado.
            np.array: Coeficientes del plano del terreno detectado.
        """
        plano, inliers = pcd.segment_plane(distance_threshold=self.distancia_thresh,
                                           ransac_n=3,
                                           num_iterations=1000)
        inlier_cloud = pcd.select_by_index(inliers)
        return inlier_cloud, plano

    def nivelar_puntos(self, plano):
        """
        Nivela una nube de puntos basándose en el plano del terreno detectado.

        Parameters:
            plano (np.array): Coeficientes del plano del terreno.

        Returns:
            np.array: Matriz de transformación para nivelar la nube de puntos.
        """
        A, B, C, D = plano
        norm = np.linalg.norm([A, B, C])
        vector_plano = np.array([A, B, C]) / norm
        up_vector = np.array([0, 0, 1])
        rot = self.matriz_rotacion(vector_plano, up_vector)
        transform = np.eye(4)
        transform[:3, :3] = rot
        return transform

    @staticmethod
    def matriz_rotacion(v1, v2):
        """
        Calcula la matriz de rotación para alinear v1 con v2.

        Parameters:
            v1 (np.array): Vector inicial.
            v2 (np.array): Vector final.

        Returns:
            np.array: Matriz de rotación que alinea v1 con v2.
        """
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)
        c = np.dot(v1, v2)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        return R
