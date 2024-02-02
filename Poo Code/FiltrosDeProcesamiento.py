import open3d as o3d
import numpy as np
import json

class PointCloudFilter:
    """
    Clase para filtrar nubes de puntos utilizando Open3D. Permite cargar una nube de puntos y una región de interés (ROI)
    desde archivos y filtrar los puntos dentro de la ROI.

    """
    def __init__(self):
        """
        Inicializa el filtro de nube de puntos sin necesidad de rutas de archivos específicos.
        Esto permite usar la clase de manera más flexible, cargando diferentes archivos según sea necesario.
        """
        self.roi_data = None
        self.pcd = None

    def load_roi_data(self, json_path):
        """
        Carga la región de interés (ROI) desde un archivo JSON.

        :param json_path: Ruta al archivo JSON que contiene las coordenadas de la región de interés.
        """
        try:
            with open(json_path, 'r') as file:
                self.roi_data = json.load(file)
        except Exception as e:
            raise IOError(f"No se pudo cargar el archivo JSON: {e}")

    def load_point_cloud(self, ply_path):
        """
        Carga la nube de puntos desde un archivo PLY.

        :param ply_path: Ruta al archivo PLY que contiene la nube de puntos a filtrar.
        """
        try:
            self.pcd = o3d.io.read_point_cloud(ply_path)
        except Exception as e:
            raise IOError(f"No se pudo cargar el archivo PLY: {e}")

    def filter_points_in_roi(self, pcd):
        """
        Filtra los puntos dentro de la región de interés especificada por los datos de ROI.

        :return: Un nuevo objeto PointCloud que contiene solo los puntos dentro de la ROI.
        """

        if pcd is None or self.roi_data is None:
            raise ValueError("La nube de puntos o los datos de ROI no han sido cargados correctamente.")

        points = np.asarray(pcd.points)
        filtered_points = points[
            (points[:, 0] >= self.roi_data['x1']) & (points[:, 0] <= self.roi_data['x2']) &
            (points[:, 1] >= self.roi_data['y1']) & (points[:, 1] <= self.roi_data['y2'])
        ]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        return filtered_pcd

    @staticmethod
    def visualize_point_cloud(pcd):
        """
        Visualiza la nube de puntos proporcionada.

        :param pcd: La nube de puntos a visualizar.
        """
        o3d.visualization.draw_geometries([pcd])

    @staticmethod
    def save_point_cloud(output_path, pcd):
        """
        Guarda la nube de puntos filtrada en un archivo PLY en la ruta especificada.

        :param output_path: Ruta del archivo PLY de salida.
        :param pcd: La nube de puntos a guardar.
        """
        try:
            o3d.io.write_point_cloud(output_path, pcd)
        except Exception as e:
            raise IOError(f"No se pudo guardar la nube de puntos: {e}")
