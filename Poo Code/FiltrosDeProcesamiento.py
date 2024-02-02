#Clase que contiene los filtros de procesamiento de la nube de puntos.
#-Mantener ROI de interes en la NP del bache
import open3d as o3d
import numpy as np
import json
class PointCloudFilter:
    def __init__(self, json_path, ply_path):
        """
        Inicializa el filtro de nube de puntos con las rutas a los archivos de la región de interés y la nube de puntos.

        :param json_path: Ruta al archivo JSON que contiene las coordenadas de la región de interés.
        :param ply_path: Ruta al archivo PLY que contiene la nube de puntos a filtrar.
        """
        self.json_path = json_path
        self.ply_path = ply_path
        self.roi_data = None
        self.pcd = None

    def load_roi_data(self):
        """
        Carga la región de interés (ROI) desde el archivo JSON.
        """
        try:
            with open(self.json_path, 'r') as file:
                self.roi_data = json.load(file)
        except Exception as e:
            raise IOError(f"No se pudo cargar el archivo JSON: {e}")

    def load_point_cloud(self):
        """
        Carga la nube de puntos desde el archivo PLY.
        """
        try:
            self.pcd = o3d.io.read_point_cloud(self.ply_path)
        except Exception as e:
            raise IOError(f"No se pudo cargar el archivo PLY: {e}")

    def filter_points_in_roi(self):
        """
        Filtra los puntos dentro de la región de interés especificada por los datos de ROI.

        :return: Un nuevo objeto PointCloud que contiene solo los puntos dentro de la ROI.
        """
        if self.pcd is None:
            raise ValueError("La nube de puntos no ha sido cargada.")
        if self.roi_data is None:
            raise ValueError("Los datos de ROI no han sido cargados.")

        points = np.asarray(self.pcd.points)
        # Asegúrate de que las coordenadas ROI están en el mismo sistema de coordenadas que la nube de puntos
        filtered_points = points[
            (points[:, 0] >= self.roi_data['x1']) & (points[:, 0] <= self.roi_data['x2']) &
            (points[:, 1] >= self.roi_data['y1']) & (points[:, 1] <= self.roi_data['y2'])
        ]

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        return filtered_pcd

    def visualize_point_cloud(self, pcd):
        """
        Visualiza la nube de puntos proporcionada.

        :param pcd: La nube de puntos a visualizar.
        """
        o3d.visualization.draw_geometries([pcd])

    def save_point_cloud(self, output_path, pcd):
        """
        Guarda la nube de puntos filtrada en un archivo PLY en la ruta especificada.

        :param output_path: Ruta del archivo PLY de salida.
        :param pcd: La nube de puntos a guardar.
        """
        try:
            o3d.io.write_point_cloud(output_path, pcd)
        except Exception as e:
            raise IOError(f"No se pudo guardar la nube de puntos: {e}")
        
#-Filtro de eliminacion de outliers
#-Filtro de Ransac