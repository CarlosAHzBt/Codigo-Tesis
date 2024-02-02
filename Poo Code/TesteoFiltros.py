# Asume que las clases RANSAC, PointCloudFilter, y FiltroOutliers ya están definidas como discutimos.
from Ransac import RANSAC
from FIltroOutliers import FiltroOutliers
from FiltrosDeProcesamiento import PointCloudFilter
from EstimacionDeSuperficie import EstimacionDeSuperficie
from EstimacionProfundidad import EstimacionProfundidad
import open3d as o3d
import numpy as np
import json
import time

# Paso 1: Cargar la nube de puntos original
ply_path = r"ProcesamientoDeBags\BacheLey\Ply\frame_00001.ply"
ransac = RANSAC()  # Configura según tus necesidades
print(f"Cargando nube de puntos desde: {ply_path}")
pcd = ransac.cargar_ply(ply_path)
print(f"Número de puntos en la nube de puntos: {len(pcd.points)}")



# Paso 2: Aplicar nivelación con RANSAC
pcd_terreno, plano = ransac.segmentar_terreno(pcd)
transformacion = ransac.nivelar_puntos(plano)
pcd_nivelada = pcd.transform(transformacion)
#Paso 2.5 Obtener la estimacion de superficie
estimacion = EstimacionDeSuperficie(pcd_nivelada)
superficie_estimada = estimacion.estimar_superficie_de_nube_precargada(pcd_nivelada)



# Paso 3: Filtrar la nube de puntos para mantener solo los puntos de la detección
json_path = r"ProcesamientoDeBags\BacheLey\ResultadosDeteccion\CoordenadasConvertidas\output_frame_00000.json"
pointCloudFilter = PointCloudFilter()
pointCloudFilter.load_roi_data(json_path)

# Asumiendo que ahora pasamos la nube de puntos directamente en lugar de cargarla desde un archivo
pcd_filtrada = pointCloudFilter.filter_points_in_roi(pcd_nivelada)

# Paso 4: Eliminar outliers dentro de la zona de detección
filtroOutliers = FiltroOutliers(nb_neighbors=500, std_ratio=0.5)  # Configura según tus necesidades
pcd_final = filtroOutliers.eliminar_outliers(pcd_filtrada)

# Paso opcional: Guardar la nube de puntos procesada
output_path = "procesada.ply"
PointCloudFilter.save_point_cloud(output_path, pcd_final)

#paso 5 aplicar estimacion de profundidad del bache apartir del punto mas bajo en el eje z de la nube de puntos procesada y la estimacion de superficie
profundidad_bache = EstimacionProfundidad(pcd_final, superficie_estimada)
profundidad_bache=profundidad_bache.estimar_profundidad_bache()
#La profundidad del bache estimada en metros es:
print("Profundidad del bachec estimada en metros es: ",profundidad_bache)
