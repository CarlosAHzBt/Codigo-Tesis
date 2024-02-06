import time
import os
import glob
import json
import open3d as o3d
import numpy as np
import json
from ProcesadorBags import ProcesadorBags
from Detector import Detector
from ConvertirDeCoordenadas import ROICoordinateConverter
from FiltrosDeProcesamiento import PointCloudFilter
from Ransac import RANSAC
from EstimacionDeSuperficie import EstimacionDeSuperficie
from EstimacionProfundidad import EstimacionProfundidad
from FIltroOutliers import FiltroOutliers


def obtener_directorios_de_bags(base_folder):
    return [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

def procesar_archivos_bag():
    base_folder = 'ProcesamientoDeBags'
    ProcesadorBags('bags').process_bag_files()
    return base_folder

def pausar_procesamiento(tiempo_en_segundos):
    time.sleep(tiempo_en_segundos)

def detectar_baches_en_imagenes_de_carpeta(source_folder, coords_folder):
    Detector(source_folder, coords_folder).process_images()

def convertir_coordenadas_en_carpeta(images_folder, ply_folder, coords_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    roi_converter = ROICoordinateConverter()
    for coords_file in os.listdir(coords_folder):
        frame_number = coords_file.split('_')[1].split('.')[0]
        ply_path = os.path.join(ply_folder, f'frame_{frame_number}.ply')
        image_path = os.path.join(images_folder, f'frame_{frame_number}.png')
        txt_path = os.path.join(coords_folder, coords_file)
        output_path = output_folder  # Cambiado aquí para usar solo el directorio de salida
        roi_converter.definir_roi_y_guardar(ply_path, image_path, output_path, txt_path)
    print("Coordenadas convertidas---------------------------------")


#

#Funcion para checar en la carpeta de coordenadas convertidas si hay archivos y buscar los que tengan el mismo numero de frame que el ply en la carpeta de ply
def checar_archivos_en_carpeta_de_coordenadas_convertidas(ply_folder, coords_folder):
    ply_files = glob.glob(os.path.join(ply_folder, '*.ply'))
    coords_files = glob.glob(os.path.join(coords_folder, '*.json'))
    ply_frame_numbers = [os.path.basename(f).split('_')[1].split('.')[0] for f in ply_files]
    coords_frame_numbers = [os.path.basename(f).split('_')[2].split('.')[0] for f in coords_files]
    return [f for f in ply_frame_numbers if f in coords_frame_numbers]

#Funcion para aplicar Ransac a las nubes de puntos que se regresen de la funcion de checar_archivos_en_carpeta_de_coordenadas_convertidas
def aplicar_ransac_a_nubes_de_puntos(folder_bag, ply_folder, coords_folder, frame_numbers):
    ransac = RANSAC()  # Asume configuración interna adecuada
    try:
        for frame_number in frame_numbers: 
            ply_path = os.path.join(ply_folder, f'frame_{frame_number}.ply')
            coords_path = os.path.join(coords_folder, f'output_frame_{frame_number}.json')
            pcd = ransac.cargar_ply(ply_path)
            pcd_terreno, plano = ransac.segmentar_terreno(pcd)
            transformacion = ransac.nivelar_puntos(plano)
            pcd_nivelada = pcd.transform(transformacion)
            estimacion = EstimacionDeSuperficie(pcd_nivelada)
            superficie_estimada = estimacion.estimar_superficie_de_nube_precargada(pcd_nivelada)
            pointCloudFilter = PointCloudFilter()
            pointCloudFilter.load_roi_data(coords_path)
            pcd_filtrada = pointCloudFilter.filter_points_in_roi(pcd_nivelada)
            filtroOutliers = FiltroOutliers(nb_neighbors=500, std_ratio=0.5)
            pcd_final = filtroOutliers.eliminar_outliers(pcd_filtrada)
            # Guardar la nube de puntos procesada
            guardar_nube_de_puntos_procesada(pcd_final, folder_bag, frame_number)
            estimacion_profundidad, punto_mas_profundo = EstimacionProfundidad(pcd_final,superficie_estimada).estimar_profundidad_bache()
            print(f"La profundidad del bache estimada en metros es: {estimacion_profundidad}, en el frame {frame_number} de la nube de puntos {ply_path} y las coordenadas {coords_path}")
    except Exception as e:
        print(f"No hay detecciones en este archivo: {e}")
def guardar_nube_de_puntos_procesada(pcd_final, folder_bag, frame_number):
    ply_procesados_folder = os.path.join('ProcesamientoDeBags', folder_bag, 'PlyProcesados')
    os.makedirs(ply_procesados_folder, exist_ok=True)
    output_path = os.path.join(ply_procesados_folder, f'frame_{frame_number}_procesada.ply')
    o3d.io.write_point_cloud(output_path, pcd_final)

#SECCION DE FILTROS DE PROCESAMIENTO
    
#Funcion para aplicar filtro Ransac a las nubes de puntos que si tienen coordenadas en el ROI
def aplicar_nivelacion_ransac(pcd):
    ransac = RANSAC()  # Asume configuración interna adecuada
    pcd_terreno, plano = ransac.segmentar_terreno(pcd)
    transformacion = ransac.nivelar_puntos(plano)
    pcd_nivelada = pcd.transform(transformacion)
    
    estimacion = EstimacionDeSuperficie(pcd_nivelada)
    superficie_estimada = estimacion.estimar_superficie_de_nube_precargada(pcd_nivelada)
    
    return pcd_nivelada, superficie_estimada

def filtrar_puntos_deteccion(pcd_nivelada, json_path):
    pointCloudFilter = PointCloudFilter()
    pointCloudFilter.load_roi_data(json_path)
    pcd_filtrada = pointCloudFilter.filter_points_in_roi(pcd_nivelada)
    return pcd_filtrada

def eliminar_outliers(pcd_filtrada):
    filtroOutliers = FiltroOutliers(nb_neighbors=500, std_ratio=0.5)  # Configura según tus necesidades
    pcd_final = filtroOutliers.eliminar_outliers(pcd_filtrada)
    return pcd_final

#Funcion para aplicar estimacion de profundidad del bache apartir del punto mas bajo en el eje z de la nube de puntos procesada y la estimacion de superficie
def aplicar_estimacion_profundidad(pcd_final, superficie_estimada):
    estimacion_profundidad = EstimacionProfundidad(pcd_final, superficie_estimada)
    profundidad_bache = estimacion_profundidad.estimar_profundidad_bache()
    return profundidad_bache



#Funcion principal
def principal():
    base_folder = procesar_archivos_bag()
    #pausar_procesamiento(60)  # Pausa para asegurar que los archivos .bag han sido procesados
    bag_folders = obtener_directorios_de_bags(base_folder)
    print("ahora toca detectar baches en las imagenes----------------------")
    for folder in bag_folders:
        source_folder = os.path.join(base_folder, folder, 'Imagenes')
        coords_folder = os.path.join(base_folder, folder, 'ResultadosDeteccion/Coordenadas')
        detectar_baches_en_imagenes_de_carpeta(source_folder, coords_folder)
        print("Coordenadas de un bag obtenidas ---------------------------------")
        images_folder = os.path.join(base_folder, folder, 'Imagenes')
        ply_folder = os.path.join(base_folder, folder, 'Ply')
        output_folder = os.path.join(base_folder, folder, 'ResultadosDeteccion', 'CoordenadasConvertidas')
        convertir_coordenadas_en_carpeta(images_folder, ply_folder, coords_folder, output_folder)
        #CoordenadasConvertidas
        print("Termine de convertir coordenadas---------------------------------")
        #Revisar que archivos hay en la carpeta de coordenadas convertidas y aplicar ransac a las nubes de puntos que si tienen correspondiente en la carpeta de ply
        frame_numbers = checar_archivos_en_carpeta_de_coordenadas_convertidas(ply_folder, output_folder)
        if len(frame_numbers) == 0:       #Si la longitud de frame_numbers = 0, no hay archivos en la carpeta de coordenadas convertidas
            print("No hay archivos en la carpeta de coordenadas convertidas")
            continue
        else:
            try:
                aplicar_ransac_a_nubes_de_puntos(folder,ply_folder, output_folder, frame_numbers)
            except Exception as e:
                print(f"Error al aplicar RANSAC: {e}")
        print("Termine Un BAG")

    print("Ya Acabe")