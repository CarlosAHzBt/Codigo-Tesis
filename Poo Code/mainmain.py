import time
import os
import glob
import json
from ProcesadorBags import ProcesadorBags
from Detector import Detector
from ConvertirDeCoordenadas import ROICoordinateConverter
from FiltrosDeProcesamiento import PointCloudFilter

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
        output_path = os.path.join(output_folder, f'output_{frame_number}.json')
        roi_converter.definir_roi_y_guardar(ply_path, image_path, output_path, txt_path)
    print("coordenadas Convertidas---------------------------------")



def filtrar_np_por_coordenadas(coor_conv_path, ply_path, output_path):
    #coor_conv_path es la ruta al archivo JSON que contiene las coordenadas de la región de interés.
    #ply_path es la ruta al archivo PLY que contiene la nube de puntos a filtrar.
    #output_path es la ruta donde se guardará el punto cloud filtrado
    coor_conv = PointCloudFilter
    coor_conv.load_roi_data(coor_conv_path)
    coor_conv.load_point_cloud(ply_path)
    filtered_pcd = coor_conv.filter_points_in_roi()
    coor_conv.save_point_cloud(output_path,filtered_pcd)


def elimador_np_sin__coordenadas(pointclouds_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for path in pointclouds_paths:
        with open(path, 'r') as file:
            data = json.load(file)
            if data['x1'] is not None:
                os.rename(path, os.path.join(output_folder, os.path.basename(path)))
    print("Archivos sin coordenadas eliminados---------------------------------")

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
        print("Termine Un BAG")

    print("Ya Acabe")