#Clase que se encarga de convertir las coordenadas de pixeles a medidas reales

import cv2
import numpy as np
import json
import math
import os
from EstimacionDeSuperficie import EstimacionDeSuperficie

class ROICoordinateConverter:
    def __init__(self):
        # Valores estándar para el campo de visión y resolución
        self.fov_horizontal = 69  # FoV horizontal en grados
        self.fov_vertical = 42   # FoV vertical en grados
        self.resolucion_ancho = 864 # Resolución en píxeles (ancho) 
        self.resolucion_alto = 512  # Resolución en píxeles (alto)

    def definir_roi_y_guardar(self, ply_path, image_path, output_path, txt_path):
        """
        Define la ROI en la imagen basada en el archivo TXT y guarda las coordenadas en metros en un Json.
        """
        alturaDeCaptura = self.estimar_altura_de_captura(ply_path)

        # Cargar imagen y obtener centro
        depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        altura, anchura = depth_image.shape[:2]
        centro_x, centro_y = anchura // 2, altura // 2

        # Leer las coordenadas desde el archivo TXT
        with open(txt_path, 'r') as file:
            coords = file.readline().strip()
        x1, y1, x2, y2 = [int(float(coord)) for coord in coords[1:-1].split(', ')]
        escala_horizontal, escala_vertical = self.calcular_escala(alturaDeCaptura)
        x1_metros, y1_metros, x2_metros, y2_metros = self.convertir_pixeles_a_metros(x1, y1, x2, y2, escala_horizontal, escala_vertical, centro_x, centro_y)

        # Asegúrate de que la carpeta para coordenadas convertidas existe
        os.makedirs(output_path, exist_ok=True)

        # Construye el nombre del archivo JSON asegurándote de que no se añade ningún separador al final
        json_filename = f"output_{os.path.splitext(os.path.basename(txt_path))[0]}.json"
        output_json_path = os.path.join(output_path, json_filename)

        # Guardar las coordenadas de la ROI en el archivo JSON
        with open(output_json_path, 'w') as file:
            json.dump({'x1': x1_metros, 'y1': y1_metros, 'x2': x2_metros, 'y2': y2_metros}, file)

    def estimar_altura_de_captura(self,ply_path):
        """
        Estima la altura de captura de la nube de puntos PLY.
        """
        altura_captura = EstimacionDeSuperficie(ply_path) # Instancia de EstimacionDeSuperficie
        return altura_captura.estimar_superficie()
    
    def calcular_escala(self, altura_captura):
        """
        Calcula las escalas de conversión de píxeles a metros basadas en la altura de captura.
        """
        ancho_real = 2 * altura_captura * math.tan(math.radians(self.fov_horizontal / 2))
        alto_real = 2 * altura_captura * math.tan(math.radians(self.fov_vertical / 2))
        escala_horizontal = ancho_real / self.resolucion_ancho
        escala_vertical = alto_real / self.resolucion_alto
        return escala_horizontal, escala_vertical

    def convertir_pixeles_a_metros(self, x1_pix, y1_pix, x2_pix, y2_pix, escala_horizontal, escala_vertical, centro_x, centro_y):
        """
        Convierte coordenadas de píxeles a metros.
        """
        x1_metros = (x2_pix - centro_x) * escala_horizontal
        y1_metros = (y2_pix - centro_y ) * escala_vertical
        x2_metros = (x1_pix - centro_x ) * escala_horizontal
        y2_metros = (y1_pix - centro_y) * escala_vertical
        return x1_metros, y1_metros, x2_metros, y2_metros

# Ejemplo de uso
# Para cada carpeta generada por ProcesadorBags, utiliza ROICoordinateConverter
#procesador_bags_folder = 'ProcesamientoDeBags'
#bag_folders = [f for f in os.listdir(procesador_bags_folder) if os.path.isdir(os.path.join(procesador_bags_folder, f))]
#
#for folder in bag_folders:
#    images_folder = os.path.join(procesador_bags_folder, folder, 'Imagenes')
#    ply_folder = os.path.join(procesador_bags_folder, folder, 'Ply')
#    coords_folder = os.path.join(procesador_bags_folder, folder, 'ResultadosDeteccion', 'Coordenadas')
#    output_folder = os.path.join(procesador_bags_folder, folder, 'ResultadosDeteccion', 'CoordenadasConvertidas')
#    #Asegurandose que la carpeta de salida exista
#    os.makedirs(output_folder, exist_ok=True)
#
#    # Asumiendo que ya se ejecutó YoloDetector y que los archivos de coordenadas están en coords_folder
#    for coords_file in os.listdir(coords_folder):
#        frame_number = coords_file.split('_')[1].split('.')[0]  # Extrae el número de frame del nombre del archivo
#        ply_path = os.path.join(ply_folder, f'frame_{frame_number}.ply')
#        image_path = os.path.join(images_folder, f'frame_{frame_number}.png')
#        txt_path = os.path.join(coords_folder, coords_file)
#        output_path = os.path.join(output_folder, f'output_{frame_number}.json')
#
#        # Aplica el conversor de coordenadas
#        roi_converter = ROICoordinateConverter()
#        roi_converter.definir_roi_y_guardar(ply_path, image_path, output_path, txt_path)