#Clase que se encarga de Detectar los baches en la imagen y devolver las coordenadas de los mismos

from ultralytics import YOLO
from PIL import Image
import os
import json

class Detector:
    def __init__(self, source_folder, coords_folder):
        """
        Inicializa una instancia de YoloDetector.

        :param model_path: Ruta al modelo de detección de YOLO.
        :param source_folder: Carpeta donde se encuentran las imágenes RGB.
        :param coords_folder: Carpeta donde se guardarán las coordenadas detectadas.
        """
        self.model = YOLO("Poo Code/ModeloDeteccion/best.pt")
        self.source_folder = source_folder
        self.coords_folder = coords_folder

        os.makedirs(self.coords_folder, exist_ok=True)

    def process_images(self):
        """
        Procesa todas las imágenes en la carpeta de origen.
        """
        
        for filename in os.listdir(self.source_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_image(filename)

        print("Processing complete, bounding boxes saved to:", self.coords_folder)

    def process_image(self, filename):
        """
        Procesa una imagen individual y guarda las detecciones.
        """
        image_path = os.path.join(self.source_folder, filename)
        image = Image.open(image_path)

        results = self.model.predict(source=image, conf=0.5)

        try:
            detections = results[0].boxes.xyxy  # Get detection bounding boxes
            if len(detections) > 0:
                detections_list = [detection.tolist() for detection in detections]

                coords_file = filename.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
                with open(os.path.join(self.coords_folder, coords_file), 'w') as f:
                    for det in detections_list:
                        f.write(f'{det}\n')
            else:
                print(f"No detections in {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Ejemplo de uso
# Para cada carpeta generada por ProcesadorBags, crea una instancia de YoloDetector y procesa las imágenes
#bag_files_folder = 'ProcesamientoDeBags'
#bag_folders = [f for f in os.listdir(bag_files_folder) if os.path.isdir(os.path.join(bag_files_folder, f))]
#
#for folder in bag_folders:
#    images_folder = os.path.join(bag_files_folder, folder, 'Imagenes')
#    coords_folder = os.path.join(bag_files_folder, folder, 'ResultadosDeteccion/Coordenadas')
#    
#    detector = Detector(images_folder, coords_folder)
#    detector.process_images()
#