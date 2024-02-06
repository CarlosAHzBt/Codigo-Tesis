import os
import glob
import pyrealsense2 as rs
import numpy as np
import cv2

class BagFile:
    def __init__(self, bag_file_path, base_folder):
        """
        Inicializa una instancia de la clase BagFile.
        """
        self.bag_file_path = bag_file_path
        self.images_folder, self.ply_folder = self.create_output_folders(bag_file_path, base_folder)

    @staticmethod
    def create_output_folders(bag_file_path, base_folder):
        """
        Crea las carpetas necesarias para almacenar los resultados.
        """
        bag_name = os.path.splitext(os.path.basename(bag_file_path))[0]
        images_folder = os.path.join(base_folder, bag_name, 'Imagenes')
        ply_folder = os.path.join(base_folder, bag_name, 'Ply')
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(ply_folder, exist_ok=True)
        return images_folder, ply_folder

    def process_bag_file(self):
        """
        Procesa el archivo .bag extrayendo y guardando los frames.
        """
        pipeline = rs.pipeline()
        self.configure_pipeline(pipeline)
        self.extract_and_save_frames(pipeline)

    def configure_pipeline(self, pipeline):
        """
        Configura la pipeline de RealSense para leer desde el archivo .bag.
        """
        config = rs.config()
        config.enable_device_from_file(self.bag_file_path, repeat_playback=False)
        pipeline.start(config)

    def extract_and_save_frames(self, pipeline):
        """
        Extrae, procesa y guarda los frames del archivo .bag.
        """
        frame_number = 0
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame, depth_frame = self.get_frames(frames)
                if color_frame and depth_frame:
                    self.save_color_image(color_frame, frame_number)
                    self.save_depth_frame_as_ply(color_frame, depth_frame, frame_number)
                    frame_number += 1
        except RuntimeError as e:
            print(f"Error al procesar frames: {e}")

    @staticmethod
    def get_frames(frames):
        """
        Obtiene los frames de color y profundidad de un conjunto de frames.
        """
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        return color_frame, depth_frame

    def save_color_image(self, color_frame, frame_number):
        """
        Guarda la imagen RGB de un frame específico.
        """
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGBA)
        cv2.imwrite(f'{self.images_folder}/frame_{frame_number:05d}.png', color_image)

    def save_depth_frame_as_ply(self, color_frame, depth_frame, frame_number):
        """
        Guarda un frame de profundidad como un archivo .ply en la carpeta correspondiente.
        """
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        ply_filename = os.path.join(self.ply_folder, f"frame_{frame_number:05d}.ply")
        points.export_to_ply(ply_filename, color_frame)

# Uso de la clase
#bag_files_folder = 'BagPrueba'
#bag_files = glob.glob(os.path.join(bag_files_folder, '*.bag'))
#base_folder = 'Datos_Extraccion_Prueba'
#
#for bag_file in bag_files:
#    bag_processor = BagFile(bag_file, base_folder)
#    bag_processor.process_bag_file()
#