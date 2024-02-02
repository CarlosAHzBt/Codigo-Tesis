import os
import glob
import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import torch
import open3d as o3d

# Estrategia de distribución para TensorFlow
strategy = tf.distribute.MirroredStrategy()

# Verifica si CUDA está disponible
if torch.cuda.is_available():
    num_cuda_devices = torch.cuda.device_count()
    print(f"Se encontraron {num_cuda_devices} dispositivos CUDA disponibles.")
    for i in range(num_cuda_devices):
        device = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print(f"Dispositivo CUDA {i}: {device} (Compute Capability {capability[0]}.{capability[1]})")
else:
    print("CUDA no está disponible en este sistema.")

# Habilitar el uso de dos GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "GPUs físicas,", len(logical_gpus), "GPUs lógicas")
    except RuntimeError as e:
        print(e)

with strategy.scope():
    # Carga el modelo de Keras y las etiquetas
    print("[INFO] Cargando el modelo...")
    model_path = "TiempoReal/Modelos/Bache v2/keras_model.h5"
    model = keras.models.load_model(model_path)

    labels_path = "TiempoReal/Modelos/Bache v2/labels.txt"
    with open(labels_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]

    # Define la carpeta donde se encuentran los archivos .bag
    bag_files_folder = 'BagPrueba'
    bag_files = glob.glob(os.path.join(bag_files_folder, '*.bag'))
    base_folder = 'Datos_Extraccion_Prueba'

    # Función para realizar la clasificacion
    def detect_image(image, model, labels):
        input_image = cv2.resize(image, (864, 512))
        input_image = input_image.astype('float32')
        input_image /= 255
        input_image = np.expand_dims(input_image, axis=0)
        predictions = model.predict(input_image)
        max_index = np.argmax(predictions[0])
        score = predictions[0][max_index]
        label = labels[max_index]

        return (label, score)

    # Función para guardar la nube de puntos como archivo .ply
    def save_to_ply(color_frame, depth_frame, ply_folder, frame_number):
        ply_filename = f"{ply_folder}/frame_{frame_number:05d}.ply"
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        points.export_to_ply(ply_filename, color_frame)
        print(f"[INFO] Frame {frame_number:05d} guardado como {ply_filename}")

    # Definición de las funciones de procesamiento de Open3D
    def cargar_ply(nombre_archivo):
        pcd = o3d.io.read_point_cloud(nombre_archivo)
        return pcd

    def eliminar_outliers(pcd, nb_neighbors=500, std_ratio=0.5):
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd

    def segmentar_entorno(pcd, punto_central, radio):
        puntos = np.asarray(pcd.points)
        diferencias = puntos - punto_central
        distancias = np.linalg.norm(diferencias, axis=1)
        indices_cercanos = np.where(distancias < radio)[0]
        return pcd.select_by_index(indices_cercanos)

    def filtrar_puntos(pcd, z_min, z_max):
        puntos = np.asarray(pcd.points)
        filtrados = puntos[(puntos[:, 2] > z_min) & (puntos[:, 2] < z_max)]
        pcd_filtrado = o3d.geometry.PointCloud()
        pcd_filtrado.points = o3d.utility.Vector3dVector(filtrados)
        return pcd_filtrado

    def segmentar_terreno(pcd, distancia_thresh=0.05):
        plano, inliers = pcd.segment_plane(distance_threshold=distancia_thresh,
                                           ransac_n=10,
                                           num_iterations=1000)
        inlier_cloud = pcd.select_by_index(inliers)
        return inlier_cloud, plano

    def matriz_rotacion(v1, v2):
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)
        c = np.dot(v1, v2)

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        return R

    def nivelar_puntos(pcd, plano):
        A, B, C, D = plano
        norm = np.linalg.norm([A, B, C])
        vector = np.array([A, B, C]) / norm
        up = np.array([0, 0, 1])

        rot = matriz_rotacion(vector, up)
        transform = np.eye(4)
        transform[:3, :3] = rot
        return transform

    # Función para procesar cada archivo PLY
    def procesar_ply(nombre_archivo_ply):
        pcd = cargar_ply(nombre_archivo_ply)
        pcd_filtrado = filtrar_puntos(pcd, z_min=-2, z_max=2)
        pcd_terreno, plano = segmentar_terreno(pcd_filtrado)
        transformacion = nivelar_puntos(pcd_terreno, plano)
        pcd_nivelado = pcd.transform(transformacion)
        pcd = eliminar_outliers(pcd_nivelado)
        superficie_fija = 0.95
        puntos = np.asarray(pcd.points)
        idx_punto_mas_profundo = np.argmin(puntos[:, 2])
        superficie_estimada = np.median(puntos[:, 2])
        profundidad_agujero = puntos[idx_punto_mas_profundo, 2] - superficie_estimada
        punto_mas_profundo = puntos[idx_punto_mas_profundo]
        pcd_entorno = segmentar_entorno(pcd, punto_mas_profundo, radio=0.2)
        return superficie_estimada, profundidad_agujero, punto_mas_profundo, pcd_entorno

    # Procesa cada archivo .bag
    for bag_file in bag_files:
        bag_name = os.path.splitext(os.path.basename(bag_file))[0]
        images_folder = os.path.join(base_folder, bag_name, 'Imagenes')
        ply_folder = os.path.join(base_folder, bag_name, 'Ply/Extraidos')
        detections_file_path = os.path.join(base_folder, bag_name, 'detections.txt')
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(ply_folder, exist_ok=True)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_file, repeat_playback=False)
        pipeline.start(config)

        # Define el nombre del archivo de resultados
    resultados_file_path = "resultados.txt"

    # Abre el archivo de resultados en modo escritura
    with open(resultados_file_path, "w") as resultados_file:
        # Procesa cada archivo .bag
        for bag_file in bag_files:
            bag_name = os.path.splitext(os.path.basename(bag_file))[0]
            images_folder = os.path.join(base_folder, bag_name, 'Imagenes')
            ply_folder = os.path.join(base_folder, bag_name, 'Ply/Extraidos')
            detections_file_path = os.path.join(base_folder, bag_name, 'detections.txt')
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(ply_folder, exist_ok=True)

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device_from_file(bag_file, repeat_playback=False)
            pipeline.start(config)

            with open(detections_file_path, "w") as detections_file:
                frame_number = 0
                detection_frames = set()

                try:
                    while True:
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        depth_frame = frames.get_depth_frame()

                        if not color_frame or not depth_frame:
                            continue

                        color_image = np.asanyarray(color_frame.get_data())
                        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                        cv2.imwrite(f'{images_folder}/frame_{frame_number:05d}.png', color_image)

                        label, score = detect_image(color_image, model, labels)

                        if score > 0.5 and (label == "0 Bache" or label == "2 Grietas"):
                            detections_file.write(f"Frame: {frame_number:05d}, Label: {label}, Score: {score:.2f}\n")
                            detection_frames.add(frame_number)

                        frame_number += 1

                except Exception as e:
                    print(f"Se ha producido una excepción: {e}")
                finally:
                    pipeline.stop()

            pipeline = rs.pipeline()
            config = rs.config()
            rs.config.enable_device_from_file(config, bag_file, repeat_playback=True)
            pipeline.start(config)

            align_to = rs.stream.color
            align = rs.align(align_to)

            try:
                current_frame_number = 0
                while True:
                    try:
                        frames = pipeline.wait_for_frames()
                    except RuntimeError as e:
                        print(f"Error al recibir el fotograma: {e}")
                        continue

                    aligned_frames = align.process(frames)
                    aligned_depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not aligned_depth_frame or not color_frame:
                        continue

                    current_frame_number += 1

                    if current_frame_number in detection_frames:
                        save_to_ply(color_frame, aligned_depth_frame, ply_folder, current_frame_number)

                        # Llamar a la función de procesamiento de la nube de puntos
                        ply_filename = f"{ply_folder}/frame_{current_frame_number:05d}.ply"
                        superficie_estimada, profundidad_agujero, punto_mas_profundo, pcd_entorno = procesar_ply(ply_filename)

                        # Realizar cualquier operación adicional con los resultados
                        print(f"Superficie estimada: {superficie_estimada}")
                        print(f"Profundidad del agujero: {profundidad_agujero}")

                        # Agregar los resultados al archivo de resultados
                        resultados_file.write(f"Bag: {bag_name}, Frame: {current_frame_number:05d}, "
                                              f"Superficie estimada: {superficie_estimada}, "
                                              f"Profundidad del agujero: {profundidad_agujero}\n")

                    if current_frame_number > max(detection_frames, default=0):
                        break

            finally:
                pipeline.stop()

            print(f"[INFO] Detecciones completadas para {bag_name} y guardadas en '{detections_file_path}'.")