#Script para vizualizar nube de puntos  usando open3d
import open3d as o3d
import numpy as np
import json
import os
import sys


#Cargar nube de puntos
ply_path = r"ProcesamientoDeBags\BacheLey\Ply\frame_00000.ply"
pcd = o3d.io.read_point_cloud(ply_path)

#Visualizar nube de puntos
o3d.visualization.draw_geometries([pcd])