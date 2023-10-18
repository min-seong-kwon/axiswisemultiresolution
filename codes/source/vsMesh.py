import os
import time
import trimesh
import numpy as np
from ChamferDistance import symmetric_face_to_point_distance, symmetric_point_to_face_distance

def normalize2(coords):
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
    coords = (coords - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    #coords *= 2.
    coords *= 20. # +-10으로 사이즈를 조정함 
    return coords

def vsMesh(mesh_path1, mesh_path2):
# mesh1 로드 & 정규화
    src_mesh = trimesh.load(mesh_path1)
    src_verts = np.array(src_mesh.vertices)
    src_faces = np.array(src_mesh.faces).astype(np.int32)
    src_norm_verts = normalize2(src_verts) # 스케일 변경 -10 ~ +10
    src_norm_verts = src_norm_verts.astype(np.float32)

# mesh2 로드 & 정규화
    dec_mesh = trimesh.load(mesh_path2)
    dec_verts = np.array(dec_mesh.vertices)
    dec_faces = np.array(dec_mesh.faces).astype(np.int32)
    dec_norm_verts = normalize2(dec_verts) # 스케일 변경 -10 ~ +10
    dec_norm_verts = dec_norm_verts.astype(np.float32)


# distance 계산 
    dist_A2B, dist_B2A = symmetric_point_to_face_distance(src_norm_verts, src_faces, 
                                                      dec_norm_verts, dec_faces)

    final_dist = 0.5 * (dist_A2B+dist_B2A).item() / 10.0 # +-1 스케일 (실제로 논문에 사용할 값)

    return final_dist

# mesh_path1 = './Dataset/Armadillo.ply'
# mesh_path2 = './DracoResults/Draco_Armadillo_11.ply'

# 
# print('+-10 스케일: ', 0.5 * (dist_A2B+dist_B2A).item())
# print('+-1 스케일 (실제로 논문에 사용할 값): ', 0.5 * (dist_A2B+dist_B2A).item() / 10.0) 

