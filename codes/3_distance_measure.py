import numpy as np
import open3d as o3d
import torch
from pytorch3d.structures import Meshes, Pointclouds
import trimesh
import os
import sys
import pandas
from pandas import Series, DataFrame
pandas.set_option('display.float_format', '{:.7f}'.format)
pandas.set_option('display.max_rows', None)
####################################################################################################
dataset_voxel_sizes = {
    'armadillo': 0.4,
    'dragon': 0.00075,
    'thai': 0.5
}
# 데이터셋 선택
dataset_name = 'dragon'
finest_voxel_size = dataset_voxel_sizes.get(dataset_name, None)
####################################################################################################
target_obj_mesh_path = f'../OriginalDataset/{dataset_name}.ply'
target_obj_mesh = trimesh.load(target_obj_mesh_path)

finest_mesh_path = f'../results/[TSDF]{dataset_name}/SingleRes/voxsize_{finest_voxel_size:.6f}/{dataset_name}_singleres=[32 32 32].ply'
finest_mesh = trimesh.load(finest_mesh_path)
####################################################################################################
mesh_bbox = target_obj_mesh.bounding_box.extents
scale_factors = 1000.0 / mesh_bbox
target_obj_mesh.apply_scale(scale_factors)
finest_mesh.apply_scale(scale_factors)

target_obj_verts = np.array(target_obj_mesh.vertices)
target_obj_faces = np.array(target_obj_mesh.faces)
finest_mesh_verts = np.array(finest_mesh.vertices)
finest_mesh_faces = np.array(finest_mesh.faces)
####################################################################################################
from source.ChamferDistance import symmetric_face_to_point_distance, custom_ChamferDistance

awmr_mesh_path = f'../results/[TSDF]{dataset_name}/SingleRes/voxsize_{finest_voxel_size:.6f}/{dataset_name}_singleres=[32 32 32].ply'
awmr_mesh = trimesh.load(awmr_mesh_path)
awmr_mesh.apply_scale(scale_factors)

awmr_verts = np.array(awmr_mesh.vertices)
awmr_faces = np.array(awmr_mesh.faces)

target_obj_verts = target_obj_verts.astype(np.float32)
target_obj_faces = target_obj_faces.astype(np.int32)
finest_mesh_verts = finest_mesh_verts.astype(np.float32)
finest_mesh_faces = finest_mesh_faces.astype(np.int32)
awmr_verts = awmr_verts.astype(np.float32)
awmr_faces = awmr_faces.astype(np.int32)

####################################################################################################
dist_original_A2B, dist_original_B2A = symmetric_face_to_point_distance(target_obj_verts, target_obj_faces, awmr_verts, awmr_faces)
dist_finest_A2B, dist_finest_B2A = symmetric_face_to_point_distance(finest_mesh_verts, finest_mesh_faces, awmr_verts, awmr_faces)

print("\n[비교하려는 대상 메시]:",os.path.basename(awmr_mesh_path))
print("[원본 obj 파일과의 distance]:",(dist_original_A2B+dist_original_B2A).item())
print("[finest mesh 파일과의 distance]:",(dist_finest_A2B+dist_finest_B2A).item())