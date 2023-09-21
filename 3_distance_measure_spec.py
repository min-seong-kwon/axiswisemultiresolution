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
####################################################################################################
target_obj_mesh_path = '../OriginalDataset/soldier_voxelized/soldier_fr0536_qp10_qt12.obj'
target_obj_mesh = trimesh.load(target_obj_mesh_path)
target_obj_verts = np.array(target_obj_mesh.vertices)
target_obj_faces = np.array(target_obj_mesh.faces)

finest_mesh_path = '../results/[TSDF]soldier/SingleRes/voxsize_1.5/soldier_singleres=[32 32 32].ply'
finest_mesh = trimesh.load(finest_mesh_path)
finest_mesh_verts = np.array(finest_mesh.vertices)
finest_mesh_faces = np.array(finest_mesh.faces)
####################################################################################################
from source.ChamferDistance import symmetric_face_to_point_distance, custom_ChamferDistance

awmr_mesh_path = '../results/[TSDF]soldier/SingleRes/voxsize_2.0/soldier_singleres=[32 32 32].ply'
awmr_mesh = trimesh.load(awmr_mesh_path)
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

