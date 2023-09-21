import numpy as np
import open3d as o3d
import torch
from pytorch3d.structures import Meshes, Pointclouds
import trimesh
import os
import sys
import pandas
from pandas import Series, DataFrame
from tqdm import tqdm
pandas.set_option('display.float_format', '{:.7f}'.format)
pandas.set_option('display.max_rows', None)
####################################################################################################
dataset_name = 'armadillo'
finest_voxel_size = 0.5
current_voxel_size = 0.75
split = 'octree'
####################################################################################################
target_obj_mesh_path = f'../OriginalDataset/{dataset_name}.ply'
target_obj_mesh = trimesh.load(target_obj_mesh_path)

finest_mesh_path = f'../results/[TSDF]{dataset_name}/SingleRes/voxsize_{finest_voxel_size:.3f}/{dataset_name}_singleres=[32 32 32].ply'
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
thres_list=[0.0005, 0.0004, 0.0003, 0.00025, 0.00022, 0.0002, 0.00018, 0.00015, 0.00012, 0.0001,
            9e-05, 8e-05, 7e-05, 5e-05, 4e-05, 3e-05, 2e-05, 1e-05]
thres_list = [str(x) for x in thres_list] 
####################################################################################################
dist_original = []
dist_finest = []
th_list = []
print(f"[dataset: {dataset_name}] [split: {split}] [voxsize: {current_voxel_size}]")
for thres in tqdm(thres_list):
    awmr_mesh_path = f'../results/[TSDF]{dataset_name}/{split}/voxsize_{current_voxel_size:.3f}/{dataset_name}_{split}_thres={thres}_pool.ply'
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
    
    dist_original_A2B, dist_original_B2A = symmetric_face_to_point_distance(target_obj_verts, target_obj_faces, awmr_verts, awmr_faces)
    dist_finest_A2B, dist_finest_B2A = symmetric_face_to_point_distance(finest_mesh_verts, finest_mesh_faces, awmr_verts, awmr_faces)
    dist_original.append((dist_original_A2B+dist_original_B2A).item())
    dist_finest.append((dist_finest_A2B+dist_finest_B2A).item())
    th_list.append(thres)

raw_data = {'thres': th_list,
        'original dist': dist_original,
        'finest dist': dist_finest}
data = DataFrame(raw_data).transpose()
data.to_excel(f'../results/[TSDF]{dataset_name}/{split}/voxsize_{current_voxel_size:.3f}/RD_{dataset_name}_{split}.xlsx', index=False)
print(data)