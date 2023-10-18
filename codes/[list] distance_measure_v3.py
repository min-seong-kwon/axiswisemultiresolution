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
from source.vsMesh import vsMesh
pandas.set_option('display.float_format', '{:.7f}'.format)
pandas.set_option('display.max_rows', None)
####################################################################################################
dataset_voxel_sizes = {
    'armadillo': 0.2,
    'dragon': 0.0002,
    'thai': 0.4,
    'asia':0.2,
    'happy': 0.0002,
    'lucy': 1.0
}
dataset_name = 'asia'
finest_voxel_size = dataset_voxel_sizes.get(dataset_name, None)
split = 'octree' # 'awmr' or 'octree' or 'SingleRes'
target_obj_mesh_path = f'../OriginalDataset/{dataset_name}.ply'
###############################################################################
dist_list = []
th_list = []
filesize_list = [] 
singleres_list = [np.array([32,32,32]), np.array([16,16,16]), np.array([8,8,8])]
thres_list = np.linspace(2e-9, 2e-6, 10) # for asia
# thres_list = np.logspace(0, 1.6, 15) * 5e-8 # for others
thres_list = [str(round(th*1e6,3)) for th in thres_list]
###############################################################################
print(f"[dataset: {dataset_name}] [split: {split}] [voxsize: {finest_voxel_size}]")

if split=='SingleRes':
    for resolution in tqdm(singleres_list):
        singleres_mesh_path = f'../0926_results/[TSDF]{dataset_name}/SingleRes/voxsize_{finest_voxel_size:.6f}/filled/{dataset_name}_singleres={resolution}_filled.ply'
        file_size = os.path.getsize(singleres_mesh_path) / 1024
        dist = vsMesh(target_obj_mesh_path, singleres_mesh_path)
        
        dist_list.append(dist)
        filesize_list.append(file_size)
        th_list.append(resolution)
else:
    for thres in tqdm(thres_list):
        awmr_mesh_path = f'../0926_results/[TSDF]{dataset_name}/{split}/voxsize_{finest_voxel_size:.6f}/filled/{dataset_name}_{split}_thres={thres}_filled.ply'
        file_size = os.path.getsize(awmr_mesh_path) / 1024
        dist = vsMesh(target_obj_mesh_path, awmr_mesh_path)
        
        dist_list.append(dist)
        filesize_list.append(file_size)
        th_list.append(thres)

raw_data = {'thres': th_list,
            'file size': filesize_list,
            'Chamfer distance': dist_list}
data = DataFrame(raw_data).transpose()
print(data)
data.to_excel(f'../0926_results/[TSDF]{dataset_name}/{split}/voxsize_{finest_voxel_size:.6f}/Distortion_filled_v2_{dataset_name}_{split}.xlsx', index=True)
###############################################################################