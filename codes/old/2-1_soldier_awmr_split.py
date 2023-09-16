
"""
Created on Sat Oct  8 21:52:04 2022

splitting a 8*8*8 volume unit into axis-wise multi-resolution blocks

@author: podragonwer
원래 4*4*4짜리 볼륨 유닛에서 작동하던 split_axiswise.py가 있었는데
볼륨 유닛을 8*8*8로 바꾸기로 결정하면서 코드도 아래와 같이 수정되었습니다.

"""

import os
import numpy as np
from tqdm import tqdm
import pickle

from source.ConfigSettings import base_volume_unit_dim, base_voxel_size, sdf_trunc
from source.VolumeUnit import VolumeUnit
from source.AWMRblock8x8 import VolumeUnit8x8
from source.AWMR_utils import pad_awmr_from_original, split_until_thres, mesh_whole_block, make_mesh, mesh_whole_block_singularize
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock # TODO

from utils.TSDFDataset import TSDFDataset
from utils.multiResUtil import assignRes, reduceRes, unitComplexity

from skimage import measure
import open3d as o3d

from utils.evalUtils import customChamfer, customHausdorff, key_is_in
import pandas as pd

#voxel_32 = np.float32(0.5)
#voxel_16 = np.float32(1.0)
#voxel_8 = np.float32(2.0)

dataset_name = 'soldier'
dataset_name_32 = f'{dataset_name}_32'
dataset_name_16 = f'{dataset_name}_16'
dataset_name_8 = f'{dataset_name}_8'

thres = 0.0002

volume_origin = np.load(f'../mpeg_vunits/{dataset_name}_vunits/volume_origin_{dataset_name_32}.npy')

#meshpath = './vunits/meshes/axisres'
#if not os.path.exists(meshpath):
#    os.makedirs(meshpath, exist_ok=True)


print("loading dataset...")

dataset_32 = TSDFDataset(scenes=[dataset_name_32],
                        dataset_name=dataset_name,
                        volume_origin=volume_origin,
                        ETRI=False)
print(f"{dataset_name_32} dataset loaded!")

dataset_16 = TSDFDataset(scenes=[dataset_name_16],
                        dataset_name=dataset_name,
                        volume_origin=volume_origin,
                        ETRI=False)
print(f"{dataset_name_16} dataset loaded!")

dataset_8 = TSDFDataset(scenes=[dataset_name_8],
                        dataset_name=dataset_name,
                        volume_origin=volume_origin,
                        ETRI=False)
print(f"{dataset_name_8} dataset loaded!")

volume_units_32 = {}
for k in dataset_32.tsdf_blocks.keys():
    new_k = (dataset_name,k[1],k[2],k[3])
    volume_units_32[new_k] = VolumeUnit(volume_unit_dim=32)
    volume_units_32[new_k].D = dataset_32.tsdf_blocks[k]
    
volume_units_16 = {}
for k in dataset_16.tsdf_blocks.keys():
    new_k = (dataset_name,k[1],k[2],k[3])
    volume_units_16[new_k] = VolumeUnit(volume_unit_dim=16)
    volume_units_16[new_k].D = dataset_16.tsdf_blocks[k]
    
volume_units_8 = {}
for k in dataset_8.tsdf_blocks.keys():
    new_k = (dataset_name,k[1],k[2],k[3])
    volume_units_8[new_k] = VolumeUnit(volume_unit_dim=8)
    volume_units_8[new_k].D = dataset_8.tsdf_blocks[k]

p1 = np.array([273.175598, 986.804321, 209.557205])
p2 = np.array([273.175598, 986.804321, 209.557205])
# p2 = np.array([218.424957, 564.081482, 393.316040])
base_voxel_size = 2
vunit_start = np.floor((p1 - np.squeeze(volume_origin))/(base_voxel_size*base_volume_unit_dim))
vunit_end = np.floor((p2 - np.squeeze(volume_origin))/(base_voxel_size*base_volume_unit_dim))
print(volume_origin, '        -         ', vunit_start, '        -         ', vunit_end)
awmr_tsdfs = {}
for k in tqdm(volume_units_32.keys()):
    if not key_is_in(k, vunit_start, vunit_end): # DEBUG
        continue
    if len(k)==3:
        print("your initial key length is 3, please modify code")
        k = (dataset_name_32, k[0], k[1], k[2])

    pad_tsdf = pad_awmr_from_original(volume_units_8[k].D,
                                    volume_units_16=volume_units_32,
                                    volume_units_8=volume_units_16,
                                    volume_units_4=volume_units_8,
                                    axisres=np.array([8,8,8]), 
                                    unit_index=k, 
                                    start_point=np.array([0,0,0]),
                                    for_mask=True)
    
    if (np.min(pad_tsdf)*np.max(pad_tsdf) >= 0) and (np.min(volume_units_32[k].D)*np.max(volume_units_32[k].D) >= 0):
        continue
    
    awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8), # edit: 4x4x4 -> 8x8x8
                                unit_index=k,
                                tsdf=volume_units_8[k].D)
    
    split_until_thres(awmr_tsdfs[k], 
                     thres,
                     volume_units_32,
                     volume_units_16,
                     volume_units_8,
                     np.array((8,8,8)),  # edit: 4x4x4 -> 8x8x8
                     k, 
                     start_point=np.array((0,0,0)),
                     max_res=32,
                     for_train=True)

#resultpath = f"Standford_Dataset/NewResults_{dataset_name}/AWMR"
resultpath = f'../results/[TSDF]{dataset_name}/AWMR/pklfiles'
if not os.path.exists(resultpath):
    os.makedirs(resultpath, exist_ok=True)
with open(resultpath + f"/{dataset_name}_awmr_Chamf={thres}.pkl", "wb") as f: 
    pickle.dump(awmr_tsdfs, f)
    
# for k in tqdm(awmr_tsdfs.keys()):
#     block_mesh = mesh_whole_block(awmr_tsdfs[k], 
#                         unit_index=k, 
#                         awmr_dict=awmr_tsdfs,
#                         node=None,
#                         volume_origin=volume_origin)
#     if not block_mesh.has_vertex_colors:
#         block_mesh.paint_uniform_color([1,1,1])
#     #meshpath = './meshes/axisres'
#     meshpath = f'../results/[TSDF]{dataset_name}/SingleRes'
#     originpath = meshpath + '/meshes'
#     if not os.path.exists(originpath):
#         os.makedirs(originpath, exist_ok=True)
#     title = os.path.join(originpath,  f'{k[-3]}_{k[-2]}_{k[-1]}_singleres={final_res}.ply')
#     o3d.io.write_triangle_mesh(title, block_mesh, write_ascii=True, write_vertex_colors=True)
