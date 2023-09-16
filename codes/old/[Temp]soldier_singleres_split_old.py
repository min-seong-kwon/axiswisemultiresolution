# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:52:04 2022

splitting a 4*4*4 volume unit into axis-wise multi-resolution blocks

@author: podragonwer

split_into_resolution: 무조건 정해진 axiswise resolution으로 split 합니다.
중대 버그가 발견되어 디버깅이 필요합니다.
"""

import os
import numpy as np
from tqdm import tqdm

from source.ConfigSettings import *
from source.VolumeUnit import VolumeUnit
from source.AWMR_utils import *
from source.AWMRblock import *

from utils.TSDFDataset import TSDFDataset

from skimage import measure
import open3d as o3d

import igl
import pickle

'''
res_list = [
    np.array([4,4,4]),
    np.array([4,4,8]), np.array([4,8,4]), np.array([8,4,4]),
    np.array([4,8,8]), np.array([8,4,8]), np.array([8,8,4]),
    np.array([8,8,8]),
    np.array([4,4,16]), np.array([4,16,4]), np.array([16,4,4]),
    np.array([4,8,16]), np.array([4,16,8]), np.array([8,4,16]), 
    np.array([8,16,4]), np.array([16,4,8]), np.array([16,8,4]),
    np.array([8,8,16]), np.array([8,16,8]), np.array([16,8,8]),
    np.array([8,16,16]), np.array([16,8,16]), np.array([16,16,8]),
    np.array([16,16,16])
    ]
'''
dataset_name = 'soldier'
dataset_name_32 = f'{dataset_name}_32'
dataset_name_16 = f'{dataset_name}_16'
dataset_name_8 = f'{dataset_name}_8'

res_list = [np.array([16,16,16])]
volume_origin = np.load(f'../mpeg_vunits/{dataset_name}_vunits/volume_origin_{dataset_name_32}.npy')

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

for final_res in res_list:
    awmr_tsdfs = {}

    for k in tqdm(volume_units_32.keys()):
        if len(k)==3:
            print("your initial key length is 3, please modify code")
            k = (scene_name, k[0], k[1], k[2])
        
        if np.min(volume_units_32[k].D)*np.max(volume_units_32[k].D) >= 0:
            continue
        awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8), # edit: 4x4x4 -> 8x8x8
                                    unit_index=k,
                                    tsdf=volume_units_8[k].D)
        
        split_into_resolution(awmrblock=awmr_tsdfs[k], 
                        thres=0, # does not matter
                        volume_units_16=volume_units_32,
                        volume_units_8=volume_units_16,
                        volume_units_4=volume_units_8,
                        axisres=(8,8,8), 
                        unit_index=k, 
                        start_point=np.array((0,0,0)),
                        final_res=final_res,
                        for_train=True)
    
    awmr_dict_2_TSDFBlockDataset(awmr_tsdfs, f"{scene_name}_{final_res[0]},{final_res[1]},{final_res[2]}",
                                res=baseres, 
                                save_path='./DeepTSDFVolCompression/Dataset/') # TODO
    
    for k in tqdm(awmr_tsdfs.keys()):
        block_mesh = mesh_whole_block(awmr_tsdfs[k], 
                            unit_index=k, 
                            awmr_dict=awmr_tsdfs,
                            node=None,
                            volume_origin=volume_origin)
        if not block_mesh.has_vertex_colors:
            block_mesh.paint_uniform_color([1,1,1])
        #meshpath = './meshes/axisres'
        meshpath = f'../results/[TSDF]{dataset_name}/SingleRes'
        originpath = meshpath + '/meshes'
        if not os.path.exists(originpath):
            os.makedirs(originpath, exist_ok=True)
        title = os.path.join(originpath,  f'{k[-3]}_{k[-2]}_{k[-1]}_singleres={final_res}.ply')
        o3d.io.write_triangle_mesh(title, block_mesh, write_ascii=True, write_vertex_colors=True)
    
    with open(meshpath + f"/pklfiles/{dataset_name}_singleres={final_res}.pkl", "wb") as f: 
        pickle.dump(awmr_tsdfs, f)

# for scene_name in ['ho_r2_3B']: #,'ho_r1_1F','re_r1_4R']:
#     print("loading dataset...")
#     volume_origin = np.load('traindata/volume_origin_' + scene_name + '.npy')
#     dataset = TSDFDataset(scenes=[scene_name],
#                           volume_origin=volume_origin,
#                           original_path='../Dataset/',
#                           rig=scene_name[3:5],
#                           face=scene_name[6:8],
#                           ETRI=True)
#     print(f"{scene_name} dataset loaded!")
    
#     # 모든 해상도의 데이터셋을 준비
#     volume_units_32 = {}
#     for k in dataset.tsdf_blocks.keys():
#         volume_units_32[k] = VolumeUnit(volume_unit_dim=32, depth=5)
#         volume_units_32[k].D = dataset.tsdf_blocks[k]
#     volume_units_16, _, _ = assignRes(dataset.tsdf_blocks, level_mode='single16')
#     _, volume_units_8, _ = assignRes(dataset.tsdf_blocks, level_mode='single8')
#     for res, vu in zip([16, 8],[volume_units_16, volume_units_8]):
#         if len(vu.items())>0:
#             print(f'└─	reducing {len(vu.items())} blocks resolution into {res}')
#             vu = reduceRes(vu, 
#                           res, 
#                           volume_origin, 
#                           dataset,
#                           base_volume_unit_dim, 
#                           base_voxel_size, 
#                           sdf_trunc,
#                           force=False)
            
#     for final_res in res_list:
#         awmr_tsdfs = {}
#         for k in tqdm(dataset.tsdf_blocks.keys()):
#             if len(k)==3:
#                 print("your initial key length is 3, please modify code")
#                 k = (scene_name, k[0], k[1], k[2])
            
#             if np.min(volume_units_32[k].D)*np.max(volume_units_32[k].D) >= 0:
#                 continue
#             awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8), # edit: 4x4x4 -> 8x8x8
#                                          unit_index=k,
#                                          tsdf=volume_units_8[k].D)
            
#             split_into_resolution(awmrblock=awmr_tsdfs[k], 
#                              thres=0, # does not matter
#                              volume_units_16=volume_units_32,
#                              volume_units_8=volume_units_16,
#                              volume_units_4=volume_units_8,
#                              axisres=(8,8,8), 
#                              unit_index=k, 
#                              start_point=np.array((0,0,0)),
#                              final_res=final_res,
#                              for_train=True)
        
#         awmr_dict_2_TSDFBlockDataset(awmr_tsdfs, f"{scene_name}_{final_res[0]},{final_res[1]},{final_res[2]}",
#                                      res=baseres, 
#                                      save_path='./DeepTSDFVolCompression/Dataset/') # TODO
            
#         for k in tqdm(awmr_tsdfs.keys()):
#             block_mesh = mesh_whole_block(awmr_tsdfs[k], 
#                                  unit_index=k, 
#                                  awmr_dict=awmr_tsdfs,
#                                  node=None,
#                                  volume_origin=volume_origin)
#             if not block_mesh.has_vertex_colors:
#                 block_mesh.paint_uniform_color([1,1,1])
#             meshpath = './meshes/axisres'
#             title = os.path.join(meshpath,  f'{k[-3]}_{k[-2]}_{k[-1]}_awmr_final_res={final_res}.ply')
#             o3d.io.write_triangle_mesh(title, block_mesh, write_ascii=True, write_vertex_colors=True)
        
#         import pickle
#         with open(f"{scene_name}_awmr_tsdfs_final_res={final_res}.pkl", "wb") as f:
#             pickle.dump(awmr_tsdfs, f)
            