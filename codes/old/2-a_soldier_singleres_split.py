# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm

from source.ConfigSettings import baseres
from source.VolumeUnit import VolumeUnit
from source.AWMR_utils import split_into_resolution, awmr_dict_2_TSDFBlockDataset, mesh_whole_block,\
    pad_awmr_from_original
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock # TODO

from utils.TSDFDataset import TSDFDataset

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
finest_voxel_size = 1.5
# dataset_name_32 = f'{dataset_name}_32'
# dataset_name_16 = f'{dataset_name}_16'
# dataset_name_8 = f'{dataset_name}_8'

res_list = [np.array([8,8,8])]
volume_origin = np.load(f'../vunits/{dataset_name}/voxsize_{finest_voxel_size:.1f}/volume_origin_{finest_voxel_size:.1f}.npy')

dataset_32 = TSDFDataset(dataset_name=dataset_name,
                        volume_unit_dim=32,
                        finest_voxel_size=finest_voxel_size,
                        volume_origin=volume_origin,
                        ETRI=False)

dataset_16 = TSDFDataset(dataset_name=dataset_name,
                        volume_unit_dim=16,
                        finest_voxel_size=finest_voxel_size,
                        volume_origin=volume_origin,
                        ETRI=False)

dataset_8 = TSDFDataset(dataset_name=dataset_name,
                        volume_unit_dim=8,
                        finest_voxel_size=finest_voxel_size,
                        volume_origin=volume_origin,
                        ETRI=False)

volume_units_32 = {}
for k in dataset_32.tsdf_blocks.keys():
    volume_units_32[k] = VolumeUnit(volume_unit_dim=32)
    volume_units_32[k].D = dataset_32.tsdf_blocks[k]

volume_units_16 = {}
for k in dataset_16.tsdf_blocks.keys():
    volume_units_16[k] = VolumeUnit(volume_unit_dim=16)
    volume_units_16[k].D = dataset_16.tsdf_blocks[k]

volume_units_8 = {}
for k in dataset_8.tsdf_blocks.keys():
    volume_units_8[k] = VolumeUnit(volume_unit_dim=8)
    volume_units_8[k].D = dataset_8.tsdf_blocks[k]
    
# for k in dataset_32.tsdf_blocks.keys():
#     new_k = (dataset_name,k[1],k[2],k[3])
#     volume_units_32[new_k] = VolumeUnit(volume_unit_dim=32)
#     volume_units_32[new_k].D = dataset_32.tsdf_blocks[k]
    
# volume_units_16 = {}
# for k in dataset_16.tsdf_blocks.keys():
#     new_k = (dataset_name,k[1],k[2],k[3])
#     volume_units_16[new_k] = VolumeUnit(volume_unit_dim=16)
#     volume_units_16[new_k].D = dataset_16.tsdf_blocks[k]
    
# volume_units_8 = {}
# for k in dataset_8.tsdf_blocks.keys():
#     new_k = (dataset_name,k[1],k[2],k[3])
#     volume_units_8[new_k] = VolumeUnit(volume_unit_dim=8)
#     volume_units_8[new_k].D = dataset_8.tsdf_blocks[k]

for final_res in res_list:
    awmr_tsdfs = {}

    for k in tqdm(volume_units_32.keys()):
        if len(k)==3:
            print("your initial key length is 3, please modify code")
            k = (dataset_name, k[0], k[1], k[2])
        
        pad_tsdf = pad_awmr_from_original(volume_units_8[k].D,
                                volume_units_16=volume_units_32,
                                volume_units_8=volume_units_16,
                                volume_units_4=volume_units_8,
                                axisres=np.array([8,8,8]), 
                                unit_index=k, 
                                start_point=np.array([0,0,0]),
                                for_mask=True)
        
        if (np.min(pad_tsdf)*np.max(pad_tsdf) >= 0):
            continue
        
        awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8), # edit: 4x4x4 -> 8x8x8
                                    unit_index=k,
                                    tsdf=volume_units_8[k].D)

        split_into_resolution(awmrblock=awmr_tsdfs[k], 
                        thres=0, # does not matter
                        volume_units_16=volume_units_32,
                        volume_units_8=volume_units_16,
                        volume_units_4=volume_units_8,
                        axisres=np.array([8,8,8]), 
                        unit_index=k, 
                        start_point=np.array((0,0,0)),
                        final_res=final_res,
                        for_train=True)
        
    resultpath = f'../results/[TSDF]{dataset_name}/SingleRes/voxsize_{finest_voxel_size}/splitted'
    if not os.path.exists(resultpath):
        os.makedirs(resultpath, exist_ok=True)
    
    with open(resultpath + f"/{dataset_name}_singleres={final_res}.pkl", "wb") as f: 
        pickle.dump(awmr_tsdfs, f)
