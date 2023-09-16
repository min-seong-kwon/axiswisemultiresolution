
import os
import numpy as np
from tqdm import tqdm

from source.ConfigSettings import base_volume_unit_dim, base_voxel_size
from source.VolumeUnit import VolumeUnit
from source.AWMR_utils import get_condition, make_mesh, mesh_whole_block_singularize, split_into_resolution, pad_awmr_from_original, pad_3d
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock # TODO

from utils.TSDFDataset import TSDFDataset
from utils.evalUtils import customChamfer, customHausdorff, key_is_in

import open3d as o3d

import igl
import pickle
import pandas as pd

evaluate = False

dataset_name = 'thai'
finest_voxel_size = 0.5
scale_factor = 0.002/finest_voxel_size
final_res = np.array([8,8,8])
print(f"[dataset = {dataset_name}] [voxsize = {finest_voxel_size}] [final res = {final_res}] [current voxel size = {finest_voxel_size*32/final_res[0]}]")

volume_origin = np.load(f'../vunits/{dataset_name}/voxsize_{finest_voxel_size:.3f}/volume_origin_{finest_voxel_size:.3f}.npy')

target_path = fr'../results/[TSDF]{dataset_name}/SingleRes/voxsize_{finest_voxel_size:.3f}'
mesh_filename = target_path + fr"/{dataset_name}_singleres={final_res}.ply"
blockmesh_path = f'../_meshes/{dataset_name}/axisres' # for debug

if not os.path.exists(target_path):
    os.makedirs(target_path, exist_ok=True)
if not os.path.exists(blockmesh_path):
    os.makedirs(blockmesh_path, exist_ok=True)

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


# p1 = np.array([308.928711, 147.270645, 227.413452])
# p2 = np.array([311.636505, 104.775398, 275.705292])
# vunit_start = np.floor((p1 - np.squeeze(volume_origin))/(finest_voxel_size*base_volume_unit_dim))
# vunit_end = np.floor((p2 - np.squeeze(volume_origin))/(finest_voxel_size*base_volume_unit_dim))
vunit_start = np.array([2, 2, 1])
vunit_end = np.array([3., 3, 0])

awmr_tsdfs = {}
for k in tqdm(volume_units_32.keys(), leave=True):
    # if not key_is_in(k, vunit_start, vunit_end): # DEBUG
        # continue
    if len(k)==3:
        print("your initial key length is 3, please modify code")
        k = (dataset_name, k[0], k[1], k[2])
    
    contd_condition = get_condition(volume_units_32,volume_units_16,volume_units_8,k)

    if contd_condition:
        continue
    
    awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8), # edit: 4x4x4 -> 8x8x8
                                unit_index=k,
                                tsdf=volume_units_8[k].D)

    split_into_resolution(awmrblock=awmr_tsdfs[k], 
                    thres=0,
                    volume_units_16=volume_units_32,
                    volume_units_8=volume_units_16,
                    volume_units_4=volume_units_8,
                    axisres=np.array([8,8,8]), 
                    unit_index=k, 
                    start_point=np.array((0,0,0)),
                    final_res=final_res,
                    for_train=True)

print("meshing...")
counter = 0
counter2 = 0
unnecessary_keys = []
wrong_keys = []
# all_hausdorff = []
# all_chamfer = []
mesh = o3d.geometry.TriangleMesh()
for k in tqdm(awmr_tsdfs.keys(), leave=True):
    block_mesh = mesh_whole_block_singularize(awmr_tsdfs[k],
                                            unit_index=k,
                                            awmr_dict=awmr_tsdfs,
                                            node=None,
                                            volume_origin=volume_origin,
                                            voxel_size=base_voxel_size,
                                            volume_unit_dim=base_volume_unit_dim,
                                            baseres=8)
    
    block_mesh.scale(1/scale_factor, center=tuple(volume_origin))
    if not block_mesh.has_vertex_colors:
        block_mesh.paint_uniform_color([1, 1, 1])

    if np.sum(np.asarray(block_mesh.triangles)) == 0:
        continue
    
    if evaluate:
        ori_mesh = make_mesh(volume_units_32[k].D,
                            (base_volume_unit_dim,
                            base_volume_unit_dim,
                            base_volume_unit_dim),
                            voxel_size=base_voxel_size,
                            volume_unit_dim=base_volume_unit_dim,
                            k=k,
                            volume_origin=volume_origin,
                            allow_degenerate=False)
        ori_mesh.scale(1/scale_factor,center=tuple(volume_origin))
        ori_mesh.paint_uniform_color([1, 1, 1])
        if np.sum(np.asarray(block_mesh.triangles)) == 0:
            if np.sum(np.asarray(ori_mesh.triangles)) == 0:
                unnecessary_keys.append(k)
                counter2 += 1
            else:
                wrong_keys.append(k)
                counter += 1
            continue
        if np.sum(np.asarray(ori_mesh.triangles)) == 0:
            continue
        # hausdorff = igl.hausdorff(np.asarray(block_mesh.vertices),
        #                         np.asarray(block_mesh.triangles),
        #                         np.asarray(ori_mesh.vertices),
        #                         np.asarray(ori_mesh.triangles))
        # chamfer = customChamfer(block_mesh, ori_mesh)

        block_mesh.paint_uniform_color([1, 1, 1])
        # all_chamfer.append(chamfer)
        # all_hausdorff.append(hausdorff)
    mesh += block_mesh
    title = os.path.join(blockmesh_path,  f'{k[-3]}_{k[-2]}_{k[-1]}.ply')
    o3d.io.write_triangle_mesh(
        title, block_mesh, write_ascii=True, write_vertex_colors=True)

# mesh.remove_duplicated_vertices()

o3d.io.write_triangle_mesh(mesh_filename,
                            mesh, write_ascii=True, write_vertex_colors=True)
