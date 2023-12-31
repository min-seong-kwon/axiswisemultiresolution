import os
import numpy as np
from tqdm import tqdm

from source.ConfigSettings import base_volume_unit_dim, base_voxel_size
from source.VolumeUnit import VolumeUnit_AWMR
from source.AWMR_utils_v3 import get_all_condition, split_into_resolution, mesh_whole_block_singularize
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock # TODO

from utils.MPEGDataset import MPEGDataset
from utils.evalUtils import key_is_in
import trimesh
import open3d as o3d

import igl
import pickle
import pandas as pd
import trimesh
from pandas import Series, DataFrame

###############################################################################
# 데이터셋, 원하는 resolution 선택
###############################################################################
evaluate = False
dataset_voxel_sizes = {
    'armadillo': 0.2,
    'dragon': 0.0002,
    'thai': 0.4,
    'asia':0.2,
    'happy': 0.0002,
    'lucy': 1.0
}
# 데이터셋 선택
dataset_name = 'asia'
finest_voxel_size = dataset_voxel_sizes.get(dataset_name, None)
scale_factor = 0.002/finest_voxel_size

###############################################################################
# 모든 해상도 데이터셋 로드
###############################################################################
volume_origin = np.load(f'../vunits/{dataset_name}/voxsize_{finest_voxel_size:.6f}/volume_origin_{finest_voxel_size:.6f}.npy')
volume_units = {}
res = np.array([8,16,32])
combinations = np.array(np.meshgrid(res, res, res)).T.reshape(-1,3)
for axisres in tqdm(combinations):
    '''
    axisres: numpy array (ex) [32,8,8] 
    axisres_str: string (ex) '32_8_8'
    k: dataset과 volume unit index (ex) ('soldier',0,10,1)
    '''
    axisres_str = '_'.join(map(str, axisres))
    dataset_axisres = MPEGDataset(dataset_name=dataset_name,
                                axisres=axisres,
                                finest_voxel_size=finest_voxel_size,
                                volume_origin=volume_origin)
    
    volume_units[axisres_str] = {}  
    
    for k in dataset_axisres.tsdf_blocks.keys():
        volume_units[axisres_str][k] = VolumeUnit_AWMR(axisres=axisres)
        volume_units[axisres_str][k].D = dataset_axisres.tsdf_blocks[k]
    del dataset_axisres

# 원하는 resolution 선택
# final_res = np.array([32,32,32])
filesize_list = [] 
num_blocks_list = []

final_res_list = [np.array([32,32,32]), np.array([16,16,16]), np.array([8,8,8])]
for final_res in final_res_list:
    print(f"[dataset = {dataset_name}_{finest_voxel_size}] [final res = {final_res}] [current voxel size = {(finest_voxel_size*32)/final_res}]")
    
    # 파일 저장 위치
    target_path = fr'../0926_results/[TSDF]{dataset_name}/SingleRes/voxsize_{finest_voxel_size:.6f}'
    mesh_filename = target_path + fr"/{dataset_name}_singleres={final_res}.ply"
    blockmesh_path = f'../_meshes/{dataset_name}/axisres' # for debug
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)
    if not os.path.exists(blockmesh_path):
        os.makedirs(blockmesh_path, exist_ok=True)
    # 디버깅용
    debug = False
    if debug:
        debug_path = f'../debug/{dataset_name}'
        mesh_filename = debug_path + fr"/{dataset_name}_singleres={final_res}_debug.ply"
        if not os.path.exists(debug_path):
            os.makedirs(debug_path, exist_ok=True)
        p1 = np.array([303.867828 ,667.890198 ,244.878784])
        p2 = np.array([282.693817, 687.614075, 266.477020])
        vunit_start = np.floor((p1 - np.squeeze(volume_origin))/(finest_voxel_size*base_volume_unit_dim))
        vunit_end = np.floor((p2 - np.squeeze(volume_origin))/(finest_voxel_size*base_volume_unit_dim))
    ###############################################################################
    # 8x8x8 블록으로부터, 원하는 해상도까지 split
    ###############################################################################
    awmr_tsdfs = {}
    for k in tqdm(volume_units['32_32_32'].keys()):
        if debug and not key_is_in(k, vunit_start, vunit_end): # DEBUG
            continue
        if len(k)==3:
            print("your initial key length is 3, please modify code")
            k = (dataset_name, k[0], k[1], k[2])
        
        if get_all_condition(volume_units, k):
            continue
        
        awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8), 
                                    unit_index=k,
                                    tsdf=volume_units['8_8_8'][k].D)

        split_into_resolution(awmrblock=awmr_tsdfs[k], 
                        thres=0,
                        volume_units=volume_units,
                        axisres=np.array([8,8,8]), 
                        unit_index=k, 
                        start_point=np.array((0,0,0)),
                        final_res=final_res,
                        for_train=True)

    n_blocks = 0
    for k, root in awmr_tsdfs.items():
        n_blocks += len(root.leaves)
    num_blocks_list.append(n_blocks)
    
    pkl_path = mesh_filename.replace('.ply', '.pkl')
    with open(pkl_path, "wb") as f: 
        pickle.dump(awmr_tsdfs, f)
    ###############################################################################
    # split된 TSDF block을 meshing
    ###############################################################################
    print("meshing...")
    counter = 0
    counter2 = 0
    unnecessary_keys = []
    wrong_keys = []
    mesh = o3d.geometry.TriangleMesh()
    for k in tqdm(awmr_tsdfs.keys()):
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

        mesh += block_mesh
        # title = os.path.join(blockmesh_path,  f'{k[-3]}_{k[-2]}_{k[-1]}.ply')
        # o3d.io.write_triangle_mesh(
        #     title, block_mesh, write_ascii=True, write_vertex_colors=True)

    o3d.io.write_triangle_mesh(mesh_filename,
                                mesh, write_ascii=True, write_vertex_colors=True)

    del mesh
    # 09.25 추가
    filled_mesh_path = mesh_filename.replace('.ply', '_filled.ply')
    processed_mesh = trimesh.load(mesh_filename)
    processed_mesh.update_faces(processed_mesh.nondegenerate_faces())
    trimesh.repair.fill_holes(processed_mesh)
    processed_mesh.export(filled_mesh_path)
    
    file_size = os.path.getsize(filled_mesh_path) / 1024
    filesize_list.append(file_size)
    print(f"filesize: {file_size} KB | # of blocks: {n_blocks}")

raw_data = {'final res': final_res_list,
            'file size': filesize_list,
            '# of blocks': num_blocks_list}
data = DataFrame(raw_data).transpose()
data.to_excel(f'{target_path}/RD_{dataset_name}_singleres.xlsx', index=True)
print(data)