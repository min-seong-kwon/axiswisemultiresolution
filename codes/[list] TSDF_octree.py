import os
import numpy as np
from tqdm import tqdm

from source.ConfigSettings import base_volume_unit_dim, base_voxel_size
from source.VolumeUnit import VolumeUnit_AWMR
from source.AWMR_utils_v3 import get_all_condition, split_octree, mesh_whole_block_singularize
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock # TODO

from utils.MPEGDataset import MPEGDataset
from utils.evalUtils import key_is_in
import trimesh
import open3d as o3d

import igl
import pickle
import pandas as pd
import gc
###############################################################################
# 데이터셋, 원하는 resolution 선택
###############################################################################
dataset_voxel_sizes = {
    'armadillo': 0.4,
    'dragon': 0.001,
    'thai': 0.5
}
# 데이터셋 선택
dataset_name = 'dragon'
finest_voxel_size = dataset_voxel_sizes.get(dataset_name, None)
scale_factor = 0.002/finest_voxel_size
# 원본 메시 로드
target_mesh_path = f'../OriginalDataset/{dataset_name}.ply'
src_mesh = trimesh.load(target_mesh_path)
src_verts = np.array(src_mesh.vertices)
src_faces = np.array(src_mesh.faces)
# 원하는 resolution 선택
volume_origin = np.load(f'../vunits/{dataset_name}/voxsize_{finest_voxel_size:.6f}/volume_origin_{finest_voxel_size:.6f}.npy')
# 파일 저장 위치
target_path = fr'../results/[TSDF]{dataset_name}/octree/voxsize_{finest_voxel_size:.6f}'
blockmesh_path = f'../_meshes/{dataset_name}/axisres' # for debug
if not os.path.exists(target_path):
    os.makedirs(target_path, exist_ok=True)
if not os.path.exists(blockmesh_path):
    os.makedirs(blockmesh_path, exist_ok=True)

###############################################################################
# 모든 해상도 데이터셋 로드
###############################################################################
dataset = {}
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
    dataset[axisres_str] = MPEGDataset(dataset_name=dataset_name,
                                        axisres=axisres,
                                        finest_voxel_size=finest_voxel_size,
                                        volume_origin=volume_origin)
    
    volume_units[axisres_str] = {}  
    
    for k in dataset[axisres_str].tsdf_blocks.keys():
        volume_units[axisres_str][k] = VolumeUnit_AWMR(axisres=axisres)
        volume_units[axisres_str][k].D = dataset[axisres_str].tsdf_blocks[k]

###############################################################################
# 8x8x8 블록으로부터, 원하는 해상도까지 split
# thres list 순회 (자동화)
###############################################################################
thres_list=[0.0005, 0.0003, 0.00025, 0.00022,
            0.0002, 0.00018, 0.00015, 0.00012,
            0.0001, 7e-05, 4e-05, 1e-05]

for thres in thres_list:
    thres2str = str(thres)
    octree_mesh_path = f'{target_path}/{dataset_name}_octree_thres={thres2str}.ply'
    awmr_tsdfs = {}
    for k in tqdm(volume_units['32_32_32'].keys(), desc=f"split octree: {dataset_name}_{finest_voxel_size:.6f}, thres={thres}"):
        if len(k)==3:
            print("your initial key length is 3, please modify code")
            k = (dataset_name, k[0], k[1], k[2])
        
        if get_all_condition(volume_units, k):
            continue
        
        awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8), 
                                    unit_index=k,
                                    tsdf=volume_units['8_8_8'][k].D)

        split_octree(awmr_tsdfs[k],
                        thres,
                        volume_units,
                        axisres=np.array([8,8,8]),
                        unit_index=k,
                        start_point=np.array((0,0,0)),
                        for_train=True)
    ###############################################################################
    # split된 TSDF block을 meshing
    ###############################################################################
    mesh = o3d.geometry.TriangleMesh()
    for k in tqdm(awmr_tsdfs.keys(), desc=f"mesh octree: {dataset_name}_{finest_voxel_size:.6f}, thres={thres}"):
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
        title = os.path.join(blockmesh_path,  f'{k[-3]}_{k[-2]}_{k[-1]}.ply')
        o3d.io.write_triangle_mesh(
            title, block_mesh, write_ascii=True, write_vertex_colors=True)

    o3d.io.write_triangle_mesh(octree_mesh_path,
                                mesh, write_ascii=True, write_vertex_colors=True)
    
    del mesh, awmr_tsdfs
