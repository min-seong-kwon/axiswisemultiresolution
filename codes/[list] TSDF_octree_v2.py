import os
import numpy as np
from tqdm import tqdm

from source.ConfigSettings import base_volume_unit_dim, base_voxel_size
from source.VolumeUnit import VolumeUnit_AWMR
from source.AWMR_utils_v3 import get_all_condition, split_octree, mesh_whole_block_singularize
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock # TODO
from source.ChamferDistance import ChamferDistance_upscaled
from utils.MPEGDataset import MPEGDataset
from utils.evalUtils import key_is_in
import trimesh
import open3d as o3d
from pandas import Series, DataFrame

import igl
import pickle
import pandas as pd
import gc
###############################################################################
# 데이터셋, 디렉토리 설정
###############################################################################
dataset_voxel_sizes = {
    'armadillo': 0.2,
    'dragon': 0.0002,
    'thai': 0.4,
    'asia':0.2,
    'happy': 0.0002,
    'lucy': 1.0
}
# 데이터셋 선택
dataset_name = 'dragon'
finest_voxel_size = dataset_voxel_sizes.get(dataset_name, None)
scale_factor = 0.002/finest_voxel_size
# GT 메시 로드
gt_mesh_path = f'../OriginalDataset/{dataset_name}.ply'
gt_mesh = trimesh.load(gt_mesh_path)
# finest mesh 로드
finest_mesh_path = f'../0926_results/[TSDF]{dataset_name}/SingleRes/voxsize_{finest_voxel_size:.6f}/{dataset_name}_singleres=[32 32 32]_filled.ply'
finest_mesh = trimesh.load(finest_mesh_path)
volume_origin = np.load(f'../vunits/{dataset_name}/voxsize_{finest_voxel_size:.6f}/volume_origin_{finest_voxel_size:.6f}.npy')
# 파일 저장 위치
target_path = fr'../0926_results/[TSDF]{dataset_name}/octree/voxsize_{finest_voxel_size:.6f}'
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
thres_list = np.logspace(0, 1.6, 15) * 5e-8
filesize_list = [] 
num_blocks_list = []
dist_original = []
dist_finest = []

for thres in thres_list:
    thres2str = str(round(thres*1e6,3))
    octree_mesh_path = f'{target_path}/{dataset_name}_octree_thres={thres2str}.ply'
    awmr_tsdfs = {}
    for k in tqdm(volume_units['32_32_32'].keys(), desc=f"split octree: {dataset_name}_{finest_voxel_size:.6f}, thres={thres2str}"):
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
    
    n_blocks = 0
    for k, root in awmr_tsdfs.items():
        n_blocks += len(root.leaves)
    num_blocks_list.append(n_blocks)
    
    pkl_path = os.path.basename(octree_mesh_path).replace('.ply', '.pkl')
    with open(pkl_path, "wb") as f: 
            pickle.dump(awmr_tsdfs, f)
    ###############################################################################
    # split된 TSDF block을 meshing
    ###############################################################################
    mesh = o3d.geometry.TriangleMesh()
    for k in tqdm(awmr_tsdfs.keys(), desc=f"mesh octree: {dataset_name}_{finest_voxel_size:.3f}, thres={thres2str}"):
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
    del mesh
    # mesh 후처리
    filled_mesh_path = os.path.basename(octree_mesh_path).replace('.ply', '_filled.ply')
    processed_mesh = trimesh.load(octree_mesh_path)
    processed_mesh.update_faces(processed_mesh.nondegenerate_faces())
    trimesh.repair.fill_holes(processed_mesh)
    processed_mesh.export(filled_mesh_path)
    
    # processed mesh 그리고 gt_mesh + finest_mesh의 거리를 측정 + append
    file_size = os.path.getsize(filled_mesh_path) / 1024
    d_gt, d_finest = ChamferDistance_upscaled(processed_mesh, gt_mesh, finest_mesh)
    dist_original.append(d_gt)
    dist_finest.append(d_finest)
    filesize_list.append(file_size)
    
    del awmr_tsdfs, processed_mesh

raw_data = {'thres': thres_list,
            'file size': filesize_list,
            '# of blocks': num_blocks_list,
            'original dist': dist_original,
            'finest dist': dist_finest}
data = DataFrame(raw_data).transpose()
data.to_excel(f'{target_path}/RD_{dataset_name}_octree.xlsx', index=False)
print(data)
