import os
import numpy as np
from tqdm import tqdm

from source.ConfigSettings import base_volume_unit_dim, base_voxel_size
from source.VolumeUnit import VolumeUnit_AWMR
from source.AWMR_utils_v3 import get_all_condition, split_until_thres, mesh_whole_block_singularize
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock # TODO
from source.ChamferDistance import ChamferDistance_upscaled
from utils.MPEGDataset import MPEGDataset
from utils.evalUtils import key_is_in
import trimesh
import open3d as o3d
from pandas import Series, DataFrame
from source.vsMesh import vsMesh

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

base_path = f'../1018_results/[TSDF]{dataset_name}/awmr/voxsize_{finest_voxel_size:.6f}/'
subdirectories = ['pklfiles', 'original', 'filled']
for subdirectory in subdirectories:
    path = os.path.join(base_path, subdirectory)
    os.makedirs(path, exist_ok=True)

blockmesh_path = f'../_blockmeshes/{dataset_name}/' # for debug
os.makedirs(blockmesh_path, exist_ok=True)
volume_origin = np.load(f'../vunits/{dataset_name}/voxsize_{finest_voxel_size:.6f}/volume_origin_{finest_voxel_size:.6f}.npy')

###############################################################################
# 모든 해상도 데이터셋 로드
###############################################################################
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
###############################################################################
# 8x8x8 블록으로부터, 원하는 해상도까지 split
# thres list 순회 (자동화)
###############################################################################
#thres_list = np.linspace(2e-9, 2e-6, 10) # np.logspace(0, 2.6, 10) * 5e-9
thres_list = np.linspace(1.8e-9, 2.2e-6, 18) # 10.18 재실험 

orifile_size_list = []
filledfile_size_list = [] 
num_blocks_list = []
chamferdist_list = []
###############################################################################
for thres in thres_list:
    thres2str = str(round(thres*1e6,3))
    
    file_format = f'{dataset_name}_awmr_thres={thres2str}'
    pklfile = os.path.join(base_path, 'pklfiles', f'{file_format}.pkl')
    meshfile = os.path.join(base_path, 'original', f'{file_format}.ply')
    filledfile = os.path.join(base_path, 'filled', f'{file_format}_filled.ply')
    
    awmr_tsdfs = {}
    for k in tqdm(volume_units['32_32_32'].keys(), desc=f"split awmr:{dataset_name}_{finest_voxel_size:.3f}, thres={thres2str}"):
        if len(k)==3:
            print("your initial key length is 3, please modify code")
            k = (dataset_name, k[0], k[1], k[2])
        
        if get_all_condition(volume_units, k):
            continue
        
        awmr_tsdfs[k] = AWMRblock(axisres=(8,8,8), 
                                    unit_index=k,
                                    tsdf=volume_units['8_8_8'][k].D)

        split_until_thres(awmr_tsdfs[k],
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
    
    with open(pklfile, "wb") as f: 
        pickle.dump(awmr_tsdfs, f)
    ###############################################################################
    # split된 TSDF block을 meshing
    ###############################################################################
    mesh = o3d.geometry.TriangleMesh()
    for k in tqdm(awmr_tsdfs.keys(), desc=f"mesh awmr: {dataset_name}_{finest_voxel_size:.3f}, thres={thres2str}"):
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
    o3d.io.write_triangle_mesh(meshfile,
                                mesh, write_ascii=True, write_vertex_colors=True)
    del mesh, awmr_tsdfs
    # mesh 후처리
    processed_mesh = trimesh.load(meshfile)
    processed_mesh.update_faces(processed_mesh.nondegenerate_faces())
    trimesh.repair.fill_holes(processed_mesh)
    processed_mesh.export(filledfile)
    del processed_mesh
    
    orifile_size = os.path.getsize(meshfile) / 1024
    filledfile_size = os.path.getsize(filledfile) / 1024
    dist = vsMesh(gt_mesh_path, filledfile)
    
    orifile_size_list.append(orifile_size)
    filledfile_size_list.append(filledfile_size)
    chamferdist_list.append(dist)

raw_data = {'thres': thres_list,
            'ori file size': orifile_size_list,
            'filled file size': filledfile_size_list,
            '# of blocks': num_blocks_list,
            'chamfer distance': chamferdist_list}

data = DataFrame(raw_data).transpose()
print(data)
data.to_excel(f'../1018_results/[TSDF]{dataset_name}/awmr/voxsize_{finest_voxel_size:.6f}/RD_{dataset_name}_awmr.xlsx', index=True)