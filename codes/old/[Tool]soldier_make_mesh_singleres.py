import os
import numpy as np
from tqdm import tqdm
import pickle

from source.ConfigSettings import base_volume_unit_dim, base_voxel_size
from source.VolumeUnit import VolumeUnit
from source.AWMRblock8x8 import VolumeUnit8x8
from source.AWMR_utils import make_mesh, mesh_whole_block_singularize
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock  # TODO

from utils.TSDFDataset import TSDFDataset
from utils.multiResUtil import assignRes, reduceRes, unitComplexity

from skimage import measure
import open3d as o3d

import igl
from utils.evalUtils import customChamfer, customHausdorff, key_is_in
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ----------------------------------------------------------------------------------------------------------
dataset_name = 'soldier'
dataset_name_32 = f'{dataset_name}_32'
dataset_name_16 = f'{dataset_name}_16'
dataset_name_8 = f'{dataset_name}_8'
scale_factor = 0.008/8

final_res = np.array([16, 16, 16])

target_path = fr'../results/[TSDF]{dataset_name}/SingleRes'
target_pkl = target_path + fr'/pklfiles/{dataset_name}_singleres={final_res}.pkl'
target_mesh = target_path + fr"/{dataset_name}_singleres={final_res}.ply"
target_txt = target_path + fr"/{dataset_name}_singleres={final_res}.txt"

volume_origin = np.load(f'../mpeg_vunits/{dataset_name}_vunits/volume_origin_{dataset_name_32}.npy')
meshpath = '../_meshes/singleres/axisres'
evaluate = True

if not os.path.exists(meshpath):
    os.makedirs(meshpath, exist_ok=True)

if evaluate:
    dataset_mono = TSDFDataset(scenes=[dataset_name_32],
                            dataset_name=dataset_name,
                            volume_origin=volume_origin,
                            ETRI=False)
    volume_units_ori = {}
    for k in dataset_mono.keys:
        new_k = (dataset_name, k[1], k[2], k[3])
        volume_units_ori[new_k] = VolumeUnit(volume_unit_dim=32)
        volume_units_ori[new_k].D = dataset_mono.tsdf_blocks[k]
        volume_units_ori[new_k].W = volume_units_ori[new_k].M = dataset_mono.mask_blocks[k]

with open(target_pkl, "rb") as f:
    awmr_tsdfs = pickle.load(f)

print("meshing...")
counter = 0
counter2 = 0
unnecessary_keys = []
wrong_keys = []
all_hausdorff = []
all_chamfer = []
mesh = o3d.geometry.TriangleMesh()
for k in tqdm(awmr_tsdfs.keys()):
    # if (k[-3], k[-2], k[-1])==(18,29,5):
    #     print("stop here")
    # else:
    #     continue
    # if not key_is_in(k, (30,19,7), (37,21,10)):
    #     continue
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

    if evaluate:
        ori_mesh = make_mesh(volume_units_ori[k].D,
                            (base_volume_unit_dim,
                            base_volume_unit_dim,
                            base_volume_unit_dim),
                            voxel_size=base_voxel_size,
                            volume_unit_dim=base_volume_unit_dim,
                            k=k,
                            volume_origin=volume_origin,
                            allow_degenerate=False)
        ori_mesh.scale(scale_factor,center=tuple(volume_origin))
        ori_mesh.paint_uniform_color([1, 1, 1])
        if np.sum(np.asarray(block_mesh.triangles)) == 0:
            print(k, "no vertices made")
            if np.sum(np.asarray(ori_mesh.triangles)) == 0:
                print("why does this block exist?")
                unnecessary_keys.append(k)
                counter2 += 1
            else:
                wrong_keys.append(k)
                counter += 1
            continue
        if np.sum(np.asarray(ori_mesh.triangles)) == 0:
            print(k, "no vertices exist, but there are AWMR vertices")
            continue
        hausdorff = igl.hausdorff(np.asarray(block_mesh.vertices),
                                np.asarray(block_mesh.triangles),
                                np.asarray(ori_mesh.vertices),
                                np.asarray(ori_mesh.triangles))
        chamfer = customChamfer(block_mesh, ori_mesh)

        block_mesh.paint_uniform_color([1, 1, 1])
        all_chamfer.append(chamfer)
        all_hausdorff.append(hausdorff)
    mesh += block_mesh
    title = os.path.join(meshpath,  f'{k[-3]}_{k[-2]}_{k[-1]}_awmr.ply')
    o3d.io.write_triangle_mesh(
        title, block_mesh, write_ascii=True, write_vertex_colors=True)
print("counter1", counter, "counter2", counter2)

table = [[np.min(all_hausdorff), np.max(all_hausdorff), np.mean(all_hausdorff), np.median(all_hausdorff)],
        [np.min(all_chamfer), np.max(all_chamfer), np.mean(all_chamfer), np.median(all_chamfer)]]
df = pd.DataFrame(table, columns=['min', 'max', 'mean', 'median'], 
                index=['Hausdorff', 'Chamfer'])
print(df)


n_blocks = 0
for k, root in awmr_tsdfs.items():
    n_blocks += len(root.leaves)
print("#TSDF: ", n_blocks)

with open(target_txt, 'w') as f:
    f.write(f"#TSDF: {n_blocks}\n")
    f.write(f"max of Hausdorff: {np.max(all_hausdorff)} \n")
    f.write(f"mean of Hausdorff: {np.mean(all_hausdorff)} \n")
    f.write(f"min of Hausdorff: {np.min(all_hausdorff)} \n\n")
    
    f.write(f"max of Chamfer: {np.max(all_chamfer)} \n")
    f.write(f"mean of Chamfer: {np.mean(all_chamfer)} \n")
    f.write(f"min of Chamfer: {np.min(all_chamfer)} \n")
o3d.io.write_triangle_mesh(target_mesh,
                            mesh, write_ascii=True, write_vertex_colors=True)

