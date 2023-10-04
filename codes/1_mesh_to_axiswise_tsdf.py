from pysdf import SDF
from numba import njit
from skimage import measure
import os
import trimesh
import numpy as np
from tqdm import tqdm
from source.VolumeUnit import VolumeUnit_AWMR
import gc
@njit
def generate_voxel_coordinates(out_coords, start_position, grid_dim, voxel_size):
    for x in range(grid_dim[0]):
        for y in range(grid_dim[1]):
            for z in range(grid_dim[2]):
                out_coords[x, y, z, 0] = start_position[0] + x * voxel_size[0]
                out_coords[x, y, z, 1] = start_position[1] + y * voxel_size[1]
                out_coords[x, y, z, 2] = start_position[2] + z * voxel_size[2]

@njit
def sign(x):
    if x >= 0:
        return 1
    else:
        return 0

@njit 
def generate_volume_mask(volume, mask):
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                
                if x+1 < volume.shape[0]:
                    if sign(volume[x, y, z]) != sign(volume[x+1, y, z]):
                        mask[x, y, z] = 1.0
                        mask[x+1, y, z] = 1.0

                if y+1 < volume.shape[1]:
                    if sign(volume[x, y, z]) != sign(volume[x, y+1, z]):
                        mask[x, y, z] = 1.0
                        mask[x, y+1, z] = 1.0
                        
                if z+1 < volume.shape[2]:
                    if sign(volume[x, y, z]) != sign(volume[x, y, z+1]):
                        mask[x, y, z] = 1.0
                        mask[x, y, z+1] = 1.0

###############################################################################
# INPUT 정보 
###############################################################################
dataset_voxel_sizes = {
    'armadillo': 0.2,
    'dragon': 0.0002,
    'thai': 0.4,
    'asia':0.2,
    'happy': 0.0002,
    'lucy': 1.0
}
DATASET_NAME = 'dragon'
TARGET_MESH_PATH = f'../OriginalDataset/{DATASET_NAME}.ply'
finest_voxel_size = dataset_voxel_sizes.get(DATASET_NAME, None)

src_mesh = trimesh.load(TARGET_MESH_PATH)
src_verts = np.array(src_mesh.vertices)
src_faces = np.array(src_mesh.faces)
mesh_min_bound = np.min(src_verts, axis=0)
mesh_max_bound = np.max(src_verts, axis=0)
###############################################################################
# CONFIG
###############################################################################
maxres = 32
res = np.array([8,16,32])
combinations = np.array(np.meshgrid(res, res, res)).T.reshape(-1,3)
###############################################################################
# 모든 해상도에 대해 축별 다해상도 볼륨 유닛 생성
###############################################################################
for axisres in tqdm(combinations):
    axisres = np.array([32,32,32])
    print(axisres)
    voxel_size = np.array(finest_voxel_size * maxres / axisres)
    SDF_TRUNC = finest_voxel_size * 6
    axisres_str = '_'.join(map(str,axisres))
    VUNIT_PATH = f'../vunits/{DATASET_NAME}/voxsize_{finest_voxel_size:.6f}'
    if not os.path.exists(VUNIT_PATH):
        os.makedirs(VUNIT_PATH, exist_ok=True)
    
    volume_origin = (np.asarray(mesh_min_bound)).reshape((3,1))
    np.save(VUNIT_PATH + f'/volume_origin_{finest_voxel_size:.6f}.npy', volume_origin)
    
    RESULT_PATH = f'../0926_results/[TSDF]{DATASET_NAME}/SingleRes/TSDF/voxsize_{finest_voxel_size:.6f}'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)
    RESULT_MESH_NAME = RESULT_PATH + \
        f'/{DATASET_NAME}_TSDF_{axisres_str}.ply' # encoding=ascii
    ###############################################################################
    # MESH -> TSDF 변환 작업 
    ###############################################################################

    voxel_grid_dim_x = (np.ceil((mesh_max_bound[0] - mesh_min_bound[0])/voxel_size[0]/axisres[0]) * axisres[0]).astype(np.int32)
    voxel_grid_dim_y = (np.ceil((mesh_max_bound[1] - mesh_min_bound[1])/voxel_size[1]/axisres[1]) * axisres[1]).astype(np.int32)
    voxel_grid_dim_z = (np.ceil((mesh_max_bound[2] - mesh_min_bound[2])/voxel_size[2]/axisres[2]) * axisres[2]).astype(np.int32)
    query_points = np.zeros((voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z, 3), dtype=np.float32)

    voxel_grid_dim = np.array([voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_dim_z]).astype(np.int32)
    
    generate_voxel_coordinates(query_points, mesh_min_bound, voxel_grid_dim, voxel_size)
    query_points_list = query_points.reshape(-1, 3).tolist()

    f = SDF(src_verts, src_faces); # (num_vertices, 3) and (num_faces, 3)
    sdf_values_list = f(query_points_list)
    sdf_values = np.array(sdf_values_list).reshape(voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2])

    tsdf_values = np.minimum(1.0, sdf_values / SDF_TRUNC)
    tsdf_values = np.maximum(-1.0, tsdf_values)
    tsdf_values = tsdf_values * (-1.0)
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    out_verts, out_faces, out_normals, out_values = measure.marching_cubes(tsdf_values, 0)
    out_verts = out_verts * voxel_size + mesh_min_bound
    out_mesh = trimesh.Trimesh(vertices=out_verts, faces=out_faces, vertex_normals=out_normals)
    _ = out_mesh.export(RESULT_MESH_NAME)

    ###############################################################################
    # volume -> RES x RES x RES block 
    ###############################################################################

    TSDF_VOL = tsdf_values.copy()
    TSDF_MASK = np.zeros_like(TSDF_VOL)
    generate_volume_mask(TSDF_VOL, TSDF_MASK)                

    if not os.path.exists(f'../vunits/{DATASET_NAME}/voxsize_{finest_voxel_size:.6f}/{DATASET_NAME}_{axisres_str}'):
        os.mkdir(f'../vunits/{DATASET_NAME}/voxsize_{finest_voxel_size:.6f}/{DATASET_NAME}_{axisres_str}')
        
    del tsdf_values, out_verts, out_faces, out_normals, out_values, out_mesh
    
    for x in range(0, TSDF_VOL.shape[0], axisres[0]):
        for y in range(0, TSDF_VOL.shape[1], axisres[1]):
            for z in range(0, TSDF_VOL.shape[2], axisres[2]):
                
                vunit_x = x // axisres[0]
                vunit_y = y // axisres[1]
                vunit_z = z // axisres[2]
            
                TSDF = TSDF_VOL[x:x+axisres[0], y:y+axisres[1], z:z+axisres[2]]
                MASK = TSDF_MASK[x:x+axisres[0], y:y+axisres[1], z:z+axisres[2]]

                vunit = VolumeUnit_AWMR(axisres)
                
                vunit.D = TSDF
                vunit.W = MASK
                vunit.save(f'../vunits/{DATASET_NAME}/voxsize_{finest_voxel_size:.6f}/{DATASET_NAME}_{axisres_str}/{vunit_x}_{vunit_y}_{vunit_z}.npz')
    del TSDF_VOL, TSDF_MASK
    gc.collect()