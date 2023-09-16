from pysdf import SDF
from numba import njit
from skimage import measure
import os
import trimesh
import numpy as np

from source.VolumeUnit import VolumeUnit

@njit
def generate_voxel_coordinates(out_coords, start_position, grid_dim, voxel_size):
    for x in range(grid_dim[0]):
        for y in range(grid_dim[1]):
            for z in range(grid_dim[2]):
                out_coords[x, y, z, 0] = start_position[0] + x * voxel_size
                out_coords[x, y, z, 1] = start_position[1] + y * voxel_size
                out_coords[x, y, z, 2] = start_position[2] + z * voxel_size

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
TARGET_MESH_PATH = '../OriginalDataset/thai.ply'
DATASET_NAME = 'thai'
###############################################################################
BASE_VOL_UNIT_DIM = 32
FINEST_VOXEL_SIZE = 0.5         # armadillo: 0.5, dragon: 0.001, thai: 0.5
BASE_VOXEL_SIZE = 16/BASE_VOL_UNIT_DIM # 복셀 해상도 
SDF_TRUNC = FINEST_VOXEL_SIZE * 6

###############################################################################
VUNIT_PATH = f'../vunits/{DATASET_NAME}/voxsize_{FINEST_VOXEL_SIZE:.3f}'
if not os.path.exists(VUNIT_PATH):
    os.makedirs(VUNIT_PATH, exist_ok=True)

RESULT_PATH = f'../results/[TSDF]{DATASET_NAME}/SingleRes/voxsize_{FINEST_VOXEL_SIZE:.3f}'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)
RESULT_MESH_NAME = RESULT_PATH + \
    f'/{DATASET_NAME}_TSDF_{BASE_VOL_UNIT_DIM}.ply' # encoding=ascii

###############################################################################
# MESH -> TSDF 변환 작업 
###############################################################################
print(f"[dataset = {DATASET_NAME}] [voxsize = {FINEST_VOXEL_SIZE}] [volume unit dim = {BASE_VOL_UNIT_DIM}] [current voxel size = {BASE_VOXEL_SIZE}]")
# Load some mesh (don't necessarily need trimesh)
src_mesh = trimesh.load(TARGET_MESH_PATH)
src_verts = np.array(src_mesh.vertices)
src_faces = np.array(src_mesh.faces)

mesh_min_bound = np.min(src_verts, axis=0)
mesh_max_bound = np.max(src_verts, axis=0)

print('min bound: %f x %f x %f' % (mesh_min_bound[0], mesh_min_bound[1], mesh_min_bound[2]))
print('max bound: %f x %f x %f' % (mesh_max_bound[0], mesh_max_bound[1], mesh_max_bound[2]))

# adjusted_mesh_min_bound = mesh_min_bound - (BASE_VOXEL_SIZE * BASE_VOL_UNIT_DIM)
# adjusted_mesh_max_bound = mesh_max_bound + (BASE_VOXEL_SIZE * BASE_VOL_UNIT_DIM)

# voxel_grid_dim = np.ceil((adjusted_mesh_max_bound - adjusted_mesh_min_bound) / BASE_VOXEL_SIZE / BASE_VOL_UNIT_DIM) * BASE_VOL_UNIT_DIM
# voxel_grid_dim = voxel_grid_dim.astype(np.int32)

voxel_grid_dim = np.ceil((mesh_max_bound - mesh_min_bound) / BASE_VOXEL_SIZE / BASE_VOL_UNIT_DIM) * BASE_VOL_UNIT_DIM
voxel_grid_dim = voxel_grid_dim.astype(np.int32)
print('voxel grid: %d x %d x %d' % (voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2]))

volume_origin = (np.asarray(mesh_min_bound)).reshape((3,1))
np.save(VUNIT_PATH + f'/volume_origin_{FINEST_VOXEL_SIZE:.3f}.npy', volume_origin)

query_points = np.zeros((voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2], 3), dtype=np.float32)
generate_voxel_coordinates(query_points, mesh_min_bound, voxel_grid_dim, BASE_VOXEL_SIZE)
query_points_list = query_points.reshape(-1, 3).tolist()

f = SDF(src_verts, src_faces); # (num_vertices, 3) and (num_faces, 3)
sdf_values_list = f(query_points_list)
sdf_values = np.array(sdf_values_list).reshape(voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2])

tsdf_values = np.minimum(1.0, sdf_values / SDF_TRUNC)
tsdf_values = np.maximum(-1.0, tsdf_values)
tsdf_values = tsdf_values * (-1.0)
# Use marching cubes to obtain the surface mesh of these ellipsoids
out_verts, out_faces, out_normals, out_values = measure.marching_cubes(tsdf_values, 0)
out_verts = out_verts * BASE_VOXEL_SIZE + mesh_min_bound
out_mesh = trimesh.Trimesh(vertices=out_verts, faces=out_faces, vertex_normals=out_normals)
_ = out_mesh.export(RESULT_MESH_NAME)


###############################################################################
# volume -> 8x8x8 block 
###############################################################################

TSDF_VOL = tsdf_values.copy()
TSDF_MASK = np.zeros_like(TSDF_VOL)
generate_volume_mask(TSDF_VOL, TSDF_MASK)                

if not os.path.exists(f'../vunits/{DATASET_NAME}/voxsize_{FINEST_VOXEL_SIZE:.3f}/{DATASET_NAME}_{BASE_VOL_UNIT_DIM}'):
    os.mkdir(f'../vunits/{DATASET_NAME}/voxsize_{FINEST_VOXEL_SIZE:.3f}/{DATASET_NAME}_{BASE_VOL_UNIT_DIM}')
    

for x in range(0, TSDF_VOL.shape[0], BASE_VOL_UNIT_DIM):
    for y in range(0, TSDF_VOL.shape[1], BASE_VOL_UNIT_DIM):
        for z in range(0, TSDF_VOL.shape[2], BASE_VOL_UNIT_DIM):
            
            vunit_x = x // BASE_VOL_UNIT_DIM
            vunit_y = y // BASE_VOL_UNIT_DIM
            vunit_z = z // BASE_VOL_UNIT_DIM
        
            TSDF = TSDF_VOL[x:x+BASE_VOL_UNIT_DIM, y:y+BASE_VOL_UNIT_DIM, z:z+BASE_VOL_UNIT_DIM]
            MASK = TSDF_MASK[x:x+BASE_VOL_UNIT_DIM, y:y+BASE_VOL_UNIT_DIM, z:z+BASE_VOL_UNIT_DIM]
            # SIGN = np.sign(TSDF)
            # SIGN[SIGN >= 0.0] = 1.0
            # SIGN[SIGN < 0.0] = 0.0
            # MAGN = np.abs(TSDF)
            
            vunit = VolumeUnit(BASE_VOL_UNIT_DIM)
            
            vunit.D = TSDF
            vunit.W = MASK
            vunit.save(f'../vunits/{DATASET_NAME}/voxsize_{FINEST_VOXEL_SIZE:.3f}/{DATASET_NAME}_{BASE_VOL_UNIT_DIM}/{vunit_x}_{vunit_y}_{vunit_z}.npz')


###############################################################################
# Chamfer 왜곡 계산 
###############################################################################

from source.ChamferDistance import symmetric_face_to_point_distance

src_verts = src_verts.astype(np.float32)
out_verts = out_verts.astype(np.float32)
src_faces = src_faces.astype(np.int32) 
out_faces = out_faces.astype(np.int32)

dist_A2B, dist_B2A = symmetric_face_to_point_distance(src_verts, src_faces, out_verts, out_faces)
print((dist_A2B+dist_B2A).item())
