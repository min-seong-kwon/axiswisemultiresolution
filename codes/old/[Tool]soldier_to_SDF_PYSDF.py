from pysdf import SDF
from numba import njit
from skimage import measure
import os
import trimesh
import numpy as np


###############################################################################
# INPUT 정보 
###############################################################################
TARGET_MESH_PATH = './soldier_voxelized/soldier_fr0536_qp10_qt12.obj'
BASE_VOLUME_UINT_DIM = 32
BASE_VOXEL_SIZE = 3 # 복셀 해상도 
SDF_TRUNC = BASE_VOXEL_SIZE * 2.0

###############################################################################
# MESH -> TSDF 변환 작업 
###############################################################################

# Load some mesh (don't necessarily need trimesh)
src_mesh = trimesh.load(TARGET_MESH_PATH)
src_verts = np.array(src_mesh.vertices)
src_faces = np.array(src_mesh.faces)

mesh_min_bound = np.min(src_verts, axis=0)
mesh_max_bound = np.max(src_verts, axis=0)

print('min bound: %f x %f x %f' % (mesh_min_bound[0], mesh_min_bound[1], mesh_min_bound[2]))
print('max bound: %f x %f x %f' % (mesh_max_bound[0], mesh_max_bound[1], mesh_max_bound[2]))


voxel_grid_dim = np.ceil((mesh_max_bound - mesh_min_bound) / BASE_VOXEL_SIZE / BASE_VOLUME_UINT_DIM) * BASE_VOLUME_UINT_DIM
voxel_grid_dim = voxel_grid_dim.astype(np.int32)
print('voxel grid: %d x %d x %d' % (voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2]))



@njit
def generate_voxel_coordinates(out_coords, start_position, grid_dim, voxel_size):
    for x in range(grid_dim[0]):
        for y in range(grid_dim[1]):
            for z in range(grid_dim[2]):
                out_coords[x, y, z, 0] = start_position[0] + x * voxel_size
                out_coords[x, y, z, 1] = start_position[1] + y * voxel_size
                out_coords[x, y, z, 2] = start_position[2] + z * voxel_size
                
query_points = np.zeros((voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2], 3), dtype=np.float32)
generate_voxel_coordinates(query_points, mesh_min_bound, voxel_grid_dim, BASE_VOXEL_SIZE)
query_points_list = query_points.reshape(-1, 3).tolist()


f = SDF(src_verts, src_faces); # (num_vertices, 3) and (num_faces, 3)
sdf_values_list = f(query_points_list)
sdf_values = np.array(sdf_values_list).reshape(voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2])

tsdf_values = np.minimum(1.0, sdf_values / SDF_TRUNC)
tsdf_values = np.maximum(-1.0, tsdf_values)

# Use marching cubes to obtain the surface mesh of these ellipsoids
out_verts, out_faces, out_normals, out_values = measure.marching_cubes(tsdf_values, 0)
out_verts = out_verts * BASE_VOXEL_SIZE + mesh_min_bound
out_mesh = trimesh.Trimesh(vertices=out_verts, faces=out_faces, vertex_normals=out_normals)
out_mesh_path = os.path.basename(TARGET_MESH_PATH).replace('.obj', '_TSDF_VS%2.1f.ply' % BASE_VOXEL_SIZE)
_ = out_mesh.export(out_mesh_path)


###############################################################################
# volume -> 8x8x8 block 
###############################################################################


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
                        
TSDF_VOL = tsdf_values.copy()
TSDF_MASK = np.zeros_like(TSDF_VOL)
generate_volume_mask(TSDF_VOL, TSDF_MASK)                       
                        
num_voxels = TSDF_VOL.shape[0] * TSDF_VOL.shape[1] * TSDF_VOL.shape[2]
num_valid_blocks = int(num_voxels / 8 /8 /8)

TSDF = np.zeros((num_valid_blocks, 8, 8, 8), dtype=np.float32)
MASK = np.zeros((num_valid_blocks, 8, 8, 8), dtype=np.float32)
SIGN = np.zeros((num_valid_blocks, 8, 8, 8), dtype=np.float32)
MAGN = np.zeros((num_valid_blocks, 8, 8, 8), dtype=np.float32)

idx_valid_blocks = 0   
for x in range(0, TSDF_VOL.shape[0], 8):
    for y in range(0, TSDF_VOL.shape[1], 8):
        for z in range(0, TSDF_VOL.shape[2], 8):                    
                   
            TSDF8 = TSDF_VOL[x:x+8, y:y+8, z:z+8]
            MASK8 = TSDF_MASK[x:x+8, y:y+8, z:z+8]
            SIGN8 = np.sign(TSDF8)
            SIGN8[SIGN8 >= 0.0] = 1.0
            SIGN8[SIGN8 < 0.0] = 0.0
            MAGN8 = np.abs(TSDF8)
        
            TSDF[idx_valid_blocks, :, :, :] = TSDF8
            MASK[idx_valid_blocks, :, :, :] = MASK8
            SIGN[idx_valid_blocks, :, :, :] = SIGN8
            MAGN[idx_valid_blocks, :, :, :] = MAGN8
            
            idx_valid_blocks += 1
          
out_npz_path = out_mesh_path.replace('.ply', '_8x8x8.npz')
np.savez_compressed(out_npz_path, TSDF=TSDF, MASK=MASK, SIGN=SIGN, MAGN=MAGN, VOXEL_SIZE=BASE_VOXEL_SIZE, VOXEL_GRID_DIM=voxel_grid_dim, ORIGIN=mesh_min_bound)        




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
