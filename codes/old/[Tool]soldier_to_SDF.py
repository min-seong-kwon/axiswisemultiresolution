import os
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from numba import njit



###############################################################################
target_mesh_path = './soldier_voxelized/soldier_fr0536_qp10_qt12.obj'
BASE_VOLUME_UINT_DIM = 32
BASE_VOXEL_SIZE = 3 # 복셀 해상도 
sdf_trunc = BASE_VOXEL_SIZE * 2.0

###############################################################################

mesh = o3d.io.read_triangle_mesh(target_mesh_path)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

mesh_min_bound = mesh.vertex.positions.min(0).numpy()
mesh_max_bound = mesh.vertex.positions.max(0).numpy()

print('min bound: %f x %f x %f' % (mesh_min_bound[0], mesh_min_bound[1], mesh_min_bound[2]))
print('max bound: %f x %f x %f' % (mesh_max_bound[0], mesh_max_bound[1], mesh_max_bound[2]))

adjusted_mesh_min_bound = mesh_min_bound - (BASE_VOXEL_SIZE * BASE_VOLUME_UINT_DIM)
adjusted_mesh_max_bound = mesh_max_bound + (BASE_VOXEL_SIZE * BASE_VOLUME_UINT_DIM)

print('adjusted min bound: %f x %f x %f' % (adjusted_mesh_min_bound[0], adjusted_mesh_min_bound[1], adjusted_mesh_min_bound[2]))
print('adjusted max bound: %f x %f x %f' % (adjusted_mesh_max_bound[0], adjusted_mesh_max_bound[1], adjusted_mesh_max_bound[2]))


voxel_grid_dim = np.ceil((adjusted_mesh_max_bound - adjusted_mesh_min_bound) / BASE_VOXEL_SIZE / BASE_VOLUME_UINT_DIM) * BASE_VOLUME_UINT_DIM
voxel_grid_dim = voxel_grid_dim.astype(np.int32)
print('voxel grid: %d x %d x %d' % (voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2]))


query_points = np.zeros((voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2], 3), dtype=np.float32)


@njit
def generate_voxel_coordinates(out_coords, start_position, grid_dim, voxel_size):
    for x in range(grid_dim[0]):
        for y in range(grid_dim[1]):
            for z in range(grid_dim[2]):
                out_coords[x, y, z, 0] = start_position[0] + x * voxel_size
                out_coords[x, y, z, 1] = start_position[1] + y * voxel_size
                out_coords[x, y, z, 2] = start_position[2] + z * voxel_size
 
generate_voxel_coordinates(query_points, adjusted_mesh_min_bound, voxel_grid_dim, BASE_VOXEL_SIZE)
                

min_voxel = np.floor((mesh_min_bound - adjusted_mesh_min_bound) / BASE_VOXEL_SIZE)
max_voxel = np.ceil((mesh_max_bound - adjusted_mesh_min_bound) / BASE_VOXEL_SIZE)
min_voxel = min_voxel.astype(np.int32) 
max_voxel = max_voxel.astype(np.int32) + 1
meshing_mask = np.zeros(voxel_grid_dim, dtype=bool)
meshing_mask[min_voxel[0]:max_voxel[0], min_voxel[1]:max_voxel[1], min_voxel[2]:max_voxel[2]] = True


# signed distance is a [32,32,32] array
signed_distance = scene.compute_signed_distance(query_points, nsamples=51)
truncated_signed_distance = np.minimum(1.0, signed_distance.numpy() / sdf_trunc)
truncated_signed_distance = np.maximum(-1.0, truncated_signed_distance)

'''              
@njit 
def clear_tsdf_volume(volume):
    for x in range(volume.shape[0]-1):
        for y in range(volume.shape[1]-1):
            for z in range(volume.shape[2]-1):
                if volume[x, y, z] >= -1.0:
                    continue
                
                if volume[x, y, z+1] > 0 or  \
                   volume[x, y+1, z] > 0 or  volume[x, y+1, z+1] > 0 or  volume[x+1, y+1, z+1] > 0 or \
                   volume[x+1, y, z] > 0 or  volume[x+1, y+1, z] > 0 or  volume[x+1, y, z+1] > 0:
                       volume[x, y, z] = 1.0
'''

@njit 
def clear_tsdf_volume(volume):
    for x in range(volume.shape[0]-1):
        for y in range(volume.shape[1]-1):
            for z in range(volume.shape[2]-1):

                num_ones = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            if np.abs(volume[x+i, y+j, z+k]) >= 1.0:
                                num_ones += 1
                
                if num_ones == 8:
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                volume[x+i, y+j, z+k] = 1.0


clear_tsdf_volume(truncated_signed_distance)

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(truncated_signed_distance, 0, mask=meshing_mask)

verts = verts * BASE_VOXEL_SIZE + adjusted_mesh_min_bound
#verts /= 100.0 # dragon only
#verts /= 100.0 # fountain only
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

output_mesh_path = os.path.basename(target_mesh_path).replace('.obj', '_tsdf.ply')
_ = mesh.export(output_mesh_path)



###############################################################################
# 8x8x8 데이터셋으로 변환 
###############################################################################
TSDF_VOL = truncated_signed_distance.copy()
TSDF_MASK = np.zeros_like(TSDF_VOL)

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
          
output_npz_path = output_mesh_path.replace('_tsdf.ply', '_tsdf.npz')
np.savez_compressed(output_npz_path, TSDF=TSDF, MASK=MASK, SIGN=SIGN, MAGN=MAGN, VOXEL_SIZE=BASE_VOXEL_SIZE, VOXEL_GRID_DIM=voxel_grid_dim, ORIGIN=adjusted_mesh_min_bound)        


