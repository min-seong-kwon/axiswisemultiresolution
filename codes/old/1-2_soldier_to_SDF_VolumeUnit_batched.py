import os
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from numba import njit
import sys
'''
soldier mesh를 32^3,16^3,8^3 BASE_VOLUME_UNIT_DIM을 가지는 VolumeUnit들로 변경
'''

class VolumeUnit:
    def __init__(self, volume_unit_BASE_VOLUME_UNIT_DIM=16, depth=None):
        self.volume_unit_BASE_VOLUME_UNIT_DIM = volume_unit_BASE_VOLUME_UNIT_DIM
        self.complexity = 0.0
        if depth==None:
            self.depth = np.log(volume_unit_BASE_VOLUME_UNIT_DIM)/np.log(2)
        else:
            self.depth = depth
        self.D = np.zeros((volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM), np.float32) # TSDF
        self.R = np.zeros((volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM), np.float32)
        self.G = np.zeros((volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM), np.float32)
        self.B = np.zeros((volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM), np.float32)
        self.W = np.zeros((volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM), np.float32)
        self.MC = np.zeros((volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM), np.uint8)
        self.M = np.zeros((volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM, volume_unit_BASE_VOLUME_UNIT_DIM), bool) # To see if it's a necessary voxel.
        
    def save(self, out_path):
        np.savez(out_path, D=self.D, R=self.R, G=self.G, B=self.B, W=self.W, MC=self.MC, M=self.M, N=self.volume_unit_BASE_VOLUME_UNIT_DIM, C=self.complexity)
        
    def load(self, in_path):
        npzfile = np.load(in_path)
        self.D = npzfile['D']
        self.R = npzfile['R']
        self.G = npzfile['G']
        self.B = npzfile['B']
        self.W = npzfile['W']
        self.MC = npzfile['MC']
        self.M = npzfile['M']
        self.volume_unit_BASE_VOLUME_UNIT_DIM = npzfile['N']
        self.complexity = npzfile['C']
###############################################################################
@njit
def generate_voxel_coordinates(out_coords, start_position, grid_BASE_VOLUME_UNIT_DIM, voxel_size):
    for x in range(grid_BASE_VOLUME_UNIT_DIM[0]):
        for y in range(grid_BASE_VOLUME_UNIT_DIM[1]):
            for z in range(grid_BASE_VOLUME_UNIT_DIM[2]):
                out_coords[x, y, z, 0] = start_position[0] + x * voxel_size
                out_coords[x, y, z, 1] = start_position[1] + y * voxel_size
                out_coords[x, y, z, 2] = start_position[2] + z * voxel_size

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


def compute_signed_distance_batch(scene, query_points, batch_size=32):
    num_query_points = query_points.shape[0]
    signed_distance_list = []

    for i in range(0, num_query_points, batch_size):
        batch_query_points = query_points[i:i+batch_size]
        batch_size_actual = batch_query_points.shape[0]

        batch_signed_distance = scene.compute_signed_distance(batch_query_points, nsamples=51)
        truncated_batch_signed_distance = np.minimum(1.0, batch_signed_distance.numpy() / sdf_trunc)
        truncated_batch_signed_distance = np.maximum(-1.0, truncated_batch_signed_distance)

        signed_distance_list.append(truncated_batch_signed_distance)

    signed_distance_array = np.concatenate(signed_distance_list, axis=0)
    return signed_distance_array

###############################################################################
target_mesh_path = '../OriginalDataset/soldier_voxelized/soldier_fr0536_qp10_qt12.obj'
dataset_name = 'soldier'
BASE_VOLUME_UNIT_DIM = 32
# 기존: (32,3m) (16,6m) (8,12m) --> 96/dim
# 기존: BASE_VOXEL_SIZE = 96/BASE_VOLUME_UNIT_DIM # og: BASE_VOXEL_SIZE = 3
# BASE_VOXEL_SIZE = 64/BASE_VOLUME_UNIT_DIM # (a)): (64,1m) (32,2m) (16,4m) (8,8m) --> 64/dim
BASE_VOXEL_SIZE = 48/BASE_VOLUME_UNIT_DIM # (b) (32,1.5) & (16,3) & (8,6) ===> 48/dim
sdf_trunc = 6.0
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

adjusted_mesh_min_bound = mesh_min_bound - (BASE_VOXEL_SIZE * BASE_VOLUME_UNIT_DIM)
adjusted_mesh_max_bound = mesh_max_bound + (BASE_VOXEL_SIZE * BASE_VOLUME_UNIT_DIM)

print('adjusted min bound: %f x %f x %f' % (adjusted_mesh_min_bound[0], adjusted_mesh_min_bound[1], adjusted_mesh_min_bound[2]))
print('adjusted max bound: %f x %f x %f' % (adjusted_mesh_max_bound[0], adjusted_mesh_max_bound[1], adjusted_mesh_max_bound[2]))


voxel_grid_BASE_VOLUME_UNIT_DIM = np.ceil((adjusted_mesh_max_bound - adjusted_mesh_min_bound) / BASE_VOXEL_SIZE / BASE_VOLUME_UNIT_DIM) * BASE_VOLUME_UNIT_DIM
voxel_grid_BASE_VOLUME_UNIT_DIM = voxel_grid_BASE_VOLUME_UNIT_DIM.astype(np.int32)
print('voxel grid: %d x %d x %d' % (voxel_grid_BASE_VOLUME_UNIT_DIM[0], voxel_grid_BASE_VOLUME_UNIT_DIM[1], voxel_grid_BASE_VOLUME_UNIT_DIM[2]))

query_points = np.zeros((voxel_grid_BASE_VOLUME_UNIT_DIM[0], voxel_grid_BASE_VOLUME_UNIT_DIM[1], voxel_grid_BASE_VOLUME_UNIT_DIM[2], 3), dtype=np.float32)

if not os.path.exists('../mpeg_vunits/%s_vunits' % (dataset_name)):
    os.mkdir('../mpeg_vunits/%s_vunits' % (dataset_name))
volume_origin = (np.asarray(adjusted_mesh_min_bound)).reshape((3,1))
# np.save(f'../mpeg_vunits/{dataset_name}_vunits/volume_origin_{dataset_name}_{BASE_VOLUME_UNIT_DIM}.npy', volume_origin)
np.save(f'../mpeg_vunits/{dataset_name}_vunits/volume_origin_{dataset_name}_d{BASE_VOLUME_UNIT_DIM}_v{BASE_VOXEL_SIZE:.0f}.npy', volume_origin)

generate_voxel_coordinates(query_points, adjusted_mesh_min_bound, voxel_grid_BASE_VOLUME_UNIT_DIM, BASE_VOXEL_SIZE)

min_voxel = np.floor((mesh_min_bound - adjusted_mesh_min_bound) / BASE_VOXEL_SIZE)
max_voxel = np.ceil((mesh_max_bound - adjusted_mesh_min_bound) / BASE_VOXEL_SIZE)
min_voxel = min_voxel.astype(np.int32) 
max_voxel = max_voxel.astype(np.int32) + 1
meshing_mask = np.zeros(voxel_grid_BASE_VOLUME_UNIT_DIM, dtype=bool)
meshing_mask[min_voxel[0]:max_voxel[0], min_voxel[1]:max_voxel[1], min_voxel[2]:max_voxel[2]] = True


# signed distance is a [32,32,32] array
# signed_distance = scene.compute_signed_distance(query_points, nsamples=51)
print("compute signed distance start")
truncated_signed_distance = compute_signed_distance_batch(scene, query_points, batch_size=32)
print("compute signed distance end")
# truncated_signed_distance = np.minimum(1.0, signed_distance / sdf_trunc)
# truncated_signed_distance = np.maximum(-1.0, truncated_signed_distance)

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


clear_tsdf_volume(truncated_signed_distance)

# Use marching cubes to obtain the surface mesh of these ellipsoids
#verts, faces, normals, values = measure.marching_cubes(truncated_signed_distance, 0, mask=meshing_mask)
verts, faces, normals, values = measure.marching_cubes(truncated_signed_distance, 0)

verts = verts * BASE_VOXEL_SIZE + adjusted_mesh_min_bound
#verts /= 100.0 # dragon only
#verts /= 100.0 # fountain only
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

output_path = '../results/[TSDF]soldier/SingleRes'
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
#output_mesh_path = output_path + '/' + os.path.basename(target_mesh_path).replace('.obj', '_tsdf.ply')
output_mesh_path = output_path + '/' + f'{dataset_name}_singleres_d{BASE_VOLUME_UNIT_DIM}_v{BASE_VOXEL_SIZE}.ply' # encoding=ascii
_ = mesh.export(output_mesh_path)


###############################################################################
# 8x8x8 데이터셋으로 변환 
###############################################################################
TSDF_VOL = truncated_signed_distance.copy()
TSDF_MASK = np.zeros_like(TSDF_VOL)

generate_volume_mask(TSDF_VOL, TSDF_MASK)

# if not os.path.exists('../mpeg_vunits/%s_vunits/%s_%d' % (dataset_name, dataset_name, BASE_VOLUME_UNIT_DIM)):
#     os.mkdir('../mpeg_vunits/%s_vunits/%s_%d' % (dataset_name, dataset_name, BASE_VOLUME_UNIT_DIM))

if not os.path.exists('../mpeg_vunits/%s_vunits/%s_d%d_v%d' % (dataset_name, dataset_name, BASE_VOLUME_UNIT_DIM, BASE_VOXEL_SIZE)):
    os.mkdir('../mpeg_vunits/%s_vunits/%s_d%d_v%d' % (dataset_name, dataset_name, BASE_VOLUME_UNIT_DIM, BASE_VOXEL_SIZE))
'''
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
np.savez_compressed(output_npz_path, TSDF=TSDF, MASK=MASK, SIGN=SIGN, MAGN=MAGN, VOXEL_SIZE=BASE_VOXEL_SIZE, VOXEL_GRID_BASE_VOLUME_UNIT_DIM=voxel_grid_BASE_VOLUME_UNIT_DIM, ORIGIN=adjusted_mesh_min_bound)        
'''

for x in range(0, TSDF_VOL.shape[0], BASE_VOLUME_UNIT_DIM):
    for y in range(0, TSDF_VOL.shape[1], BASE_VOLUME_UNIT_DIM):
        for z in range(0, TSDF_VOL.shape[2], BASE_VOLUME_UNIT_DIM):
            
            vunit_x = x // BASE_VOLUME_UNIT_DIM
            vunit_y = y // BASE_VOLUME_UNIT_DIM
            vunit_z = z // BASE_VOLUME_UNIT_DIM
        
            TSDF = TSDF_VOL[x:x+BASE_VOLUME_UNIT_DIM, y:y+BASE_VOLUME_UNIT_DIM, z:z+BASE_VOLUME_UNIT_DIM]
            MASK = TSDF_MASK[x:x+BASE_VOLUME_UNIT_DIM, y:y+BASE_VOLUME_UNIT_DIM, z:z+BASE_VOLUME_UNIT_DIM]
            
            vunit = VolumeUnit(BASE_VOLUME_UNIT_DIM)
            
            vunit.D = TSDF
            vunit.W = MASK

            # vunit.save('../mpeg_vunits/%s_vunits/%s_%d/%d_%d_%d.npz' % (dataset_name, dataset_name, BASE_VOLUME_UNIT_DIM,
            #                                             vunit_x, vunit_y, vunit_z))
            vunit.save('../mpeg_vunits/%s_vunits/%s_d%d_v%d/%d_%d_%d.npz' % (dataset_name, 
                                                                            dataset_name, BASE_VOLUME_UNIT_DIM, BASE_VOXEL_SIZE,
                                                                            vunit_x, vunit_y, vunit_z))


