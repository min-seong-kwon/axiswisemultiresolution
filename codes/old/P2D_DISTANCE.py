import os
import trimesh
import numpy as np
import open3d as o3d
import DracoPy
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from numba import njit

import torch
from pytorch3d.structures import Meshes, Pointclouds



###############################################################################
# distance 측정 소스 코드 
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/point_mesh_distance.html#point_mesh_face_distance
###############################################################################
    
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable

"""
This file defines distances between meshes and pointclouds.
The functions make use of the definition of a distance between a point and
an edge segment or the distance of a point and a triangle (face).

The exact mathematical formulations and implementations of these
distances can be found in `csrc/utils/geometry_utils.cuh`.
"""

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


point_face_distance = _PointFaceDistance.apply


# FacePointDistance
class _FacePointDistance(Function):
    """
    Torch autograd Function wrapper FacePointDistance Cuda implementation
    """

    @staticmethod
    def forward(
        ctx,
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_tris,
        min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_tris: Scalar equal to maximum number of faces in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(T,)`, where `dists[t]` is the squared
                euclidean distance of `t`-th triangular face to the closest point in the
                corresponding example in the batch
            idxs: LongTensor of shape `(T,)` indicating the closest point in the
                corresponding example in the batch.

            `dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])`,
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`.
        """
        dists, idxs = _C.face_point_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.face_point_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


face_point_distance = _FacePointDistance.apply


def point_mesh_face_distance(
    meshes: Meshes,
    pcls: Pointclouds,
    min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
    )

    # weight each example by the inverse of number of faces in the example
    tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
    num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
    weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
    weights_t = 1.0 / weights_t.float()
    face_to_point = face_to_point * weights_t
    face_dist = face_to_point.sum() / N

    return point_dist, face_dist





###############################################################################
# 원본 메쉬 
###############################################################################
target_mesh_path = './soldier_voxelized/soldier_fr0536_qp10_qt12.obj'
mesh = o3d.io.read_triangle_mesh(target_mesh_path)

src_mesh = Meshes(verts=[torch.from_numpy(np.array(mesh.vertices).astype(np.float32))], 
                  faces=[torch.from_numpy(np.array(mesh.triangles).astype(np.float32))])

src_points = Pointclouds(points=[torch.from_numpy(np.array(mesh.vertices).astype(np.float32))])
src_points = src_points.cuda()
###############################################################################
# Draco
###############################################################################

for draco_qp in range(8, 15):
    binary = DracoPy.encode(
        mesh.vertices, mesh.triangles, 
        quantization_bits=draco_qp, compression_level=10,
        quantization_range=-1, quantization_origin=None,
        create_metadata=False, preserve_order=False
    )

    with open('temp.drc', 'wb') as draco_file:
        draco_file.write(binary)
        
    with open('temp.drc', 'rb') as draco_file:
        decoded = DracoPy.decode(draco_file.read())
      
    '''
    dec_points = Pointclouds(points=[torch.from_numpy(decoded.points.astype(np.float32))])  
    with torch.no_grad(): 
        loss = point_mesh_face_distance(src_mesh, dec_points)
    '''
    dec_mesh = Meshes(verts=[torch.from_numpy(decoded.points.astype(np.float32))],
                      faces=[torch.from_numpy(decoded.faces.astype(np.int32))])  
    dec_mesh = dec_mesh.cuda()
    with torch.no_grad(): 
        loss = point_mesh_face_distance(dec_mesh, src_points)
        
    print(loss)
    
    
###############################################################################
# TSDF
###############################################################################

@njit
def generate_voxel_coordinates(out_coords, start_position, grid_dim, voxel_size):
    for x in range(grid_dim[0]):
        for y in range(grid_dim[1]):
            for z in range(grid_dim[2]):
                out_coords[x, y, z, 0] = start_position[0] + x * voxel_size
                out_coords[x, y, z, 1] = start_position[1] + y * voxel_size
                out_coords[x, y, z, 2] = start_position[2] + z * voxel_size
                
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
                                
                                
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

BASE_VOLUME_UINT_DIM = 32
for BASE_VOXEL_SIZE in [4, 3.5, 3, 2.5, 2]:
#for BASE_VOXEL_SIZE in [4, 3.5, 3, 2.5]:
    sdf_trunc = BASE_VOXEL_SIZE * 2.0

    mesh_min_bound = mesh.vertex.positions.min(0).numpy()
    mesh_max_bound = mesh.vertex.positions.max(0).numpy()

    #print('min bound: %f x %f x %f' % (mesh_min_bound[0], mesh_min_bound[1], mesh_min_bound[2]))
    #print('max bound: %f x %f x %f' % (mesh_max_bound[0], mesh_max_bound[1], mesh_max_bound[2]))

    adjusted_mesh_min_bound = mesh_min_bound - (BASE_VOXEL_SIZE * BASE_VOLUME_UINT_DIM)
    adjusted_mesh_max_bound = mesh_max_bound + (BASE_VOXEL_SIZE * BASE_VOLUME_UINT_DIM)

    #print('adjusted min bound: %f x %f x %f' % (adjusted_mesh_min_bound[0], adjusted_mesh_min_bound[1], adjusted_mesh_min_bound[2]))
    #print('adjusted max bound: %f x %f x %f' % (adjusted_mesh_max_bound[0], adjusted_mesh_max_bound[1], adjusted_mesh_max_bound[2]))


    voxel_grid_dim = np.ceil((adjusted_mesh_max_bound - adjusted_mesh_min_bound) / BASE_VOXEL_SIZE / BASE_VOLUME_UINT_DIM) * BASE_VOLUME_UINT_DIM
    voxel_grid_dim = voxel_grid_dim.astype(np.int32)
    print('voxel grid: %d x %d x %d' % (voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2]))

    query_points = np.zeros((voxel_grid_dim[0], voxel_grid_dim[1], voxel_grid_dim[2], 3), dtype=np.float32)

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
    
    clear_tsdf_volume(truncated_signed_distance)
    
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(truncated_signed_distance, 0, mask=meshing_mask)
    
    verts = verts * BASE_VOXEL_SIZE + adjusted_mesh_min_bound

    #tsdf_points = Pointclouds(points=[torch.from_numpy(verts.astype(np.float32))])
    #loss = point_mesh_face_distance(src_mesh, tsdf_points)
    
    tsdf_mesh = Meshes(verts=[torch.from_numpy(verts.astype(np.float32))],
                      faces=[torch.from_numpy(faces.astype(np.int32))])  
    tsdf_mesh = tsdf_mesh.cuda()
    with torch.no_grad(): 
        loss = point_mesh_face_distance(tsdf_mesh, src_points)
        
    print(loss)
    