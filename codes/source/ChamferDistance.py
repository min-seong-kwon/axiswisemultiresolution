import numpy as np
import open3d as o3d

###############################################################################
# distance 측정 소스 코드 
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/point_mesh_distance.html#point_mesh_face_distance
###############################################################################

from typing import Union

import torch
import torch.nn.functional as F    
import pytorch3d
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures import Meshes, Pointclouds


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



def face_to_point_distance(pts_pt, mesh_pt, min_triangle_area=5e-3):
    # packed representation for pointclouds
    points = pts_pt.points_packed()  # (P, 3)
    points_first_idx = pts_pt.cloud_to_packed_first_idx()
    max_points = pts_pt.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = mesh_pt.verts_packed()
    faces_packed = mesh_pt.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = mesh_pt.mesh_to_faces_packed_first_idx()
    max_tris = mesh_pt.num_faces_per_mesh().max().item()

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris, min_triangle_area
    )
    
    return face_to_point


def point_to_face_distance(pts_pt, mesh_pt, min_triangle_area=5e-3):
    # packed representation for pointclouds
    points = pts_pt.points_packed()  # (P, 3)
    points_first_idx = pts_pt.cloud_to_packed_first_idx()
    max_points = pts_pt.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = mesh_pt.verts_packed()
    faces_packed = mesh_pt.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = mesh_pt.mesh_to_faces_packed_first_idx()
    max_tris = mesh_pt.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    
    return point_to_face
    


def symmetric_face_to_point_distance(verts1_np, tris1_np, verts2_np, tris2_np, voxel_size=None):
    pts1_pt = Pointclouds(points=[torch.from_numpy(verts1_np)])
    pts2_pt = Pointclouds(points=[torch.from_numpy(verts2_np)])
    mesh1_pt = Meshes(verts=[torch.from_numpy(verts1_np)], 
                      faces=[torch.from_numpy(tris1_np)])
    mesh2_pt = Meshes(verts=[torch.from_numpy(verts2_np)], 
                      faces=[torch.from_numpy(tris2_np)])

    pts1_pt = pts1_pt.cuda()
    pts2_pt = pts2_pt.cuda()
    mesh1_pt = mesh1_pt.cuda()
    mesh2_pt = mesh2_pt.cuda()
    
    dist1 = face_to_point_distance(pts1_pt, mesh2_pt)
    dist2 = face_to_point_distance(pts2_pt, mesh1_pt)
    
    if voxel_size:
        final_dist1 = dist1[dist1 < voxel_size].mean()
        final_dist2 = dist2[dist2 < voxel_size].mean()
    else:
        final_dist1 = dist1.mean()
        final_dist2 = dist2.mean()
        

    return final_dist1, final_dist2


def symmetric_face_to_point_distance(verts1_np, tris1_np, verts2_np, tris2_np, voxel_size=None):
    pts1_pt = Pointclouds(points=[torch.from_numpy(verts1_np)])
    pts2_pt = Pointclouds(points=[torch.from_numpy(verts2_np)])
    mesh1_pt = Meshes(verts=[torch.from_numpy(verts1_np)], 
                      faces=[torch.from_numpy(tris1_np)])
    mesh2_pt = Meshes(verts=[torch.from_numpy(verts2_np)], 
                      faces=[torch.from_numpy(tris2_np)])

    pts1_pt = pts1_pt.cuda()
    pts2_pt = pts2_pt.cuda()
    mesh1_pt = mesh1_pt.cuda()
    mesh2_pt = mesh2_pt.cuda()
    
    dist1 = point_to_face_distance(pts1_pt, mesh2_pt)
    dist2 = point_to_face_distance(pts2_pt, mesh1_pt)
    
    # print(dist1.max())
    # print(dist2.max())
    
    if voxel_size:
        final_dist1 = dist1[dist1 < voxel_size].mean()
        final_dist2 = dist2[dist2 < voxel_size].mean()
    else:
        final_dist1 = dist1.mean()
        final_dist2 = dist2.mean()
        

    return final_dist1, final_dist2

def custom_ChamferDistance(mesh1, mesh2, voxel_size=None):
    verts1_np = np.array(mesh1.vertices)
    tris1_np = np.array(mesh1.triangles)
    
    verts2_np = np.array(mesh2.vertices)
    tris2_np = np.array(mesh2.triangles)
    
    verts1_np = verts1_np.astype(np.float32)
    tris1_np = tris1_np.astype(np.int32)
    verts2_np = verts2_np.astype(np.float32)
    tris2_np = tris2_np.astype(np.int32)
    
    dist_A2B, dist_B2A = symmetric_face_to_point_distance(verts1_np, tris1_np, verts2_np, tris2_np, voxel_size)
    distance = dist_A2B + dist_B2A
    distance = distance.cpu().numpy()
    return float(distance)

def ChamferDistance_upscaled(my_mesh,
                            gt_mesh,
                            finest_mesh,
                            scaler=1000.0,
                            voxel_size=None):
    mesh_bbox = gt_mesh.bounding_box.extents
    scale_factor = scaler / mesh_bbox
    
    my_mesh.apply_scale(scale_factor)
    gt_mesh.apply_scale(scale_factor)
    finest_mesh.apply_scale(scale_factor)
    
    my_verts = (np.array(my_mesh.vertices)).astype(np.float32)
    my_faces = (np.array(my_mesh.triangles)).astype(np.int32)
    gt_verts = (np.array(gt_mesh.vertices)).astype(np.float32)
    gt_faces = (np.array(gt_mesh.triangles)).astype(np.int32)
    finest_verts = (np.array(finest_mesh.vertices)).astype(np.float32)
    finest_faces = (np.array(finest_mesh.triangles)).astype(np.int32)
    
    d_gt_A2B, d_gt_B2A = symmetric_face_to_point_distance(my_verts, my_faces, gt_verts, gt_faces, voxel_size)
    d_gt = float((d_gt_A2B + d_gt_B2A).cpu().numpy())
    d_finest_A2B, d_finest_B2A = symmetric_face_to_point_distance(my_verts, my_faces, finest_verts, finest_faces, voxel_size)
    d_finest = float((d_finest_A2B + d_finest_B2A).cpu().numpy())
    
    return d_gt, d_finest