import numpy as np
import open3d as o3d
import torch
from pytorch3d.structures import Meshes, Pointclouds
import trimesh


target_mesh_path = '../results/[TSDF]armadillo/SingleRes/voxsize_0.500/armadillo_singleres=[32 32 32].ply'
src_mesh = trimesh.load(target_mesh_path)
src_verts = np.array(src_mesh.vertices)
src_faces = np.array(src_mesh.faces)

awmr_mesh_path1 = '../results/[TSDF]armadillo/SingleRes/voxsize_0.500/armadillo_singleres=[8 8 8].ply'

awmr_mesh = trimesh.load(awmr_mesh_path1)
awmr_verts = np.array(awmr_mesh.vertices)
awmr_faces = np.array(awmr_mesh.faces)

from source.ChamferDistance import symmetric_face_to_point_distance, custom_ChamferDistance

src_verts = src_verts.astype(np.float32)
awmr_verts = awmr_verts.astype(np.float32)
src_faces = src_faces.astype(np.int32)
awmr_faces = awmr_faces.astype(np.int32)

dist_A2B, dist_B2A = symmetric_face_to_point_distance(src_verts, src_faces, awmr_verts, awmr_faces)
print((dist_A2B+dist_B2A).item())