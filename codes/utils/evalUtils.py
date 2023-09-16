# -*- coding: utf-8 -*-
"""
Created on Fri May 13 18:40:21 2022

@author: alienware
"""

import numpy as np
from tqdm import tqdm
from datetime import datetime

from source.TriMesh import TriMesh
from source.VolumeUnit import *
from source.MCLutClassic import *
from source.VolumeUnit import *
from source.ConfigSettings import *

from utils.TSDFDataset import TSDFDataset
from utils.TSDFCoder import TSDFCoder
from utils.conditional import *

import torch
from torch.utils.data import DataLoader

import sys
import os
import numpy as np
# import igl
#from chamferdist import ChamferDistance
from ArithmeticEncodingPython.pyae import ArithmeticEncoding
from decimal import *

import trimesh
import open3d as o3d

def key_is_in(key, tlf, brb):
    x = int(key[-3])
    y = int(key[-2])
    z = int(key[-1])
    
    l = int(min(tlf[-3],brb[-3]))
    t = int(min(tlf[-2],brb[-2]))
    f = int(min(tlf[-1],brb[-1]))

    r = int(max(tlf[-3],brb[-3]))
    bo = int(max(tlf[-2],brb[-2]))
    ba = int(max(tlf[-1],brb[-1]))
    
    if (l <= x <= r) and (t <= y <= bo) and (f <= z <= ba):
        return True
    return False

def slice_trimesh_by_volume_units(mesh: trimesh.base.Trimesh,
                                  vunit_start: tuple,
                                  vunit_end: tuple,
                                  voxel_size_mm,
                                  volume_unit_dim,
                                  volume_origin=np.array([[0],[0],[0]])):
    p1 = np.array(vunit_start[-3:]) * (voxel_size_mm*volume_unit_dim) + np.squeeze(volume_origin)
    vunit_end = np.array(vunit_end[-3:])
    vunit_end += 1
    p2_ori = np.array(vunit_end) * (voxel_size_mm*volume_unit_dim) + np.squeeze(volume_origin)
    return slice_trimesh_between(mesh, p1, p2_ori)


def slice_trimesh_between(mesh:trimesh.base.Trimesh, 
                          p1,
                          p2):
    x_normal = np.array([1,0,0])
    y_normal = np.array([0,1,0])
    z_normal = np.array([0,0,1])
    
    minpoint = np.min(np.stack([p1,p2]),axis=0)
    maxpoint = np.max(np.stack([p1,p2]),axis=0)
    
    normals = [x_normal, y_normal, z_normal, -x_normal, -y_normal, -z_normal]
    points = [minpoint, minpoint, minpoint, maxpoint, maxpoint, maxpoint]
    new_mesh = mesh
    for n, p in zip(normals, points):
        new_mesh = trimesh.intersections.slice_mesh_plane(new_mesh,n,p)
    return new_mesh


def dist2col(dist): # 나중에 util로 옮길 것
    # Distance between [0, 2.5mm] is mapped to [0, 255] on the red channel
    # @ dee p implicit volume compression Figure 8
    # 2.5mm = 0.0025 m
    return min(0.0025, dist) * 255 / 0.0025


def customHausdorff(mesh_a, mesh_b):
    trimesh_a = trimesh.Trimesh(vertices=mesh_a.vertices[:mesh_a.n_vertices,:3],
                            faces=mesh_a.faces[:mesh_b.n_faces,:])
    trimesh_b = trimesh.Trimesh(vertices=mesh_b.vertices[:mesh_b.n_vertices,:3],
                            faces=mesh_b.faces[:mesh_b.n_faces,:])
    
    (_, distances_a, _) = trimesh_a.nearest.on_surface(mesh_b.vertices[:,:3])
    (_, distances_b, _) = trimesh_b.nearest.on_surface(mesh_a.vertices[:,:3])
    hausdorff = max(np.max(distances_a), np.max(distances_b))
    return hausdorff


def customChamfer(mesh_a, mesh_b):
    if type(mesh_a)==o3d.cpu.pybind.geometry.TriangleMesh:
        trimesh_a = trimesh.Trimesh(vertices=np.asarray(mesh_a.vertices),
                                    faces=np.asarray(mesh_a.triangles))
        trimesh_b = trimesh.Trimesh(vertices=np.asarray(mesh_b.vertices),
                                    faces=np.asarray(mesh_b.triangles))
    else:
        trimesh_a = trimesh.Trimesh(vertices=mesh_a.vertices[:mesh_a.n_vertices,:3],
                                faces=mesh_a.faces[:mesh_b.n_faces,:])
        trimesh_b = trimesh.Trimesh(vertices=mesh_b.vertices[:mesh_b.n_vertices,:3],
                                faces=mesh_b.faces[:mesh_b.n_faces,:])
    (_, distances_a, _) = trimesh_a.nearest.on_surface(np.asarray(trimesh_b.vertices))
    (_, distances_b, _) = trimesh_b.nearest.on_surface(np.asarray(trimesh_a.vertices))
    a_isnan = list(np.where(np.isnan(distances_a))[0].reshape(-1))
    b_isnan = list(np.where(np.isnan(distances_b))[0].reshape(-1))
    if a_isnan:
        print(len(a_isnan), "degenerate triangles excluded.")
        a_isnan.reverse()
        for i in a_isnan:
            distances_a = np.delete(distances_a, i)
    if b_isnan:
        print(len(b_isnan), "degenerate triangles excluded.")
        b_isnan.reverse()
        for i in b_isnan:
            distances_b = np.delete(distances_b, i)
    chamfer_a = np.sum(distances_a)/(2*len(distances_a))
    chamfer_b = np.sum(distances_b)/(2*len(distances_b))
    return chamfer_a + chamfer_b

def compareS(s, shat):
    shat_itm = (shat>0.5).double() - (shat<=0.5).double()
    result = torch.eq(s, shat_itm) # 같으면 0, 틀렸으면 1
    return torch.logical_not(result).double()


def run_model(num_epochs,
             code_length=None,
             model_path=None,
             dataset=None,
             precision=None):
    volume_origin = dataset.volume_origin
    
    model = TSDFCoder(code_length).to('cuda')
    model_path = model_path %  num_epochs
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    batch_size = len(dataset) #TODO - this must be the whole data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    batch = next(iter(dataloader))
    m, s, s_bin, x, k = batch
    m = m.cuda()
    s = s.cuda()
    s_bin = s_bin.cuda()
    x = x.cuda()
    
    prob_zhat, zhat, shat, xhat = model(x)
    serr = compareS(s, shat)
    
    zhat = zhat.detach().cpu().numpy()
    shat = shat.cpu()
    x = x.cpu()
    xhat = xhat.cpu()
    
    volume_units = {}
    ori_units = {}
    check = [] # loss check
    check2 = []
    while True:
        print(f"current precision for s is {precision}")
        for idx in tqdm(range(batch_size)):
            d_idx = list(range(batch_size))[idx]
            vunit_x = k[1][idx].item()
            vunit_y = k[2][idx].item()
            vunit_z = k[3][idx].item()
            
            volume_units[vunit_x, vunit_y, vunit_z] = VolumeUnit(base_volume_unit_dim)
            ori_units[vunit_x, vunit_y, vunit_z] = VolumeUnit(base_volume_unit_dim)
            
            if precision is not None:
                getcontext().prec = precision
            
            # encode signs
            msg_serr = serr[idx].flatten().tolist()
            size_ori_s = sys.getsizeof(msg_serr)
            
            frequency_table = calcErrorConditional(serr[idx])
            AE_serr = ArithmeticEncoding(frequency_table=frequency_table,
                                        save_stages=False)
            en_serr, _, _, _ = AE_serr.encode(msg=msg_serr, 
                                       probability_table=AE_serr.probability_table)
            size_en_s = sys.getsizeof(en_serr)
            
            # compute binary - # TODO
            # bin_code, en_bin = AE.encode_binary(float_interval_min=interval_min_value,
            #                                      float_interval_max=interval_max_value)
            # size_en = sys.getsizeof(bin_code)
            
            # decode signs
            de_serr, _ = AE_serr.decode(encoded_msg=en_serr, 
                                  msg_length=len(msg_serr),
                                  probability_table=AE_serr.probability_table)
            
            getcontext().prec = 1000
            
            # encode Z hat
            msg_zhat = zhat[idx].flatten().tolist()
            size_ori_zhat = sys.getsizeof(msg_zhat)
            zmax = int(max(msg_zhat))
            zmin = int(min(msg_zhat))
            frequency_table_2 = {k:msg_zhat.count(k) for k in range(zmin-1,zmax+1)}
            AE_zhat = ArithmeticEncoding(frequency_table=frequency_table_2,
                                        save_stages=False)
            en_zhat, _, _, _= AE_zhat.encode(msg=msg_zhat,
                                      probability_table=AE_zhat.probability_table)
            size_en_zhat = sys.getsizeof(en_zhat)
            
            # decode Z hat
            de_zhat, _ = AE_zhat.decode(encoded_msg=en_zhat,
                                    msg_length=len(msg_zhat),
                                    probability_table=AE_zhat.probability_table)
            
            # reconstruct TSDF
            de_serr = torch.Tensor(de_serr)
            shat_itm = (shat[idx]>0.5).double() - (shat[idx]<=0.5).double()
            correction = torch.logical_not(de_serr).double() - de_serr
            correction = correction.reshape([base_volume_unit_dim,
                                            base_volume_unit_dim,
                                            base_volume_unit_dim])
            rec_s = torch.mul(correction, shat_itm) # set shat sigmoid thres 0.5
    
            # reconstruct s from serr
            x_rec = torch.mul(rec_s, xhat[idx].abs())
            
            volume_units[vunit_x, vunit_y, vunit_z].D = torch.squeeze(x_rec)
            volume_units[vunit_x, vunit_y, vunit_z].W = torch.squeeze(x_rec)!=0 
            volume_units[vunit_x, vunit_y, vunit_z].s = size_en_zhat + size_en_s
            
            ori_units[vunit_x, vunit_y, vunit_z].D = torch.squeeze(x[idx])
            ori_units[vunit_x, vunit_y, vunit_z].W = torch.squeeze(x[idx])!=0
            ori_units[vunit_x, vunit_y, vunit_z].s = size_ori_zhat + size_ori_s
            
            # check lossless?
            c = [msg_zhat[i]==de_zhat[i] for i in range(len(msg_zhat))]
            c2 = [de_serr[i]==msg_serr[i] for i in range(len(msg_serr))]
            check += [sum(c)]
            check2 += [sum(c2)]
    
        print("\nthere is %d lossy units for Z"%sum([c<len(msg_zhat) for c in check]))
        print("\nthere is %d lossy units for S"%sum([c<len(msg_serr) for c in check2]))
        if sum([c<len(msg_serr) for c in check2])<=1:
            break
        precision += 1000
    return volume_units, ori_units


def getVoxelValues(volume_values, volume_unit_dim, global_voxel_ix, global_voxel_iy, global_voxel_iz):
    volume_unit_ix = int(global_voxel_ix // volume_unit_dim)
    volume_unit_iy = int(global_voxel_iy // volume_unit_dim)
    volume_unit_iz = int(global_voxel_iz // volume_unit_dim)
    
    local_voxel_ix = int(global_voxel_ix % volume_unit_dim)
    local_voxel_iy = int(global_voxel_iy % volume_unit_dim)
    local_voxel_iz = int(global_voxel_iz % volume_unit_dim)
    
    if (volume_unit_ix, volume_unit_iy, volume_unit_iz) not in volume_values:
        scene_name = list(volume_values.keys())[0][0]
        if (scene_name, volume_unit_ix, volume_unit_iy, volume_unit_iz) not in volume_values:
            return None, None, None, None, None
        else:
            k = (scene_name, volume_unit_ix, volume_unit_iy, volume_unit_iz)
            d = volume_values[k].D[
                    local_voxel_ix, local_voxel_iy, local_voxel_iz]
            w = volume_values[k].W[
                    local_voxel_ix, local_voxel_iy, local_voxel_iz]
            r = volume_values[k].R[
                    local_voxel_ix, local_voxel_iy, local_voxel_iz]
            g = volume_values[k].G[
                    local_voxel_ix, local_voxel_iy, local_voxel_iz]
            b = volume_values[k].B[
                    local_voxel_ix, local_voxel_iy, local_voxel_iz]
    else:
        d = volume_values[volume_unit_ix, volume_unit_iy, volume_unit_iz].D[
                local_voxel_ix, local_voxel_iy, local_voxel_iz]
        w = volume_values[volume_unit_ix, volume_unit_iy, volume_unit_iz].W[
                local_voxel_ix, local_voxel_iy, local_voxel_iz]
        r = volume_values[volume_unit_ix, volume_unit_iy, volume_unit_iz].R[
                local_voxel_ix, local_voxel_iy, local_voxel_iz]
        g = volume_values[volume_unit_ix, volume_unit_iy, volume_unit_iz].G[
                local_voxel_ix, local_voxel_iy, local_voxel_iz]
        b = volume_values[volume_unit_ix, volume_unit_iy, volume_unit_iz].B[
                local_voxel_ix, local_voxel_iy, local_voxel_iz]
    return d, w, r, g, b   
    

###############################################################################
# 단일 해상도 (4mm) 메쉬 생성
###############################################################################


class singleVoxelMesh(object):
    def __init__(self):
        pass
    def __call__(self, 
                  volume_units,
                  volume_origin,
                  volume_unit_dim,
                  voxel_size_mm,
                  mesh,
                  voxel_to_vertices,
                  offset_voxel_indices,
                  num_voxels_per_volume_unit,
                  volume_unit_ix=None,
                  volume_unit_iy=None,
                  volume_unit_iz=None,
                  k=None):
        assert ((volume_unit_ix is None) and \
                (volume_unit_iy is None) and \
                (volume_unit_iz is None) and (k is not None)) \
            or ((volume_unit_ix is not None) and \
                (volume_unit_iy is not None) and \
                (volume_unit_iz is not None) and (k is None))
        if k is None:
            k = (volume_unit_ix, volume_unit_iy, volume_unit_iz)
        else:
            volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
        voxel = TriMesh()
        
        # 볼륨 유닛의 첫번째 복셀좌표 
        anchor_voxel_ix = volume_unit_ix * volume_unit_dim
        anchor_voxel_iy = volume_unit_iy * volume_unit_dim
        anchor_voxel_iz = volume_unit_iz * volume_unit_dim
        
        # 볼륨 유닛 안의 모든 복셀들을 순회 
        for i in range(num_voxels_per_volume_unit):
            offset_voxel_ix = offset_voxel_indices[0, i]
            offset_voxel_iy = offset_voxel_indices[1, i]
            offset_voxel_iz = offset_voxel_indices[2, i]
            
            global_voxel_ix = anchor_voxel_ix + offset_voxel_ix
            global_voxel_iy = anchor_voxel_iy + offset_voxel_iy
            global_voxel_iz = anchor_voxel_iz + offset_voxel_iz
            
            marching_cube_index = 0
            Ds = np.zeros(8, dtype=np.float32) # 인접 8개 복셀 TSDF 정보를 저장할 공간 
            Ws = np.zeros(8, dtype=np.float32) # 인접 8개 복셀 W 정보를 저장할 공간 
            Rs = np.zeros(8, dtype=np.float32) # 인접 8개 복셀 R컬러 정보를 저장할 공간 
            Gs = np.zeros(8, dtype=np.float32) # 인접 8개 복셀 G컬러 정보를 저장할 공간 
            Bs = np.zeros(8, dtype=np.float32) # 인접 8개 복셀 B컬러 정보를 저장할 공간
            
            # 현재 복셀을 포함한 인접 8개 복셀들을 순회 
            for j in range(8):
                neighbor_voxel_ix = global_voxel_ix + shift[j][0]
                neighbor_voxel_iy = global_voxel_iy + shift[j][1]
                neighbor_voxel_iz = global_voxel_iz + shift[j][2]
                
                d, w, r, g, b = getVoxelValues(volume_units, volume_unit_dim, neighbor_voxel_ix, neighbor_voxel_iy, neighbor_voxel_iz)
                
                # 없는 복셀이거나 한번도 업데이트 되지 않은 복셀일 때
                if (w is None) or (w <= 0.0):
                    marching_cube_index = 0 # empty cube
                    break
                
                Ds[j] = d
                Ws[j] = w
                Rs[j] = r
                Gs[j] = g
                Bs[j] = b
                
                if Ds[j] < 0.0: # k번째 복셀의 부호를 k번째 비트에 기록 
                    marching_cube_index |= (1<<j)
            
            try:
                volume_units[k].MC[offset_voxel_ix, offset_voxel_iy, offset_voxel_iz] = marching_cube_index
            except:
                pass
 
            # empty cube
            if (marching_cube_index == 0) or (marching_cube_index == 255):
                continue
            
            # mesh vertex 정보를 생성 
            vertex_indicies = -np.ones(12, np.int32)
            for j in range(12):
                if edge_table[marching_cube_index] & (1<<j): # vertex 생성이 필요한 경우 
                    
                    e1, e2 = edge_to_vert[j]
                    
                    e1_voxel_ix = global_voxel_ix + shift[e1][0]
                    e1_voxel_iy = global_voxel_iy + shift[e1][1]
                    e1_voxel_iz = global_voxel_iz + shift[e1][2]
                    e2_voxel_ix = global_voxel_ix + shift[e2][0]
                    e2_voxel_iy = global_voxel_iy + shift[e2][1]
                    e2_voxel_iz = global_voxel_iz + shift[e2][2]
                    
                    if (e1_voxel_ix, e1_voxel_iy, e1_voxel_iz, e2_voxel_ix, e2_voxel_iy, e2_voxel_iz) in voxel_to_vertices:
                        vertex_indicies[j] = voxel_to_vertices[e1_voxel_ix, e1_voxel_iy, e1_voxel_iz, e2_voxel_ix, e2_voxel_iy, e2_voxel_iz]
                    elif (e2_voxel_ix, e2_voxel_iy, e2_voxel_iz, e1_voxel_ix, e1_voxel_iy, e1_voxel_iz) in voxel_to_vertices:
                        vertex_indicies[j] = voxel_to_vertices[e2_voxel_ix, e2_voxel_iy, e2_voxel_iz, e1_voxel_ix, e1_voxel_iy, e1_voxel_iz]
                    else:
                        edge_direction = edge_shift[j][3]
                        d1 = np.abs(Ds[e1])
                        d2 = np.abs(Ds[e2])
                        r1 = Rs[e1]
                        r2 = Rs[e2]
                        g1 = Gs[e1]
                        g2 = Gs[e2]
                        b1 = Bs[e1]
                        b2 = Bs[e2]
                        
                        vertex_x = e1_voxel_ix * voxel_size_mm + volume_origin[0, 0] 
                        vertex_y = e1_voxel_iy * voxel_size_mm + volume_origin[1, 0] 
                        vertex_z = e1_voxel_iz * voxel_size_mm + volume_origin[2, 0] 
                        
                        if edge_direction == 0: # X
                            vertex_x += (d1 * voxel_size_mm) / (d1 + d2)
                        elif edge_direction == 1: # Y
                            vertex_y += (d1 * voxel_size_mm) / (d1 + d2)
                        elif edge_direction == 2: # Z
                            vertex_z += (d1 * voxel_size_mm) / (d1 + d2)
                        
                        vertex_r = (d2 * r1 + d1 * r2) / (d1 + d2)
                        vertex_g = (d2 * g1 + d1 * g2) / (d1 + d2)
                        vertex_b = (d2 * b1 + d1 * b2) / (d1 + d2)
                        
                        vertex_indicies[j] = mesh.add_vertex(
                                    vertex_x,
                                    vertex_y,
                                    vertex_z,
                                    vertex_r * 255.0,
                                    vertex_g * 255.0,
                                    vertex_b * 255.0,
                                )
                        voxel.add_vertex(
                                    vertex_x,
                                    vertex_y,
                                    vertex_z,
                                    vertex_r * 255.0,
                                    vertex_g * 255.0,
                                    vertex_b * 255.0,
                                )
                        
                        voxel_to_vertices[e1_voxel_ix, e1_voxel_iy, e1_voxel_iz, e2_voxel_ix, e2_voxel_iy, e2_voxel_iz] = vertex_indicies[j]
                        voxel_to_vertices[e2_voxel_ix, e2_voxel_iy, e2_voxel_iz, e1_voxel_ix, e1_voxel_iy, e1_voxel_iz] = vertex_indicies[j]
                                        
            # mesh connectity 정보를 생성 
            for j in range(0, 12, 3):
                if tri_table[marching_cube_index][j] == -1:
                    break
                
                mesh.add_face(vertex_indicies[tri_table[marching_cube_index][j]],
                              vertex_indicies[tri_table[marching_cube_index][j+2]],
                              vertex_indicies[tri_table[marching_cube_index][j+1]])
                voxel.add_face(vertex_indicies[tri_table[marching_cube_index][j]],
                              vertex_indicies[tri_table[marching_cube_index][j+2]],
                              vertex_indicies[tri_table[marching_cube_index][j+1]])
        return voxel
    
    
class fastVoxelMesh(object):
    def __init__(self):
        pass
    def __call__(self, 
                volume_units,
                volume_origin,
                volume_unit_dim,
                voxel_size_mm,
                mesh,
                voxel_to_vertices,
                offset_voxel_indices,
                num_voxels_per_volume_unit,
                offset_positions,
                dirs,
                pos_s,
                volume_unit_ix,
                volume_unit_iy,
                volume_unit_iz):
        # this might be fast but it will use large memory (...perchance?)
        voxel = TriMesh()
        
        # 볼륨 유닛의 첫번째 복셀좌표 
        anchor_voxel_ix = volume_unit_ix * volume_unit_dim
        anchor_voxel_iy = volume_unit_iy * volume_unit_dim
        anchor_voxel_iz = volume_unit_iz * volume_unit_dim
        anchor_position = np.array([anchor_voxel_ix,
                                    anchor_voxel_iy,
                                    anchor_voxel_iz])*voxel_size_mm
        
        all_tsdf = np.zeros((volume_unit_dim+1,
                               volume_unit_dim+1,
                               volume_unit_dim+1))
        typeoftsdf = type(volume_units[volume_unit_ix, 
                                       volume_unit_iy, 
                                       volume_unit_iz].D)
        if typeoftsdf is not np.ndarray:
            all_tsdf[:volume_unit_dim,
                     :volume_unit_dim,
                     :volume_unit_dim] = volume_units[volume_unit_ix, 
                                                    volume_unit_iy, 
                                                    volume_unit_iz].D.detach()
        else:
            all_tsdf[:volume_unit_dim,
                     :volume_unit_dim,
                     :volume_unit_dim] = volume_units[volume_unit_ix, 
                                                    volume_unit_iy, 
                                                    volume_unit_iz].D
                                                          
        for i in range(1,8):
            sx, sy, sz = shift[i]
            if (volume_unit_ix + sx, volume_unit_iy + sy, volume_unit_iz + sz) in volume_units.keys():
                if typeoftsdf is not np.ndarray:
                    all_tsdf[-sx:volume_unit_dim+sx,
                             -sy:volume_unit_dim+sy,
                             -sz:volume_unit_dim+sz] = volume_units[volume_unit_ix+sx,
                                                                   volume_unit_iy+sy,
                                                                   volume_unit_iz+sz].D[:sx or volume_unit_dim,
                                                                                         :sy or volume_unit_dim,
                                                                                         :sz or volume_unit_dim].detach()
                else:
                    all_tsdf[-sx:volume_unit_dim+sx,
                             -sy:volume_unit_dim+sy,
                             -sz:volume_unit_dim+sz] = volume_units[volume_unit_ix+sx,
                                                                   volume_unit_iy+sy,
                                                                   volume_unit_iz+sz].D[:sx or volume_unit_dim,
                                                                                         :sy or volume_unit_dim,
                                                                                         :sz or volume_unit_dim]
    
        # 볼륨 유닛 안의 모든 복셀들의 인접 8개쌍 부호
        tsdf_cubes = np.zeros((volume_unit_dim,
                               volume_unit_dim,
                               volume_unit_dim,
                               8))
        # vertex를 지정하지 않을 곳: 해당 큐브에 nan값이 껴있을 경우???
        nan_cubes = np.ones((volume_unit_dim,
                              volume_unit_dim,
                              volume_unit_dim))
        
        # 인접 복셀의 부호
        for i in range(8):
            sx, sy, sz = shift[i]
            tsdf_cubes[:,:,:,i] = all_tsdf[sx:volume_unit_dim+sx,
                                            sy:volume_unit_dim+sy,
                                            sz:volume_unit_dim+sz]
            if (volume_unit_ix + sx, volume_unit_iy + sy, volume_unit_iz + sz) not in volume_units.keys():
                nan_cubes[-sx:,-sy:,-sz:] *= 0
        sign_cubes = tsdf_cubes>=0
    
        # 부호에 따라 마칭큐브 인덱스를 결정
        mc_ids = np.packbits(sign_cubes, bitorder='little') * nan_cubes.flatten().astype(int)
        # 각 엣지의 위치를 결정하기 위해 위치.. 위치...
        pos_i = pos_s + anchor_position
        
        # 부호와 TSDF에 따라 12개 엣지의 버텍스 유무 및 위치를 결정
        v_s = np.empty((volume_unit_dim,
                        volume_unit_dim,
                        volume_unit_dim,
                        12, 3))
        
        # tsdf 열람
        tsdf_s = np.empty((volume_unit_dim,
                        volume_unit_dim,
                        volume_unit_dim,
                        12, 2))
        for i in range(12):
            tsdf_s[:,:,:,i,0] = tsdf_cubes[:,:,:,edge_to_vert[i][0]]
            tsdf_s[:,:,:,i,1] = tsdf_cubes[:,:,:,edge_to_vert[i][1]]
        
        len_s = (np.abs(tsdf_s[:,:,:,:,0])*voxel_size_mm) / (np.abs(tsdf_s[:,:,:,:,0])+np.abs(tsdf_s[:,:,:,:,1]))
        np.nan_to_num(len_s, copy=False, nan=1.0327, posinf=1.0327, neginf=1.0327)
        len_s = np.stack([len_s, len_s, len_s], axis=-1)
        v_s = pos_i[:,:,:,:,0,:] + (len_s * dirs)
        
        # MC 인덱스에 해당하는것만 저장
        # -1 인덱스가 (0,0,0)을 저장하도록 만듦
        v_s = np.concatenate([v_s, np.zeros_like(v_s[:,:,:,-1:,:])], axis=-2)
        # 각 복셀에 대해서, tri_table의 리스트의 노드들
        mc_nodes = map(lambda e: tri_table[e], mc_ids)
        mc_nodes = np.reshape(np.array(list(mc_nodes)), (volume_unit_dim,
                                                         volume_unit_dim,
                                                         volume_unit_dim,
                                                         -1))
        save_vertices = list(map(lambda k: v_s[k][list(mc_nodes[k])], 
                                        np.ndindex(volume_unit_dim,
                                                   volume_unit_dim,
                                                   volume_unit_dim)))
        save_vertices = np.reshape(save_vertices, [-1, 3])
        # RGB 채널은 일단 (0,0,0)을 줌
        save_vertices = np.concatenate([save_vertices, np.zeros_like(save_vertices)], axis=-1)
        save_vertices = save_vertices[~np.all(save_vertices == 0, axis=1)]
        # np.nan_to_num(save_vertices, copy=False, nan=1.0327)
        uniq_v = np.unique(save_vertices, axis=-2)
        ver_dict = {tuple(v):i+mesh.n_vertices for (i,v) in enumerate(uniq_v)}
        mesh.add_vertices(uniq_v)
        voxel.add_vertices(uniq_v)
        
        # face는 버텍스 인덱스를 필요로한다
        save_faces = np.array(list(map(lambda e: ver_dict[tuple(e)], save_vertices)))
        save_faces = np.reshape(save_faces, (-1,3))
        if (1.0327,1.0327,1.0327,0, 0, 0) in ver_dict.keys():
            idx = ver_dict[(1.0327,1.0327,1.0327,0, 0, 0)]
            save_faces = save_faces[~np.any(save_faces==idx, axis=-1)]
        mesh.add_faces(save_faces)
        voxel.add_faces(save_faces)
        return voxel


def singleResMesh(volume_units,
                  dataset,
                  volume_origin,
                  volume_unit_dim,
                  voxel_size_mm=0.004,
                  num_epochs=None,
                  lam=None,
                  fast=False,
                  save=True,
                  evaluate=False,
                  ori_units=None):
    if fast:
        voxelfunc = fastVoxelMesh()
    else:
        voxelfunc = singleVoxelMesh()
    
    mesh = TriMesh()   
    voxel_to_vertices = {}
    ix, iy, iz = np.meshgrid(range(0, volume_unit_dim), 
                             range(0, volume_unit_dim), 
                             range(0, volume_unit_dim), indexing='ij')
    offset_voxel_indices = np.vstack((ix.flatten(), iy.flatten(), iz.flatten()))
    num_voxels_per_volume_unit = volume_unit_dim * volume_unit_dim * volume_unit_dim
    kwargs = dict(volume_units=volume_units,
                  volume_origin=volume_origin,
                  volume_unit_dim=volume_unit_dim,
                  voxel_size_mm=voxel_size_mm,
                  mesh=mesh,
                  voxel_to_vertices=voxel_to_vertices,
                  offset_voxel_indices=offset_voxel_indices,
                  num_voxels_per_volume_unit=num_voxels_per_volume_unit)
    
    if fast:
        ix, iy, iz = np.meshgrid(range(0, volume_unit_dim+1), 
                                 range(0, volume_unit_dim+1), 
                                 range(0, volume_unit_dim+1), indexing='ij')
        offset_voxel_indices = np.vstack((ix.flatten(), iy.flatten(), iz.flatten()))
        offset_positions = np.stack([ix,iy,iz]).transpose([1,2,3,0])*voxel_size_mm
        
        dirs = np.array([[1,0,0],[0,1,0],[1,0,0],[0,1,0],
                         [1,0,0],[0,1,0],[1,0,0],[0,1,0],
                         [0,0,1],[0,0,1],[0,0,1],[0,0,1]])
        
        # 각 엣지의 위치를 결정하기 위해 위치
        pos_s = np.empty((volume_unit_dim,
                        volume_unit_dim,
                        volume_unit_dim,
                        12, 2, 3))
        for i in range(12):
            sx, sy, sz = shift[edge_to_vert[i][0]]
            pos_s[:,:,:,i,0,:] = offset_positions[sx:volume_unit_dim+sx,
                                                sy:volume_unit_dim+sy,
                                                sz:volume_unit_dim+sz,:]
            
            sx, sy, sz = shift[edge_to_vert[i][1]]
            pos_s[:,:,:,i,1,:] = offset_positions[sx:volume_unit_dim+sx,
                                                sy:volume_unit_dim+sy,
                                                sz:volume_unit_dim+sz,:]
            
        kwargs.update(dict(offset_positions=offset_positions,
                           offset_voxel_indices=offset_voxel_indices,
                           dirs=dirs,
                           pos_s=pos_s))
    
    if evaluate:
        hausdorff = [0] # TODO
        chamfer = []
        sizes_en = []
        sizes_ori = []
        count = 0
        
    for k in tqdm(volume_units.keys()):   
        volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
        if len(k)==3:
            kwargs.update(dict(mesh=mesh,
                               volume_unit_ix=volume_unit_ix,
                               volume_unit_iy=volume_unit_iy,
                               volume_unit_iz=volume_unit_iz,
                               volume_units=volume_units))
        else:
            kwargs.update(dict(mesh=mesh,
                               k=k,
                               volume_units=volume_units))
        voxel_rec = voxelfunc(**kwargs)
    
        if evaluate:
            kwargs.update(dict(volume_units=ori_units,
                               mesh=TriMesh()))
            voxel_ori = voxelfunc(**kwargs)
            length = len(np.all(voxel_ori.vertices[:,:3], axis=1))
            ori_v = voxel_ori.vertices[:,:3]
            ori_v = ori_v[:length]
            rec_v = voxel_rec.vertices[:,:3]
            rec_v = rec_v[:length]
        
            # hausdorff += [igl.hausdorff(rec_v,
            #                             voxel_rec.faces[:length],
            #                             ori_v,
            #                             voxel_ori.faces[:length])]
            hausdorff += [customHausdorff(voxel_rec, voxel_ori)]
            # chamfer_dist = ChamferDistance()
            # dist1 = chamfer_dist(torch.Tensor([ori_v]), 
            #                      torch.Tensor([rec_v]))
            # dist2 = chamfer_dist(torch.Tensor([rec_v]), 
            #                      torch.Tensor([ori_v]))
            # chamfer += [0.5*(torch.mean(dist1) + torch.mean(dist2))]
            chamfer += [customChamfer(voxel_rec, voxel_ori)]
            sizes_en += [volume_units[volume_unit_ix, volume_unit_iy, volume_unit_iz].s]
            sizes_ori += [ori_units[volume_unit_ix, volume_unit_iy, volume_unit_iz].s]
            count += 1
        
    if save:
        print("start saving")
        if not os.path.exists('./meshes'):
            os.mkdir('./meshes')
        mesh.save_ply('./meshes/reconstructed_Mesh_%s_%s_%s.ply' % (dataset.scenes, num_epochs, lam))
        print("end saving")
    if evaluate:
        print("maxes:", np.max(hausdorff), np.max(chamfer), np.max(sizes_en), np.max(sizes_ori))
        print("mins:", np.min(hausdorff), np.min(chamfer), np.min(sizes_en), np.min(sizes_ori))
        print("medians:", np.median(hausdorff), np.median(chamfer), np.median(sizes_en), np.median(sizes_ori))
        return mesh, np.average(hausdorff), np.average(chamfer), np.average(sizes_en), np.average(sizes_ori)
    return mesh
