# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:31:05 2022

@author: alienware
"""
import numpy as np
# from tqdm import tqdm

from source.TriMesh import TriMesh
from source.VolumeUnit import *
from source.MCLutClassic import *
from datetime import datetime


def getVoxelValues(volume_values, volume_unit_dim, global_voxel_ix, global_voxel_iy, global_voxel_iz):
    volume_unit_ix = global_voxel_ix // volume_unit_dim
    volume_unit_iy = global_voxel_iy // volume_unit_dim
    volume_unit_iz = global_voxel_iz // volume_unit_dim
    
    local_voxel_ix = global_voxel_ix % volume_unit_dim
    local_voxel_iy = global_voxel_iy % volume_unit_dim
    local_voxel_iz = global_voxel_iz % volume_unit_dim
    
    if (volume_unit_ix, volume_unit_iy, volume_unit_iz) not in volume_values:
        return None, None, None, None, None
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


def singleResMesh(volume_units,
                  dataset,
                  num_epochs,
                  volume_origin,
                  volume_unit_dim,
                  voxel_size_mm=0.004):
                  
    mesh = TriMesh()   
    voxel_to_vertices = {}
    
    ix, iy, iz = np.meshgrid(range(0, volume_unit_dim), 
                             range(0, volume_unit_dim), 
                             range(0, volume_unit_dim), indexing='ij')
    offset_voxel_indices = np.vstack((ix.flatten(), iy.flatten(), iz.flatten()))
    num_voxels_per_volume_unit = volume_unit_dim * volume_unit_dim * volume_unit_dim  
    
    for volume_unit_ix, volume_unit_iy, volume_unit_iz in volume_units.keys():   
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

            volume_units[volume_unit_ix, volume_unit_iy, volume_unit_iz].MC[offset_voxel_ix, offset_voxel_iy, offset_voxel_iz] = marching_cube_index
        
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
                        
                        voxel_to_vertices[e1_voxel_ix, e1_voxel_iy, e1_voxel_iz, e2_voxel_ix, e2_voxel_iy, e2_voxel_iz] = vertex_indicies[j]
                        voxel_to_vertices[e2_voxel_ix, e2_voxel_iy, e2_voxel_iz, e1_voxel_ix, e1_voxel_iy, e1_voxel_iz] = vertex_indicies[j]
                                        
            # mesh connectity 정보를 생성 
            for j in range(0, 12, 3):
                if tri_table[marching_cube_index][j] == -1:
                    break
                
                mesh.add_face(vertex_indicies[tri_table[marching_cube_index][j]],
                              vertex_indicies[tri_table[marching_cube_index][j+2]],
                              vertex_indicies[tri_table[marching_cube_index][j+1]])
                
    mesh.save_ply('reconstruced_Mesh_%s_%d.ply' % (dataset.scenes, num_epochs))
    return mesh

def fastSRMesh(volume_units,
                      dataset,
                      num_epochs,
                      volume_origin,
                      volume_unit_dim,
                      voxel_size_mm=0.004,
                      device='cuda'):
    # this might be fast but it will use large memory (...perchance?)
    mesh = TriMesh()   
    # voxel_to_vertices = {}
    
    ix, iy, iz = np.meshgrid(range(0, volume_unit_dim+1), 
                              range(0, volume_unit_dim+1), 
                              range(0, volume_unit_dim+1), indexing='ij')
    # offset_voxel_indices = np.vstack((ix.flatten(), iy.flatten(), iz.flatten()))
    # num_voxels_per_volume_unit = volume_unit_dim * volume_unit_dim * volume_unit_dim  
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

    for volume_unit_ix, volume_unit_iy, volume_unit_iz in volume_units.keys():
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
        all_tsdf[:volume_unit_dim,
                 :volume_unit_dim,
                 :volume_unit_dim] = volume_units[volume_unit_ix, 
                                                volume_unit_iy, 
                                                volume_unit_iz].D.detach()
        for i in range(1,8):
            sx, sy, sz = shift[i]
            if (volume_unit_ix + sx, volume_unit_iy + sy, volume_unit_iz + sz) in volume_units.keys():
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
        
        # face는 버텍스 인덱스를 필요로한다
        save_faces = np.array(list(map(lambda e: ver_dict[tuple(e)], save_vertices)))
        save_faces = np.reshape(save_faces, (-1,3))
        if (1.0327,1.0327,1.0327,0, 0, 0) in ver_dict.keys():
            idx = ver_dict[(1.0327,1.0327,1.0327,0, 0, 0)]
            save_faces = save_faces[~np.any(save_faces==idx, axis=-1)]
        mesh.add_faces(save_faces)
    now = datetime.now().strftime("%d%m%y%H%M%S")
    mesh.save_ply(now+'_fast_reconstruced_Mesh_%s_%d.ply' % (dataset.scenes, num_epochs))
    return mesh


