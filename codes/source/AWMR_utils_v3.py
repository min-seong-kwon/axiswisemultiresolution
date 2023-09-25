# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:02:49 2022

@author: alienware
"""
import os
import numpy as np
import itertools
from skimage import measure
import open3d as o3d
import igl
from source.ConfigSettings import base_voxel_size, base_volume_unit_dim, baseres # TODO
from source.AWMRblock8x8 import AWMRblock8x8 as AWMRblock
from source.MCLutClassic import edge_table, edge_to_vert, tri_table 
from utils.evalUtils import customChamfer
from source.ChamferDistance import custom_ChamferDistance
from pysdf import SDF
from numba import njit
from skimage import measure
import trimesh


def slice_list(where:str,
               sx: int,
               sy: int,
               sz: int):
    shifts = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1],
                       [1,1,0],
                       [1,0,1],
                       [0,1,1],
                       [1,1,1]])
    if where=="back": # padding slices for brb
        slices_from = [(slice(0, 1), slice(0,sy), slice(0,sz)),
                      (slice(0,sx), slice(0, 1), slice(0,sz)),
                      (slice(0,sx), slice(0,sy), slice(0, 1)),
                      (slice(0, 1), slice(0, 1), slice(0,sz)),
                      (slice(0, 1), slice(0,sy), slice(0, 1)),
                      (slice(0,sx), slice(0, 1), slice(0, 1)),
                      (slice(0, 1), slice(0, 1), slice(0, 1))]
        slices_to  = [(slice(-1,sx+1), slice(0,sy), slice(0,sz)),
                      (slice(0,sx), slice(-1,sy+1), slice(0,sz)),
                      (slice(0,sx), slice(0,sy), slice(-1,sz+1)),
                      (slice(-1,sx+1), slice(-1,sy+1), slice(0,sz)),
                      (slice(-1,sx+1), slice(0,sy), slice(-1,sz+1)),
                      (slice(0,sx), slice(-1,sy+1), slice(-1,sz+1)),
                      (slice(-1,sx+1), slice(-1,sy+1), slice(-1,sz+1))]
        return shifts, slices_from, slices_to
    if where=="front": # padding slices for tlf
        slices_from = [(slice(-1,sx), slice(0,sy), slice(0,sz)),
                      (slice(0,sx), slice(-1,sy), slice(0,sz)),
                      (slice(0,sx), slice(0,sy), slice(-1,sz)),
                      (slice(-1,sx), slice(-1,sy), slice(0,sz)),
                      (slice(-1,sx), slice(0,sy), slice(-1,sz)),
                      (slice(0,sx), slice(-1,sy), slice(-1,sz)),
                      (slice(-1,sx), slice(-1,sy), slice(-1,sz))]
        slices_to  = [(slice(0, 1), slice(1,sy+1), slice(1,sz+1)),
                      (slice(1,sx+1), slice(0, 1), slice(1,sz+1)),
                      (slice(1,sx+1), slice(1,sy+1), slice(0, 1)),
                      (slice(0, 1), slice(0, 1), slice(1,sz+1)),
                      (slice(0, 1), slice(1,sy+1), slice(0, 1)),
                      (slice(1,sx+1), slice(0, 1), slice(0, 1)),
                      (slice(0, 1), slice(0, 1), slice(0, 1))]
        return shifts, slices_from, slices_to


# 두 블럭으로 나누어 공분산 구함
def get_2_complexity(block1, block2):
    assert(block1.shape==block2.shape)
    Gx1, Gy1, Gz1 = np.gradient(block1)
    Gx2, Gy2, Gz2 = np.gradient(block2)
    Gxyz1 = np.stack((Gx1.flatten(), Gy1.flatten(), Gz1.flatten()))
    Gxyz2 = np.stack((Gx2.flatten(), Gy2.flatten(), Gz2.flatten()))
    covG1 = np.cov(Gxyz1)
    covG2 = np.cov(Gxyz2)
    # complexity1 = np.abs(np.linalg.det(covG1))
    # complexity2 = np.abs(np.linalg.det(covG2))
    return covG1, covG2


def get_4_complexity(block1, block2, block3, block4):
    # deprecated
    assert(block1.shape==block2.shape==block3.shape==block4.shape)
    Gx1, Gy1, Gz1 = np.gradient(block1)
    Gx2, Gy2, Gz2 = np.gradient(block2)
    Gx3, Gy3, Gz3 = np.gradient(block3)
    Gx4, Gy4, Gz4 = np.gradient(block4)
    Gxyz1 = np.stack((Gx1.flatten(), Gy1.flatten(), Gz1.flatten()))
    Gxyz2 = np.stack((Gx2.flatten(), Gy2.flatten(), Gz2.flatten()))
    Gxyz3 = np.stack((Gx3.flatten(), Gy3.flatten(), Gz3.flatten()))
    Gxyz4 = np.stack((Gx4.flatten(), Gy4.flatten(), Gz4.flatten()))
    covG1 = np.cov(Gxyz1)
    covG2 = np.cov(Gxyz2)
    covG3 = np.cov(Gxyz3)
    covG4 = np.cov(Gxyz4)
    # complexity1 = np.abs(np.linalg.det(covG1))
    # complexity2 = np.abs(np.linalg.det(covG2))
    # complexity3 = np.abs(np.linalg.det(covG3))
    # complexity4 = np.abs(np.linalg.det(covG4))
    return covG1, covG2, covG3, covG4


def pad_3d(arr, k, dict_arr):
    if type(arr)!=np.ndarray:
        arr = arr.D
    size = arr.shape[0] # for cubic arr
    
    shift2from = {1:slice(0,1), 0:slice(0,size), -1:slice(size-1, size)}
    shift2to = {1:slice(size+1,size+2), 0:slice(1,size+1), -1:slice(0,1)}
    
    scene_name = k[0]
    k = k[-3:]
    
    padded = np.zeros([size+2,size+2,size+2])

    shifts = itertools.product([1,0,-1], [1,0,-1], [1,0,-1])
    for shift in shifts:
        xs, ys, zs = shift
        kshift = tuple(np.array(k)+np.array(shift))
        kshift = (scene_name, *kshift)
        if kshift in dict_arr:
            get_from = dict_arr[kshift]
            if type(get_from)!=np.ndarray:
                get_from = get_from.D
            padded[shift2to[xs],
                   shift2to[ys],
                   shift2to[zs]] = get_from[shift2from[xs],
                                            shift2from[ys],
                                            shift2from[zs]]
    return padded

def get_condition(volume_units_32, volume_units_16, volume_units_8 , k):
	pad8 = pad_3d(volume_units_8[k].D, k, volume_units_8)
	pad16 = pad_3d(volume_units_16[k].D, k, volume_units_16)
	pad32 = pad_3d(volume_units_32[k].D, k, volume_units_32)
	
	condition = (np.min(pad8)*np.max(pad8)>=0) and (np.min(pad16)*np.max(pad16)>=0) and (np.min(pad32)*np.max(pad32)>=0)
	
	return condition

# def get_all_condition_old(volume_units, unit_index):
#     condition_list = []
#     for res in volume_units.keys():
#         pad = pad_3d(volume_units[res][unit_index].D, unit_index, volume_units[res])
#         isntmesh = (np.min(pad)*np.max(pad) >= 0)
#         condition_list.append(isntmesh)
#     condition = np.logical_and(*condition_list)
#     return condition

def get_all_condition(volume_units, k):
    condition_list = []
    for res in volume_units.keys():
        current_block = volume_units[res][k].D
        isntmesh = (np.min(current_block)*np.max(current_block)>=0)
        condition_list.append(isntmesh)
    condition_list.append(get_condition(volume_units['32_32_32'],
                                   volume_units['16_16_16'],
                                   volume_units['8_8_8'],
                                   k))
    condition = all(condition_list)
    return condition

# 주변 메쉬 패딩
def get_padded(vunit, k, vunits_ori, for_mask=False):
    # this function pads a AWMR block from highest resolution volume_units dict
    # this is dirty and may cause error, but until now, it worked fine
    # IMPORTANT this code ONLY pads in back-right-bottom direction
    # IMPORTANT this code AUTOMATICALLY clips out padded face
    # if the target volume unit does not exist
    # IMPORTANT this code can, and will ONLY pad depth
    # if other attributes are needed to be padded, refer to
    # getPaddedAttr() in RGBD2TSDFFusion.py and implement similarly
    
    assert len(k)==4, "please use length 4 type key for volume units"
    if type(vunit)==np.ndarray:
        sx, sy, sz = vunit.shape
    else:
        sx, sy, sz = vunit.D.shape
    
    ks = [(k[-3]+1, k[-2], k[-1]),
          (k[-3], k[-2]+1, k[-1]),
          (k[-3], k[-2], k[-1]+1),
          (k[-3]+1, k[-2]+1, k[-1]),
          (k[-3]+1, k[-2], k[-1]+1),
          (k[-3], k[-2]+1, k[-1]+1),
          (k[-3]+1, k[-2]+1, k[-1]+1)]
    
    scene_name = k[0]
    ks_exists = [(scene_name, *kd) in vunits_ori.keys() for kd in ks]
    
    depth_new = np.ones([sx+1,sy+1,sz+1])*505
    if type(vunit)==np.ndarray:
        depth_new[:sx,:sy,:sz] = vunit
    else:
        depth_new[:sx,:sy,:sz] = vunit.D
    if sx==sy==sz:
        if ks_exists[0]:
            depth_new[-1, :sy, :sz] = vunits_ori[k[0], k[1]+1, k[2], k[3]].D[0,:,:]
        if ks_exists[1]:
            depth_new[:sx, -1, :sz] = vunits_ori[k[0], k[1], k[2]+1, k[3]].D[:,0,:]
        if ks_exists[2]:
            depth_new[:sx, :sy, -1] = vunits_ori[k[0], k[1], k[2], k[3]+1].D[:,:,0]
        if ks_exists[3]:
            depth_new[-1, -1, :sz] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]].D[0,0,:]
        if ks_exists[4]:
            depth_new[-1, :sy, -1] = vunits_ori[k[0], k[1]+1, k[2], k[3]+1].D[0,:,0]
        if ks_exists[5]:
            depth_new[:sx, -1, -1] = vunits_ori[k[0], k[1], k[2]+1, k[3]+1].D[:,0,0]
        if ks_exists[6]:
            depth_new[-1, -1, -1] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]+1].D[0,0,0]
            
        # if for_mask:
        #     ks_fw = [(k[-3]-1, k[-2], k[-1]),
        #             (k[-3], k[-2]-1, k[-1]),
        #             (k[-3], k[-2], k[-1]-1),
        #             (k[-3]-1, k[-2]-1, k[-1]),
        #             (k[-3]-1, k[-2], k[-1]-1),
        #             (k[-3], k[-2]-1, k[-1]-1),
        #             (k[-3]-1, k[-2]-1, k[-1]-1)]
        #     ks_fw_exists = [(scene_name, *kd) in vunits_ori.keys() for kd in ks_fw]
        #     depth_new_fw = np.ones([sx+2,sy+2,sz+2]) * 327
        #     depth_new_fw[1:, 1:, 1:] = depth_new
        #     if ks_fw_exists[0]:
        #         depth_new_fw[0, 1:1+sy, 1:1+sz] = vunits_ori[k[0], k[1]+1, k[2], k[3]].D[-1,:,:]
        #     if ks_fw_exists[1]:
        #         depth_new_fw[1:1+sx, 0, 1:1+sz] = vunits_ori[k[0], k[1], k[2]+1, k[3]].D[:,-1,:]
        #     if ks_fw_exists[2]:
        #         depth_new_fw[1:1+sx, 1:1+sy, 0] = vunits_ori[k[0], k[1], k[2], k[3]+1].D[:,:,-1]
        #     if ks_fw_exists[3]:
        #         depth_new_fw[0, 0, 1:1+sz] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]].D[-1,-1,:]
        #     if ks_fw_exists[4]:
        #         depth_new_fw[0, 1:1+sy, 0] = vunits_ori[k[0], k[1]+1, k[2], k[3]+1].D[-1,:,-1]
        #     if ks_fw_exists[5]:
        #         depth_new_fw[1:1+sx, 0, 0] = vunits_ori[k[0], k[1], k[2]+1, k[3]+1].D[:,-1,-1]
        #     if ks_fw_exists[6]:
        #         depth_new_fw[0, 0, 0] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]+1].D[-1,-1,-1]
        #     return depth_new_fw
            
    elif sx==sy and sz//2==sx:
        if ks_exists[0]:
            depth_new[-1, :sy, :sz] = vunits_ori[k[0], k[1]+1, k[2], k[3]].D[0,::2,:]
        if ks_exists[1]:
            depth_new[:sx, -1, :sz] = vunits_ori[k[0], k[1], k[2]+1, k[3]].D[::2,0,:]
        if ks_exists[2]:
            depth_new[:sx, :sy, -1] = vunits_ori[k[0], k[1], k[2], k[3]+1].D[::2,::2,0]
        if ks_exists[3]:
            depth_new[-1, -1, :sz] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]].D[0,0,:]
        if ks_exists[4]:
            depth_new[-1, :sy, -1] = vunits_ori[k[0], k[1]+1, k[2], k[3]+1].D[0,::2,0]
        if ks_exists[5]:
            depth_new[:sx, -1, -1] = vunits_ori[k[0], k[1], k[2]+1, k[3]+1].D[::2,0,0]
        if ks_exists[6]:
            depth_new[-1, -1, -1] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]+1].D[0,0,0]
    elif sx==sz and sy//2==sx:
        if ks_exists[0]:
            depth_new[-1, :sy, :sz] = vunits_ori[k[0], k[1]+1, k[2], k[3]].D[0,:,::2]
        if ks_exists[1]:
            depth_new[:sx, -1, :sz] = vunits_ori[k[0], k[1], k[2]+1, k[3]].D[::2,0,::2]
        if ks_exists[2]:
            depth_new[:sx, :sy, -1] = vunits_ori[k[0], k[1], k[2], k[3]+1].D[::2,:,0]
        if ks_exists[3]:
            depth_new[-1, -1, :sz] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]].D[0,0,::2]
        if ks_exists[4]:
            depth_new[-1, :sy, -1] = vunits_ori[k[0], k[1]+1, k[2], k[3]+1].D[0,:,0]
        if ks_exists[5]:
            depth_new[:sx, -1, -1] = vunits_ori[k[0], k[1], k[2]+1, k[3]+1].D[::2,0,0]
        if ks_exists[6]:
            depth_new[-1, -1, -1] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]+1].D[0,0,0]
    elif sy==sz and sx//2==sz:
        if ks_exists[0]:
            depth_new[-1, :sy, :sz] = vunits_ori[k[0], k[1]+1, k[2], k[3]].D[0,::2,::2]
        if ks_exists[1]:
            depth_new[:sx, -1, :sz] = vunits_ori[k[0], k[1], k[2]+1, k[3]].D[:,0,::2]
        if ks_exists[2]:
            depth_new[:sx, :sy, -1] = vunits_ori[k[0], k[1], k[2], k[3]+1].D[:,::2,0]
        if ks_exists[3]:
            depth_new[-1, -1, :sz] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]].D[0,0,::2]
        if ks_exists[4]:
            depth_new[-1, :sy, -1] = vunits_ori[k[0], k[1]+1, k[2], k[3]+1].D[0,::2,0]
        if ks_exists[5]:
            depth_new[:sx, -1, -1] = vunits_ori[k[0], k[1], k[2]+1, k[3]+1].D[:,0,0]
        if ks_exists[6]:
            depth_new[-1, -1, -1] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]+1].D[0,0,0]
    elif sx==sy and sz//4==sx:
        if ks_exists[0]:
            depth_new[-1, :sy, :sz] = vunits_ori[k[0], k[1]+1, k[2], k[3]].D[0,1::4,:]
        if ks_exists[1]:
            depth_new[:sx, -1, :sz] = vunits_ori[k[0], k[1], k[2]+1, k[3]].D[1::4,0,:]
        if ks_exists[2]:
            depth_new[:sx, :sy, -1] = vunits_ori[k[0], k[1], k[2], k[3]+1].D[1::4,1::4,0]
        if ks_exists[3]:
            depth_new[-1, -1, :sz] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]].D[0,0,:]
        if ks_exists[4]:
            depth_new[-1, :sy, -1] = vunits_ori[k[0], k[1]+1, k[2], k[3]+1].D[0,1::4,0]
        if ks_exists[5]:
            depth_new[:sx, -1, -1] = vunits_ori[k[0], k[1], k[2]+1, k[3]+1].D[1::4,0,0]
        if ks_exists[6]:
            depth_new[-1, -1, -1] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]+1].D[0,0,0]
    elif sx==sz and sy//4==sx:
        if ks_exists[0]:
            depth_new[-1, :sy, :sz] = vunits_ori[k[0], k[1]+1, k[2], k[3]].D[0,:,1::4]
        if ks_exists[1]:
            depth_new[:sx, -1, :sz] = vunits_ori[k[0], k[1], k[2]+1, k[3]].D[::4,0,1::4]
        if ks_exists[2]:
            depth_new[:sx, :sy, -1] = vunits_ori[k[0], k[1], k[2], k[3]+1].D[1::4,:,0]
        if ks_exists[3]:
            depth_new[-1, -1, :sz] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]].D[0,0,1::4]
        if ks_exists[4]:
            depth_new[-1, :sy, -1] = vunits_ori[k[0], k[1]+1, k[2], k[3]+1].D[0,:,0]
        if ks_exists[5]:
            depth_new[:sx, -1, -1] = vunits_ori[k[0], k[1], k[2]+1, k[3]+1].D[1::4,0,0]
        if ks_exists[6]:
            depth_new[-1, -1, -1] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]+1].D[0,0,0]
    elif sy==sz and sx//4==sz:
        if ks_exists[0]:
            depth_new[-1, :sy, :sz] = vunits_ori[k[0], k[1]+1, k[2], k[3]].D[0,1::4,1::4]
        if ks_exists[1]:
            depth_new[:sx, -1, :sz] = vunits_ori[k[0], k[1], k[2]+1, k[3]].D[:,0,1::4]
        if ks_exists[2]:
            depth_new[:sx, :sy, -1] = vunits_ori[k[0], k[1], k[2], k[3]+1].D[:,1::4,0]
        if ks_exists[3]:
            depth_new[-1, -1, :sz] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]].D[0,0,1::4]
        if ks_exists[4]:
            depth_new[-1, :sy, -1] = vunits_ori[k[0], k[1]+1, k[2], k[3]+1].D[0,1::4,0]
        if ks_exists[5]:
            depth_new[:sx, -1, -1] = vunits_ori[k[0], k[1], k[2]+1, k[3]+1].D[:,0,0]
        if ks_exists[6]:
            depth_new[-1, -1, -1] = vunits_ori[k[0], k[1]+1, k[2]+1, k[3]+1].D[0,0,0]
    else:
        print(sx, sy, sz)
    
    if not ks_exists[0]:# or not ks_exists[3] or not ks_exists[4]:
       depth_new = depth_new[:sx, :, :]
    if not ks_exists[1]:# or not ks_exists[3] or not ks_exists[5]:
       depth_new = depth_new[:, :sy, :]
    if not ks_exists[2]:# or not ks_exists[4] or not ks_exists[5]:
       depth_new = depth_new[:, :, :sz]
    
    # depth_new[depth_new==0] 는 hausdorff distance에ㅔ는 영향을 미치지 않을 것임
    return depth_new

    
def make_mesh(depth, 
              axisres, 
              voxel_size=base_voxel_size, 
              volume_unit_dim=base_volume_unit_dim, 
              volume_origin=np.array((0,0,0)),
              k=(0,0,0),
              start_point=(0,0,0),
              allow_degenerate=True):
    if np.min(depth)*np.max(depth) >= 0:
        return o3d.geometry.TriangleMesh()
    verts, faces, _, _ = measure.marching_cubes(depth,
                                                0,
                                                allow_degenerate=allow_degenerate)
    verts = np.divide(verts, axisres)*voxel_size*volume_unit_dim \
          + volume_origin.T \
          + volume_unit_dim*voxel_size*np.array(k[-3:]) \
          + np.array(start_point)*volume_unit_dim*voxel_size
    # Use Open3D for visualization
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    # o3d.visualization.draw_geometries([mesh])
    return mesh


def generate_mask(arr: np.ndarray):
    # this function assumes that arr is 6-way padded
    left = np.sign(arr[:-2, 1:-1, 1:-1])
    right = np.sign(arr[2:, 1:-1, 1:-1])
    top = np.sign(arr[1:-1, :-2, 1:-1])
    bottom = np.sign(arr[1:-1, 2:, 1:-1])
    front = np.sign(arr[1:-1, 1:-1, :-2])
    back = np.sign(arr[1:-1, 1:-1, 2:])
    center = np.sign(arr[1:-1, 1:-1, 1:-1])
    
    l = (left*center)<0
    r = (right*center)<0
    t = (top*center)<0
    b = (bottom*center)<0
    f = (front*center)<0
    B = (back*center)<0
    
    mask = (l + r + t + b + f + B) > 0
    
    return mask


def split_until_thres(awmrblock,
                    thres,
                    volume_units,
                    axisres: np.array, 
                    unit_index, 
                    start_point=np.array((0,0,0)),
                    baseres=8,
                    max_res=32,
                    for_train=False
                    ):
    version=2
    # mask generation, if for_train is set to True
    if for_train and (awmrblock.tsdf.M is None):
        pad_for_mask = pad_awmr_from_original(awmrblock.tsdf.D,
                                              volume_units,
                                              axisres,
                                              unit_index,
                                              start_point,
                                              for_mask=True)
        awmrblock.tsdf.M = generate_mask(pad_for_mask)

    # 원본(최고해상도) TSDF를 가져와서 해당 블록에 맞게 쪼개고 pad합니다.
    # 기하정확도의 기준이 되는 원본 메쉬를 만듭니다.
    vunit_ori = volume_units['32_32_32'][unit_index]
    ori_shape = np.array(vunit_ori.D.shape)
    pad_tsdf = get_padded(vunit_ori, unit_index, volume_units['32_32_32'])
    x_width, y_width, z_width = (baseres*ori_shape/axisres).astype(int)
    x_start, y_start, z_start = (start_point*ori_shape).astype(int)
    ori_tsdf = pad_tsdf[x_start:x_start+x_width+1,
                        y_start:y_start+y_width+1,
                        z_start:z_start+z_width+1]
    
    ori_mesh = make_mesh(ori_tsdf,
                     ori_shape, 
                     voxel_size=base_voxel_size, 
                     volume_unit_dim=base_volume_unit_dim, 
                     volume_origin=np.array((0,0,0)),
                     k=unit_index,
                     start_point=start_point)
    # 해당 블록에 GT 메쉬가 없다면, 자동으로 현재 해상도로 저장합니다
    if len(np.asarray(ori_mesh.vertices))==0:
        return

    current_block = get_TSDF_block(volume_units,
                           axisres,
                           unit_index,
                           start_point=start_point,
                           baseres=baseres)
    
    current_pad = pad_awmr_from_original(current_block,
                                volume_units,
                                axisres, 
                                unit_index, 
                                start_point)
    
    current_mesh = make_mesh(current_pad,
                        axisres, 
                        voxel_size=base_voxel_size, 
                        volume_unit_dim=base_volume_unit_dim, 
                        volume_origin=np.array((0,0,0)), # 주의
                        k=unit_index,
                        start_point=start_point)
    
    if len(np.asarray(current_mesh.vertices)):
        if version==1:
            currenterror = customChamfer(ori_mesh, current_mesh)
        else:
            currenterror = custom_ChamferDistance(ori_mesh,current_mesh)
    else:
        currenterror = np.inf
        
    if currenterror < thres:
        return
    
    # 해당 블록을 x, y, z 각 방향으로 한 단계씩 해상도를 높여서 기하 오차를 잽니다
    split_tsdf = {}
    for ax in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
        newaxisres = (ax+np.array([1,1,1]))*axisres
        if max(newaxisres) <= max_res:
            start_point1 = start_point
            start_point2 = np.array(start_point) + ax*baseres*(1/newaxisres)
            
            block1 = get_TSDF_block(volume_units,
                                newaxisres,
                                unit_index,
                                start_point=start_point1)
            
            block2 = get_TSDF_block(volume_units,
                                newaxisres,
                                unit_index,
                                start_point=start_point2)
            
            pad1 = pad_awmr_from_original(block1,
                                        volume_units,
                                        newaxisres, 
                                        unit_index, 
                                        start_point1)
            # print(pad1)
            pad2 = pad_awmr_from_original(block2,
                                        volume_units,
                                        newaxisres, 
                                        unit_index, 
                                        start_point2)
            cutx, cuty, cutz = np.array(block1.shape) + np.array([1,1,1]) - ax
            pad1 = pad1[:cutx, :cuty, :cutz] # pad2와 중복되는 부분을 제거
            pad_new = np.concatenate([pad1,pad2], axis=np.argmax(ax))
            newmesh = make_mesh(pad_new,
                                newaxisres, 
                                voxel_size=base_voxel_size, 
                                volume_unit_dim=base_volume_unit_dim, 
                                volume_origin=np.array((0,0,0)), # 주의
                                k=unit_index,
                                start_point=start_point)
            # 해당 axisres의 mesh가 존재할 경우, 기하정확도를 잰다.
            # mesh가 존재하지 않으면 기하정확도는 inf이다.
            if len(np.asarray(newmesh.vertices)):
                if version == 1:
                    error = customChamfer(ori_mesh, newmesh)
                else:
                    error = custom_ChamferDistance(ori_mesh,newmesh)
            else:
                error = np.inf
            split_tsdf[tuple(ax)] = error
    
    if len(split_tsdf)==0:
        return
    
    minerroraxis = min(split_tsdf, key=split_tsdf.get)
    minerror = split_tsdf[minerroraxis]
    newaxisres = np.array(axisres)*(np.array(minerroraxis)+np.array([1,1,1]))
    start_point1 = start_point
    start_point2 = np.array(start_point) + np.array(minerroraxis)*baseres*(1/np.array(newaxisres))
    tsdf1 = get_TSDF_block(volume_units,
                        newaxisres,
                        unit_index,
                        start_point=start_point1)

    pad1 = pad_awmr_from_original(tsdf1,
                                volume_units,
                                newaxisres, 
                                unit_index, 
                                start_point1,
                                True)
    
    if np.sign(np.min(pad1))*np.sign(np.max(pad1)) <= 0:
        awmrblock.left = AWMRblock(newaxisres, 
                                   unit_index, 
                                   start_point=start_point1,
                                   tsdf=tsdf1)
        if for_train:
            awmrblock.left.tsdf.M = generate_mask(pad1)
            
    tsdf2 = get_TSDF_block(volume_units,
                        newaxisres,
                        unit_index,
                        start_point=start_point2)
    
    pad2 = pad_awmr_from_original(tsdf2,
                                volume_units,
                                newaxisres, 
                                unit_index, 
                                start_point2,
                                True)

    if np.sign(np.min(pad2))*np.sign(np.max(pad2)) <= 0:
        awmrblock.right = AWMRblock(newaxisres, 
                                    unit_index, 
                                    start_point=start_point2,
                                    tsdf=tsdf2)
        if for_train:
            awmrblock.right.tsdf.M = generate_mask(pad2)
            
    if (awmrblock.left is None) and (awmrblock.right is None):
        print(f"somehow, both child is gone at: {unit_index}, start point: {start_point}, axiswise resolution :{axisres}")
        return
    
    awmrblock.tsdf = None
    
    # if minerror>thres:
    if True:
        if awmrblock.left is not None:
            split_until_thres(awmrblock=awmrblock.left,
                            thres=thres,
                            volume_units=volume_units,
                            axisres=newaxisres,
                            unit_index=unit_index,
                            start_point=start_point1,
                            baseres=baseres,
                            max_res=max_res,
                            for_train=for_train)

        if awmrblock.right is not None:
            split_until_thres(awmrblock=awmrblock.right,
                            thres=thres,
                            volume_units=volume_units,
                            axisres=newaxisres,
                            unit_index=unit_index,
                            start_point=start_point2,
                            baseres=baseres,
                            max_res=max_res,
                            for_train=for_train)
    return


def split_octree(awmrblock,
                    thres,
                    volume_units,
                    axisres: np.array, 
                    unit_index, 
                    start_point=np.array((0,0,0)),
                    baseres=8,
                    max_res=32,
                    for_train=False
                    ):
    version=2
    # mask generation, if for_train is set to True
    if for_train and (awmrblock.tsdf.M is None):
        pad_for_mask = pad_awmr_from_original(awmrblock.tsdf.D,
                                              volume_units,
                                              axisres,
                                              unit_index,
                                              start_point,
                                              for_mask=True)
        awmrblock.tsdf.M = generate_mask(pad_for_mask)

    # 원본(최고해상도) TSDF를 가져와서 해당 블록에 맞게 쪼개고 pad합니다.
    # 기하정확도의 기준이 되는 원본 메쉬를 만듭니다.
    vunit_ori = volume_units['32_32_32'][unit_index]
    ori_shape = np.array(vunit_ori.D.shape)
    pad_tsdf = get_padded(vunit_ori, unit_index, volume_units['32_32_32'])
    x_width, y_width, z_width = (baseres*ori_shape/axisres).astype(int)
    x_start, y_start, z_start = (start_point*ori_shape).astype(int)
    ori_tsdf = pad_tsdf[x_start:x_start+x_width+1,
                        y_start:y_start+y_width+1,
                        z_start:z_start+z_width+1]
    
    ori_mesh = make_mesh(ori_tsdf,
                     ori_shape, 
                     voxel_size=base_voxel_size, 
                     volume_unit_dim=base_volume_unit_dim, 
                     volume_origin=np.array((0,0,0)),
                     k=unit_index,
                     start_point=start_point)
    ######################################################
    split_tsdf = {}
    if np.all(axisres == axisres[0]): # 축별 해상도가 같을 때 
        current_block = get_TSDF_block(volume_units,
                            axisres,
                            unit_index,
                            start_point=np.array([0,0,0]),
                            baseres=baseres)
        
        current_pad = pad_awmr_from_original(current_block,
                                    volume_units,
                                    axisres, 
                                    unit_index, 
                                    start_point)
        
        current_mesh = make_mesh(current_pad,
                            axisres, 
                            voxel_size=base_voxel_size, 
                            volume_unit_dim=base_volume_unit_dim, 
                            volume_origin=np.array((0,0,0)), # 주의
                            k=unit_index,
                            start_point=start_point)
        

        if  len(np.asarray(ori_mesh.vertices))==0 or \
            len(np.asarray(current_mesh.vertices))==0:
            return
        else: # currentmesh와 mesh 모두 존재한다고 가정
            if version==1:
                currenterror = customChamfer(ori_mesh,current_mesh)
            else:
                currenterror = custom_ChamferDistance(ori_mesh,current_mesh)
        
        if currenterror < thres or axisres[0] == max_res:
            return
        else: # 한 단계 내려가겠다
            all_axis = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    

    else: # 축별 해상도가 다를 때
        minidx = np.argmin(axisres)
        minerroraxis = [0,0,0]
        minerroraxis[minidx] = 1
        minerroraxis = np.array(minerroraxis)
        all_axis = [minerroraxis] # 이경우 all_axis = [array([0,1,0])] 형태
    
    for ax in all_axis:
        newaxisres = (ax+np.array([1,1,1]))*np.array(axisres)
        if max(newaxisres) <= max_res:
            start_point1 = start_point
            start_point2 = np.array(start_point) + ax*baseres*(1/newaxisres)
            block1 = get_TSDF_block(volume_units,
                                newaxisres,
                                unit_index,
                                start_point=start_point1)
            
            block2 = get_TSDF_block(volume_units,
                                newaxisres,
                                unit_index,
                                start_point=start_point2)
            
            pad1 = pad_awmr_from_original(block1,
                                        volume_units,
                                        newaxisres, 
                                        unit_index, 
                                        start_point1)
            # print(pad1)
            pad2 = pad_awmr_from_original(block2,
                                        volume_units,
                                        newaxisres, 
                                        unit_index, 
                                        start_point2)
            cutx, cuty, cutz = np.array(block1.shape) + np.array([1,1,1]) - ax
            pad1 = pad1[:cutx, :cuty, :cutz] # pad2와 중복되는 부분을 제거
            pad_new = np.concatenate([pad1,pad2], axis=np.argmax(ax))
            newmesh = make_mesh(pad_new,
                                newaxisres, 
                                voxel_size=base_voxel_size, 
                                volume_unit_dim=base_volume_unit_dim, 
                                volume_origin=np.array((0,0,0)), # 주의
                                k=unit_index,
                                start_point=start_point)
            # 해당 axisres의 mesh가 존재할 경우, 기하정확도를 잰다.
            # mesh가 존재하지 않으면 기하정확도는 inf이다.
            
            if  len(np.asarray(ori_mesh.vertices))==0 or \
                len(np.asarray(newmesh.vertices))==0:
                error = np.inf
            else:
                if version==1:
                    error = customChamfer(ori_mesh, newmesh)
                else:
                    error = custom_ChamferDistance(ori_mesh, newmesh)
                
            split_tsdf[tuple(ax)] = error

    minerroraxis = min(split_tsdf, key=split_tsdf.get)
    minerror = split_tsdf[minerroraxis]

    newaxisres = np.array(axisres)*(np.array(minerroraxis)+np.array([1,1,1]))
    start_point1 = start_point
    start_point2 = np.array(start_point) + np.array(minerroraxis)*baseres*(1/np.array(newaxisres))
    
    tsdf1 = get_TSDF_block(volume_units,
                        newaxisres,
                        unit_index,
                        start_point=start_point1)

    pad1 = pad_awmr_from_original(tsdf1,
                                volume_units,
                                newaxisres, 
                                unit_index, 
                                start_point1,
                                True)
    if np.sign(np.min(pad1))*np.sign(np.max(pad1)) <= 0:
        awmrblock.left = AWMRblock(newaxisres, 
                                   unit_index, 
                                   start_point=start_point1,
                                   tsdf=tsdf1)
        if for_train:
            awmrblock.left.tsdf.M = generate_mask(pad1)
        
    tsdf2 = get_TSDF_block(volume_units,
                        newaxisres,
                        unit_index,
                        start_point=start_point2)
    
    pad2 = pad_awmr_from_original(tsdf2,
                                volume_units,
                                newaxisres, 
                                unit_index, 
                                start_point2,
                                True)
    
    if np.sign(np.min(pad2))*np.sign(np.max(pad2)) <= 0:
        awmrblock.right = AWMRblock(newaxisres, 
                                    unit_index, 
                                    start_point=start_point2,
                                    tsdf=tsdf2)
        if for_train:
            awmrblock.right.tsdf.M = generate_mask(pad2)
            
    if (awmrblock.left is None) and (awmrblock.right is None):
        print(f"somehow, both child is gone at: {unit_index}, start point: {start_point}, axiswise resolution :{axisres}")
        return
    
    awmrblock.tsdf = None
    
    if (minerror>thres) or not np.all(newaxisres == newaxisres[0]):
        if awmrblock.left is not None:
            split_octree(awmrblock=awmrblock.left,
                    thres=thres,
                    volume_units=volume_units,
                    axisres=newaxisres, 
                    unit_index=unit_index, 
                    start_point=start_point1,
                    baseres=baseres,
                    max_res=max_res,
                    for_train=for_train)
            
        if awmrblock.right is not None:
            split_octree(awmrblock=awmrblock.right,
                    thres=thres,
                    volume_units=volume_units,
                    axisres=newaxisres, 
                    unit_index=unit_index, 
                    start_point=start_point2,
                    baseres=baseres,
                    max_res=max_res,
                    for_train=for_train)
    return

def split_into_resolution(awmrblock, 
                        thres,
                        volume_units,
                        axisres, 
                        unit_index, 
                        start_point=np.array((0,0,0)),
                        final_res=np.array((16,16,16)),
                        for_train=False):
    thres = -np.inf
    
    # mask generation, if for_train is set to True
    if for_train and (awmrblock.tsdf.M is None):
        pad_for_mask = pad_awmr_from_original(awmrblock.tsdf.D,
                                              volume_units,
                                              axisres,
                                              unit_index,
                                              start_point,
                                              for_mask=True)
        awmrblock.tsdf.M = generate_mask(pad_for_mask)
    
    # max_res에 다다르기 전까지 계속 쪼갤 수 있도록 axisres를 체크
    can_split = False
    for ax in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
        newaxisres = (ax+np.array([1,1,1]))*axisres
        if np.all(newaxisres<=final_res):
            can_split = True
            break
    if can_split is False:
        return
    
    minerroraxis = ax
    minerror = 1
    newaxisres = np.array(axisres)*(np.array(minerroraxis)+np.array([1,1,1]))
    start_point1 = start_point
    start_point2 = np.array(start_point) + np.array(minerroraxis)*baseres*(1/np.array(newaxisres))
    
    tsdf1 = get_TSDF_block(volume_units,
                        newaxisres,
                        unit_index,
                        start_point=start_point1)
    pad1 = pad_awmr_from_original(tsdf1,
                                  volume_units,
                                  newaxisres, 
                                  unit_index, 
                                  start_point1,
                                  True)
    if np.sign(np.min(pad1))*np.sign(np.max(pad1)) <= 0:
        awmrblock.left = AWMRblock(newaxisres, 
                                   unit_index, 
                                   start_point=start_point1,
                                   tsdf=tsdf1)
        if for_train:
            awmrblock.left.tsdf.M = generate_mask(pad1)
        
    tsdf2 = get_TSDF_block(volume_units,
                        newaxisres,
                        unit_index,
                        start_point=start_point2)
    
    pad2 = pad_awmr_from_original(tsdf2,
                                  volume_units,
                                  newaxisres, 
                                  unit_index, 
                                  start_point2,
                                  True)
    if np.sign(np.min(pad2))*np.sign(np.max(pad2)) <= 0:
        awmrblock.right = AWMRblock(newaxisres, 
                                    unit_index, 
                                    start_point=start_point2,
                                    tsdf=tsdf2)
        if for_train:
            awmrblock.right.tsdf.M = generate_mask(pad2)
            
    if (awmrblock.left is None) and (awmrblock.right is None):
        print(f"somehow, both child is gone at: {unit_index}, start point: {start_point}, axiswise resolution :{axisres}")
        return
    
    awmrblock.tsdf = None
    
    if minerror>thres:
        if awmrblock.left is not None:
            split_into_resolution(awmrblock=awmrblock.left, 
                                thres=thres,
                                volume_units=volume_units,
                                axisres=newaxisres, 
                                unit_index=unit_index, 
                                start_point=start_point1,
                                final_res=final_res,
                                for_train=for_train)
        if awmrblock.right is not None:
            split_into_resolution(awmrblock=awmrblock.right, 
                                thres=thres,
                                volume_units=volume_units,
                                axisres=newaxisres, 
                                unit_index=unit_index, 
                                start_point=start_point2,
                                final_res=final_res,
                                for_train=for_train)
    return


def get_TSDF_block(volume_units,
                    axisres,
                    unit_index,
                    start_point=np.array((0,0,0)),
                    baseres=baseres
                    ):
    res2str = '_'.join(map(str, axisres))

    vunit = volume_units[res2str][unit_index]
    if type(vunit)!=np.ndarray:
        vunit = vunit.D
    
    start_x, start_y, start_z = (np.array(axisres)*start_point).astype(int)
    return vunit[start_x:start_x+baseres,
                 start_y:start_y+baseres,
                 start_z:start_z+baseres]

def get_AWMR_block(volume_units_16, 
                   volume_units_8,
                   volume_units_4, 
                   axisres, 
                   unit_index,
                   option,
                   start_point=np.array((0,0,0)),
                   baseres=baseres,
                   ):
    
    maxisres = max(axisres)
    xres, yres, zres = axisres
    if maxisres==8*baseres/4:
        vunit = volume_units_8[unit_index]
    elif maxisres==16*baseres/4:
        vunit = volume_units_16[unit_index]
    else:
        return volume_units_4[unit_index].D
    vunit = volume_units_16[unit_index]
    if type(vunit)!=np.ndarray:
        vunit = vunit.D
        
    if option=='pool':
        '''POOL'''
        if axisres[0]==maxisres//4:
            vunit = vunit[1::4,:,:]
            # vunit = np.median([vunit[0::4,:,:], vunit[1::4,:,:], vunit[2::4,:,:], vunit[3::4,:,:]], axis=0)
        elif axisres[0]==maxisres//2:
            vunit = vunit[::2,:,:]
            # vunit = np.median([vunit[0::2,:,:], vunit[1::2,:,:]], axis=0)
        # y axis sampling
        if axisres[1]==maxisres//4:
            vunit = vunit[:,1::4,:]
            # vunit = np.median([vunit[:,0::4,:], vunit[:,1::4,:], vunit[:,2::4,:], vunit[:,3::4,:]], axis=0)
        elif axisres[1]==maxisres//2:
            vunit = vunit[:,::2,:]
            # vunit = np.median([vunit[:,0::2,:], vunit[:,1::2,:]], axis=0)
        # z axis sampling
        if axisres[2]==maxisres//4:
            vunit = vunit[:,:,1::4]
            # vunit = np.median([vunit[:,:,0::4], vunit[:,:,1::4], vunit[:,:,2::4], vunit[:,:,3::4]], axis=0)
        elif axisres[2]==maxisres//2:
            vunit = vunit[:,:,::2]
            # vunit = np.median([vunit[:,:,0::2], vunit[:,:,1::2]], axis=0)
    
    elif option=='mean':
        '''MEAN'''
        if axisres[0]==maxisres//4:
            vunit = np.mean([vunit[0::4,:,:], vunit[1::4,:,:], vunit[2::4,:,:], vunit[3::4,:,:]], axis=0)
        elif axisres[0]==maxisres//2:
            vunit = np.mean([vunit[0::2,:,:], vunit[1::2,:,:]], axis=0)
        # y axis sampling
        if axisres[1]==maxisres//4:
            vunit = np.mean([vunit[:,0::4,:], vunit[:,1::4,:], vunit[:,2::4,:], vunit[:,3::4,:]], axis=0)
        elif axisres[1]==maxisres//2:
            vunit = np.mean([vunit[:,0::2,:], vunit[:,1::2,:]], axis=0)
        # z axis sampling
        if axisres[2]==maxisres//4:
            vunit = np.mean([vunit[:,:,0::4], vunit[:,:,1::4], vunit[:,:,2::4], vunit[:,:,3::4]], axis=0)
        elif axisres[2]==maxisres//2:
            vunit = np.mean([vunit[:,:,0::2], vunit[:,:,1::2]], axis=0)
            
    elif option=='weighted':
        '''WEIGHTED'''
        if axisres[0]==maxisres//4:
            vunit = np.average([vunit[0::4,:,:], vunit[1::4,:,:], vunit[2::4,:,:], vunit[3::4,:,:]], axis=0, weights=[.1,.4,.4,.1])
        elif axisres[0]==maxisres//2:
            vunit = np.mean([vunit[0::2,:,:], vunit[1::2,:,:]], axis=0)
        # y axis sampling
        if axisres[1]==maxisres//4:
            vunit = np.average([vunit[:,0::4,:], vunit[:,1::4,:], vunit[:,2::4,:], vunit[:,3::4,:]], axis=0,  weights=[.1,.4,.4,.1])
        elif axisres[1]==maxisres//2:
            vunit = np.mean([vunit[:,0::2,:], vunit[:,1::2,:]], axis=0)
        # z axis sampling
        if axisres[2]==maxisres//4:
            vunit = np.average([vunit[:,:,0::4], vunit[:,:,1::4], vunit[:,:,2::4], vunit[:,:,3::4]], axis=0,  weights=[.1,.4,.4,.1])
        elif axisres[2]==maxisres//2:
            vunit = np.mean([vunit[:,:,0::2], vunit[:,:,1::2]], axis=0)


    start_x, start_y, start_z = (np.array(axisres)*start_point).astype(int)
    return vunit[start_x:start_x+baseres,
                 start_y:start_y+baseres,
                 start_z:start_z+baseres]


def pad_awmr_from_original_old(tsdf,
                           src_verts,
                           src_faces,
                           finest_voxel_size,
                           volume_units_16, 
                           volume_units_8, 
                           volume_units_4,
                           axisres, 
                           unit_index, 
                           start_point=np.array((0,0,0)),
                           for_mask=False):
    eps = 1e-5
    lowest_res = 8
    sx, sy, sz = (baseres,baseres,baseres) # TODO this WILL cause a bug someday
    shifts, slices_from, slices_to = slice_list("back", sx,sy,sz)
    if len(unit_index)==4:
        scene_name = unit_index[0]
    padded = np.zeros((sx+1, sy+1, sz+1))
    padded[:sx,:sy,:sz] = tsdf
    pad_exists = []
    for i in range(7):
        sample_from = np.array(unit_index[-3:])
        shift = shifts[i]
        new_start_point = start_point + lowest_res*shift/np.array(axisres)
        for axis in [0,1,2]:
            if abs(new_start_point[axis] - 1) <= eps:
                sample_from[axis] += 1
                new_start_point[axis]  = 0
        if len(unit_index)==4:
            sample_from = (scene_name, sample_from[0], sample_from[1], sample_from[2])
        else:
            sample_from = tuple(sample_from)
        if (sample_from in volume_units_16.keys()) or (sample_from in volume_units_8.keys()) or (sample_from in volume_units_4.keys()):
            pad_exists.append(list(shift))
            #temp_arr = get_AWMR_block(volume_units_16, volume_units_8, volume_units_4, axisres, sample_from, 'mean', new_start_point)
            temp_arr = get_TSDF_block(src_verts, src_faces, finest_voxel_size, axisres, sample_from, new_start_point)
            padded[slices_to[i]] = temp_arr[slices_from[i]]
    
    if not for_mask:
        if [1,0,0] not in pad_exists:
            padded = padded[:sx, :, :]
        if [0,1,0] not in pad_exists:
            padded = padded[:, :sy, :]
        if [0,0,1] not in pad_exists:
            padded = padded[:, :, :sz]
        return padded
    
    else: # in this case, we preserve new shape regardless of pad_exists
        _, slices_from, slices_to = slice_list("front", sx,sy,sz)
        
        pad_more = np.zeros([padded.shape[0]+1,
                             padded.shape[1]+1, 
                             padded.shape[2]+1])
        pad_more[1:,1:,1:] = padded
        # pad_more_exists = []
        for i in range(7):
            sample_from = np.array(unit_index[-3:])
            shift = shifts[i] * (-1)
            new_start_point = start_point + lowest_res*shift/np.array(axisres)
            # print(sample_from ,start_point, axisres, new_start_point)
            for axis in [0,1,2]:
                if new_start_point[axis] <= -eps:
                    sample_from[axis] -= 1
                    new_start_point[axis]  = 1+new_start_point[axis]
                if abs(new_start_point[axis]) <= eps:
                    new_start_point[axis] = 0
            # print(new_start_point, sample_from )
            if len(unit_index)==4:
                sample_from = (scene_name, sample_from[0], sample_from[1], sample_from[2])
            else:
                sample_from = tuple(sample_from)
            if (sample_from in volume_units_16.keys()) or (sample_from in volume_units_8.keys()) or (sample_from in volume_units_4.keys()):
                # pad_more_exists.append(list(shift))
                # temp_arr = get_AWMR_block(volume_units_16, volume_units_8, volume_units_4, axisres, sample_from, 'mean', new_start_point)
                temp_arr = get_TSDF_block(src_verts, src_faces, finest_voxel_size, axisres, sample_from, new_start_point)
                # print(temp_arr)
                try:
                    pad_more[slices_to[i]] = temp_arr[slices_from[i]]
                except:
                    print("debug hrere")
        return pad_more



def pad_awmr_from_original(tsdf,
                            volume_units,
                            axisres, 
                            unit_index, 
                            start_point=np.array((0,0,0)),
                            for_mask=False):
    eps = 1e-5
    lowest_res = 8
    sx, sy, sz = (baseres,baseres,baseres) # TODO this WILL cause a bug someday
    shifts, slices_from, slices_to = slice_list("back", sx,sy,sz)
    if len(unit_index)==4:
        scene_name = unit_index[0]
    padded = np.zeros((sx+1, sy+1, sz+1))
    padded[:sx,:sy,:sz] = tsdf
    pad_exists = []
    for i in range(7):
        sample_from = np.array(unit_index[-3:])
        shift = shifts[i]
        new_start_point = start_point + lowest_res*shift/np.array(axisres)
        for axis in [0,1,2]:
            if abs(new_start_point[axis] - 1) <= eps:
                sample_from[axis] += 1
                new_start_point[axis]  = 0
        if len(unit_index)==4:
            sample_from = (scene_name, sample_from[0], sample_from[1], sample_from[2])
        else:
            sample_from = tuple(sample_from)
        if (sample_from in volume_units['32_32_32'].keys()) or (sample_from in volume_units['16_16_16'].keys()) or (sample_from in volume_units['8_8_8'].keys()):
            pad_exists.append(list(shift))
            #temp_arr = get_AWMR_block(volume_units_16, volume_units_8, volume_units_4, axisres, sample_from, 'mean', new_start_point)
            temp_arr = get_TSDF_block(volume_units, axisres, sample_from, new_start_point)
            padded[slices_to[i]] = temp_arr[slices_from[i]]
    
    if not for_mask:
        if [1,0,0] not in pad_exists:
            padded = padded[:sx, :, :]
        if [0,1,0] not in pad_exists:
            padded = padded[:, :sy, :]
        if [0,0,1] not in pad_exists:
            padded = padded[:, :, :sz]
        return padded
    
    else: # in this case, we preserve new shape regardless of pad_exists
        _, slices_from, slices_to = slice_list("front", sx,sy,sz)
        
        pad_more = np.zeros([padded.shape[0]+1,
                             padded.shape[1]+1, 
                             padded.shape[2]+1])
        pad_more[1:,1:,1:] = padded
        # pad_more_exists = []
        for i in range(7):
            sample_from = np.array(unit_index[-3:])
            shift = shifts[i] * (-1)
            new_start_point = start_point + lowest_res*shift/np.array(axisres)
            # print(sample_from ,start_point, axisres, new_start_point)
            for axis in [0,1,2]:
                if new_start_point[axis] <= -eps:
                    sample_from[axis] -= 1
                    new_start_point[axis]  = 1+new_start_point[axis]
                if abs(new_start_point[axis]) <= eps:
                    new_start_point[axis] = 0
            # print(new_start_point, sample_from )
            if len(unit_index)==4:
                sample_from = (scene_name, sample_from[0], sample_from[1], sample_from[2])
            else:
                sample_from = tuple(sample_from)
            if (sample_from in volume_units['32_32_32'].keys()) or (sample_from in volume_units['16_16_16'].keys()) or (sample_from in volume_units['8_8_8'].keys()):
                # pad_more_exists.append(list(shift))
                # temp_arr = get_AWMR_block(volume_units_16, volume_units_8, volume_units_4, axisres, sample_from, 'mean', new_start_point)
                temp_arr = get_TSDF_block(volume_units, axisres, sample_from, new_start_point)
                # print(temp_arr)
                try:
                    pad_more[slices_to[i]] = temp_arr[slices_from[i]]
                except:
                    print("debug hrere")
        return pad_more
'''
def get_adjacent_in_awmr(tsdf,
                         axisres, 
                         unit_index, 
                         awmr_dict=None,
                         node=None,
                         start_point=np.array((0,0,0))):
    # this function does not work properly!!! do not use!!!
    assert (awmr_dict is not None) or (node is not None)
    eps = 1e-5
    shifts = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1],
                       [1,1,0],
                       [1,0,1],
                       [0,1,1],
                       [1,1,1]])
    sx, sy, sz = (baseres,baseres,baseres)
    slices_from = [[slice(0, 1), slice(0,sy), slice(0,sz)],
                  [slice(0,sx), slice(0, 1), slice(0,sz)],
                  [slice(0,sx), slice(0,sy), slice(0, 1)],
                  [slice(0, 1), slice(0, 1), slice(0,sz)],
                  [slice(0, 1), slice(0,sy), slice(0, 1)],
                  [slice(0,sx), slice(0, 1), slice(0, 1)],
                  [slice(0, 1), slice(0, 1), slice(0, 1)]]
    slices_to  = [[slice(-1,sx), slice(0,sy), slice(0,sz)],
                  [slice(0,sx), slice(-1,sy), slice(0,sz)],
                  [slice(0,sx), slice(0,sy), slice(-1,sz)],
                  [slice(-1,sx), slice(-1,sy), slice(0,sz)],
                  [slice(-1,sx), slice(0,sy), slice(-1,sz)],
                  [slice(0,sx), slice(-1,sy), slice(-1,sz)],
                  [slice(-1,sx), slice(-1,sy), slice(-1,sz)]]
    if awmr_dict is None:
        awmr_dict = {tuple(unit_index): node}
    if len(unit_index)==4:
        scene_name = unit_index[0]
    padded = np.zeros((sx+1, sy+1, sz+1))
    padded[:sx,:sy,:sz] = tsdf
    pad_exists = []
    for i in range(7):
        sample_from = np.array(unit_index[-3:])
        shift = shifts[i]
        new_start_point = start_point + 4*shift/np.array(axisres)
        for axis in [0,1,2]:
            if abs(new_start_point[axis] - 1) <= eps:
                sample_from[axis] += 1
                new_start_point[axis]  = 0
        if len(unit_index)==4:
            sample_from = (scene_name, sample_from[0], sample_from[1], sample_from[2])
        else:
            sample_from = tuple(sample_from)
        if (sample_from in awmr_dict.keys()):
            node = awmr_dict[sample_from].find_node(start_point=new_start_point)
            # print(node.tsdf, node.right,new_start_point )
            if node is None:
                continue # TODO
            temp_arr = node.tsdf.D
            padded[slices_to[i]] = temp_arr[slices_from[i]]
            pad_exists.append(list(shift))
    
    if [1,0,0] not in pad_exists:
        padded = padded[:sx, :, :]
    if [0,1,0] not in pad_exists:
        padded = padded[:, :sy, :]
    if [0,0,1] not in pad_exists:
        padded = padded[:, :, :sz]
    
    return padded
'''

def mesh_whole_block(root, 
                     unit_index=(0,0,0), 
                     awmr_dict=None,
                     node=None,
                     volume_origin=np.array((0,0,0)),
                     voxel_size=base_voxel_size,
                     volume_unit_dim=base_volume_unit_dim):
    color = {4:0, 8:0.33, 16:0.67, 32:1}
    # this does not use dual marching cube
    # for now, this will make empty space between each small awmr blocks
    assert (awmr_dict is not None) or (node is not None)
    if awmr_dict is None:
        awmr_dict = {tuple(unit_index): node}

    mesh = o3d.geometry.TriangleMesh()
    for node in root.leaves:
        # pad_tsdf = get_adjacent_in_awmr(node.tsdf.D,
        #                                 node.axisres, 
        #                                 unit_index, 
        #                                 awmr_dict=awmr_dict,
        #                                 start_point=node.start_point)
        smallmesh = make_mesh(node.tsdf.D, 
                              node.axisres, 
                              voxel_size=voxel_size, 
                              volume_unit_dim=volume_unit_dim, 
                              volume_origin=volume_origin,
                              k=node.unit_index,
                              start_point=node.start_point)
        r = color[node.axisres[0]]
        g = color[node.axisres[1]]
        b = color[node.axisres[2]]
        smallmesh.paint_uniform_color([r,g,b])
        mesh += smallmesh
    
    return mesh

def mesh_whole_block_singularize(root, 
                     unit_index=(0,0,0), 
                     awmr_dict=None,
                     node=None,
                     volume_origin=np.array((0,0,0)),
                     volume_unit_dim=base_volume_unit_dim,
                     voxel_size=base_voxel_size,
                     baseres=baseres):
    # color = {4:0, 8:0.33, 16:0.67, 32:1}
    # this does not use dual marching cube
    # for now, this will make empty space between each small awmr blocks
    assert (awmr_dict is not None) or (node is not None)
    if awmr_dict is None:
        awmr_dict = {tuple(unit_index): node}
    # if sum([not np.all(n.axisres==n.axisres[0]) for n in root.leaves]):
    #     print ("정방형 아닌것 있음.")
    dict_xyz2voxel, occupancy = singularize_block(unit_index,
                                               awmr_dict,
                                               volume_unit_dim,
                                               baseres,
                                               voxel_size)
    voxel_range = np.array(list(dict_xyz2voxel.keys()), dtype='int')
    x_min, y_min, z_min = np.min(voxel_range,axis=0)
    x_max, y_max, z_max = np.max(voxel_range,axis=0)
    del voxel_range
    
    shifts, slices_from, slices_to = slice_list("back", *occupancy.shape)
    occupancy_pad = np.zeros(np.array(occupancy.shape)+1, dtype=bool) # for padding occupancy
    occupancy_pad[:base_volume_unit_dim,
                  :base_volume_unit_dim,
                  :base_volume_unit_dim] = occupancy
    
    for i in range(7):
        shift = shifts[i]
        temp_unit_index = tuple(np.array(unit_index[-3:]) + shift)
        if len(unit_index)==4:
            temp_unit_index = (unit_index[0],) + temp_unit_index
        if temp_unit_index in awmr_dict.keys():
            new_dict, occupancy = singularize_block(temp_unit_index,
                                                    awmr_dict,
                                                    volume_unit_dim,
                                                    baseres,
                                                    voxel_size)
            dict_xyz2voxel.update(new_dict)
            occupancy_pad[slices_to[i]] = occupancy[slices_from[i]]
        # else:
        #     pass
    depth = np.zeros([x_max-x_min+2, y_max-y_min+2, z_max-z_min+2]) # include brb neighbors
    temp_arr = np.array([k+(v.D,) for (k,v) in dict_xyz2voxel.items()])
    temp_arr = np.delete(temp_arr, 
                          np.where((temp_arr[:,0]<x_min)|(temp_arr[:,0]>x_max+1)| \
                                  (temp_arr[:,1]<y_min)|(temp_arr[:,1]>y_max+1)| \
                                  (temp_arr[:,2]<z_min)|(temp_arr[:,2]>z_max+1)),
                        axis=0)
    temp_arr[:,:3] = temp_arr[:,:3] - np.array([x_min, y_min, z_min])
    depth[temp_arr[:,0].astype(int),
          temp_arr[:,1].astype(int),
          temp_arr[:,2].astype(int)] = temp_arr[:,3]
    
    dict_voxIdPair2vertIndex = {}
    vertices = []
    faces = []
    #shifts = np.r_[np.array([[0,0,0]]), shifts] # add [0,0,0] to shift
    # np.vstack([np.array([[0,0,0]]), shifts]) # 위에줄이랑 똑같은건데 둘중에 뭐쓸지 결정장애 옴
    
    shifts = [
            [0, 0, 0], [1, 0, 0],
            [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1],
            [1, 1, 1], [0, 1, 1],
            ]
    
    occupancy = occupancy_pad
    del occupancy_pad
    dim_x, dim_y, dim_z = (np.array(unit_index[-3:], dtype='int') + np.ones([3,], dtype='int')) * volume_unit_dim #TODO
    for x in range(dim_x-volume_unit_dim, dim_x):
        for y in range(dim_y-volume_unit_dim, dim_y):
            for z in range(dim_z-volume_unit_dim, dim_z):
                
                mc_index = 0
                Ps = np.zeros((8, 3), dtype=np.float32)
                Ds = np.zeros(8, dtype=np.float32)
                # Rs = np.zeros(8, dtype=np.float32)
                # Gs = np.zeros(8, dtype=np.float32)
                # Bs = np.zeros(8, dtype=np.float32)
                Vs = np.zeros(8, dtype=np.int64)
                Vs = [0]*8 # 안 예쁨. 알아서 고치세요.
                
                for i in range(8):
                    neighbor_x = x + shifts[i][0]
                    neighbor_y = y + shifts[i][1]
                    neighbor_z = z + shifts[i][2]
                    
                    occupancy_key = (neighbor_x-dim_x+volume_unit_dim,
                                     neighbor_y-dim_y+volume_unit_dim,
                                     neighbor_z-dim_z+volume_unit_dim)
                    if min(occupancy_key)<0:
                        print(f"{occupancy_key}, {x}, {y}, {z}")
                        continue # 계산상 뜰 수 없음
                        # break
                    if not occupancy[occupancy_key]:
                        if occupancy_key[0] >32:
                            print(f"not occupied at {occupancy_key}, {x}, {y}, {z}")
                        mc_index = 0
                        break
                    
                    #try: # TODO -- i don't remember when this caused error
                    Ps[i] = dict_xyz2voxel[neighbor_x, neighbor_y, neighbor_z].pos
                    Ds[i] = dict_xyz2voxel[neighbor_x, neighbor_y, neighbor_z].D
                    # Rs[i] = dict_xyz2voxel[neighbor_x, neighbor_y, neighbor_z].R
                    # Gs[i] = dict_xyz2voxel[neighbor_x, neighbor_y, neighbor_z].G
                    # Bs[i] = dict_xyz2voxel[neighbor_x, neighbor_y, neighbor_z].B
                    Vs[i] = dict_xyz2voxel[neighbor_x, neighbor_y, neighbor_z].voxid
                    
                    #except:
                    #    pass
                    # if Ds[i] == 0.0:
                    #     mc_index = 0
                    #     break
                    #el
                    if Ds[i] < 0.0: # i번째 복셀의 부호를 i번째 비트에 기록 
                        mc_index |= (1<<i)
                
                # empty case
                if (mc_index == 0) or (mc_index == 255):
                    continue
                
                # mesh vertex 정보를 생성 
                vert_indices = -np.ones(12, np.int64)
                for i in range(12):
                    if edge_table[mc_index] & (1<<i): # vertex 생성이 필요한 경우 
                        e1, e2 = edge_to_vert[i]
                        
                        voxid1 = Vs[e1]
                        voxid2 = Vs[e2]
                        
                        pos1 = Ps[e1]
                        pos2 = Ps[e2]
                        
                        D1 = np.abs(Ds[e1])
                        D2 = np.abs(Ds[e2])
                        # R1 = np.abs(Rs[e1])
                        # R2 = np.abs(Rs[e2])
                        # G1 = np.abs(Gs[e1])
                        # G2 = np.abs(Gs[e2])
                        # B1 = np.abs(Bs[e1])
                        # B2 = np.abs(Bs[e2])
                        
                        e1_x = pos1[0] * voxel_size
                        e1_y = pos1[1] * voxel_size
                        e1_z = pos1[2] * voxel_size
                    
                        e2_x = pos2[0] * voxel_size
                        e2_y = pos2[1] * voxel_size
                        e2_z = pos2[2] * voxel_size

                        vert_x = (D2 * e1_x + D1 * e2_x) / (D1 + D2)
                        vert_y = (D2 * e1_y + D1 * e2_y) / (D1 + D2)
                        vert_z = (D2 * e1_z + D1 * e2_z) / (D1 + D2)
                        
                        # vert_R = (D2 * R1 + D1 * R2) / (D1 + D2)
                        # vert_G = (D2 * G1 + D1 * G2) / (D1 + D2)
                        # vert_B = (D2 * B1 + D1 * B2) / (D1 + D2)
                        
                        # voxIdPair = (np.minimum(voxid1, voxid2), 
                        #              np.maximum(voxid1, voxid2))
                        
                        voxIdPair = [voxid1, voxid2]
                        voxIdPair.sort()
                        voxIdPair = tuple(voxIdPair)
                        
                        if voxIdPair in dict_voxIdPair2vertIndex:
                            vert_indices[i] = dict_voxIdPair2vertIndex[voxIdPair]
                        else:
                            vertIndex = len(vertices)
                            vertices.append([vert_x, vert_y, vert_z])#, vert_R, vert_G, vert_B])
                            dict_voxIdPair2vertIndex[voxIdPair] = vertIndex           
                            vert_indices[i] = vertIndex
                        
                # face 정보를 생성
                for i in range(0, 12, 3):                
                    if tri_table[mc_index][i] == -1:
                        break
                    
                    v1 = vert_indices[tri_table[mc_index][i]]
                    v2 = vert_indices[tri_table[mc_index][i+2]]
                    v3 = vert_indices[tri_table[mc_index][i+1]]
                    
                    # if v1==32 or v2==32 or v3==32:
                    #     print(x, y, z, v1,v2,v3, mc_index, i)
                    #     print(vert_indices)
                    
                    if v1 == v2 or v2 == v3 or v1 == v3: # line은 제외 
                        continue
                    
                    faces.append([v1, v2, v3])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh = mesh.translate(volume_origin)
    return mesh


def get_pedigrees(node, curpath, pedigrees, leaves):
    # left=-1, right=1
    if node.left is not None:
        curpath_new = curpath + [-1]
        pedigrees, leaves = get_pedigrees(node.left, curpath_new, pedigrees, leaves)
    if node.right is not None:
        curpath_new = curpath + [1]
        pedigrees, leaves = get_pedigrees(node.right, curpath_new, pedigrees, leaves)
    if (node.left is None) and (node.right is None):
        pedigrees += [curpath]
        leaves += [node]
    return pedigrees, leaves


''' 한 복셀을 표현하는 클래스 '''
class Voxel(object):
    def __init__(self, pos, D, R, G, B):
        self.pos = pos # 실제 복셀 인덱스
        self.D = D 
        # self.R = R
        # self.G = G
        # self.B = B
        self.voxid = 0
        
    def setVoxid(self, dims): # TODO: Unique ID
        # (x,y,z) -> int
        #self.voxid = self.pos[0] + dims[0] * (self.pos[1] + dims[1] * self.pos[2])
        #self.voxid = int(self.pos[0]*1e5 + self.pos[1]*1e4 + self.pos[2]*1e3 + self.dims[0]*1e2 + self.dims[1]*1e1 + self.dims[0])
        self.voxid = f"{self.pos[0]:02d}{self.pos[1]:02d}{self.pos[2]:02d}{dims[0]:02d}{dims[1]:02d}{dims[2]:02d}"
    def reverse_unique_id(self):
        self.pos[0] = int(self.voxid[0:2])
        self.pos[1] = int(self.voxid[2:4])
        self.pos[2] = int(self.voxid[4:6])
        dims = [int(self.voxid[6:8]), int(self.voxid[8:10]), int(self.voxid[10:12])]
        return dims
    
def singularize_block(key: tuple, 
                      awmr_tsdfs: dict,
                      volume_unit_dim: int,
                      baseres: int,
                      voxel_size: int):
    # dim_x, dim_y, dim_z = (np.array(key[-3:], dtype='int') + np.ones([3,], dtype='int')) * volume_unit_dim #TODO
    occupancy = np.zeros((volume_unit_dim, volume_unit_dim, volume_unit_dim),
                         dtype=bool)
    
    dim_x, dim_y, dim_z = np.array(key[-3:], dtype='int') * volume_unit_dim
    dict_xyz2voxel = {} # 단일해상도 볼륨 정보를 저장하는 딕셔너리
    
    ahchor_x, ahchor_y, ahchor_z = np.array(key[-3:], dtype='int') * volume_unit_dim
    node = awmr_tsdfs[key]
    
    # Leaf Node까지 순회하여 복셀 정보를 추출 단일해상도 표현으로 저장
    for leaf in node.leaves:
        offset_x, offset_y, offset_z = np.array(leaf.start_point) * volume_unit_dim
        step_x, step_y, step_z = volume_unit_dim // np.array(leaf.axisres, dtype='int') * 1 # TODO:int(0.004/voxel_size)
        
        # baseres x baseres x baseres 복셀들 
        for ix in range(baseres):
            for iy in range(baseres):
                for iz in range(baseres):
                    x = ahchor_x + offset_x.astype(int) + step_x * ix
                    y = ahchor_y + offset_y.astype(int) + step_y * iy
                    z = ahchor_z + offset_z.astype(int) + step_z * iz
                    
                    # 해상도를 고려하여 단일해상도 표현으로 변환
                    for jx in range(step_x):
                        for jy in range(step_y):
                            for jz in range(step_z):
                    
                                dict_xyz2voxel[x+jx, y+jy, z+jz] = Voxel(
                                    [x, y, z], 
                                    leaf.tsdf.D[ix, iy, iz], None,None,None
                                    # leaf.tsdf.R[ix, iy, iz], 
                                    # leaf.tsdf.G[ix, iy, iz], 
                                    # leaf.tsdf.B[ix, iy, iz]
                                )
                                dict_xyz2voxel[x+jx, y+jy, z+jz].setVoxid([dim_x, dim_y, dim_z])
                                
                                occupancy[int(x+jx)-dim_x, 
                                          int(y+jy)-dim_y, 
                                          int(z+jz)-dim_z] = True
        # print("ppiyong")
    return dict_xyz2voxel, occupancy

def fast_rstrip(L, hit):
    for i in range(len(L) - 1, -1, -1):
        if L[i] != hit:
            break
    del L[i + 1:]


def reconstruct_bintree(root, new_pedigs, new_nodes):
    for ped, node in zip(new_pedigs, new_nodes):
        if sum([abs(p) for p in ped])==0:
            root = node
            continue
        
        fast_rstrip(ped,0)
        n = root
        
        for i in range(len(ped)):
            d = ped[i]
            ended = (i==len(ped)-1)
            
            if ended:
                if d==-1:
                    n.left = node
                elif d==1:
                    n.right = node
                
            else:
                if d==-1:
                    if n.left is None:
                        n.left = AWMRblock(np.array([4,4,4]), root.unit_index)
                    n = n.left
                elif d==1:
                    if n.right is None:
                        n.right = AWMRblock(np.array([4,4,4]), root.unit_index)
                    n = n.right
    return root


def AWMRBlock_2_batch(awmr: AWMRblock):
    '''
    generate 4*4*4 tsdf ndarrays and metadata from an awmr block 

    Parameters
    ----------
    awmr : AWMRblock
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # traverse to have list of leaf nodes
    pedigrees, leafnodes =  get_pedigrees(awmr, [], [], [])
    
    # from list of leaf nodes
    tsdfs =  np.array([n.tsdf.D for n in leafnodes])
    masks = np.array([n.tsdf.M for n in leafnodes])
    axisress = [n.axisres for n in leafnodes]
    start_points = [n.start_point for n in leafnodes]
    unit_indices = [n.unit_index for n in leafnodes]
    
    return tsdfs, masks, axisress, start_points, pedigrees, unit_indices


def awmr_dict_2_dataset_mono(awmr_dict: dict,
                            volume_unit_dim=4,
                            save_path=None):
    '''
    generate input data to send from awmr_dict

    Parameters
    ----------
    awmr_dict : dict
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if save_path==None:
        raise ValueError("need save path")
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for k, awmr in awmr_dict.items():
        ts, ms, ars, sps, peds, indices = AWMRBlock_2_batch(awmr)
        scene_name = indices[0][0]
        if not os.path.exists(os.path.join(save_path, scene_name)):
            os.mkdir(os.path.join(save_path, scene_name))
        filename = '%d_%d_%d.npz' % (indices[0][1],indices[0][2],indices[0][3])
        out_path = os.path.join(save_path, scene_name, filename)
        np.savez(out_path,
                 TSDF=ts,
                 MASK=ms,
                 SIGN=np.sign(ts),
                 MAGN=np.abs(ts),
                 AXRES=ars,
                 START=sps,
                 PEDIG=peds)
    return


def awmr_dict_2_TSDFBlockDataset(awmr_dict: dict,
                                scene_name: str,
                                res=4, # TODO
                                save_path=None):
    '''
    Parameters
    ----------
    awmr_dict : dict
    scene_name : str
    res : TYPE, optional
        this is actually dimension than resolution. The default is 4.
    save_path : STR, THIS IS ACTUALLY NOT optional IM SORRY

    Raises
    ------
    ValueError

    Returns
    -------
    None.

    '''
    if save_path==None:
        raise ValueError("need save path")
        
    TSDF = np.zeros((0, res, res, res), dtype=np.float32)
    MASK = np.zeros((0, res, res, res), dtype=np.float32)
    SIGN = np.zeros((0, res, res, res), dtype=np.float32)
    MAGN = np.zeros((0, res, res, res), dtype=np.float32)
    KEYS = np.zeros((0, 4))
    AXRES = np.zeros((0, 3))
    START = np.zeros((0, 3))
    PEDIG = np.zeros((0, 6)) # xyzxyz
    for k, awmr in awmr_dict.items():
        # print(k)
        ts, ms, ars, sps, peds, indices = AWMRBlock_2_batch(awmr)
        _TSDF = ts
        _MASK = ms
        _KEYS = np.array(indices)
        _SIGN = np.sign(ts)
        _MAGN = np.abs(ts)
        _AXRES = np.array(ars)
        _START = np.array(sps)
        _PEDIG = np.array([ped+[0]*(6-len(ped)) for ped in peds])
        
        TSDF = np.vstack((TSDF, _TSDF))
        MASK = np.vstack((MASK, _MASK))
        SIGN = np.vstack((SIGN, _SIGN))
        MAGN = np.vstack((MAGN, _MAGN))
        KEYS = np.vstack((KEYS, _KEYS))
        AXRES = np.vstack((AXRES, _AXRES))
        START = np.vstack((START, _START))
        PEDIG = np.vstack((PEDIG, _PEDIG))
    
    # save as npz
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    npzfile = f"{scene_name}_{res}x{res}x{res}.npz"
    npzfile = os.path.join(save_path, npzfile)
    np.savez(npzfile,
             TSDF=TSDF,
             MASK=MASK,
             SIGN=SIGN,
             MAGN=MAGN,
             KEYS=KEYS,
             AXRES=AXRES,
             START=START,
             PEDIG=PEDIG)
    