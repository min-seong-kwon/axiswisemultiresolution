# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:21:09 2022

1차년도때 만들었던 볼륨유닛 복잡도 계산 코드를 function으로 사용할수있도록 수정

@author: alienware
"""
import os
import re
from glob import glob
import shutil
import numpy as np
from tqdm import tqdm
from utils.evalUtils import singleVoxelMesh
from source.TriMesh import TriMesh
from source.VolumeUnit import VolumeUnit
from source.ConfigSettings import render_6x6
import trimesh

CPL_MODE = {0: 'TSDFGradient',
            1: 'FaceNormals',
            2: 'EdgeSharpness'}


def neighbors(k, unit_dim=16):
    b, d, x, y, z = k
    n_list = [(b, d, x+1,   y,   z), (b, d, x-1,   y,   z),
              (b, d,   x, y+1,   z), (b, d,   x, y-1,   z),
              (b, d,   x,   y, z+1), (b, d,   x,   y, z-1)]
    n_list = [n for n in n_list if (0<=n[2]<unit_dim) and (0<=n[3]<unit_dim) and (0<=n[4]<unit_dim)]
    return n_list


def unitComplexity(unit,
                   cpl_mode=0):
    '''
    inputs
    vunit: (ndarray) 16*16*16
    cpl_mode: (integer) complexity calculation function
    ---
    outputs
    (float) complexity
    '''
    vunit = VolumeUnit()
    vunit.D = unit
    if CPL_MODE[cpl_mode]=='TSDFGradient':
        Gx, Gy, Gz = np.gradient(vunit.D)
        Gxyz = np.stack((Gx.flatten(), Gy.flatten(), Gz.flatten()))
        complexity = np.abs(np.linalg.det(np.cov(Gxyz)))
        
    elif CPL_MODE[cpl_mode]=='FaceNormals' or \
        CPL_MODE[cpl_mode]=='EdgeSharpness':
        # make mesh from voxel
        mesh = TriMesh()
        mesh = singleVoxelMesh({(0,0,0):vunit},
                               (0,0,0),
                               16,
                               0.004,
                               mesh,
                               {},
                               16*16*16,
                               0,0,0)
        if CPL_MODE[cpl_mode]=='FaceNormals':
            complexity = mesh.calculate_face_normals()
        else:
            complexity = mesh.calculate_sharpnesses()
    else:
        raise Exception('wrong complexity mode')
    return complexity


def calcComplexities(tsdf_dict,
                     cpl_mode=0):
    '''
    complexity values are saved in volume_units as well as retval
    inputs
    volume_units: (dictionary) key - (x,y,z)index,
                               val - volume unit object
    cpl_mode: (integer) complexity calculation function
    ---
    outputs
    (dictionary) key - (x,y,z)index, val - complexity(float)
    '''
    complexities = {}
    for k in tsdf_dict.keys():
        complexity = unitComplexity(tsdf_dict[k],
                                    cpl_mode)
        if hasattr(tsdf_dict[k], 'complexity'):
            tsdf_dict[k].complexity = complexity
        complexities[k] = complexity
    return complexities


def assignRes(tsdf_dict,
              level_mode='use_complexity',
              dims=[16,8,4],
              ori_mask_dict=None):
    '''
    assign resolutions to volume units
    Parameters
    ----------
    volume_units : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    volume_units_d2 = {}
    volume_units_d1 = {}
    volume_units_d0 = {}
    
    complexities = calcComplexities(tsdf_dict, cpl_mode=0)
    
    percents = np.arange(1, 100, 1)
    percentiles = np.zeros(percents.shape)
    temp = list(complexities.values())
    for i, p in enumerate(percents):
        percentiles[i] = np.percentile(temp, p)

    percentile40 = np.percentile(temp, 40)
    percentile60 = np.percentile(temp, 60)
    percentile80 = np.percentile(temp, 80)
    print(percentile40, percentile60, percentile80, np.max(temp))
    # # 4.6154248460222924e-11 5.227742478766329e-09 1.5324780310450562e-07 1.204452924725995e-05
    # percentile60 = 1e-11
    # percentile80 = 1e-8

    if level_mode=='single16':
        percentile80 = 0
        level_mode = 'use_complexity'
    if level_mode=='single8':
        percentile80 = np.inf
        percentile60 = 0
        level_mode = 'use_complexity'
    if level_mode=='single4':
        percentile80 = np.inf
        percentile60 = np.inf
        percentile40 = 0
        level_mode = 'use_complexity'
    
    dim2, dim1, dim0 = dims
    if level_mode=='use_complexity':
        for k in tsdf_dict.keys():
            complexity = complexities[k]
            
            if complexity >= percentile80: 
                volume_units_d2[k] = VolumeUnit(volume_unit_dim=dim2, depth=None)
                if tsdf_dict[k].shape[0]==dim2:
                    volume_units_d2[k].D = tsdf_dict[k]
                if tsdf_dict[k].shape[0]==dim2 and (ori_mask_dict is not None):
                    volume_units_d2[k].W = volume_units_d2[k].M = ori_mask_dict[k]
                volume_units_d2[k].complexity = complexity
            elif complexity >= percentile60: 
                volume_units_d1[k] = VolumeUnit(volume_unit_dim=dim1, depth=None) # 해상도를 한단계 낮춤
                volume_units_d1[k].complexity = complexity
            else: #if complexity >= percentile40:
                volume_units_d0[k] = VolumeUnit(volume_unit_dim=dim0, depth=None) # 해상도를 두단계 낮춤
                volume_units_d0[k].complexity = complexity
            # else:
            #     volume_units_16.pop(k)
            #     volume_units_2[k] = VolumeUnit(1) # 해상도를 두단계 낮춤
            #     volume_units_2[k].complexity = complexity
    # elif level_mode=='use_distance':
    #     for volume_unit_ix, volume_unit_iy, volume_unit_iz in volume_units.keys():
    #         distance = distances[volume_unit_ix, volume_unit_iy, volume_unit_iz]
    #         if distance < dist_thres1:
    #             continue
    #         elif dist_thres1 <= distance < dist_thres2:
    #             volume_units_8[volume_unit_ix, volume_unit_iy, volume_unit_iz] = VolumeUnit(3) # 해상도를 한단계 낮춤
    #             volume_units_8[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity = complexity
    #         elif dist_thres2 <= distance <= dist_thres3:
    #             volume_units_4[volume_unit_ix, volume_unit_iy, volume_unit_iz] = VolumeUnit(2) # 해상도를 두단계 낮춤
    #             volume_units_4[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity = complexity
    #         else:
    #             volume_units_2[volume_unit_ix, volume_unit_iy, volume_unit_iz] = VolumeUnit(1) # 해상도를 두단계 낮춤
    #             volume_units_2[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity = complexity
    
    # elif level_mode=='use_both': #TODO
    #     for volume_unit_ix, volume_unit_iy, volume_unit_iz in volume_units.keys():
    #         abs_comp_thres = 100000000000 #TODO
    #         complexity = volume_units[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity
    #         distance = distances[volume_unit_ix, volume_unit_iy, volume_unit_iz]
    #         if distance < dist_thres1:
    #             continue
    #         elif dist_thres1 <= distance < dist_thres2:
    #             if complexity >= abs_comp_thres:
    #                 continue
    #             else:
    #                 volume_units_8[volume_unit_ix, volume_unit_iy, volume_unit_iz] = VolumeUnit(3) # 해상도를 한단계 낮춤
    #                 volume_units_8[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity = complexity
    #         elif dist_thres2 <= distance <= dist_thres3:
    #             if complexity >= abs_comp_thres:
    #                 volume_units_8[volume_unit_ix, volume_unit_iy, volume_unit_iz] = VolumeUnit(3) # 해상도를 한단계 낮춤
    #                 volume_units_8[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity = complexity
    #             else:
    #                 volume_units_4[volume_unit_ix, volume_unit_iy, volume_unit_iz] = VolumeUnit(2) # 해상도를 두단계 낮춤
    #                 volume_units_4[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity = complexity
    #         elif dist_thres3 <= distance:
    #             if complexity >= abs_comp_thres:
    #                 volume_units_4[volume_unit_ix, volume_unit_iy, volume_unit_iz] = VolumeUnit(2) # 해상도를 두단계 낮춤
    #                 volume_units_4[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity = complexity
    #             else:
    #                 volume_units_2[volume_unit_ix, volume_unit_iy, volume_unit_iz] = VolumeUnit(1) # 해상도를 두단계 낮춤
    #                 volume_units_2[volume_unit_ix, volume_unit_iy, volume_unit_iz].complexity = complexity
 
    return volume_units_d2, volume_units_d1, volume_units_d0#, volume_units_2


def reduceRes(volume_units_re,
              resolution,
              volume_origin,
              dataset,
              volume_unit_dim,
              voxel_size_mm,
              sdf_trunc,
              force=False):
    datadir = './_VolumeUnits_%d_%s' % (resolution, dataset.scenes[0])
    if os.path.exists(datadir) and force==False and len(volume_units_re)!=0:
        print('loading saved volume units...')
        if len(os.listdir(datadir)) != 0:
            npz_files = glob(datadir+'/*.npz')
            for npz in npz_files:
                k = re.findall(r'\d+', npz)
                volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
                volume_unit_ix = int(volume_unit_ix)
                volume_unit_iy = int(volume_unit_iy)
                volume_unit_iz = int(volume_unit_iz)
                
                if len(list(volume_units_re.keys())[0])==3:
                    k = volume_unit_ix, volume_unit_iy, volume_unit_iz
                elif len(list(volume_units_re.keys())[0])==4:
                    k = dataset.scenes[0], volume_unit_ix, volume_unit_iy, volume_unit_iz
                if k in volume_units_re.keys():
                    volume_units_re[k].load(npz)
        return volume_units_re
    
    K = dataset.getCameraMatrix()
    image_width, image_height = dataset.getImageSize()

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    ix, iy = np.meshgrid(range(0, image_height), 
                             range(0, image_width), indexing='ij')
    ones = np.ones((image_height, image_width), dtype=np.float32)
    depth_to_camera_distance_multiplier = np.sqrt(ones + ((ix - cy)/fy)**2 + ((iy - cx)/fx)**2)
    depth_to_camera_distance_multiplier = depth_to_camera_distance_multiplier.astype(np.float32)
    IDX_OFFSET = {32:0, 16:0.5, 8:1.5, 4:3.5} #TODO 16:0? 2mm에서도?
    idx_offset = IDX_OFFSET[resolution]
    
    ix, iy, iz = np.meshgrid(range(0, resolution), 
                             range(0, resolution), 
                             range(0, resolution), indexing='ij')
    
    offset_voxel_index = np.vstack((ix.flatten(), iy.flatten(), iz.flatten()))
    
    print('Integrating multiview RGB-D images...\n')       
    for view in tqdm(render_6x6):#range(1, 21*21+1)):
        dep = dataset.getDepthMap(view)
        col = dataset.getRgbImage(view)
        R, T = dataset.getCameraPosition(view)
        inv_R = np.linalg.inv(R)
        # T = T + np.array([[0.3], [0], [0.01]])
    
        for k in volume_units_re.keys():
            volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
                
            anchor_voxel_index = np.array([[volume_unit_ix * volume_unit_dim], 
                                           [volume_unit_iy * volume_unit_dim],
                                           [volume_unit_iz * volume_unit_dim]], dtype=np.int32)
        
            voxel_indices = anchor_voxel_index + offset_voxel_index * (volume_unit_dim//resolution) + idx_offset
            voxel_world = voxel_indices * voxel_size_mm + volume_origin
            voxel_camera = np.dot(inv_R, voxel_world) - np.dot(inv_R, T) 
            voxel_pixel = np.dot(K, voxel_camera)
            
            x_pixel = voxel_pixel[0, :] / voxel_pixel[2, :] + 0.5
            y_pixel = voxel_pixel[1, :] / voxel_pixel[2, :] + 0.5 
            x_pixel = x_pixel.astype(np.int32)
            y_pixel = y_pixel.astype(np.int32)
        
            valid_pixel = np.logical_and(x_pixel >= 0,
                            np.logical_and(x_pixel <= image_width-1,
                            np.logical_and(y_pixel >= 0,
                            np.logical_and(y_pixel <= image_height-1,
                                           voxel_camera[2, :] > 0))))
            
            #------------------ depth 정보 통합 ------------------#
            dep_valid_pixel = np.zeros(valid_pixel.shape)
            dep_valid_pixel[valid_pixel] = dep[y_pixel[valid_pixel], x_pixel[valid_pixel]]
            
            multiplier = np.zeros(valid_pixel.shape)
            multiplier[valid_pixel] = depth_to_camera_distance_multiplier[y_pixel[valid_pixel], x_pixel[valid_pixel]]
            
            sdf = (dep_valid_pixel - voxel_camera[2, :]) * multiplier
            valid_sdf = np.logical_and(dep_valid_pixel > 0, sdf > -sdf_trunc)
            # if (volume_unit_ix, volume_unit_iy, volume_unit_iz)==(23, 26, 7) and sum(valid_sdf)>=512:
            #     print("sss", min(sdf))
            tsdf = np.minimum(1.0, sdf / sdf_trunc)
            tsdf = tsdf[valid_sdf]
            
            volume_units_re[k].D[
                    offset_voxel_index[0, valid_sdf],
                    offset_voxel_index[1, valid_sdf],
                    offset_voxel_index[2, valid_sdf]] += tsdf
                    
            volume_units_re[k].W[
                    offset_voxel_index[0, valid_sdf],
                    offset_voxel_index[1, valid_sdf],
                    offset_voxel_index[2, valid_sdf]] += 1.0
                    
            #------------------ color 정보 통합 ------------------#        
            #R
            R_valid_pixel = np.zeros(valid_pixel.shape)            
            R_valid_pixel[valid_pixel] = col[y_pixel[valid_pixel], x_pixel[valid_pixel], 0]
            volume_units_re[k].R[
                    offset_voxel_index[0, valid_sdf],
                    offset_voxel_index[1, valid_sdf],
                    offset_voxel_index[2, valid_sdf]] += R_valid_pixel[valid_sdf]
            
            # G     
            G_valid_pixel = np.zeros(valid_pixel.shape)          
            G_valid_pixel[valid_pixel] = col[y_pixel[valid_pixel], x_pixel[valid_pixel], 1]  
            volume_units_re[k].G[
                    offset_voxel_index[0, valid_sdf],
                    offset_voxel_index[1, valid_sdf],
                    offset_voxel_index[2, valid_sdf]] += G_valid_pixel[valid_sdf]
                    
            # B
            B_valid_pixel = np.zeros(valid_pixel.shape)
            B_valid_pixel[valid_pixel] = col[y_pixel[valid_pixel], x_pixel[valid_pixel], 2]
            volume_units_re[k].B[
                    offset_voxel_index[0, valid_sdf],
                    offset_voxel_index[1, valid_sdf],
                    offset_voxel_index[2, valid_sdf]] += B_valid_pixel[valid_sdf]
            
    # TSDF, R, G, B를 weight으로 나눠 평균값으로 바꾼다.
    for k in volume_units_re.keys():
        # if len(k)==3:
        #     volume_unit_ix, volume_unit_iy, volume_unit_iz = k
        # elif len(k)==4:
        #     _, volume_unit_ix, volume_unit_iy, volume_unit_iz = k
        # if (volume_unit_ix, volume_unit_iy, volume_unit_iz)==(23, 26, 7):
        #     print('www')
        D = volume_units_re[k].D
        W = volume_units_re[k].W
        R = volume_units_re[k].R
        G = volume_units_re[k].G
        B = volume_units_re[k].B
    
        nonzeros = np.where(W > 0.0)
        
        D[nonzeros] = D[nonzeros] / W[nonzeros]
        R[nonzeros] = R[nonzeros] / W[nonzeros]
        G[nonzeros] = G[nonzeros] / W[nonzeros]
        B[nonzeros] = B[nonzeros] / W[nonzeros]
        
        volume_units_re[k].D = D
        volume_units_re[k].R = R
        volume_units_re[k].G = G
        volume_units_re[k].B = B
        
    # 계산된 TSDF 볼륨 유닛들을 저장
    if os.path.exists(datadir):
        shutil.rmtree(datadir)
        
    os.mkdir('./_VolumeUnits_%d_%s' % (resolution, dataset.scenes[0]))
    for k in volume_units_re.keys():
        if len(k)==3:
            volume_unit_ix, volume_unit_iy, volume_unit_iz = k
        elif len(k)==4:
            _, volume_unit_ix, volume_unit_iy, volume_unit_iz = k
        
        # if (volume_unit_ix, volume_unit_iy, volume_unit_iz)==(23, 26, 7):
        #     print('www')
        
        W = volume_units_re[k].W
        
        out_path = '%d_%d_%d.npz' % (volume_unit_ix, volume_unit_iy, volume_unit_iz)
        out_path = os.path.join(datadir, out_path)
        volume_units_re[k].save(out_path)

    return volume_units_re

