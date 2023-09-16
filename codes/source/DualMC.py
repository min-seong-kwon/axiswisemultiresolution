 # -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:36:20 2022

@author: alienware
"""
import os
import re
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import torch

from source.TriMesh import TriMesh
from source.MVDDataSet import MVDDataSet 
from source.VolumeUnit import VolumeUnit
from source.MCLutClassic import edge_table, edge_to_vert, shift, edge_shift, tri_table
from source.Octree_old import BranchNode, LeafNode, octree_orders
from utils.evalUtils import singleVoxelMesh, fastVoxelMesh, customHausdorff, customChamfer
# from parameters import *
import matplotlib.cm as cm #added
import matplotlib.colors as colors #added
import matplotlib.pyplot as plt

import igl
# from chamferdist import ChamferDistance


def construct_octree_depth5(volume_units_32, k, volume_unit_dim=16):
    volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
    octree = BranchNode()
    base_log = 5
    depth=5
    if hasattr(volume_units_32[k], 's'):
        s = volume_units_32[k].s
        octree.s = s
    
    # depth1
    octree.child = [
            BranchNode(), BranchNode(), BranchNode(), BranchNode(),
            BranchNode(), BranchNode(), BranchNode(), BranchNode(),
        ] # 2x2x2
    
    # depth2
    for d1 in range(8):
        bn1 = octree.child[d1]
        bn1.child = [
                BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                BranchNode(), BranchNode(), BranchNode(), BranchNode(),
            ] # 4x4x4
        
    # depth3
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            bn2.child = [
                    BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                    BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                ] # 8x8x8
    
    # depth4
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            for d3 in range(8):
                bn3 = bn2.child[d3]
                bn3.child = [
                        BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                        BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                    ] # 16x16x16
                
    # depth5
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            for d3 in range(8):
                bn3 = bn2.child[d3]
                for d4 in range(8):
                    bn4 = bn3.child[d4]
                    bn4.child = [
                            LeafNode(), LeafNode(), LeafNode(), LeafNode(),
                            LeafNode(), LeafNode(), LeafNode(), LeafNode(),
                        ] # 32x32x32
            
    offset_voxel_ix = volume_unit_ix * volume_unit_dim
    offset_voxel_iy = volume_unit_iy * volume_unit_dim
    offset_voxel_iz = volume_unit_iz * volume_unit_dim             
    
    for d1 in range(8):
        bn1 = octree.child[d1]
        x1 = octree_orders[d1][0]
        y1 = octree_orders[d1][1]
        z1 = octree_orders[d1][2]
        
        for d2 in range(8):
            bn2 = bn1.child[d2]
            x2 = octree_orders[d2][0]
            y2 = octree_orders[d2][1]
            z2 = octree_orders[d2][2]
            
            for d3 in range(8):
                bn3 = bn2.child[d3]
                x3 = octree_orders[d3][0]
                y3 = octree_orders[d3][1]
                z3 = octree_orders[d3][2]  
                
                for d4 in range(8):
                    bn4 = bn3.child[d4]
                    x4 = octree_orders[d4][0]
                    y4 = octree_orders[d4][1]
                    z4 = octree_orders[d4][2]  
                    
                    for d5 in range(8):
                        ln5 = bn4.child[d5]
                        x5 = octree_orders[d5][0]
                        y5 = octree_orders[d5][1]
                        z5 = octree_orders[d5][2]  
                    
                        x = (x1 * 16) + (x2 * 8) + (x3 * 4) + (x4 * 2) + x5
                        y = (y1 * 16) + (y2 * 8) + (y3 * 4) + (y4 * 2) + y5
                        z = (z1 * 16) + (z2 * 8) + (z3 * 4) + (z4 * 2) + z5
                
                        vertex = [
                            offset_voxel_ix + x * (2 ** (base_log-depth)),# TODO: offset?????
                            offset_voxel_iy + y * (2 ** (base_log-depth)),
                            offset_voxel_iz + z * (2 ** (base_log-depth)),
                        ]
                        value = volume_units_32[k].D[x, y, z]
                        weight = volume_units_32[k].W[x, y, z]
                        R = volume_units_32[k].R[x, y, z]
                        G = volume_units_32[k].G[x, y, z]
                        B = volume_units_32[k].B[x, y, z]
                        complexity = volume_units_32[k].complexity
                    
                        ln5.vertex = vertex
                        ln5.value = value
                        ln5.weight = weight
                        ln5.R = R
                        ln5.G = G
                        ln5.B = B
                        #ln5.R = volume_uint_to_color[k][0]
                        #ln5.G = volume_uint_to_color[k][1]
                        #ln5.B = volume_uint_to_color[k][2]
                        ln5.depth = depth #added
                        # print(ln5.depth, vertex, value, R, G, B)
                        ln5.complexity = complexity
    return octree


def construct_octree_depth4(volume_units_16, k, volume_unit_dim=16):
    volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
    octree = BranchNode()
    depth = 4
    base_log = 5
    if hasattr(volume_units_16[k], 's'):
        s = volume_units_16[k].s
        octree.s = s
    
    # depth1
    octree.child = [
            BranchNode(), BranchNode(), BranchNode(), BranchNode(),
            BranchNode(), BranchNode(), BranchNode(), BranchNode(),
        ] # 2x2x2
    
    # depth2
    for d1 in range(8):
        bn1 = octree.child[d1]
        bn1.child = [
                BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                BranchNode(), BranchNode(), BranchNode(), BranchNode(),
            ] # 4x4x4
        
    # depth3
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            bn2.child = [
                    BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                    BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                ] # 8x8x8
    
    # depth4
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            for d3 in range(8):
                bn3 = bn2.child[d3]
                bn3.child = [
                        LeafNode(), LeafNode(), LeafNode(), LeafNode(),
                        LeafNode(), LeafNode(), LeafNode(), LeafNode(),
                    ] # 16x16x16
            
    offset_voxel_ix = volume_unit_ix * volume_unit_dim
    offset_voxel_iy = volume_unit_iy * volume_unit_dim
    offset_voxel_iz = volume_unit_iz * volume_unit_dim             
    
    for d1 in range(8):
        bn1 = octree.child[d1]
        x1 = octree_orders[d1][0]
        y1 = octree_orders[d1][1]
        z1 = octree_orders[d1][2]
        
        for d2 in range(8):
            bn2 = bn1.child[d2]
            x2 = octree_orders[d2][0]
            y2 = octree_orders[d2][1]
            z2 = octree_orders[d2][2]
            
            for d3 in range(8):
                bn3 = bn2.child[d3]
                x3 = octree_orders[d3][0]
                y3 = octree_orders[d3][1]
                z3 = octree_orders[d3][2]  
                
                for d4 in range(8):
                    ln4 = bn3.child[d4]
                    x4 = octree_orders[d4][0]
                    y4 = octree_orders[d4][1]
                    z4 = octree_orders[d4][2]  
                    
                    x = (x1 * 8) + (x2 * 4) + (x3 * 2) + x4
                    y = (y1 * 8) + (y2 * 4) + (y3 * 2) + y4
                    z = (z1 * 8) + (z2 * 4) + (z3 * 2) + z4
                
                    vertex = [
                        offset_voxel_ix + x * (2 ** (base_log-depth)),
                        offset_voxel_iy + y * (2 ** (base_log-depth)),
                        offset_voxel_iz + z * (2 ** (base_log-depth)),
                    ]
                    value = volume_units_16[k].D[x, y, z]
                    weight = volume_units_16[k].W[x, y, z]
                    R = volume_units_16[k].R[x, y, z]
                    G = volume_units_16[k].G[x, y, z]
                    B = volume_units_16[k].B[x, y, z]
                    complexity = volume_units_16[k].complexity
                    
                    ln4.vertex = vertex
                    ln4.value = value
                    ln4.weight = weight
                    ln4.R = R
                    ln4.G = G
                    ln4.B = B
                    #ln4.R = volume_uint_to_color[k][0]
                    #ln4.G = volume_uint_to_color[k][1]
                    #ln4.B = volume_uint_to_color[k][2]
                    ln4.depth = depth #added
                    ln4.complexity = complexity
    return octree

###
def construct_octree_depth3(volume_units_8, k, volume_unit_dim=16):
    volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
    octree = BranchNode()
    depth = 3
    base_log = 5
    if hasattr(volume_units_8[k], 's'):
        s = volume_units_8[k].s
        octree.s = s
    
    # depth1
    octree.child = [
            BranchNode(), BranchNode(), BranchNode(), BranchNode(),
            BranchNode(), BranchNode(), BranchNode(), BranchNode(),
        ] # 2x2x2
    
    # depth2
    for d1 in range(8):
        bn1 = octree.child[d1]
        bn1.child = [
                BranchNode(), BranchNode(), BranchNode(), BranchNode(),
                BranchNode(), BranchNode(), BranchNode(), BranchNode(),
            ] # 4x4x4
        
    # depth3
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            bn2.child = [
                    LeafNode(), LeafNode(), LeafNode(), LeafNode(),
                    LeafNode(), LeafNode(), LeafNode(), LeafNode(),
                ] # 8x8x8
            
            
    offset_voxel_ix = volume_unit_ix * volume_unit_dim
    offset_voxel_iy = volume_unit_iy * volume_unit_dim
    offset_voxel_iz = volume_unit_iz * volume_unit_dim        
            
    for d1 in range(8):
        bn1 = octree.child[d1]
        x1 = octree_orders[d1][0]
        y1 = octree_orders[d1][1]
        z1 = octree_orders[d1][2]
        
        for d2 in range(8):
            bn2 = bn1.child[d2]
            x2 = octree_orders[d2][0]
            y2 = octree_orders[d2][1]
            z2 = octree_orders[d2][2]
            
            for d3 in range(8):
                ln3 = bn2.child[d3]
                x3 = octree_orders[d3][0]
                y3 = octree_orders[d3][1]
                z3 = octree_orders[d3][2]
            
                x = (x1 * 4) + (x2 * 2) + x3
                y = (y1 * 4) + (y2 * 2) + y3
                z = (z1 * 4) + (z2 * 2) + z3
            
                vertex = [
                    offset_voxel_ix + x * (2 ** (base_log-depth)) + 0.5,
                    offset_voxel_iy + y * (2 ** (base_log-depth)) + 0.5,
                    offset_voxel_iz + z * (2 ** (base_log-depth)) + 0.5,
                ]
                value = volume_units_8[k].D[x, y, z]
                weight = volume_units_8[k].W[x, y, z]
                R = volume_units_8[k].R[x, y, z]
                G = volume_units_8[k].G[x, y, z]
                B = volume_units_8[k].B[x, y, z]
                if hasattr(volume_units_8[k], 's'):
                    s = volume_units_8[k].s
                complexity = volume_units_8[k].complexity
                
                ln3.vertex = vertex
                ln3.value = value
                ln3.weight = weight
                ln3.R = R
                ln3.G = G
                ln3.B = B
                #ln3.R = volume_uint_to_color[k][0]
                #ln3.G = volume_uint_to_color[k][1]
                #ln3.B = volume_uint_to_color[k][2]
                ln3.depth = depth #added
                ln3.complexity = complexity
    return octree

###
def construct_octree_depth2(volume_units_4, k, volume_unit_dim=16):
    volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
    octree = BranchNode()
    depth = 2
    base_log = 5
    if hasattr(volume_units_4[k], 's'):
        s = volume_units_4[k].s
        octree.s = s
    
    # depth1
    octree.child = [
            BranchNode(), BranchNode(), BranchNode(), BranchNode(),
            BranchNode(), BranchNode(), BranchNode(), BranchNode(),
        ] # 2x2x2
    
    # depth2
    for d1 in range(8):
        bn1 = octree.child[d1]
        bn1.child = [
                LeafNode(), LeafNode(), LeafNode(), LeafNode(),
                LeafNode(), LeafNode(), LeafNode(), LeafNode(),
            ] # 4x4x4
    
    offset_voxel_ix = volume_unit_ix * volume_unit_dim
    offset_voxel_iy = volume_unit_iy * volume_unit_dim
    offset_voxel_iz = volume_unit_iz * volume_unit_dim        
            
    for d1 in range(8):
        bn1 = octree.child[d1]
        x1 = octree_orders[d1][0]
        y1 = octree_orders[d1][1]
        z1 = octree_orders[d1][2]
        
        for d2 in range(8):
            ln2 = bn1.child[d2]
            x2 = octree_orders[d2][0]
            y2 = octree_orders[d2][1]
            z2 = octree_orders[d2][2]
    
            x = (x1 * 2) + x2
            y = (y1 * 2) + y2
            z = (z1 * 2) + z2
            
            vertex = [
                offset_voxel_ix + x * (2 ** (base_log-depth)) + 1.5,
                offset_voxel_iy + y * (2 ** (base_log-depth)) + 1.5,
                offset_voxel_iz + z * (2 ** (base_log-depth)) + 1.5,
            ]
            value = volume_units_4[k].D[x, y, z]
            weight = volume_units_4[k].W[x, y, z]
            R = volume_units_4[k].R[x, y, z]
            G = volume_units_4[k].G[x, y, z]
            B = volume_units_4[k].B[x, y, z]
            if hasattr(volume_units_4[k], 's'):
                s = volume_units_4[k].s
            complexity = volume_units_4[k].complexity
            
            ln2.vertex = vertex
            ln2.value = value
            ln2.weight = weight
            ln2.R = R
            ln2.G = G
            ln2.B = B
            #ln2.R = volume_uint_to_color[k][0]
            #ln2.G = volume_uint_to_color[k][1]
            #ln2.B = volume_uint_to_color[k][2]
            ln2.depth = depth #added       
            ln2.complexity = complexity
    return octree        

###
def construct_octree_depth1(volume_units_2, k, volume_unit_dim=16):
    volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
    octree = BranchNode()
    depth = 2
    base_log = 5
    if hasattr(volume_units_2[k], 's'):
        s = volume_units_2[k].s
        octree.s = s
    
    # depth1
    octree.child = [
            LeafNode(), LeafNode(), LeafNode(), LeafNode(),
            LeafNode(), LeafNode(), LeafNode(), LeafNode(),
        ] # 2x2x2
        
    offset_voxel_ix = volume_unit_ix * volume_unit_dim
    offset_voxel_iy = volume_unit_iy * volume_unit_dim
    offset_voxel_iz = volume_unit_iz * volume_unit_dim        
            
    for d1 in range(8):
        ln1 = octree.child[d1]
        x1 = octree_orders[d1][0]
        y1 = octree_orders[d1][1]
        z1 = octree_orders[d1][2]
        
        x = x1
        y = y1
        z = z1
            
        vertex = [
            offset_voxel_ix + x * (2 ** (base_log-depth)) + 3.5,
            offset_voxel_iy + y * (2 ** (base_log-depth)) + 3.5,
            offset_voxel_iz + z * (2 ** (base_log-depth)) + 3.5,
        ]
        value = volume_units_2[k].D[x, y, z]
        weight = volume_units_2[k].W[x, y, z]
        R = volume_units_2[k].R[x, y, z]
        G = volume_units_2[k].G[x, y, z]
        B = volume_units_2[k].B[x, y, z]
        complexity = volume_units_2[k].complexity
                
        ln1.vertex = vertex
        ln1.value = value
        ln1.weight = weight
        ln1.R = R
        ln1.G = G
        ln1.B = B
        #ln1.R = volume_uint_to_color[k][0]
        #ln1.G = volume_uint_to_color[k][1]
        #ln1.B = volume_uint_to_color[k][2]
        ln1.depth = depth #added
        ln1.complexity = complexity
    return octree


def construct_octree(volume_units_16,
                     volume_units_8,
                     volume_units_4,
                     volume_units_2):
    octrees = {}
    for k in volume_units_16.keys():
        octree = construct_octree_depth4(volume_units_16, k)
        octrees[k] = octree        

    for k in volume_units_8.keys():
        octree = construct_octree_depth3(volume_units_8, k)
        octrees[k] = octree        
        
    for k in volume_units_4.keys():
        octree = construct_octree_depth2(volume_units_4, k)
        octrees[k] = octree      
        
    for k in volume_units_2.keys():
        octree = construct_octree_depth1(volume_units_2, k)
        octrees[k] = octree
    return octrees


def construct_octree_gen(vu_dicts_list:list,
                     depths:list,
                     volume_unit_dim=16):
    octrees = {}
    CONSTRUCT_FUNC = {5: construct_octree_depth5,
                       4: construct_octree_depth4,
                       3: construct_octree_depth3,
                       2: construct_octree_depth2,
                       1: construct_octree_depth1}
    
    for vu,depth in zip(vu_dicts_list,depths):
        construct_func = CONSTRUCT_FUNC[depth]
        for k in vu.keys():
            octree = construct_func(vu, k, volume_unit_dim)
            octrees[k] = octree
    return octrees

###############################################################################
# 각 볼륨마다 따로 메쉬 생성
############################################################################### 

def enumerate_impl_c(node):
    #print('enumerate_impl_c')
    
    b = node.is_branch()
    
    if (b):
        enumerate_impl_c(node.child[0])
        enumerate_impl_c(node.child[1])
        enumerate_impl_c(node.child[2])
        enumerate_impl_c(node.child[3])
        enumerate_impl_c(node.child[4])
        enumerate_impl_c(node.child[5])
        enumerate_impl_c(node.child[6])
        enumerate_impl_c(node.child[7])
        
        enumerate_impl_f_x(node.child[0], node.child[1])
        enumerate_impl_f_x(node.child[2], node.child[3])
        enumerate_impl_f_x(node.child[4], node.child[5])
        enumerate_impl_f_x(node.child[6], node.child[7])
        
        enumerate_impl_f_y(node.child[0], node.child[2])
        enumerate_impl_f_y(node.child[1], node.child[3])
        enumerate_impl_f_y(node.child[4], node.child[6])
        enumerate_impl_f_y(node.child[5], node.child[7])
        
        enumerate_impl_f_z(node.child[0], node.child[4])
        enumerate_impl_f_z(node.child[1], node.child[5])
        enumerate_impl_f_z(node.child[2], node.child[6])
        enumerate_impl_f_z(node.child[3], node.child[7])
        
        enumerate_impl_e_xy(node.child[0], node.child[1], node.child[2], node.child[3])
        enumerate_impl_e_xy(node.child[4], node.child[5], node.child[6], node.child[7])
        
        enumerate_impl_e_yz(node.child[0], node.child[2], node.child[4], node.child[6])
        enumerate_impl_e_yz(node.child[1], node.child[3], node.child[5], node.child[7])
        
        enumerate_impl_e_xz(node.child[0], node.child[1], node.child[4], node.child[5])
        enumerate_impl_e_xz(node.child[2], node.child[3], node.child[6], node.child[7])
        
        enumerate_impl_v(
                node.child[0],
                node.child[1],
                node.child[2],
                node.child[3],
                node.child[4],
                node.child[5],
                node.child[6],
                node.child[7],
            )
        

# 완료 
def enumerate_impl_f_x(node1, node2):
    #print('enumerate_impl_f_x')
    
    b1 = node1.is_branch()
    b2 = node2.is_branch()
    
    if (b1 or b2):
        enumerate_impl_f_x(node1.child[1] if b1 else node1, node2.child[0] if b2 else node2)
        enumerate_impl_f_x(node1.child[3] if b1 else node1, node2.child[2] if b2 else node2)
        enumerate_impl_f_x(node1.child[5] if b1 else node1, node2.child[4] if b2 else node2)
        enumerate_impl_f_x(node1.child[7] if b1 else node1, node2.child[6] if b2 else node2)
        
        enumerate_impl_e_xy(
                node1.child[1] if b1 else node1,
                node2.child[0] if b2 else node2,
                node1.child[3] if b1 else node1,
                node2.child[2] if b2 else node2,
            )
        
        enumerate_impl_e_xy(
                node1.child[5] if b1 else node1,
                node2.child[4] if b2 else node2,
                node1.child[7] if b1 else node1,
                node2.child[6] if b2 else node2,
            )
        
        enumerate_impl_e_xz(
                node1.child[1] if b1 else node1,
                node2.child[0] if b2 else node2,
                node1.child[5] if b1 else node1,
                node2.child[4] if b2 else node2,
            )
        
        enumerate_impl_e_xz(
                node1.child[3] if b1 else node1,
                node2.child[2] if b2 else node2,
                node1.child[7] if b1 else node1,
                node2.child[6] if b2 else node2,
            )
        
        enumerate_impl_v(
                node1.child[1] if b1 else node1,
                node2.child[0] if b2 else node2,
                node1.child[3] if b1 else node1,
                node2.child[2] if b2 else node2,
                node1.child[5] if b1 else node1,
                node2.child[4] if b2 else node2,
                node1.child[7] if b1 else node1,
                node2.child[6] if b2 else node2,
            )
        
# 완료  
def enumerate_impl_f_y(node1, node2):
    #print('enumerate_impl_f_y')
    
    b1 = node1.is_branch()
    b2 = node2.is_branch()
    
    if (b1 or b2):
        enumerate_impl_f_y(node1.child[2] if b1 else node1, node2.child[0] if b2 else node2)
        enumerate_impl_f_y(node1.child[3] if b1 else node1, node2.child[1] if b2 else node2)
        enumerate_impl_f_y(node1.child[6] if b1 else node1, node2.child[4] if b2 else node2)
        enumerate_impl_f_y(node1.child[7] if b1 else node1, node2.child[5] if b2 else node2)
        
        enumerate_impl_e_xy(
                node1.child[2] if b1 else node1,
                node1.child[3] if b1 else node1,
                node2.child[0] if b2 else node2,
                node2.child[1] if b2 else node2,
            )
        
        enumerate_impl_e_xy(
                node1.child[6] if b1 else node1,
                node1.child[7] if b1 else node1,
                node2.child[4] if b2 else node2,
                node2.child[5] if b2 else node2,
            )
        
        enumerate_impl_e_yz(
                node1.child[3] if b1 else node1,
                node2.child[1] if b2 else node2,
                node1.child[7] if b1 else node1,
                node2.child[5] if b2 else node2,
            )
        
        enumerate_impl_e_yz(
                node1.child[2] if b1 else node1,
                node2.child[0] if b2 else node2,
                node1.child[6] if b1 else node1,
                node2.child[4] if b2 else node2,
            )
        
        enumerate_impl_v(
                node1.child[2] if b1 else node1,
                node1.child[3] if b1 else node1,
                node2.child[0] if b2 else node2,
                node2.child[1] if b2 else node2,
                node1.child[6] if b1 else node1,
                node1.child[7] if b1 else node1,
                node2.child[4] if b2 else node2,
                node2.child[5] if b2 else node2,
            )
        
# 완료  
def enumerate_impl_f_z(node1, node2):
    #print('enumerate_impl_f_z')
    
    b1 = node1.is_branch()
    b2 = node2.is_branch()
    
    if (b1 or b2):
        enumerate_impl_f_z(node1.child[4] if b1 else node1, node2.child[0] if b2 else node2)
        enumerate_impl_f_z(node1.child[5] if b1 else node1, node2.child[1] if b2 else node2)
        enumerate_impl_f_z(node1.child[6] if b1 else node1, node2.child[2] if b2 else node2)
        enumerate_impl_f_z(node1.child[7] if b1 else node1, node2.child[3] if b2 else node2)

        enumerate_impl_e_xz(
                node1.child[4] if b1 else node1,
                node1.child[5] if b1 else node1,
                node2.child[0] if b2 else node2,
                node2.child[1] if b2 else node2,
            )
        
        enumerate_impl_e_xz(
                node1.child[6] if b1 else node1,
                node1.child[7] if b1 else node1,
                node2.child[2] if b2 else node2,
                node2.child[3] if b2 else node2,
            )
        
        enumerate_impl_e_yz(
                node1.child[4] if b1 else node1,
                node1.child[6] if b1 else node1,
                node2.child[0] if b2 else node2,
                node2.child[2] if b2 else node2,
            )
        
        enumerate_impl_e_yz(
                node1.child[5] if b1 else node1,
                node1.child[7] if b1 else node1,
                node2.child[1] if b2 else node2,
                node2.child[3] if b2 else node2,
            )
    
        enumerate_impl_v(
                node1.child[4] if b1 else node1,
                node1.child[5] if b1 else node1,
                node1.child[6] if b1 else node1,
                node1.child[7] if b1 else node1,
                node2.child[0] if b2 else node2,
                node2.child[1] if b2 else node2,
                node2.child[2] if b2 else node2,
                node2.child[3] if b2 else node2,
            )

def enumerate_impl_e_xy(node1, node2, node3, node4):
    #print('enumerate_impl_e_xy')
    
    b1 = node1.is_branch()
    b2 = node2.is_branch()
    b3 = node3.is_branch()
    b4 = node4.is_branch()
    
    if (b1 or b2 or b3 or b4):
        enumerate_impl_e_xy(
                node1.child[3] if b1 else node1,
                node2.child[2] if b2 else node2,
                node3.child[1] if b3 else node3,
                node4.child[0] if b4 else node4,
            )
        
        enumerate_impl_e_xy(
                node1.child[7] if b1 else node1,
                node2.child[6] if b2 else node2,
                node3.child[5] if b3 else node3,
                node4.child[4] if b4 else node4,
            )
        
        enumerate_impl_v(
                node1.child[3] if b1 else node1,
                node2.child[2] if b2 else node2,
                node3.child[1] if b3 else node3,
                node4.child[0] if b4 else node4,
                node1.child[7] if b1 else node1,
                node2.child[6] if b2 else node2,
                node3.child[5] if b3 else node3,
                node4.child[4] if b4 else node4,
            )
        
def enumerate_impl_e_xz(node1, node2, node3, node4):
    #print('enumerate_impl_e_xz')
    
    b1 = node1.is_branch()
    b2 = node2.is_branch()
    b3 = node3.is_branch()
    b4 = node4.is_branch()
    
    if (b1 or b2 or b3 or b4):
        enumerate_impl_e_xz(
                node1.child[5] if b1 else node1,
                node2.child[4] if b2 else node2,
                node3.child[1] if b3 else node3,
                node4.child[0] if b4 else node4,
            )
        
        enumerate_impl_e_xz(
                node1.child[7] if b1 else node1,
                node2.child[6] if b2 else node2,
                node3.child[3] if b3 else node3,
                node4.child[2] if b4 else node4,
            )
        
        enumerate_impl_v(
                node1.child[5] if b1 else node1,
                node2.child[4] if b2 else node2,
                node1.child[7] if b1 else node1,
                node2.child[6] if b2 else node2,
                node3.child[1] if b3 else node3,
                node4.child[0] if b4 else node4,
                node3.child[3] if b3 else node3,
                node4.child[2] if b4 else node4,
            )
        
        
def enumerate_impl_e_yz(node1, node2, node3, node4):
    #print('enumerate_impl_e_yz')
    
    b1 = node1.is_branch()
    b2 = node2.is_branch()
    b3 = node3.is_branch()
    b4 = node4.is_branch()
    
    if (b1 or b2 or b3 or b4):
        enumerate_impl_e_yz(
                node1.child[6] if b1 else node1,
                node2.child[4] if b2 else node2,
                node3.child[2] if b3 else node3,
                node4.child[0] if b4 else node4,
            )
        
        enumerate_impl_e_yz(
                node1.child[7] if b1 else node1,
                node2.child[5] if b2 else node2,
                node3.child[3] if b3 else node3,
                node4.child[1] if b4 else node4,
            )
        
        enumerate_impl_v(
                node1.child[6] if b1 else node1,
                node1.child[7] if b1 else node1,
                node2.child[4] if b2 else node2,
                node2.child[5] if b2 else node2,
                node3.child[2] if b3 else node3,
                node3.child[3] if b3 else node3,
                node4.child[0] if b4 else node4,
                node4.child[1] if b4 else node4,
            )
        
def enumerate_impl_v(node1, node2, node3, node4, node5, node6, node7, node8):
    #print('enumerate_impl_v')
    
    b1 = node1.is_branch()
    b2 = node2.is_branch()
    b3 = node3.is_branch()
    b4 = node4.is_branch()
    b5 = node5.is_branch()
    b6 = node6.is_branch()
    b7 = node7.is_branch()
    b8 = node8.is_branch()
    
    if (b1 or b2 or b3 or b4 or b5 or b6 or b7 or b8):
        enumerate_impl_v(
                node1.child[7] if b1 else node1,
                node2.child[6] if b2 else node2,
                node3.child[5] if b3 else node3,
                node4.child[4] if b4 else node4,
                node5.child[3] if b5 else node5,
                node6.child[2] if b6 else node6,
                node7.child[1] if b7 else node7,
                node8.child[0] if b8 else node8,
            )
    else:
        vertices = [
                node1.vertex, node2.vertex, node3.vertex, node4.vertex,
                node5.vertex, node6.vertex, node7.vertex, node8.vertex,
            ]
        
        values = [
                node1.value, node2.value, node3.value, node4.value,
                node5.value, node6.value, node7.value, node8.value,
            ]
        
        weights = [
                node1.weight, node2.weight, node3.weight, node4.weight,
                node5.weight, node6.weight, node7.weight, node8.weight,
            ]
        
        Rs = [
                node1.R, node2.R, node3.R, node4.R,
                node5.R, node6.R, node7.R, node8.R,
            ]
        
        Gs = [
                node1.G, node2.G, node3.G, node4.G,
                node5.G, node6.G, node7.G, node8.G,
            ]
        
        Bs = [
                node1.B, node2.B, node3.B, node4.B,
                node5.B, node6.B, node7.B, node8.B,
            ]
        depths = [
                node1.depth, node2.depth, node3.depth, node4.depth,
                node5.depth, node6.depth, node7.depth, node8.depth,
            ] #added
        complexities = [
                node1.complexity, node2.complexity, node3.complexity, node4.complexity,
                node5.complexity, node6.complexity, node7.complexity, node8.complexity,
            ] #added, #TODO

        mc = marching_cubes() # added
        # marching_cubes.voxel_size_mm = 0.004 * 2**(4 - node1.depth)
        # marching_cubes.volume_unit_dim = 2**(node1.depth)
        # print("before call", marching_cubes.voxel_size_mm, marching_cubes.volume_unit_dim)
        if type(values[0])!=np.float32:
            values = [v.detach().cpu() for v in values]
        # mc(vertices, values, weights, Rs, Gs, Bs)
        # mc(vertices, values, weights, Rs, Gs, Bs, depths=depths) #added
        mc(vertices, values, weights, Rs, Gs, Bs, complexities=complexities) #added


class marching_cubes:
    vmax = 1e-5
    cmap = cm.gist_rainbow
    norm = colors.LogNorm(vmin=1e-15, vmax=vmax)
    # norm = colors.Normalize(vmin=0, vmax=vmax)
    colormap = cm.ScalarMappable(norm=norm, cmap=cmap)
    # fig,ax = plt.subplots(figsize=(2,3))
    # plt.colorbar(colormap)
    # ax.remove()
    # plt.savefig('plot_onlycbar_tight.png',bbox_inches='tight')
    
    def __init__(self,
                volume_origin=None,
                volume_unit_dim=None,
                voxel_size_mm=None,
                voxel_to_vertices=None,
                mesh=None):
        if volume_origin is not None:
            marching_cubes.volume_origin = volume_origin
            marching_cubes.volume_unit_dim = volume_unit_dim
            marching_cubes.voxel_size_mm = voxel_size_mm
            marching_cubes.voxel_to_vertices = voxel_to_vertices
            marching_cubes.mesh = mesh
    
    def __call__(self,
                vertices, 
                values, 
                weights, 
                Rs, Gs, Bs,
                depths=[], 
                complexities=[]):
        #vertices = np.array(vertices)
        #print(np.mean(vertices, axis=0))
        #points.append(np.mean(vertices, axis=0))
        
        # print("in call", marching_cubes.voxel_size_mm, marching_cubes.volume_unit_dim)
        order = [0, 1, 3, 2, 4, 5, 7, 6]
        vertices = [vertices[i] for i in order]
        values = [values[i] for i in order]
        weights = [weights[i] for i in order]
        if len(depths): # added - from here..
            depths = [depths[i] for i in order]
            color_depth={2:[255,0,0], 2.5:[255,255,0], 3:[0,255,0], 3.5:[0,255,255], 4:[0,0,255], 4.5: [0,0,0], 5:[255,255,255]}
        elif len(complexities):
            complexities = [complexities[i] for i in order]
            colormap = marching_cubes.colormap
        # ...to here
        
        marching_cube_index = 0
        for i in range(8):
            if values[i] < 0.0:
                marching_cube_index |= (1 << i)
                
            if (weights[i] <= 0.0) and (values[i] == 0.0):
                return
                
        if (marching_cube_index == 0) or (marching_cube_index == 255):
            return
        
        vertex_indicies = -np.ones(12, np.int32)
        for i in range(12):
            if (edge_table[marching_cube_index] & (1 << i)): # vertex가 존재하는 에지라면
                e1, e2 = edge_to_vert[i]
                d1 = np.abs(values[e1])
                d2 = np.abs(values[e2])
                r1 = Rs[e1]
                r2 = Rs[e2]
                g1 = Gs[e1]
                g2 = Gs[e2]
                b1 = Bs[e1]
                b2 = Bs[e2]
            
                vertex1 = vertices[e1].copy()
                vertex2 = vertices[e2].copy()
    
                vertex1[0] = vertex1[0] * marching_cubes.voxel_size_mm + marching_cubes.volume_origin[0, 0] 
                vertex1[1] = vertex1[1] * marching_cubes.voxel_size_mm + marching_cubes.volume_origin[1, 0] 
                vertex1[2] = vertex1[2] * marching_cubes.voxel_size_mm + marching_cubes.volume_origin[2, 0] 
                vertex2[0] = vertex2[0] * marching_cubes.voxel_size_mm + marching_cubes.volume_origin[0, 0] 
                vertex2[1] = vertex2[1] * marching_cubes.voxel_size_mm + marching_cubes.volume_origin[1, 0] 
                vertex2[2] = vertex2[2] * marching_cubes.voxel_size_mm + marching_cubes.volume_origin[2, 0] 
                
                vertex_x = vertex1[0] + (d1 * (vertex2[0] - vertex1[0])) / (d1 + d2)
                vertex_y = vertex1[1] + (d1 * (vertex2[1] - vertex1[1])) / (d1 + d2)
                vertex_z = vertex1[2] + (d1 * (vertex2[2] - vertex1[2])) / (d1 + d2)
                
                vertex_r = (d2 * r1 + d1 * r2) / (d1 + d2)
                vertex_g = (d2 * g1 + d1 * g2) / (d1 + d2)
                vertex_b = (d2 * b1 + d1 * b2) / (d1 + d2)
                
                '''
                vertex_indicies[i] = mesh.add_vertex(
                        vertex_x,
                        vertex_y,
                        vertex_z,
                        255, 0, 0,
                    )
            
                
                '''
                if (vertex_x, vertex_y, vertex_z) in marching_cubes.voxel_to_vertices:
                    vertex_indicies[i] = marching_cubes.voxel_to_vertices[vertex_x, vertex_y, vertex_z]
                else:
                    if len(depths):
                        color = color_depth[(depths[e1]+depths[e2])/2]
                        if abs(depths[e1]-depths[e2])>=1:
                            color = [255, 255, 255]
                    elif len(complexities):
                        if (complexities[e1]+complexities[e2])/2==0:
                            print("what hapapened?")
                        color = [c*255 for c in colormap.to_rgba((complexities[e1]+complexities[e2])/2)[:3]]
                    else:
                        color = [
                            vertex_r * 255.0,
                            vertex_g * 255.0,
                            vertex_b * 255.0
                            ]
                    vertex_indicies[i] = marching_cubes.mesh.add_vertex(
                            vertex_x,
                            vertex_y,
                            vertex_z,
                            *color
                        )
                    marching_cubes.voxel_to_vertices[vertex_x, vertex_y, vertex_z] = vertex_indicies[i]
                
                    
        for i in range(0, 12, 3):
            if tri_table[marching_cube_index][i] == -1:
                break
            
            marching_cubes.mesh.add_face(vertex_indicies[tri_table[marching_cube_index][i]],
                                          vertex_indicies[tri_table[marching_cube_index][i+2]],
                                          vertex_indicies[tri_table[marching_cube_index][i+1]])


def incrementKey(k, dim, val):
    assert(type(k)==tuple)
    temp_k = list(k)
    
    if type(dim)==int:
        temp_k[dim] += val
    elif type(dim)==list or type(dim)==tuple:
        assert len(val)==len(dim)
        for d,v in zip(dim,val):
            temp_k[d] += v
    return tuple(temp_k)


def dualMCMesh(octrees,
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
    # Octree에 저장된 모든 볼륨 유닛마다 메쉬 생성
    if os.path.exists('./Meshes_%s' % (dataset.scenes[0])):
        shutil.rmtree('./Meshes_%s' % (dataset.scenes[0]))
    os.mkdir('./Meshes_%s' % (dataset.scenes[0]))
    
    if evaluate:
        ix, iy, iz = np.meshgrid(range(0, volume_unit_dim), 
                                 range(0, volume_unit_dim), 
                                 range(0, volume_unit_dim), indexing='ij')
        offset_voxel_indices = np.vstack((ix.flatten(), iy.flatten(), iz.flatten()))
        num_voxels_per_volume_unit = volume_unit_dim * volume_unit_dim * volume_unit_dim
        kwargs = dict(volume_units=ori_units,
                      volume_origin=volume_origin,
                      volume_unit_dim=volume_unit_dim,
                      offset_voxel_indices=offset_voxel_indices,
                      num_voxels_per_volume_unit=num_voxels_per_volume_unit)
        hausdorff = [] # TODO
        chamfer = []
        sizes_en = []
        sizes_ori = []
        debug_keys = []
        count = 0
        if fast:
            voxelfunc = fastVoxelMesh()
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
        else:
            voxelfunc = singleVoxelMesh()
            
    for k in tqdm(octrees.keys()):
        volume_unit_ix, volume_unit_iy, volume_unit_iz = k[-3:]
        mesh = TriMesh()
        voxel_to_vertices = {}
        mc = marching_cubes(volume_origin,
                            volume_unit_dim,
                            voxel_size_mm,
                            voxel_to_vertices,
                            mesh)
        
        enumerate_impl_c(octrees[k])
        
        if incrementKey(k, -3, 1) in octrees:
            enumerate_impl_f_x(
                    octrees[k],
                    octrees[incrementKey(k, -3, 1)]
                )
            
        if incrementKey(k, -2, 1) in octrees:
            enumerate_impl_f_y(
                    octrees[k],
                    octrees[incrementKey(k, -2, 1)]
                )
            
        if incrementKey(k, -1, 1) in octrees:
            enumerate_impl_f_z(
                    octrees[k],
                    octrees[incrementKey(k, -1, 1)]
                )
            
        if (incrementKey(k, -3, 1) in octrees) and \
            (incrementKey(k, -2, 1) in octrees) and \
            (incrementKey(k, [-3,-2], [1,1]) in octrees):
            enumerate_impl_e_xy(
                    octrees[k],
                    octrees[incrementKey(k, -3, 1)],
                    octrees[incrementKey(k, -2, 1)],
                    octrees[incrementKey(k, [-3,-2], [1,1])],
                )
            
        if (incrementKey(k, -2, 1) in octrees) and \
            (incrementKey(k, -1, 1) in octrees) and \
            (incrementKey(k, [-2,-1], [1,1]) in octrees):
            enumerate_impl_e_yz(
                    octrees[k],
                    octrees[incrementKey(k, -2, 1)],
                    octrees[incrementKey(k, -1, 1)],
                    octrees[incrementKey(k, [-2,-1], [1,1])],
                )
            
        if (incrementKey(k, -3, 1) in octrees) and \
            (incrementKey(k, -1, 1) in octrees) and \
            (incrementKey(k, [-3,-1], [1,1]) in octrees):
            enumerate_impl_e_xz(
                    octrees[k],
                    octrees[incrementKey(k, -3, 1)],
                    octrees[incrementKey(k, -1, 1)],
                    octrees[incrementKey(k, [-3,-1], [1,1])],
                )
        
        if (incrementKey(k, -2, 1) in octrees) and \
            (incrementKey(k, -1, 1) in octrees) and \
            (incrementKey(k, [-2,-1], [1,1]) in octrees) and \
            (incrementKey(k, -3, 1)in octrees) and \
            (incrementKey(k, [-3,-2], [1,1])in octrees) and \
            (incrementKey(k, [-3,-1], [1,1]) in octrees) and \
            (incrementKey(k, [-3,-2,-1], [1,1,1]) in octrees):
            enumerate_impl_v(
                    octrees[k],
                    octrees[incrementKey(k, -3, 1)],
                    octrees[incrementKey(k, -2, 1)],
                    octrees[incrementKey(k, [-3,-2], [1,1])],
                    octrees[incrementKey(k, -1, 1)],
                    octrees[incrementKey(k, [-3,-1], [1,1])],
                    octrees[incrementKey(k, [-2,-1], [1,1])],
                    octrees[incrementKey(k, [-3,-2,-1], [1,1,1])],
                )
            
        if mesh.n_vertices == 0:
            continue
        
        mesh.save_ply('Meshes_%s/%d_%d_%d.ply' % (dataset.scenes[0], volume_unit_ix, volume_unit_iy, volume_unit_iz))
        
        if evaluate: #TODO
            # vunit = ori_units[k]
            if len(k)==3:
                kwargs.update(mesh=TriMesh(),
                              volume_unit_ix=volume_unit_ix,
                              volume_unit_iy=volume_unit_iy,
                              volume_unit_iz=volume_unit_iz,
                              voxel_to_vertices={},
                              voxel_size_mm=voxel_size_mm) # TODO
            else:
                kwargs.update(mesh=TriMesh(),
                              k=k,
                              voxel_to_vertices={},
                              voxel_size_mm=voxel_size_mm) # TODO
            voxel_ori = voxelfunc(**kwargs)
            # 원본 mesh와 다해상도압축된 mesh의 기하정확도를 구해서 accumulate한다
            len_o = max(1, sum(np.all(voxel_ori.vertices[:,:3], axis=1)))
            ori_v = voxel_ori.vertices[:,:3]
            ori_v = ori_v[:len_o]
            len_r = max(1, sum(np.all(mesh.vertices[:,:3], axis=1)))
            rec_v = mesh.vertices[:,:3]
            rec_v = rec_v[:len_r]
            if len_o>=3 and len_r>=3:
                # hausdorff += [customHausdorff(voxel_ori, mesh)]
                chamfer += [customChamfer(voxel_ori, mesh)]
            # try:
            hausdorff += [igl.hausdorff(rec_v,
                                        mesh.faces[:len_r],
                                        ori_v,
                                        voxel_ori.faces[:len_o])]
            # except:
            #     print("wehy????")
            
            # chamfer_dist = ChamferDistance()
            # dist1 = chamfer_dist(torch.Tensor([ori_v]), 
            #                       torch.Tensor([rec_v]))
            # dist2 = chamfer_dist(torch.Tensor([rec_v]), 
            #                       torch.Tensor([ori_v]))
            # chamfer += [0.5*(torch.mean(dist1) + torch.mean(dist2))]
            
            if hasattr(octrees[k], 's'):
                s = octrees[k].s
            else:
                s = None
            sizes_en += [s]
            if hasattr(ori_units[k], 's'):
                s_ori = ori_units[k].s
            else:
                s_ori = None
            sizes_ori += [s_ori]
            voxel_ori.save_ply_without_properties('Meshes_%s/%d_%d_%d_ori.ply' % (dataset.scenes[0], volume_unit_ix, volume_unit_iy, volume_unit_iz))
            debug_keys += [k]
            count += 1
            
    if evaluate:
        try: # 죄송합니다...
            print("maxes:", np.max(hausdorff), np.max(chamfer), np.max(sizes_en), np.max(sizes_ori))
            print("mins:", np.min(hausdorff), np.min(chamfer), np.min(sizes_en), np.min(sizes_ori))
            print("medians:", np.median(hausdorff), np.median(chamfer), np.median(sizes_en), np.median(sizes_ori))
            # # find bad mesh
            # badhaus = np.argmax(hausdorff)
            # badcham = np.argmax(chamfer)
            # print("max error at:", debug_keys[badhaus], debug_keys[badcham])
            return mesh, np.average(hausdorff), np.average(chamfer), np.average(sizes_en), np.average(sizes_ori)
        except:
            print("maxes:", np.max(hausdorff), np.max(chamfer), None, None)
            print("mins:", np.min(hausdorff), np.min(chamfer), None, None)
            print("medians:", np.median(hausdorff), np.median(chamfer), None, None)
            # # find bad mesh
            # badhaus = np.argmax(hausdorff)
            # badcham = np.argmax(chamfer)
            # print("max error at:", debug_keys[badhaus], debug_keys[badcham])
            return mesh, np.average(hausdorff), np.average(chamfer), None, None
    return mesh
