# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 21:52:35 2022

@author: alienware
"""

import numpy as np
from binarytree import Node
from source.VolumeUnit import VolumeUnit
from source.ConfigSettings import base_voxel_size, base_volume_unit_dim


class VolumeUnit4x4(VolumeUnit):
    # using the existing VolumeUnit class, just in case
    def __init__(self, tsdf=None, mask=None):
        super().__init__(volume_unit_dim=4, depth=np.nan)
        self.D = tsdf
        self.M = mask
        return
    

class AWMRblock(Node):
    def __init__(self, 
                 axisres,
                 unit_index,
                 left=None, 
                 right=None,
                 start_point=np.array((0,0,0)),
                 tsdf=None):
        assert(tsdf is None) or type(tsdf)==np.ndarray
        assert(len(unit_index)==4) # to prevent mess
        super().__init__(str(axisres), left=None, right=None)
        '''
        Parameters
        ----------
        unit_index: tuple
            (scene_name(string), vunit_x(int), vunit_y(int), vunit_z(int))
        axisres: tuple
            (res_x, res_y, res_z), possible values are 16, 8, 4
        start_point: numpy.ndarray
            the front-top-leftmost point within the block,
            stored as fractions. (eg. 0, 0.125, 0.25, 0.5, 0.625, ...)
            eg. np.array([0, 0.5, 0.25]) the front-top-leftmost point
            of this block is located at the far left of the block,
            at the middle of the height, and a quarter of the volume
            unit's length behind.
            이 축별다해상도블록의 가장 전-좌-상단의 위치를 복셀 유닛의 길이
            기준으로 분수로 나타낸 것입니다. 예를 들어 64*64*64mm 복셀유닛
            에서 [0, 0.5, 0.25]의 start_point를 가졌다면 해당 블록은 유닛
            왼쪽에서 0mm, 위에서 32mm, 앞에서 16mm에 해당하는 지점이 전-
            좌-상단이 됩니다. 후-우-하단은 axisres를 사용해서 계산할 수 있
            습니다. 이를테면, axisres가 (4, 8, 16)이면 블록의 크기는 전체
            유닛의 (1, 0.5, 0.25)가 되므로, 후-우-하단은 (1,1,0.5)입니다.
        tsdf: numpy.ndarray
            tsdf of size 4x4x4.
            Will be stored as VolumeUnit4x4 object -- in attribute "D".
        '''
        self.unit_index = unit_index
        self.start_point = start_point
        self.tsdf = VolumeUnit4x4(tsdf=tsdf)
        self.axisres = axisres
        
    def voxel_size(self):
        '''
        Returns
        -------
        numpy.ndarray
            returns the size of a voxel as numpy.ndarray in millimeters
            - np.array([size_x, size_y, size_z])
        '''
        return base_volume_unit_dim*base_voxel_size/np.array(self.axisres)
   
    def find_node(self, start_point):
        '''
        Parameters
        ----------
        start_point : numpy.ndarray
            The start point of the block we want to search for.

        Returns
        -------
        n : AWMRblock
            Note that this method only finds node from under this node.
            If there is no node with start_point, it returns None.

        '''
        for n in self.leaves:
            if np.sum(np.abs((n.start_point - start_point))) < 1e-5:
                if n.tsdf is not None:
                    return n
        print ("No such node, returning None")
        return None