from abc import ABC, abstractmethod
from .ConfigSettings import *
from .MCLutClassic import *

octree_orders = [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1]
            ]

###
class OctTreeNode(object):
    def is_branch(self):
        return (type(self) == OctBranchNode)

    def is_leaf(self):
        return type(self) == OctLeafNodeDWRGB
    
###
class OctBranchNode(OctTreeNode):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.child = []
                
###
class OctLeafNodeDWRGB(OctTreeNode):
    def __init__(self, index=[0.0,0.0,0.0], d=0.0, w=0.0, r=0.0, g=0.0, b=0.0, parent=None):
        super().__init__()
        self.index  = index
        self.d      = d
        self.w      = w
        self.r      = r
        self.g      = g
        self.b      = b
        self.parent = parent
        
    def setValues(self, index=[0.0,0.0,0.0], d=0.0, w=0.0, r=0.0, g=0.0, b=0.0):
        self.index  = index
        self.d      = d
        self.w      = w
        self.r      = r
        self.g      = g
        self.b      = b


###############################################################################
# 다해상도 TSDF 볼륨 유닛 마다 Octree 생성 
###############################################################################
 
###       
def convertVolumeUnit16ToOctree(vunit, vunit_x, vunit_y, vunit_z):
    octree = OctBranchNode()
    
    # depth1
    octree.child = [
        OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode(),
        OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode()
    ] # 2x2x2
    
    # depth2
    for d1 in range(8):
        bn1 = octree.child[d1]
        bn1.child = [
            OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode(),
            OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode()
        ] # 4x4x4
    
    # depth3
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            bn2.child = [
                OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode(),
                OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode()
            ] # 8x8x8
            
    # depth4
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            for d3 in range(8):
                bn3 = bn2.child[d3]
                bn3.child = [
                    OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(),
                    OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB()    
                ] # 16x16x16
                
    anchor_voxel_x = vunit_x * base_volume_unit_dim
    anchor_voxel_y = vunit_y * base_volume_unit_dim
    anchor_voxel_z = vunit_z * base_volume_unit_dim
    
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
                    
                    offset_voxel_x = (x1 * 8) + (x2 * 4) + (x3 * 2) + x4
                    offset_voxel_y = (y1 * 8) + (y2 * 4) + (y3 * 2) + y4
                    offset_voxel_z = (z1 * 8) + (z2 * 4) + (z3 * 2) + z4
                    
                    index = [
                        anchor_voxel_x + offset_voxel_x,
                        anchor_voxel_y + offset_voxel_y,
                        anchor_voxel_z + offset_voxel_z
                    ]
                    
                    d = vunit.D[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                    w = vunit.W[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                    r = vunit.R[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                    g = vunit.G[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                    b = vunit.B[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                    
                    ln4.setValues(index, d, w, r, g, b)
        
    return octree

###    
def convertVolumeUnit8ToOctree(vunit, vunit_x, vunit_y, vunit_z):
    octree = OctBranchNode()
    
    # depth1
    octree.child = [
        OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode(),
        OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode()
    ] # 2x2x2
    
    # depth2
    for d1 in range(8):
        bn1 = octree.child[d1]
        bn1.child = [
            OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode(),
            OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode()
        ] # 4x4x4
    
    # depth3
    for d1 in range(8):
        bn1 = octree.child[d1]
        for d2 in range(8):
            bn2 = bn1.child[d2]
            bn2.child = [
                OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(),
                OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB()   
            ] # 8x8x8
                            
    anchor_voxel_x = vunit_x * base_volume_unit_dim
    anchor_voxel_y = vunit_y * base_volume_unit_dim
    anchor_voxel_z = vunit_z * base_volume_unit_dim
    
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
            
                offset_voxel_x = (x1 * 4) + (x2 * 2) + x3
                offset_voxel_y = (y1 * 4) + (y2 * 2) + y3
                offset_voxel_z = (z1 * 4) + (z2 * 2) + z3
            
                index = [
                    anchor_voxel_x + offset_voxel_x * 2 + 0.5,
                    anchor_voxel_y + offset_voxel_y * 2 + 0.5,
                    anchor_voxel_z + offset_voxel_z * 2 + 0.5
                ]
                
                d = vunit.D[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                w = vunit.W[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                r = vunit.R[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                g = vunit.G[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                b = vunit.B[offset_voxel_x, offset_voxel_y, offset_voxel_z]
                
                ln3.setValues(index, d, w, r, g, b)
                
    return octree

###
def convertVolumeUnit4ToOctree(vunit, vunit_x, vunit_y, vunit_z):
    octree = OctBranchNode()
    
    # depth1
    octree.child = [
        OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode(),
        OctBranchNode(), OctBranchNode(), OctBranchNode(), OctBranchNode()
    ] # 2x2x2
    
    # depth2
    for d1 in range(8):
        bn1 = octree.child[d1]
        bn1.child = [
            OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(),
            OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB(), OctLeafNodeDWRGB()   
        ] # 4x4x4
                            
    anchor_voxel_x = vunit_x * base_volume_unit_dim
    anchor_voxel_y = vunit_y * base_volume_unit_dim
    anchor_voxel_z = vunit_z * base_volume_unit_dim
        
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
    
            offset_voxel_x = (x1 * 2) + x2
            offset_voxel_y = (y1 * 2) + y2
            offset_voxel_z = (z1 * 2) + z2
                            
            index = [
                anchor_voxel_x + offset_voxel_x * 4 + 1.5,
                anchor_voxel_y + offset_voxel_y * 4 + 1.5,
                anchor_voxel_z + offset_voxel_z * 4 + 1.5
            ]
            
            d = vunit.D[offset_voxel_x, offset_voxel_y, offset_voxel_z]
            w = vunit.W[offset_voxel_x, offset_voxel_y, offset_voxel_z]
            r = vunit.R[offset_voxel_x, offset_voxel_y, offset_voxel_z]
            g = vunit.G[offset_voxel_x, offset_voxel_y, offset_voxel_z]
            b = vunit.B[offset_voxel_x, offset_voxel_y, offset_voxel_z]
            
            ln2.setValues(index, d, w, r, g, b)
                
    return octree
        
    
    
    
    
    
    
    
    
    
    
                
                    
    