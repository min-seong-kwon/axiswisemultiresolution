import cv2
import numpy as np
from abc import ABC, abstractmethod
import os
from glob import glob

class MVDDataSet(ABC):
    
    @abstractmethod
    def getNumberOfFrames(self):
        pass
    
    @abstractmethod
    def getNumberOfViews(self):
        pass
    
    @abstractmethod
    def getRGBImage(self, frame, view):
        pass
    
    @abstractmethod
    def getDepthImage(self, frame, view):
        pass
    
    @abstractmethod
    def getImageSize(self):
        pass
    
    @abstractmethod
    def getIntrinsicMatrix(self, view):
        pass
    
    @abstractmethod
    def getExtrinsicMatrix(self, view):
        pass
    
    
class CubeMVDDataSet(MVDDataSet):
    def __init__(self, 
                 scene_name, 
                 rgb_scale=255.0, 
                 depth_scale=10000.0,
                 original_path='../Dataset/',
                 rig=None,
                 face='1F'):
        self.scene_name = scene_name
        self.rgb_scale = rgb_scale
        self.depth_scale = depth_scale
        self.original_path = original_path
        self.rig = rig
        self.face = face
        
        STARTVIEW = {'1F': 441*0,
                     '2L': 441*1,
                     '3B': 441*2,
                     '4R': 441*3,
                     '5U': 441*4,
                     '6D': 441*5}
        self.startview = STARTVIEW[face]
        
        # camera intrinsic
        self.focal_length_mm  = 12.6037245
        self.sensor_width_mm  = 36.0
        self.sensor_height_mm = 20.25
        self.image_width = 3840
        self.image_height = 2160
        
        # camera matrix
        fx = self.image_width * self.focal_length_mm / self.sensor_width_mm
        fy = self.image_height * self.focal_length_mm / self.sensor_height_mm
        cx = self.image_width / 2.0 
        cy = self.image_height / 2.0 

        self.K = np.array(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float32)
        
    def getNumberOfFrames(self):
        return 1
    
    def getNumberOfViews(self):
        return 21*21
        
    def getImageSize(self):
        return self.image_width, self.image_height
    
    def getRGBImage(self, frame, view):
        rgb_list = glob(os.path.join(self.original_path, self.scene_name, 'col', "*"))
        rgb_list = [rgb for rgb in rgb_list if '%04d'%(view+self.startview) in rgb]
        rgb_path = rgb_list[0]
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / self.rgb_scale
        return rgb
    
    def getDepthImage(self, frame, view):
        dep_list = glob(os.path.join(self.original_path, self.scene_name, 'dep', "*"))
        # print(dep_list[0], dep_list[-1])
        # print(view,self.startview)
        dep_list = [dep for dep in dep_list if '%04d'%(view+self.startview) in dep]
        dep_path = dep_list[0]
        dep = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH) # 16bit depth map
        dep = dep.astype(np.float32)
        dep = dep / self.depth_scale # 1.0 = 1 meter가 되도록 스케일링 
        return dep
    
    def getIntrinsicMatrix(self, view):
        return self.K
    
    def getExtrinsicMatrix(self, view):
        # 이 함수는 기존에 1F, 3B와 2L, 4R의 카메라 배열 방향이 다르다는 점이
        # 반영되지 않았습니다. 따라서 face를 반드시 받고, 해당 face에 따라
        # Extrinsic matrix를 반환하도록 수정하였습니다.
        R = self._getRotationMatrix3x3(0.0, 0.0, 0.0)
        
        if self.face in ['1F', '3B']:
            i = (view - 1) // 21
            j = (view - 1) % 21
        elif self.face in ['2L', '4R']:
            i = (view - 1) // 21
            j = 21 - ((view - 1) % 21)
        elif self.face in ['5U', '6D']:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        Tx = 0.06 * j
        Ty = 0.06 * i
        Tz = 0.0
        T = np.array([[Tx], [Ty], [Tz]], dtype=np.float32)
        
        return R, T
        
    def _getRotationMatrix3x3(self, deg_Rx, deg_Ry, deg_Rz):
        # degree to radian
        rad_Rx = deg_Rx / 180. * np.pi
        rad_Ry = deg_Ry / 180. * np.pi
        rad_Rz = deg_Rz / 180. * np.pi
        
        Rx = np.array([ [1, 0, 0],
                        [0, np.cos(rad_Rx), -np.sin(rad_Rx)],
                        [0, np.sin(rad_Rx), np.cos(rad_Rx)] ], dtype=np.float32)
        
        Ry = np.array([ [np.cos(rad_Ry), 0, np.sin(rad_Ry)],
                        [0, 1, 0],
                        [-np.sin(rad_Ry), 0, np.cos(rad_Ry)] ], dtype=np.float32)
        
        Rz = np.array([ [np.cos(rad_Rz), -np.sin(rad_Rz), 0],
                        [np.sin(rad_Rz), np.cos(rad_Rz), 0],
                        [0, 0, 1] ], dtype=np.float32)
        
        R = np.dot(np.dot(Rz, Ry), Rx)
        return R
        

'''
import matplotlib.pyplot as plt

dataset = CubeMVDDataSet('Hotel-Far')
dep = dataset.getDepthImage(0, 1)
print(dep.shape)
plt.imshow(dep)
plt.show()
rgb = dataset.getRGBImage(0, 1)
print(rgb.shape)
plt.imshow(rgb)
plt.show()
'''    
