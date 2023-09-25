import re
import os
import pickle
import numpy as np
import cv2
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class TSDFDataset(Dataset):
    def __init__(self, scenes=['Hotel-Near'],
                dataset_name='soldier',
                volume_unit_dim=32,
                finest_voxel_size=1.5,
                volume_origin = np.array([0,0,0]),
                original_path=None, 
                _rgb_scale=255.0,
                _dep_scale=10000.0,
                rig='r1',
                face='1F',
                ETRI=True):
        self.ETRI = ETRI
        self.scenes = scenes
        self.rig = rig
        self.face = face
        self.dataset_name = dataset_name
        self.volume_unit_dim = volume_unit_dim
        self.finest_voxel_size = finest_voxel_size
        
        STARTVIEW = {'1F': 441*0,
                '2L': 441*1,
                '3B': 441*2,
                '4R': 441*3,
                '5U': 441*4,
                '6D': 441*5}
        self.startview = STARTVIEW[face]
        
        #self.scenes = ['Hotel-Near', 'Hotel-Far', 'Restr-Near', 'Restr-Far']
        # if len(scenes)>1:
        #     print("it is strongly recommended to use only 1 scene per use")
        #     print("the code is not tested under multi-scene dataset condition")
        #     print("it's probably not going to work")
            
        self.block_size = 16 # 16 x 16 x 16
        self.mask_blocks = {}
        self.tsdf_blocks = {}
        self.keys = []
        
        self.volume_origin = volume_origin
        self.original_path = original_path
        self.rgb_scale = _rgb_scale
        self.dep_scale = _dep_scale
        
        self._setCameraMatrix()
        self._loadDataset()
        
    def _loadDataset(self):
        # npzfiles = glob('../vunits/%s/voxsize_%.1f/SDF/soldier_%d/*.npz' % (
        #     self.dataset_name, self.finest_voxel_size, self.volume_unit_dim
        # ))
        npzfiles = glob('../vunits/%s/voxsize_%.6f/%s_%d/*.npz' % (
            self.dataset_name, self.finest_voxel_size, self.dataset_name, self.volume_unit_dim
        ))
        for npz in tqdm(npzfiles, desc=f"loading {self.dataset_name}, dim={self.volume_unit_dim}", leave=True):
            strlist = re.findall(r'\d+', os.path.split(npz)[-1])
            vunit_x, vunit_y, vunit_z = strlist[-3:]
            vunit_x = int(vunit_x)
            vunit_y = int(vunit_y)
            vunit_z = int(vunit_z)
            
            vunit_data = np.load(npz)
            tsdf = vunit_data['D']
            mask = vunit_data['M']
    
            key = (self.dataset_name, vunit_x, vunit_y, vunit_z)
            self.keys.append(key)
            self.tsdf_blocks[key] = tsdf
            self.mask_blocks[key] = mask
    
    def _setCameraMatrix(self):
        if self.ETRI:
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
        else:
            pass
            
    def getDepthMap(self, iframe):
        for scene in self.scenes:
            dep_list = glob(os.path.join(self.original_path, scene, 'dep', "*"))
            # print(dep_list[0], dep_list[-1])
            # print(view,self.startview)
            dep_list = [dep for dep in dep_list if '%04d'%(iframe+self.startview) in dep]
            dep_path = dep_list[0]
            # dep_path = self.original_path + scene + '/dep/dep%04d.png' % iframe
            # dep_path = '../Dataset/' + self.dataset_name + '/dep/Cu_ho_r2_dep_1F_cam%04d.png' % iframe
            dep = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH) # 16bit depth map
            dep = dep.astype(np.float32)
            dep = dep / self.dep_scale # 1.0 = 1 meter가 되도록 스케일링 
        return dep
    
    def getRgbImage(self, iframe):
        for scene in self.scenes:
            rgb_list = glob(os.path.join(self.original_path, scene, 'col', "*"))
            rgb_list = [rgb for rgb in rgb_list if '%04d'%(iframe+self.startview) in rgb]
            rgb_path = rgb_list[0]
            # rgb_path = self.original_path + scene + '/col/col%04d.png' % iframe
            # rgb_path = '../Dataset/' + self.dataset_name + '/col/Cu_ho_r2_col_1F_cam%04d.png' % iframe
            bgr = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / self.rgb_scale
        return rgb
    
    def getCameraPosition(self, iframe):
        # 이 함수는 기존에 1F, 3B와 2L, 4R의 카메라 배열 방향이 다르다는 점이
        # 반영되지 않았습니다. 따라서 face를 반드시 받고, 해당 face에 따라
        # Extrinsic matrix를 반환하도록 수정하였습니다.
        R = self._getRotationMatrix3x3(0.0, 0.0, 0.0)
        
        if self.face in ['1F', '3B']:
            i = (iframe - 1) // 21
            j = (iframe - 1) % 21
        elif self.face in ['2L', '4R']:
            i = (iframe - 1) // 21
            j = 21 - ((iframe - 1) % 21)
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
    
    def getCameraMatrix(self):
        return self.K
    
    def getImageSize(self):
        return self.image_width, self.image_height
                
    def __len__(self):
        return len(self.keys)
            
    def __getitem__(self, idx):
        key = self.keys[idx]
        mask = self.mask_blocks[key]
        tsdf = self.tsdf_blocks[key]
        
        mask = mask.reshape((1, self.block_size, self.block_size, self.block_size))
        tsdf = tsdf.reshape((1, self.block_size, self.block_size, self.block_size))
        
        mask = (mask).astype(np.float32)
        sign_bin = (tsdf >= 0).astype(np.float32) # negative: 0, positive 1
        sign = sign_bin.copy()
        sign[sign==0.] = -1.  # negative: -1, positive 1
        
        return mask, sign, sign_bin, tsdf, key #edited
    

#dataset = TSDFDataset()