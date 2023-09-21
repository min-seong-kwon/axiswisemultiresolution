import re
import os
import pickle
import numpy as np
import cv2
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class MPEGDataset(Dataset):
    def __init__(self,
                dataset_name='soldier',
                axisres=np.array([8,8,8]),
                finest_voxel_size=1.5,
                volume_origin = np.array([0,0,0]),
                original_path=None):
        self.dataset_name = dataset_name
        self.axisres = axisres
        self.finest_voxel_size = finest_voxel_size
            
        self.block_size = 16 
        self.mask_blocks = {}
        self.tsdf_blocks = {}
        self.keys = []
        
        self.volume_origin = volume_origin
        self.original_path = original_path
        self.axisres_str = '_'.join(map(str,self.axisres))
        self._loadDataset()
        
    def _loadDataset(self):
        npzfiles = glob(f'../vunits/{self.dataset_name}/voxsize_{self.finest_voxel_size:.3f}/{self.dataset_name}_{self.axisres_str}/*.npz')

        for npz in npzfiles: #tqdm(npzfiles, desc=f"loading {self.dataset_name}, dim={self.axisres}"):
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