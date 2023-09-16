import numpy as np

class VolumeUnit:
    def __init__(self, volume_unit_dim=16, depth=None):
        self.volume_unit_dim = volume_unit_dim
        self.complexity = 0.0
        if depth==None:
            self.depth = np.log(volume_unit_dim)/np.log(2)
        else:
            self.depth = depth
        self.D = np.zeros((volume_unit_dim, volume_unit_dim, volume_unit_dim), np.float32) # TSDF
        # self.R = np.zeros((volume_unit_dim, volume_unit_dim, volume_unit_dim), np.float32)
        # self.G = np.zeros((volume_unit_dim, volume_unit_dim, volume_unit_dim), np.float32)
        # self.B = np.zeros((volume_unit_dim, volume_unit_dim, volume_unit_dim), np.float32)
        self.W = np.zeros((volume_unit_dim, volume_unit_dim, volume_unit_dim), np.float32)
        self.MC = np.zeros((volume_unit_dim, volume_unit_dim, volume_unit_dim), np.uint8)
        self.M = np.zeros((volume_unit_dim, volume_unit_dim, volume_unit_dim), bool) # To see if it's a necessary voxel.
        
    def save(self, out_path):
        np.savez(out_path, D=self.D, W=self.W, MC=self.MC, M=self.M, N=self.volume_unit_dim, C=self.complexity)
        #R=self.R, G=self.G, B=self.B,
        
    def load(self, in_path):
        npzfile = np.load(in_path)
        self.D = npzfile['D']
        # self.R = npzfile['R']
        # self.G = npzfile['G']
        # self.B = npzfile['B']
        self.W = npzfile['W']
        self.MC = npzfile['MC']
        self.M = npzfile['M']
        self.volume_unit_dim = npzfile['N']
        self.complexity = npzfile['C']