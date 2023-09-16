import torch
import torch.nn as nn
import torch.nn.functional as F


'''
class TSDFEncoder(nn.Module):
    def __init__(self, code_length):
        super(TSDFEncoder, self).__init__()
        
        self.conv1 = nn.Conv3d(1, code_length, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv3d(code_length, code_length, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv3d(code_length, code_length, kernel_size=3, stride=2, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.lrelu(y)
        y = self.conv2(y)
        y = self.lrelu(y)
        z = self.conv3(y)

        return z
    
class TSDFDecoder(nn.Module):
    def __init__(self, code_length):
        super(TSDFDecoder, self).__init__()
        
        self.conv1 = nn.ConvTranspose3d(code_length, code_length, kernel_size=2, stride=2, bias=True)
        self.conv2 = nn.ConvTranspose3d(code_length, code_length, kernel_size=2, stride=2, bias=True)
        self.conv3 = nn.ConvTranspose3d(code_length, code_length, kernel_size=2, stride=2, bias=True)    
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv_head_sign = nn.Conv3d(code_length, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        self.conv_head_magnitude = nn.Conv3d(code_length, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, zhat):
        y = self.conv1(zhat)
        y = self.lrelu(y)
        y = self.conv2(y)
        y = self.lrelu(y)
        y = self.conv3(y)
        y = self.lrelu(y)

        shat = self.conv_head_sign(y)
        shat = self.sigmoid(shat)
        
        xhat_mag = self.conv_head_magnitude(y)
        #xhat_mag = self.sigmoid(xhat_mag)

        return shat, xhat_mag
'''    

class TSDFEncoder(nn.Module):
    def __init__(self, code_length):
        super(TSDFEncoder, self).__init__()
        
        self.conv1 = nn.Conv3d(1, code_length, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv3d(code_length, code_length, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv3d(code_length, code_length, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv3d(code_length, code_length, kernel_size=3, stride=2, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.lrelu(y)
        y = self.conv2(y)
        y = self.lrelu(y)
        y = self.conv3(y)
        y = self.lrelu(y)
        z = self.conv4(y)

        return z
    
class TSDFDecoder(nn.Module):
    def __init__(self, code_length):
        super(TSDFDecoder, self).__init__()
        
        self.conv1 = nn.ConvTranspose3d(code_length, code_length, kernel_size=2, stride=2, bias=True)
        self.conv2 = nn.ConvTranspose3d(code_length, code_length, kernel_size=2, stride=2, bias=True)
        self.conv3 = nn.ConvTranspose3d(code_length, code_length, kernel_size=2, stride=2, bias=True)    
        self.conv4 = nn.ConvTranspose3d(code_length, code_length, kernel_size=2, stride=2, bias=True)    
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        
        self.conv_head_sign = nn.Conv3d(code_length, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
        self.conv_head_magnitude = nn.Conv3d(code_length, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, zhat):
        y = self.conv1(zhat)
        y = self.lrelu(y)
        y = self.conv2(y)
        y = self.lrelu(y)
        y = self.conv3(y)
        y = self.lrelu(y)
        y = self.conv4(y)
        y = self.lrelu(y)

        shat = self.conv_head_sign(y)
        shat = self.sigmoid(shat)
        
        xhat_mag = self.conv_head_magnitude(y)
        #xhat_mag = self.sigmoid(xhat_mag)

        return shat, xhat_mag
    
    
# I referred to https://github.com/liujiaheng/iclr_17_compression/blob/master/models/bitEstimator.py
class NonParamDensityModel(nn.Module):
    def __init__(self, code_length):
        super(NonParamDensityModel, self).__init__()
        
        self.h1 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        self.h2 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        self.h3 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        self.h4 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        
        self.b1 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        self.b2 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        self.b3 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        self.b4 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        
        self.a1 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        self.a2 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        self.a3 = nn.Parameter(torch.nn.init.normal_(torch.empty(code_length).view(1, -1, 1, 1, 1), 0, 0.01))
        
    def forward(self, zhat):
        zhat_upper = zhat + 0.5
        zhat_lower = zhat - 0.5
        
        # k = 1
        y = zhat_upper * F.softplus(self.h1) + self.b1
        y = y + torch.tanh(y) * torch.tanh(self.a1)
        # k = 2
        y = y * F.softplus(self.h2) + self.b2
        y = y + torch.tanh(y) * torch.tanh(self.a2)
        # k = 3
        y = y * F.softplus(self.h3) + self.b3
        y = y + torch.tanh(y) * torch.tanh(self.a3)
        # k = 4
        cdf_upper = torch.sigmoid(y * F.softplus(self.h4) + self.b4)
    
        # k = 1
        y = zhat_lower * F.softplus(self.h1) + self.b1
        y = y + torch.tanh(y) * torch.tanh(self.a1)
        # k = 2
        y = y * F.softplus(self.h2) + self.b2
        y = y + torch.tanh(y) * torch.tanh(self.a2)
        # k = 3
        y = y * F.softplus(self.h3) + self.b3
        y = y + torch.tanh(y) * torch.tanh(self.a3)
        # k = 4
        cdf_lower = torch.sigmoid(y * F.softplus(self.h4) + self.b4)
        
        prob = cdf_upper - cdf_lower
        
        return prob
        

class TSDFCoder(nn.Module):
    def __init__(self, code_length):
        super(TSDFCoder, self).__init__()
        
        self.code_length = code_length
        self.enc = TSDFEncoder(code_length)
        self.dec = TSDFDecoder(code_length)
        self.density_model = NonParamDensityModel(code_length)
    
    def forward(self, x):
        z = self.enc(x)
        
        if self.training:
            zhat = z + torch.nn.init.uniform_(torch.zeros_like(z), -0.5, 0.5)
        else:
            zhat = torch.round(z)
        
        prob_zhat = self.density_model(zhat)
        shat, xhat_mag = self.dec(zhat)
        
        return prob_zhat, zhat, shat, xhat_mag

        
