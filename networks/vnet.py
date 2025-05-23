import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 
from torch import nn 
import torch.nn.functional as F 

class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, n_stages, normalization= 'none'): 
        super(ConvBlock, self).__init__() 

        layers = []
        for i in range(n_stages): 
            if i == 0: 
                input_channels = in_channels
            else: 
                input_channels = out_channels
        
            layers.append(nn.Conv3d(input_channels, out_channels, kernel_size=3, padding=1))
            
            # normalizatoin 
            if normalization == 'batchnorm': 
                layers.append(nn.BatchNorm3d(out_channels))
            elif normalization == 'groupnorm': 
                layers.append(nn.GroupNorm(num_groups=16, num_channels= out_channels))
            elif normalization == 'instancenorm': 
                layers.append(nn.InstanceNorm3d(out_channels))
            elif normalization != 'none': 
                raise ValueError(f'Unknown normalization: {normalization}')

            # Activation
            layers.append(nn.ReLU(inplace= True))

        self.conv = nn.Sequential(*layers)


    def forward(self, x): 
        x = self.conv(x) 
        return x  
    

class ResidualBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, n_stages, normalization= 'none'): 
        super(ResidualBlock, self).__init__() 

        layers = []
        for i in range(n_stages): 
            if i == 0: 
                input_channels = in_channels
            else: 
                input_channels = out_channels
        
            layers.append(nn.Conv3d(input_channels, out_channels, kernel_size=3, padding=1))
            
            # normalizatoin 
            if normalization == 'batchnorm': 
                layers.append(nn.BatchNorm3d(out_channels))
            elif normalization == 'groupnorm': 
                layers.append(nn.GroupNorm(num_groups=16, num_channels= out_channels))
            elif normalization == 'instancenorm': 
                layers.append(nn.InstanceNorm3d(out_channels))
            elif normalization != 'none': 
                raise ValueError(f'Unknown normalization: {normalization}')

            # Activation 
            if i != n_stages - 1: 
                layers.append(nn.ReLU(inplace= True))

        self.conv = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace= True)


    def forward(self, x): 
        x = self.conv(x) + x 
        x = self.relu(x)  
        return x  
    
class DownsamplingConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, normalization='none'): 
        super(DownsamplingConvBlock, self).__init__() 

        layers = [] 
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=2, stride= 2, padding= 0))
        if normalization == 'batchnorm': 
            layers.append(nn.BatchNorm3d(out_channels))
        elif normalization == 'groupnorm': 
            layers.append(nn.GroupNorm(num_groups=16, num_channels= out_channels))
        elif normalization == 'instancenorm': 
            layers.append(nn.InstanceNorm3d(out_channels))
        elif normalization != 'none': 
            raise ValueError(f'Unknown normalization: {normalization}')

        layers.append(nn.ReLU(inplace= True)) 

        self.down = nn.Sequential(*layers)     

    def forward(self, x): 
        x = self.down(x) 
        return x  
    
class UpsamplingDeconvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, normalization='none'): 
        super(UpsamplingDeconvBlock, self).__init__() 

        layers = [] 
        layers.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride= 2, padding= 0))
        if normalization == 'batchnorm': 
            layers.append(nn.BatchNorm3d(out_channels))
        elif normalization == 'groupnorm': 
            layers.append(nn.GroupNorm(num_groups=16, num_channels= out_channels))
        elif normalization == 'instancenorm': 
            layers.append(nn.InstanceNorm3d(out_channels))
        elif normalization != 'none': 
            raise ValueError(f'Unknown normalization: {normalization}')

        layers.append(nn.ReLU(inplace= True)) 
        self.updeconv = nn.Sequential(*layers)     

    def forward(self, x): 
        x = self.updeconv(x) 
        return x  
    
class Upsampling(nn.Module): 
    def __init__(self, in_channels, out_channels, normalization='none'): 
        super(Upsampling, self).__init__() 

        layers = [] 
        layers.append(nn.Upsample(scale_factor= 2, mode= 'trilinear', align_corners= False))
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size= 3, padding=1))

        if normalization == 'batchnorm': 
            layers.append(nn.BatchNorm3d(out_channels))
        elif normalization == 'groupnorm': 
            layers.append(nn.GroupNorm(num_groups=16, num_channels= out_channels))
        elif normalization == 'instancenorm': 
            layers.append(nn.InstanceNorm3d(out_channels))
        elif normalization != 'none': 
            raise ValueError(f'Unknown normalization: {normalization}')
        
        layers.append(nn.ReLU(inplace=True)) 
        self.up =  nn.Sequential(*layers)

    def forward(self, x):
        x = self.up(x) 
        return x  
    

class VNet(nn.Module): 
    def __init__(self, in_channels=3, out_channels= 16, n_class=2 , normalization= 'none', has_dropout= False): 
        super(VNet, self).__init__() 
        self.has_dropout = has_dropout

        # Enconder 
        self.block_one = ConvBlock(in_channels, out_channels, n_stages=1, normalization= normalization)
        self.block_one_dw = DownsamplingConvBlock(out_channels, out_channels*2, normalization= normalization) 

        self.block_two = ConvBlock(out_channels*2, out_channels*2, n_stages= 2, normalization= normalization)
        self.block_two_dw = DownsamplingConvBlock(out_channels*2, out_channels*4, normalization=normalization)

        self.block_three = ConvBlock(out_channels*4, out_channels * 4, n_stages= 3, normalization= normalization)
        self.block_three_dw = DownsamplingConvBlock(out_channels*4, out_channels*8, normalization= normalization)

        self.block_four = ConvBlock(out_channels * 8, out_channels*8, n_stages= 3, normalization= normalization) 
        self.block_four_dw = DownsamplingConvBlock(out_channels*8, out_channels*16, normalization= normalization)

        # Middle ? 
        self.block_five = ConvBlock(out_channels * 16, out_channels*16, n_stages= 3, normalization= normalization)
        self.block_five_up = UpsamplingDeconvBlock(out_channels * 16, out_channels*8, normalization= normalization)

        # Decoder 
        self.block_six = ConvBlock(out_channels *8, out_channels*8, n_stages=3, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(out_channels*8, out_channels*4, normalization= normalization)

        self.block_seven = ConvBlock(out_channels*4, out_channels*4, n_stages=3, normalization= normalization)
        self.block_seven_up = UpsamplingDeconvBlock(out_channels*4, out_channels*2,normalization= normalization )

        self.block_eight = ConvBlock(out_channels*2, out_channels*2, n_stages= 2, normalization= normalization)
        self.block_eight_up = Upsampling(out_channels *2, out_channels, normalization= normalization)

        self.block_nine = ConvBlock(out_channels, out_channels, n_stages=1, normalization= normalization)
        
        self.out_conv = nn.Conv3d(out_channels, n_class, kernel_size=1, padding=0) 

        self.dropout = nn.Dropout3d(p= 0.5, inplace= True)

        self.__init_weight__() 

    def encoder(self, input): 
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2) 

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3) 

        x4 = self.block_four(x3_dw) 
        x4_dw = self.block_four_dw(x4) 

        x5 = self.block_five(x4_dw)

        if self.has_dropout: 
            x5 = self.dropout(x5) 
        
        res = [x1, x2, x3, x4, x5]

        return res  

    def decoder(self, features): 
        x1 = features[0] 
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4] 

        x5_up = self.block_five_up(x5) 
        x5_up = x5_up + x4 

        x6 = self.block_six(x5_up) 
        x6_up = self.block_six_up(x6) 
        x6_up = x6_up + x3 

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7) 
        x7_up = x7_up + x2 

        x8 = self.block_eight(x7_up) 
        x8_up = self.block_eight_up(x8) 
        x8_up = x8_up + x1 

        x9 = self.block_nine(x8_up) 

        if self.has_dropout: 
            x9 = self.dropout(x9) 
        
        out = self.out_conv(x9) 
        return out 

    def forward(self, x, turnoff_dropout): 
        if turnoff_dropout: 
            has_dropout = self.has_dropout 
            self.has_dropout = False 
        
        features = self.encoder(x) 
        out = self.decoder(features) 
        if turnoff_dropout: 
            self.has_dropout = has_dropout
        
        return out 

    def __init_weight__(self): 
        for m in self.modules(): 
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d): 
                torch.nn.init.kaiming_normal_(m.weight) 
            elif isinstance(m, nn.BatchNorm3d): 
                m.weight.data.fill_(1) 
                m.bias.data.zero_()