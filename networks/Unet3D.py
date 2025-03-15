from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid
import torch
import pdb

class UNet(Module):
    def __init__(self, in_dimension=1, out_dimension=2, ft_channels=[64, 256, 256, 512, 1024], residual='conv'):
        super(UNet, self).__init__()
        
        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))
        
        # Encoder convolutions
        self.conv_block1 = Conv3D_Block(in_dimension, ft_channels[0], residual=residual)
        self.conv_block2 = Conv3D_Block(ft_channels[0], ft_channels[1], residual=residual)
        self.conv_block3 = Conv3D_Block(ft_channels[1], ft_channels[2], residual=residual)
        self.conv_block4 = Conv3D_Block(ft_channels[2], ft_channels[3], residual=residual)
        self.conv_block5 = Conv3D_Block(ft_channels[3], ft_channels[4], residual=residual)
        
        # Decoderr convolutions
        self.decoder_conv_block4 = Conv3D_Block(2 * ft_channels[3], ft_channels[3], residual=residual)
        self.decoder_conv_block3 = Conv3D_Block(2 * ft_channels[2], ft_channels[2], residual=residual)
        self.decoder_conv_block2 = Conv3D_Block(2 * ft_channels[1], ft_channels[1], residual=residual)
        self.decoder_conv_block1 = Conv3D_Block(2 * ft_channels[0], ft_channels[0], residual=residual)
        
        # Decoder upsamplers
        self.deconv_block4 = Deconv3D_Block(ft_channels[4], ft_channels[3])
        self.deconv_block3 = Deconv3D_Block(ft_channels[3], ft_channels[2])
        self.deconv_block2 = Deconv3D_Block(ft_channels[2], ft_channels[1])
        self.deconv_block1 = Deconv3D_Block(ft_channels[1], ft_channels[0])
        
        # Final 1*1 Convolutions segmentation map
        self.one_conv = Conv3d(ft_channels[0], out_dimension, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Activation function
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        
        # Encoder part
        x1 = self.conv_block1(x)
        x_low1 = self.pool1(x1)
        
        x2 = self.conv_block2(x_low1)
        x_low2 = self.pool2(x2)
        
        x3 = self.conv_block3(x_low2)
        x_low3 = self.pool3(x3)
        
        x4 = self.conv_block4(x_low3)
        x_low4 = self.pool4(x4)
        
        base = self.conv_block5(x_low4)
        
        # Decoder part
        d4 = torch.cat([self.deconv_block4(base), x4], dim=1)
        d_high4 = self.decoder_conv_block4(d4)
        
        d3 = torch.cat([self.deconv_block3(d_high4), x3], dim=1)
        d_high3 = self.decoder_conv_block3(d3)
        d_high3 = Dropout3d(p=0.05)(d_high3)
        
        d2 = torch.cat([self.deconv_block2(d_high3), x2], dim=1)
        d_high2 = self.decoder_conv_block2(d2)
        d_high2 = Dropout3d(p=0.05)(d_high2)
        
        d1 = torch.cat([self.deconv_block1(d_high2), x1], dim=1)
        d_high1 = self.decoder_conv_block1(d1)
        
        seg = self.one_conv(d_high1)
        
        return seg

        
class Conv3D_Block(Module):
    def __init__(self, in_features, out_features, kernel=3, stride=1, padding=1, residual=None):
        super(Conv3D_Block, self).__init__()
        
        self.conv1 = Sequential(
            Conv3d(in_features, out_features, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_features),
            ReLU()
        )
        
        self.conv2 = Sequential(
            Conv3d(out_features, out_features, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_features),
            ReLU()
        )
        
        self.residual = residual
        
        if self.residual is not None:
            self.residual_upsampler = Conv3d(in_features, out_features, kernel_size=1, bias=False)
            
    def forward(self, x):
        
        res = x
        
        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)
        
class Deconv3D_Block(Module):
    
    def __init__(self, in_features, out_features, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()
        
        self.deconv = Sequential(
            ConvTranspose3d(in_features, out_features, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            ReLU()
        )
        
    def forward(self, x):
        return self.deconv(x)