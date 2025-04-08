from networks.UNet2D import UNet_2d
from networks.ResNet2d import ResUNet_2d
import torch.nn as nn 

def BCP_net(model='unet', in_chns=1, num_classes= 4, ema= False): 
    if model == 'unet': 
        net = UNet_2d(in_chns, num_classes).cuda()
        if ema: 
            for param in net.parameters(): 
                param.detach_() 
        
    elif model == 'ResUnet': 
        net = ResUNet_2d(in_chns, num_classes).cuda()
        if ema: 
            for param in net.parameters(): 
                param.detach_()

    return net 