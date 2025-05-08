from networks.UNet2D import UNet_2d, MBCP
from networks.ResNet2d import ResUNet_2d, ResUnetMAE
import torch.nn as nn 

def BCP_net(model='MUnet', in_chns=1, num_classes= 4, ema= False): 
    if model == 'MUnet': 
        net = MBCP(in_chns, num_classes).cuda()
        if ema: 
            for param in net.parameters(): 
                param.detach_() 
        
    elif model == 'MResUnet': 
        net = ResUnetMAE(in_chns, num_classes).cuda()
        if ema: 
            for param in net.parameters(): 
                param.detach_()

    return net 