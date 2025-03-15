import torch
import numpy as np 
from networks.UNet2D import UNet_2d

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'optim': optimizer.state_dict()
    }
    torch.save(state, str(path))

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net']) 

def load_net_opt(net, optimizer, path): 
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optim'])

def BCP_net(in_chns=1, class_num=4, ema=False):
    net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

# CauSSL utils 
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0: 
        return 1.0 
    else:
        current = np.clip(current, 0, rampup_length)
        phase = 1 - (current / rampup_length)
        return float(np.exp(-5 * phase * phase))

def get_current_consistency_weight(args, epoch): 
    return 5 * args.consistency + sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variable(model, ema_model, alpha, global_step): 
    alpha = min(1 - 1/(global_step + 1), alpha) 
    with torch.no_grad(): 
        for ema_param, param in zip(ema_model.parameters(), model.parameters()): 
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

