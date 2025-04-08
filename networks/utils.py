import torch
import numpy as np 

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


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)
