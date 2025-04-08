import numpy as np 
import torch 

def generate_mask(img): 
    """
    Use to generate the random mask 
    Parameters: 
        - img (torch.Tensor).shape = (batch_size, channels, W, H): the input image
    
    Return
    """
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda() 
    mask = torch.ones(img_x, img_y).cuda() 

    patch_x, patch_y = int(img_x * 2/3), int(img_y * 2/3) 
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w: w + patch_x, h : h + patch_y] = 0 
    loss_mask[:, w + w: patch_x, h : h + patch_y] = 0 
    return mask.long(), loss_mask.long() 
