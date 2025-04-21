import numpy as np 
import torch 

def generate_mask(img): 
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda() 
    mask = torch.ones(img_x, img_y).cuda() 

    patch_x, patch_y = int(img_x * 2/3), int(img_y * 2/3) 
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w: w + patch_x, h : h + patch_y] = 0 
    loss_mask[:, w: w+patch_x, h : h + patch_y] = 0 
    return mask.long(), loss_mask.long() 


def contact_mask(img): 
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3] 
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda() 
    mask = torch.ones(img_x, img_y).cuda() 
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h : h + patch_y, :] = 0 
    loss_mask[:, h : h + patch_y, :] = 0 
    return mask.long(), loss_mask.long

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()
    