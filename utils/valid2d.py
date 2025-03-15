import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 
from torch.utils.data import DataLoader
from medpy import metric 
from scipy.ndimage import zoom 

#TODO: should I have a threshold 
def calculalte_metric_percase(pred, gt): 
    pred[pred > 0] = 1 
    gt[gt > 0] = 1 

    if pred.sum() > 0: 
        dice = metric.binary.dc(pred, gt) 
        hd95 = metric.binary.hd95(pred, gt) 
        return dice, hd95
    else: 
        return 0, 0 

def test_single_volume(image, label, model, classes, patch_size = [256,256]):
    """
    Use to validate ACDC dataset 
    1. Valid for 2D image. Shape = (1, H, W)
    2. Valid metric = [dice, hd95]
    Params: 
        - image (torch.Tensor): valid image. Shape = (1, num_slices, H, W) 
        - label (torch.Tensor): valid label. Shape = (1, num_slices, H, W) 

    """ 
    image = image.squeeze(0).cpu().detach().numpy() 
    label = label.squeeze(0).cpu().detach().numpy() # label.shape = (n_slices, H, W), label.range = range(0, 4)

    prediction = np.zeros_like(label) # shape = (n_slices, H, W) 
    for ind in range(image.shape[0]): 
        slice = image[ind, :, :] # shape = (image.H, image.W) 

        # zoom 
        x, y = slice.shape[0], slice.shape[1] 
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order= 0)
        
        # Evaluate
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda() # (1, 1, 256, 256) 
        model.eval() 
        with torch.no_grad(): 
            output = model(input) # output.shape = (1, 3, 256, 256) with n_classes = 3 - logits 
            if len(output) > 1: 
                output = output[0] 
            
            out = torch.argmax(torch.softmax(output, dim= 1), dim= 1).squeeze(0)  # out.shape = (256, 256), probabilites
            out = out.cpu().detach().numpy() 
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order= 0) 
            prediction[ind] = pred 
    
    metric_list = [] 
    for i in range(1, classes): 
        metric_list.append(calculalte_metric_percase(prediction == i, label == i))
    
    return metric_list