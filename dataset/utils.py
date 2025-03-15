import numpy as np 
from scipy.ndimage import zoom, rotate
import itertools

import torch 
from torch.utils.data import Sampler

def random_rot_flip(image, label): 
    """
    Random rotate and Random flip 
    """
    
    # Random rotate
    k = np.random.randint(0, 4) 
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    # Random flip 
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis).copy() 
    label = np.flip(label, axis).copy() 

    return image, label 

def random_rotate(image, label):
    angle = np.random.randint(-20, 20) 
    image = rotate(image, angle, order= 0, reshape= False)
    label = rotate(label,angle, order=0, reshape= False )
    return image, label

class RandomGenerator: 
    def __init__(self, output_size): 
        self.output_size = output_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if np.random.random() > 0.5: 
            image, label = random_rot_flip(image, label)
        
        if np.random.random() > 0.5: 
            image, label = random_rotate(image, label) 
        
        # Zoom image to -> [256,256]
        x,y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order= 0)
        label = zoom(label, (self.output_size[0] /x , self.output_size[1] / y), order= 0)

        # Convert to pytorch 
        imageTensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0) # image.shape = (1, H, W)
        labelTensor = torch.from_numpy(label.astype(np.uint8)) # label.shape = (H, W)
        sample = {'image': imageTensor, 'label': labelTensor}
        
        return sample
    
def iterate_once(indices): 
    """
    Permutate the iterable once 
    (permutate the labeled_idxs once)
    """
    return np.random.permutation(indices) 

def iterate_externally(indices): 
    """
    Create an infinite iterator that repeatedly permutes the indices.
    ( permutate the unlabeled_idxs to make different)
    """
    def infinite_shuffles(): 
        while True: 
            yield np.random.permutation(indices)
            
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n): 
    args = [iter(iterable)] * n 
    return zip(*args)

class TwoStreamBatchSampler(Sampler): 
    def __init__(self, primary_indicies, secondary_indicies, batchsize, secondary_batchsize): 
        self.primary_indicies = primary_indicies
        self.secondary_indicies = secondary_indicies
        self.primary_batchsize = batchsize - secondary_batchsize
        self.secondary_batchsize = secondary_batchsize

        assert len(self.primary_indicies) >= self.primary_batchsize > 0 
        assert len(self.secondary_indicies) >= self.secondary_batchsize > 0 

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indicies)
        secondary_iter = iterate_externally(self.secondary_indicies)

        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) 
            in zip(grouper(primary_iter, self.primary_batchsize),
                   grouper(secondary_iter, self.secondary_batchsize))
        )

    def __len__(self): 
        return len(self.primary_indicies) // self.primary_batchsize
    

# Split the data 
def patients_to_slices(dataset, patients_num): 
    ref_dict = {} 
    if "ACDC" in dataset: 
        ref_dict = {'1': 32, '3': 68, '7': 136, '14': 256, '21': 396, '28': 512, '35': 664, '70': 1312}
    else:
        print('Error')
    
    return ref_dict[str(patients_num)]