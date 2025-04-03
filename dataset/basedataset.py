import numpy as np 
import os 
import h5py
from torch.utils.data import Dataset

class ACDCDataset(Dataset): 
    def __init__(self, base_dir, split='train_lab', reverse=None, transform=None): 
        super(ACDCDataset, self).__init__() 
        self.base_dir = base_dir
        self.split = split
        self.reverse = reverse
        self.transform = transform
        self.sample_list = []

        # Read the file 
        if self.split == 'train_lab': 
            with open(os.path.join(self.base_dir, 'train_lab.list'), 'r') as file: 
                self.sample_list = file.readlines() 
        elif self.split == 'train_unlab': 
            with open(os.path.join(self.base_dir, 'train_unlab.list'), 'r') as file: 
                self.sample_list = file.readlines() 
        elif self.split == 'val': 
            with open(os.path.join(self.base_dir, 'val.list'), 'r') as file: 
                self.sample_list = file.readlines() 
        else: 
            raise ValueError(f'Split: {self.split} is not support for ACDC dataset')
        
        self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        print(f'Mode: {self.split}: {len(self.sample_list)} samples in total')


    def __len__(self): 
        if (self.split == 'train_lab') | (self.split == 'train_unlab'): 
            return len(self.sample_list) * 10 # Why use it ???? 
        
        return len(self.sample_list)

    def __getitem__(self, idx): 
        case = self.sample_list[idx%len(self.sample_list)] # Avoid problem of __len__ 
        if self.reverse: 
            case = self.sample_list[len(self.sample_list) - idx%len(self.sample_list) - 1] 

        # read the file 
        if (self.split == 'train_lab') | (self.split == 'train_unlab'): 
            h5f = h5py.File((self.base_dir + f'/data/slices/{case}.h5'), 'r')         
        elif (self.split == 'val'): 
            h5f = h5py.File((self.base_dir + f'/data/{case}.h5'), 'r')
        
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}

        if self.transform: 
            sample = self.transform(sample)
        image_, label_ = sample['image'], sample['label']
        return image_, label_