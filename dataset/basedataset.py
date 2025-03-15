import numpy as np 
import os 
from scipy.ndimage import zoom, rotate
import h5py

from torch.utils.data import Dataset, Sampler

class BaseDataset(Dataset): 
    def __init__(self, root_path, split= 'train', transform= None, num= None): 
        """
        Use to load name of file.list 

        """
        self.root_path = root_path
        self.split = split 
        self.transform = transform
        self.sample_list = []
        
        # Select dataset ( ACDC or LA )
        if self.root_path == 'ACDC': 
            train_list = os.path.join(self.root_path, 'train_slices.list')
            val_list = os.path.join(self.root_path, 'val.list')
            self.train_files = os.path.join(self.root_path, 'data/slices')
            self.valid_files = os.path.join(self.root_path, 'data')
        elif self.root_path == 'LA': 
            train_list = os.path.join(self.root_path, 'train.list')
        
        modes = ['train', 'val'] if self.root_path == 'ACDC' else ['train']
        if self.split in modes: 
            # Create sample_list 
            if self.split == 'train': 
                with open(train_list, 'r') as file: 
                    self.sample_list = file.readlines() 
                
                self.sample_list = [item.replace('\n', '') for item in self.sample_list]
            elif self.split == 'val': 
                with open(val_list, 'r') as file: 
                    self.sample_list = file.readlines() 
                
                self.sample_list = [item.replace('\n','') for item in self.sample_list]
        else: 
            raise ValueError(f'Mode: {self.split} is not supported')
        
        # Use number of dataset only 
        if isinstance(num, int): 
            self.sample_list = self.sample_list[:num]
        
        # print(f'Total slices: {len(self.sample_list)}')
        

    def __len__(self): 
        return len(self.sample_list)

    def __getitem__(self, index):
        case = self.sample_list[index]

        if self.root_path == 'ACDC':
            if self.split == 'train':  
                file_path = os.path.join(self.train_files, f'{case}.h5')
            else: 
                file_path = os.path.join(self.valid_files, f'{case}.h5')
        elif self.root_path == 'LA': 
            file_path = os.path.join(self.train_files, f'{case}/mri_norm2.h5')

        h5f = h5py.File(file_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        sample = {'image': image, 'label': label}
        if self.split == 'train' and self.transform is not None: 
            sample = self.transform(sample)
        
        sample['case'] = case 
        return sample 