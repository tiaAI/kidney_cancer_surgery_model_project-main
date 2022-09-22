# -*- coding:utf-8 -*-
# Author: Kai Wu
# Email: wukai1990@hotmail.com
# Date: 2021.10.09

from math import nan
import os
import torch 
from torch.utils.data import Dataset
import argparse
import numpy as np
import pandas as pd
from glob import glob
#import torchio as tio

class TrainLoader(Dataset):
    def __init__(self, args):
        self.args = args
        self.images_dir = os.path.join(args.data_path, 'cropped')
        self.images_list = glob(os.path.join(self.images_dir, '*.npz'))
        self.images_list = [image_dir for image_dir in self.images_list\
                             if os.path.basename(image_dir).split('_')[0].replace('.npz','') in args.training_cases]
        assert len(self.images_list) > 0, 'can not find any images'
        # target file
        target_table = pd.read_csv(args.target_file, index_col=0)
        self.target_dict = {}
        for case_id in args.training_cases:
            if target_table[target_table['case_id'] == case_id][args.task].item() == 1:
                type = 1
            else:
                type = 0
            self.target_dict[case_id] = type
        print('Done')
        
    def __getitem__(self, index):
        image_dir = self.images_list[index]
        assert os.path.exists(image_dir), 'Can not find file %s'%image_dir
        case_id = os.path.basename(image_dir).split('_')[0].replace('.npz','')
        target = self.target_dict.get(case_id)
        assert target is not None, 'Can not find the target value'
        #1.load image
        image = np.expand_dims(np.load(image_dir)['data'],0)
        #2.augmentation
        #random_multitrans = RandomTransform(self.args)
        #image = random_multitrans(image)
        return image, target

    def __len__(self):
        return len(self.images_list)

class TestLoader(Dataset):
    def __init__(self, args):
        self.args = args
        self.images_dir = os.path.join(args.data_path, 'cropped')
        self.images_list = glob(os.path.join(self.images_dir, '*.npz'))
        self.images_list = [image_dir for image_dir in self.images_list\
                             if os.path.basename(image_dir).split('_')[0].replace('.npz','') in args.testing_cases]
        assert len(self.images_list) > 0, 'can not find any images'
        # target file
        target_table = pd.read_csv(args.target_file, index_col=0)
        self.target_dict = {}

        for case_id in args.testing_cases:
            if target_table[target_table['case_id'] == case_id][args.task].item() == 1:
                type = 1
            else:
                type = 0
            self.target_dict[case_id] = type

    def __getitem__(self, index):
        image_dir = self.images_list[index]
        assert os.path.exists(image_dir), 'Can not find file %s'%image_dir
        case_id = os.path.basename(image_dir).split('_')[0].replace('.npz','')
        target = self.target_dict.get(case_id)
        assert target is not None, 'Can not find the target value'
        #1.load image
        image = np.expand_dims(np.load(image_dir)['data'],0)
        return image, target

    def __len__(self):
        return len(self.images_list)


class RandomTransform():
    def __init__(self,args):
        pass

    def __call__(self,subject):
        spacial_transform_list = self.transform_spacial()
        intensity_transform_list = self.transform_intensity()
        transform =  tio.transforms.Compose(spacial_transform_list + intensity_transform_list)
        return  transform(subject)

    @staticmethod
    def transform_intensity():
        transform_list = [
            # intensity transform
            tio.transforms.RandomGhosting(num_ghosts=int(np.random.choice(range(3,10))), p = 0.1),
            tio.transforms.RandomSpike(num_spikes = int(np.random.choice(range(1,3))), p = 0.1),
            tio.transforms.RandomBiasField(coefficients=np.random.uniform(0, 0.1), p = 0.1),
            tio.transforms.RandomSwap(patch_size = tuple(np.random.choice(range(3,10),3)), num_iterations = int(np.random.choice(range(1,5))),  p = 0.1),
            tio.transforms.RandomGamma(log_gamma = (-np.random.uniform(0, 0.1), np.random.uniform(0, 0.2)), p = 0.1),
            tio.transforms.RandomBlur(std = np.random.uniform(0, 0.1), p = 0.1),
            tio.transforms.RandomNoise(std = np.random.uniform(0, 0.1), p = 0.1)
            ]
        idx1, idx2, idx3, idx4= np.random.choice(range(len(transform_list)),4, replace = False)
        return [transform_list[i] for i in [idx1, idx2, idx3, idx4]]

    @staticmethod
    def transform_spacial():
        transform_list = [tio.transforms.RandomFlip(axes = 0, flip_probability = 0.1),
                        tio.transforms.RandomFlip(axes = 1, flip_probability = 0.1),
                        tio.transforms.RandomFlip(axes = 2, flip_probability = 0.1),
                        tio.transforms.RandomElasticDeformation(num_control_points=5, locked_borders=2, p = 0.1)]
        idx1, idx2 = list(np.random.choice(range(len(transform_list)),2, replace = False))

        return [transform_list[i] for i in [idx1, idx2]]

def CT_numpy_image_save(image):
    import matplotlib.pyplot as plt
    for i,img in enumerate(image):
        plt.imshow(img)
        plt.savefig('%d.png'%i)

if __name__ == '__main__':
    # for debug
    pass