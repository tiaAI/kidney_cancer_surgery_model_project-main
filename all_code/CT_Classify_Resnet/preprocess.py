'''
@author: Kai Wu
@email: wukai1990@hotmail.com
'''
import os
import pickle
import numpy as np
import json
import argparse
import pickle
import glob
from multiprocessing import Pool
from functools import partial
import pandas as pd
from feature_extraction.load_data_and_resize_to_target_spacing import load_data_and_resize
from feature_extraction.image_analyze import image_analyze
from feature_extraction.mask_to_gtboxes import gt_boxes_generate
from feature_extraction.image_region_crop import tumor_size_stats,crop_image,crop_tumor_image
from feature_extraction.image_normlize import image_normalize


def parse_arguments():
    parser = argparse.ArgumentParser()

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data_path", type=str, default="./dataset",
                        help="path to dataset repository")
    parser.add_argument("--process_num", type=int, default=16,
                        help="threads of process")
    parser.add_argument("--target_spacing", type=list, default=[2.5,0.85,0.85],
                        help="threads of process")
    parser.add_argument("--crop_size", type=list, default=[60,120,120],
                        help="threads of process")                        
    parser.add_argument("--crop_tumor_size", type=list, default=[24,48,48],
                        help="threads of process")                        

    return parser.parse_args()

def generate_files_dir(args):
    images_dir = glob.glob(os.path.join(args.data_path, 'images','*.nii.gz'))
    images_id = [dir.split('/')[-1].replace('_0000.nii.gz','') for dir in images_dir]
    image_label_dir_dict={}
    for image_id in images_id:
        image_dir = os.path.join(args.data_path, 'images','%s_0000.nii.gz'%image_id)
        label_dir = os.path.join(args.data_path, 'labels','%s.nii.gz'%image_id)
        assert os.path.isfile(image_dir),'Can not find image file'
        assert os.path.isfile(label_dir),'Can not find label file'
        image_label_dir_dict[image_id] = [image_dir, label_dir]
    return image_label_dir_dict
    

if __name__ == '__main__':
    args = parse_arguments()

    ## step1: kidney and tumor segmentation for raw CT images ##
    ## segmentation model was supported by nnUNet: https://github.com/MIC-DKFZ/nnunet, and trained before
    '''
    cmd = 'nnUNet_predict -i %s -o %s -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_lowres -p nnUNetPlansv2.1 -t Task100_KidneyTumor'%(args.image_dir, args.mask_dir)
    os.system(cmd)
    '''
    ## step2: crop kidney + tumor
    args.resample_dir = os.path.join(args.data_path, 'resampled')
    args.cropped_dir = os.path.join(args.data_path, 'cropped')
    args.tumor_cropped_dir = os.path.join(args.data_path, 'tumor_cropped')    
    args.normalized_dir = os.path.join(args.data_path, 'normalized')
    if not os.path.exists(args.resample_dir):
        os.mkdir(args.resample_dir,)
    if not os.path.exists(args.cropped_dir):
        os.mkdir(args.cropped_dir)
    if not os.path.exists(args.tumor_cropped_dir):
        os.mkdir(args.tumor_cropped_dir)
    if not os.path.exists(args.normalized_dir):
        os.mkdir(args.normalized_dir)
    '''
    # 1) load image & mask and resize files into target_spacing
    print('-------load and image & mask into .npz------')
    images_labels_dir = generate_files_dir(args)

    with Pool(args.process_num) as pool:
        nifti_info_list = pool.map(partial(load_data_and_resize, args = args), images_labels_dir.values())  # 并行
        pool.close()
        pool.join() 
    print('-------load data completed------')

    nifti_dict = {}
    for info in nifti_info_list:
        nifti_dict.update(info)
    with open(os.path.join(args.resample_dir,'original_nifti_info.json'),'w') as OUT:
        json.dump(nifti_dict, OUT)
    '''
    '''
    # 2) analyze image
    files_dir = glob.glob(os.path.join(args.resample_dir, '*.npz'))
    #file_list = tumor_size_filter(files_dir[12],args)
    voxels_meta_dict = image_analyze(files_dir,args)
    with open(os.path.join(args.normalized_dir, 'voxels_meta_dict.pkl'),'wb') as OUT:
        pickle.dump(voxels_meta_dict, OUT)
    '''
    # 3) normalize image CT value by mean & SD
    '''
    print('-------Normalize tumor image by clip_extremum_value, mean and sd------')
    files_dir = glob.glob(os.path.join(args.resample_dir, '*.npz'))
    args.voxels_meta_dir = os.path.join(args.normalized_dir, 'voxels_meta_dict.pkl')

    with Pool(args.process_num) as pool:
        pool.map(partial(image_normalize, args = args), files_dir)  # 并行
        pool.close()
        pool.join()
    print('-------Normalization completed------')

    # 4) tumor&renal region crop
    print('-------crop raw image by tumor&renal mask------')
    files_list = glob.glob(os.path.join(args.normalized_dir, '*.npz'))
    #crop_image(os.path.join(args.normalized_dir,'C3L-01034_0.npz'), args = args)
    #for file_dir in files_list:
    #    crop_image(file_dir, args = args)

    with Pool(args.process_num) as pool:
        pool.map(partial(crop_image, args = args), files_list)  # 并行
        pool.close()
        pool.join()
    print('-------crop tumor images completed------')
    '''
    # 5) tumor&renal region crop
    print('-------crop raw image by tumor mask------')
    files_list = glob.glob(os.path.join(args.normalized_dir, '*.npz'))
    #crop_image(os.path.join(args.normalized_dir,'C3L-01034_0.npz'), args = args)
    #for file_dir in files_list:
    #    crop_image(file_dir, args = args)

    with Pool(args.process_num) as pool:
        pool.map(partial(crop_tumor_image, args = args), files_list)  # 并行
        pool.close()
        pool.join()
    print('-------crop tumor images completed------')
    