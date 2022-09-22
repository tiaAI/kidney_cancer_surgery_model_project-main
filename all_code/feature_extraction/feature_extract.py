import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import glob
import pickle
from multiprocessing import Pool
from functools import partial
from config import config
from feature_extraction.load_data_and_resize_to_target_spacing import load_data_and_resize
from feature_extraction.image_analyze import image_analyze, tumor_size_filter
from feature_extraction.mask_to_gtboxes import gt_boxes_generate
from feature_extraction.image_region_crop import tumor_size_stats,crop_image
from feature_extraction.image_normlize import image_normalize
from feature_extraction.image_augmentation import image_augment

def prepare_dataset(images_dir, masks_dir):
    dataset_list = []
    files_dir = glob.glob(os.path.join(masks_dir, '*nii.gz'))
    files_id = [dir.split('/')[-1].replace('.nii.gz', '') for dir in files_dir]
    for file_id in files_id:
        case_id = file_id.split('_')[0]
        image_dir = os.path.join(images_dir, file_id+'_0000.nii.gz')
        mask_dir = os.path.join(masks_dir, file_id+'.nii.gz')
        assert os.path.exists(image_dir), 'Can not find image file of %s'%(file_id)
        assert os.path.exists(mask_dir), 'Can not find mask file of %s'%(file_id)

        dataset_list.append({'case_id':case_id,
                            'file_id':file_id,
                            'image_dir':image_dir,
                            'mask_dir':mask_dir})
    return dataset_list

def prepare_testset(images_dir, masks_dir):
    dataset_list = []
    files_dir = glob.glob(os.path.join(masks_dir, '*nii.gz'))
    files_id = [dir.split('/')[-1].replace('.nii.gz', '') for dir in files_dir]
    for file_id in files_id:
        case_id = file_id.split('_')[0]
        image_dir = os.path.join(images_dir, file_id+'_0000.nii.gz')
        mask_dir = os.path.join(masks_dir, file_id+'.nii.gz')
        assert os.path.exists(image_dir), 'Can not find image file of %s'%(file_id)
        assert os.path.exists(mask_dir), 'Can not find mask file of %s'%(file_id)

        dataset_list.append({'case_id':case_id,
                            'file_id':file_id,
                            'image_dir':image_dir,
                            'mask_dir':mask_dir})
    return dataset_list


if __name__ == '__main__':
    cf = config()
    if not os.path.exists(cf.testset_resample_dir):
        os.mkdir(cf.testset_resample_dir)
    if not os.path.exists(cf.testset_cropped_dir):
        os.mkdir(cf.testset_cropped_dir)
    if not os.path.exists(cf.testset_normalized_dir):
        os.mkdir(cf.testset_normalized_dir)


    dataset_list = []
    '''
    # for training
    dataset_list += prepare_dataset(cf.image_dir0, cf.mask_dir0)
    dataset_list += prepare_dataset(cf.image_dir1, cf.mask_dir1)
    dataset_list += prepare_dataset(cf.image_dir2, cf.mask_dir2)
    dataset_list += prepare_dataset(cf.image_dir3, cf.mask_dir3)
    dataset_list += prepare_dataset(cf.image_dir4, cf.mask_dir4)
    '''
    # for testing
    dataset_list = prepare_testset(cf.testset_image_dir, cf.testset_mask_dir)
    cf.resample_dir = cf.testset_resample_dir
    cf.normalized_dir = cf.testset_normalized_dir
    cf.cropped_dir = cf.testset_cropped_dir

    #1.load image & mask and resize files into target_spacing

    # load_data_and_resize(dataset_list[0], cf)
    '''
    print('-------load and image & mask into .npz------')
    with Pool(cf.process_num) as pool:
        nifti_info_list = pool.map(partial(load_data_and_resize, cf = cf), dataset_list)  # 并行
        pool.close() #关闭进程池，不再接受新的进程
        pool.join() #主进程阻塞等待子进程的退出
    print('-------load data completed------')
    nifti_dict = {}
    for info in nifti_info_list:
        nifti_dict.update(info)
    with open(os.path.join(cf.testset_resample_dir,'original_nifti_info.pkl'),'wb') as OUT:
        pickle.dump(nifti_dict, OUT)
    '''
    #2. filter the over-size tumor sample and analyze image
    '''
    files_dir = glob.glob(os.path.join(cf.resample_dir, '*.npz'))
    #file_list = tumor_size_filter(files_dir[12],cf)

    with Pool(cf.process_num) as pool:
        files_list = pool.map(partial(tumor_size_filter, cf = cf), files_dir)  # 并行
        pool.close()
        pool.join()
    files_list = list(filter(None, files_list))
    print('%d files left'%(len(files_list)))
    with open(os.path.join(cf.normalized_dir, 'filter_files_dir.pkl'),'wb') as OUT:
        pickle.dump(files_list, OUT)

    voxels_meta_dict = image_analyze(files_dir,cf)
    with open(os.path.join(cf.normalized_dir, 'voxels_meta_dict.pkl'),'wb') as OUT:
        pickle.dump(voxels_meta_dict, OUT)
    '''
    #3. normalize image CT value by mean & SD
    '''
    print('-------Normalize tumor image by clip_extremum_value, mean and sd------')
    files_dir = glob.glob(os.path.join(cf.resample_dir, '*.npz'))
    cf.voxels_meta_dir = os.path.join(cf.normalized_dir, 'voxels_meta_dict.pkl')

    with Pool(cf.process_num) as pool:
        pool.map(partial(image_normalize, cf = cf), files_dir)  # 并行
        pool.close()
        pool.join()
    print('-------Normalization completed------')
    '''
    #4.tumor&renal region crop

    print('-------crop raw image by tumor&renal mask------')
    files_list = glob.glob(os.path.join(cf.normalized_dir, '*.npz'))
    #crop_image(os.path.join(cf.normalized_dir,'C3L-01034_0.npz'), cf = cf)
    for file_dir in files_list:
        crop_image(file_dir, cf = cf)
    '''
    with Pool(cf.process_num) as pool:
        pool.map(partial(crop_image, cf = cf), files_list)  # 并行
        pool.close()
        pool.join()
    '''
    print('-------crop tumor images completed------')

    #5. data augmentation
    '''
    print('-------Augmentation start------')
    files_list = glob.glob(os.path.join(cf.cropped_dir, '*.npz'))
    image_augment(os.path.join(cf.cropped_dir, 'TCGA-B9-4113_0a.npz'), cf = cf)
    #for file_dir in files_list:
    #    image_augment(file_dir, cf = cf)

    with Pool(cf.process_num) as pool:
        pool.map(partial(image_augment, cf = cf), files_list)  # 并行
        pool.close()
        pool.join()

    print('-------Augmentation completed------')
    '''