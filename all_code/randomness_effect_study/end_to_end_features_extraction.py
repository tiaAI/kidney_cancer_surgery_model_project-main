'''
The end_to_end pipline for kidney clinical predict
@author: Kai Wu
@email: wukai1990@hotmail.com
'''
import os
import pickle
import numpy as np
import pickle
import json
import glob
from multiprocessing import Pool
from functools import partial
import SimpleITK as sitk
import pandas as pd
from config import  config

from pyradiomics.pyradiomics_run import pyradiomics_run
from pyradiomics.pyradiomics_output_process import pyradiomics_out_process

from feature_extraction.feature_extract import prepare_dataset
from feature_extraction.load_data_and_resize_to_target_spacing import load_data_and_resize
from feature_extraction.image_analyze import image_analyze
from feature_extraction.mask_to_gtboxes import gt_boxes_generate
from feature_extraction.image_region_crop import tumor_size_stats,crop_image
from feature_extraction.image_normlize import image_normalize
from feature_extraction.images_pixel_transform import  pca_reduce_dimension,svd_reduce_dimension

if __name__ == '__main__':
    cf = config()

    ## step1: kidney and tumor segmentation for raw CT images ##
    ## segmentation model was supported by nnUNet: https://github.com/MIC-DKFZ/nnunet, and trained before
    '''
    cmd = 'nnUNet_predict -i %s -o %s -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_lowres -p nnUNetPlansv2.1 -t Task100_KidneyTumor'%(cf.image_dir, cf.mask_dir)
    os.system(cmd)
    '''
    ## step2: feature extraction by radiomics and dimension reduction ##
    # 1. Pyradiomics

    #cf = pyradiomics_run(cf)
    #radiomics_feature_table = pyradiomics_out_process(cf)
    #radiomics_feature_table.to_csv(os.path.join(cf.radiomics_dir, 'radiomics_features.csv'))
    radiomics_feature_table = pd.read_csv(cf.radiomics_feature_table,header=0,index_col=0)
    # 2. dimension reduction
    if not os.path.exists(cf.resample_dir):
        os.mkdir(cf.resample_dir)
    if not os.path.exists(cf.cropped_dir):
        os.mkdir(cf.cropped_dir)
    if not os.path.exists(cf.normalized_dir):
        os.mkdir(cf.normalized_dir)

    dataset_list = prepare_dataset(cf.image_dir, cf.mask_dir) # prepare dataset
    # 1) load image & mask and resize files into target_spacing

    print('-------load and image & mask into .npz------')
    with Pool(cf.process_num) as pool:
        nifti_info_list = pool.map(partial(load_data_and_resize, cf = cf), dataset_list)  # 并行
        pool.close()
        pool.join() 
    print('-------load data completed------')

    nifti_dict = {}
    for info in nifti_info_list:
        nifti_dict.update(info)
    with open(os.path.join(cf.resample_dir,'original_nifti_info.json'),'w') as OUT:
        json.dump(nifti_dict, OUT)

    # 2) analyze image

    files_dir = glob.glob(os.path.join(cf.resample_dir, '*.npz'))
    #file_list = tumor_size_filter(files_dir[12],cf)

    #voxels_meta_dict = image_analyze(files_dir,cf)
    #with open(os.path.join(cf.normalized_dir, 'voxels_meta_dict.pkl'),'wb') as OUT:
    #    pickle.dump(voxels_meta_dict, OUT)

    # 3) normalize image CT value by mean & SD

    print('-------Normalize tumor image by clip_extremum_value, mean and sd------')
    files_dir = glob.glob(os.path.join(cf.resample_dir, '*.npz'))

    with Pool(cf.process_num) as pool:
        pool.map(partial(image_normalize, cf = cf), files_dir)  # 并行
        pool.close()
        pool.join()
    print('-------Normalization completed------')

    # 4) tumor&renal region crop
    print('-------crop raw image by tumor&renal mask------')
    files_list = glob.glob(os.path.join(cf.normalized_dir, '*.npz'))
    #crop_image(os.path.join(cf.normalized_dir,'C3L-01034_0.npz'), cf = cf)
    for file_dir in files_list:
        crop_image(file_dir, cf = cf)

    with Pool(cf.process_num) as pool:
        pool.map(partial(crop_image, cf = cf), files_list)  # 并行
        pool.close()
        pool.join()

    print('-------crop tumor images completed------')

    # 5) dimension reduction
    files_dir = glob.glob(os.path.join(cf.cropped_dir, '*.npz'))
    files_id = [file_dir.split('/')[-1].replace('.npz','') for file_dir in files_dir]

    with Pool(cf.process_num) as pool:
        features_list = pool.map(partial(pca_reduce_dimension), files_dir) # 并行
        pool.close()
        pool.join()
    rownames = []
    features_table = []
    for features_dict in features_list:
        for key, value in features_dict.items():
            rownames.append(key)
            features_table.append(pd.DataFrame(value))
    PCA_table = pd.concat(features_table, axis=0)
    PCA_table.index = rownames
    PCA_table.columns = ['pca_'+str(i) for i in range(256)]

    with Pool(cf.process_num) as pool:
        features_list = pool.map(partial(svd_reduce_dimension), files_dir) # 并行
        pool.close()
        pool.join()
    rownames = []
    features_table = []
    for features_dict in features_list:
        for key, value in features_dict.items():
            rownames.append(key)
            features_table.append(pd.DataFrame(value))
    SVD_table = pd.concat(features_table, axis=0)
    SVD_table.index = rownames
    SVD_table.columns = ['svd_'+str(i) for i in range(64)]

    reduce_dimension_feature_table = pd.concat((PCA_table, SVD_table), axis=1)
    reduce_dimension_feature_table.to_csv(os.path.join(cf.cropped_dir, 'reduce_dimension_features.csv'))

    # 3. feature fusion
    total_feature_table = pd.concat((radiomics_feature_table, reduce_dimension_feature_table), axis=1)
    total_feature_table.to_csv(os.path.join(cf.base_dir, 'total_features_cort_pred_nephro.csv'))
    '''
    ## step3: tumor histologic subtype and stage predict ##
    # 1) load model 
    model_subtype = pickle.load(open('/home/wukai/Desktop/RCC_classification/xgboost/tumor_subtype_predict_model.pkl','rb'))
    model_stage = pickle.load(open('/home/wukai/Desktop/RCC_classification/xgboost/tumor_stage_predict_model.pkl','rb'))
    
    # 2) load data
    input_features = pd.read_csv(os.path.join(cf.base_dir, 'total_features.csv'),index_col=0, header=0)
    cases_id = input_features.index.to_list()
    input_features = np.array(input_features)
    # 3) predict
    subtype_pred = model_subtype.predict_proba(input_features)
    subtype_pred_proba = subtype_pred[:,1] 
    subtype_pred_class = ['non-ccRCC' if proba < 0.5 else 'ccRCC' for proba in subtype_pred_proba]

    stage_pred = model_stage.predict_proba(input_features)
    stage_pred_proba = stage_pred[:,1] 
    stage_pred_class = ['early' if proba < 0.5 else 'advanced' for proba in stage_pred_proba]


    for i,case_id in enumerate(cases_id):
        print(case_id+'\t'+'tumor_subtype:'+'\t'+str(subtype_pred_class[i])+'\t'+str(subtype_pred_proba[i])+'\t'+
        'tumor stage:'+'\t'+str(stage_pred_class[i])+'\t'+str(stage_pred_proba[i])+'\t')
    '''
