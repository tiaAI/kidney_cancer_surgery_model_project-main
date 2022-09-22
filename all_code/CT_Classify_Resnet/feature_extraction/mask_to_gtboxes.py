import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing,binary_dilation
from skimage.measure import label,regionprops

def gt_boxes_generate(case_id,cf):
    #resample_dir = '/share/Data01/wukai/Renal_cancer_classification/images_resampled'
    print('process '+case_id)
    all_data = np.load(os.path.join(cf.resample_dir, case_id+'.npz'))['data']
    mask = all_data[1]
    rois, kidney_num = split_mask(mask)
    #assert kidney_num <= 2, 'The file %s has %d kidney'%(case_id, kidney_num)
    bbox_info = bounding_box_coord(case_id, rois, kidney_num)
    return bbox_info

def split_mask(mask):
    tumor_mask = np.zeros_like(mask)
    kidney_mask = np.zeros_like(mask)
    tumor_mask[mask == 2] = 1
    kidney_mask[mask == 1] = 1
    tumor_mask = split_region(tumor_mask)
    kidney_mask = split_region(kidney_mask)
    kidney_num = len(kidney_mask)
    return np.vstack((kidney_mask, tumor_mask)), kidney_num


def split_region(mask):
    #用于处理多个肿瘤的mask分割
    mask = binary_closing(mask)
    mask = binary_dilation(mask)
    labels = label(mask)
    regions = regionprops(labels)
    split_mask = []
    for region in regions:
        if len(region.coords) > 300: # 去除像素点总个数小于300的孤立标记，过小应该不是真实
            region_mask = np.zeros(mask.shape)
            for coord in region.coords: # 联通区域坐标
                region_mask[coord.tolist()[0],coord.tolist()[1],coord.tolist()[2]] = 1 # 该坐标更改为1
            split_mask.append(region_mask)
    return np.stack(split_mask,axis=0)

def bounding_box_coord(case_id, rois, kidney_num):
    '''
    output
    'bbox' bounding box coordinates: case_id, "[z1, y1, x1, z2, y2, x2]", kidney
    '''
    bbox_info = pd.DataFrame({'index':[],'coord':[],'label':[]})
    bbox_info['index'] = [case_id for i in range(rois.shape[0])]

    coord_list = []
    for i, mask in enumerate(rois):
        assert mask.ndim == 3, 'Need input 3-dim image array!'
        assert np.sum(mask !=0) > 0, 'The label is empty!'
        seg_ixs = np.argwhere(mask != 0)
        coord = [np.min(seg_ixs[:, 0])-1, np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1,
                        np.max(seg_ixs[:, 0])+1, np.max(seg_ixs[:, 1])+1, np.max(seg_ixs[:, 2])+1]
        coord_list.append(coord)
    bbox_info['coord'] =  coord_list
    bbox_info['label'] = ['kidney']*kidney_num + ['tumor' for i in range(rois.shape[0]- kidney_num)] # 用于单肾和双肾样本处理
    print(case_id)
    print(bbox_info['coord'])
    print(bbox_info['label'])
    return bbox_info


if __name__ == '__main__':
    case_id = 'case00030'

    gt_boxes_generate(case_id)
