import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing,binary_dilation
from skimage.measure import label,regionprops

def tumor_size_filter(file_dir,cf):
    '''
    输出肿瘤体积小于阈值的样本地址
    '''
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('process '+ file_id)
    assert os.path.exists(file_dir), 'Can not find file %s'%(file_dir)
    mask = np.load(file_dir)['data'][1]
    tumor_mask = np.zeros_like(mask)
    tumor_mask[mask == 2] = 1
    tumor_mask = binary_closing(tumor_mask)
    labels = label(tumor_mask)
    regions = regionprops(labels)
    for region in regions:
        if len(region.coords) > cf.tumor_size_threshold:
            print('%s tumor over size'%(file_id))
            return None
    return file_dir


def image_analyze(files_dir,cf):
    print('analyze total %d files' %len(files_dir))
    voxels = []
    for file_dir in files_dir:
        all_data = np.load(file_dir)['data']
        image = all_data[0]
        mask = all_data[1]
        image = image[mask > 0]
        voxels += list(image)
    voxels = np.array(voxels) # 重新转为np.array

    image_meta_dict = {'max':np.max(voxels),
                        'min':np.min(voxels),
                        'mean':np.mean(voxels),
                        'median':np.median(voxels),
                        'sd':np.std(voxels),
                        'percentile_99_5' :np.percentile(voxels, 99.5),
                        'percentile_00_5' :np.percentile(voxels, 0.5)
                        }
    print(image_meta_dict)
    return image_meta_dict