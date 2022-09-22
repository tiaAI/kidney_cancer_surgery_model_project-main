import os
import glob
import numpy as np
import pickle
#import matplotlib.pyplot as plt


def image_normalize(file_dir, cf):
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('process '+file_id)
    assert os.path.exists(cf.voxels_meta_dir), 'Can not find voxels analysis files'
    with open(cf.voxels_meta_dir, 'rb') as voxels_file:
        voxels_meta = pickle.load(voxels_file)
    mean_intensity = voxels_meta['mean']
    std_intensity = voxels_meta['sd']
    lower_bound = voxels_meta['percentile_00_5']
    upper_bound = voxels_meta['percentile_99_5']

    all_data = np.load(file_dir)['data']
    image = all_data[0]
    mask = all_data[1]
    assert len(image.shape) == 3, 'The image shape need to be 3, and H,Y,X'
    for i in range(image.shape[0]):
        image[i] = np.clip(image[i], lower_bound, upper_bound) # np.clip将bound范围外的数值转为上下限数
        image[i] = (image[i] - mean_intensity) / std_intensity # 每个样本foreground部位的均值与标准差进行矫正
    new_data = np.stack((image,mask), axis=0)
    np.savez_compressed(os.path.join(cf.normalized_dir, file_id+'.npz'), data = new_data)


