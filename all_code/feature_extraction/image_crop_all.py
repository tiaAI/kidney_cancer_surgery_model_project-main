import os
import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing,binary_dilation
from skimage.measure import label,regionprops

def crop_image(file_dir, cf):
    file_id = file_dir.split('/')[-1].replace('.npz','')
    assert os.path.exists(file_dir), 'Can not find file %s'%(file_id)
    # load image & mask
    all_data = np.load(file_dir)['data']
    image = all_data[0]
    mask = all_data[1]
    assert 2 in mask, 'The label 2 not in mask of %s'%(file_id)
    #计算需要剪裁区域坐标
    crop_boxes, new_mask = compute_crop_boxes_coord(mask, cf)


    # 将image向外扩边，以防止剪裁时超边缘
    new_image = image_padding(image, cf)
    # 由于原始image的H,Y,X轴向两侧均pading,crop_boxes坐标需要重新计算
    crop_boxes = boxes_padding(crop_boxes, cf)
    # 按crop坐标剪裁image
    image_cropped = crop_image_by_boxes(new_image, crop_boxes)
    '''
    # for debug
    for i,img in enumerate(image_cropped):
        plt.imshow(img)
        plt.savefig('%s.png'%(str(i)))
    '''
    np.savez_compressed(os.path.join(cf.cropped_dir, file_id+'.npz'), data = image_cropped)
    print('cropped %s'%file_id)

def compute_crop_boxes_coord(mask, cf):
    # 计算包含肿瘤的肾
    tumor_mask = np.zeros_like(mask)
    tumor_mask[mask == 2] = 1
    kidney_mask = np.zeros_like(mask)
    kidney_mask[mask == 1] = 1

    # tumor取最大的一个
    labels = label(tumor_mask)
    regions = regionprops(labels)
    tumor_index = np.argmax([region.area for region in regions])
    tumor_region = regions[tumor_index]
    # kidney取最大的前两个
    labels = label(kidney_mask)
    regions = regionprops(labels)
    kidney_index = np.argsort([region.area for region in regions])[::-1][:2]
    kidney_regions = [regions[i] for i in kidney_index]

    # 判断哪个肾为肿瘤所在肾
    for kidney_region in kidney_regions:
        intersection = compute_intersection_3D(tumor_region.bbox, kidney_region.bbox)
        if intersection > 0:
            kidney_tumor_loc_region = kidney_region
    try: # 判断取得kidney
        kidney_tumor_loc_region
    except:
        print('Tumor not attached to any kidney!')

    # 合并肿瘤与病灶肾
    new_mask = np.zeros_like(mask)
    for coord in tumor_region.coords:
        new_mask[coord.tolist()[0],coord.tolist()[1],coord.tolist()[2]] = 1
    for coord in kidney_tumor_loc_region.coords:
        new_mask[coord.tolist()[0],coord.tolist()[1],coord.tolist()[2]] = 1
    # 计算target区域中心点坐标
    new_mask = binary_closing(new_mask)
    labels = label(new_mask)
    regions = regionprops(labels)
    assert len(regions) == 1, 'Wrong New mask with tumor and kidney'
    center_point = np.round(np.array(regions[0].centroid)) # 质心坐标

    # 计算需剪切box的坐标
    crop_boxes = [coord - cf.crop_size[i]/2 for i,coord in enumerate(center_point)] + [coord + cf.crop_size[i]/2 for i,coord in enumerate(center_point)]
    return crop_boxes, new_mask

def image_padding(array, cf):
    '''
    return a padded data array with cval
    在图像三个维度的最外边缘填充，值为cval
    '''
    pad_H, pad_Y, pad_X = [int(i/2) for i in cf.crop_size]
    pad_width = ((pad_H, pad_H),(pad_Y, pad_Y),(pad_X, pad_X))
    new_array = np.pad(array, pad_width = pad_width, mode = 'constant', constant_values = np.min(array))
    return new_array

def boxes_padding(box, cf):
    pad_H, pad_Y, pad_X = [int(i/2) for i in cf.crop_size]
    # boxes中每个轴的两个坐标均需要加上pad数值
    box[0], box[3] = box[0] + pad_H, box[3] + pad_H
    box[1], box[4] = box[1] + pad_Y, box[4] + pad_Y
    box[2], box[5] = box[2] + pad_X, box[5] + pad_X
    return box

def crop_image_by_boxes(image, crop_boxes):
    h1,y1,x1,h2,y2,x2 = [int(i) for i in crop_boxes]
    return image[h1:h2,y1:y2,x1:x2]


def compute_intersection_3D(box1, box2):
    # Calculate intersection areas
    z1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x1 = np.maximum(box1[2], box2[2])
    z2 = np.minimum(box1[3], box2[3])
    y2 = np.minimum(box1[4], box2[4])
    x2 = np.minimum(box1[5], box2[5])
    if (z2-z1) < 0:
        Z_distance = 0
    else:
        Z_distance = 1
    if (y2-y1) < 0:
        Y_distance = 0
    else:
        Y_distance = 1
    if (x2-x1) < 0:
        X_distance = 0
    else:
        X_distance = 1
    intersection = Z_distance*Y_distance*X_distance
    return intersection

def tumor_size_stats(gt_box_dir):
    gt_box = pd.read_csv(gt_box_dir)
    tumor_info = gt_box[gt_box['label'] == 'tumor']
    tumor_boxes = np.array([json.loads(coord) for coord in  tumor_info['coord']]).astype('float32')
    H = tumor_boxes[:,3] - tumor_boxes[:,0]
    Y = tumor_boxes[:,4] - tumor_boxes[:,1]
    X = tumor_boxes[:,5] - tumor_boxes[:,2]
    tumor_size = np.stack((H,Y,X),axis=1)
    # 最大值，最小值，中位数，平均值
    tumor_stats_dict = {'H_min':np.min(tumor_size[:,0]),
                        'H_max':np.max(tumor_size[:,0]),
                        'H_mean':np.mean(tumor_size[:,0]),
                        'H_median':np.median(tumor_size[:,0]),
                        'Y_min':np.min(tumor_size[:,1]),
                        'Y_max':np.max(tumor_size[:,1]),
                        'Y_mean':np.mean(tumor_size[:,1]),
                        'Y_median':np.median(tumor_size[:,1]),
                        'X_min':np.min(tumor_size[:,2]),
                        'X_max':np.max(tumor_size[:,2]),
                        'X_mean':np.mean(tumor_size[:,2]),
                        'X_median':np.median(tumor_size[:,2]),
                        }

    print(tumor_stats_dict.items())
    return tumor_stats_dict

if __name__ == '__main__':
    import sys
    sys.path.append('/share/Data01/wukai/rcc_classify')
    from config import config 
    cf = config()

    image_source_dir = '/share/service04/wukai/CT_image/images_normalized'
    cf.cropped_dir = '/share/service04/wukai/CT_image/images_cropped_all'
    if not os.path.exists(cf.cropped_dir):
        os.mkdir(cf.cropped_dir)
    files_dir = glob.glob(os.path.join(image_source_dir, '*.npz'))
    files_id = [file_dir.split('/')[-1].replace('.npz','') for file_dir in files_dir]

    for file_dir in files_dir:
        crop_image(file_dir, cf)

    print('Process complete')
