import json
import os
import pickle
import shutil
import SimpleITK as sitk
import numpy as np
from skimage.transform import resize # 用于图像的缩放

def load_data_and_resize(info, cf):
    '''
    info: a dict containing image_id, image dir, mask dir
    '''
    file_id = info['file_id']
    image_dir = info['image_dir']
    mask_dir = info['mask_dir']
    assert os.path.exists(image_dir), 'Given image directory is wrong'
    assert os.path.exists(mask_dir), 'Given mask directory is wrong'

    print('process '+info['file_id'])
    image_sitk = sitk.ReadImage(image_dir)
    mask_sitk = sitk.ReadImage(mask_dir)

    if 'case' in file_id: #判断是否为kits19样本，仅该数据集数据维度排列顺序不同
        image = sitk.GetArrayFromImage(image_sitk).transpose(2,1,0).astype(np.float32) #默认是float64,转为32位以减少数据大小,维度[H,Y,X]
        mask = sitk.GetArrayFromImage(mask_sitk).transpose(2,1,0).astype(np.float32)
        shape = image_sitk.GetSize()
        spacing = image_sitk.GetSpacing()
    else:
        image = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        assert len(image.shape) == 3, 'wrong shape %s'%(file_id)
        mask = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)
        shape = image_sitk.GetSize()[::-1]
        spacing = image_sitk.GetSpacing()[::-1]

    new_shape =  np.round(((np.array(spacing) / np.array(cf.target_spacing)).astype(float) * shape)).astype(int)

    new_image = image_resize(image, new_shape)
    new_mask = mask_resize(mask,new_shape)
    
    all_data = np.stack((new_image, new_mask), axis=0)
    np.savez_compressed(os.path.join(cf.resample_dir, file_id+'.npz'), data = all_data)
    file_info = prepare_nifti_info(image_sitk, file_id)

    return file_info


def prepare_nifti_info(image_sitk,file_id):
    shape = image_sitk.GetSize()
    spacing = image_sitk.GetSpacing()
    origin = image_sitk.GetOrigin()
    direction = image_sitk.GetDirection()

    nifti_info = {}
    nifti_info['shape'] = shape
    nifti_info['spacing'] = spacing
    nifti_info['origin'] = origin
    nifti_info['direction'] = direction
    file_info = {}
    file_info[file_id] = nifti_info
    return file_info


def image_resize(image, new_shape, order=3, cval=0):
    new_image = resize(image, new_shape,order=order, cval = cval)
    return new_image


def mask_resize(mask, new_shape, order=0, cval=0):
    '''
    根据skimage的resize函数，对segmentation Mask进行resize操作
    不直接用resize函数的原因可能是：直接resize后，Mask label中数值[0,1,2]会发生改变，不能再作为Mask用
    该函数将每个label拆分运算resize,最后将所有resize的label合并起来，成为seg的reszie结果

    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = mask.dtype
    unique_labels = np.unique(mask)
    assert len(mask.shape) == len(new_shape), "new shape must have same dimensionality as mask"
    if order == 0:
        return resize(mask, new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=mask.dtype)
        for i, c in enumerate(unique_labels):
            mask = mask == c
            reshaped_multihot = resize(mask, new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped