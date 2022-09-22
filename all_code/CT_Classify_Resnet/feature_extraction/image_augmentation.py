import os
import json
import cv2 as cv
import glob
import numpy as np
import shutil
from scipy.ndimage import rotate
from matplotlib import pyplot as plt
from skimage.filters import gaussian

def image_augment(file_dir, cf):
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('process '+file_id)
    assert os.path.exists(file_dir), 'Can not find file %s'%(file_id)
    shutil.copyfile(file_dir, os.path.join(cf.enhanced_dir, file_id+'.npz'))

    # flip Z axis
    files_dir = glob.glob(os.path.join(cf.enhanced_dir, file_id+'*.npz'))
    if cf.flip_Z:
        for file_dir in files_dir:
            file_name = file_dir.split('/')[-1].replace('.npz','')
            image = np.load(file_dir)['data']
            new_image = flip_z_axis(image)
            np.savez_compressed(os.path.join(cf.enhanced_dir, file_name+'_flipZ.npz'), data = new_image)
    
    # horizontal flip for each single image
    files_dir = glob.glob(os.path.join(cf.enhanced_dir, file_id+'*.npz'))
    if cf.flip_H:
        for file_dir in files_dir:
            file_name = file_dir.split('/')[-1].replace('.npz','')
            image = np.load(file_dir)['data']
            new_image = flip_horizontal(image)
            np.savez_compressed(os.path.join(cf.enhanced_dir, file_name+'_flipH.npz'), data = new_image)
    
    # rotate for each single image
    files_dir = glob.glob(os.path.join(cf.enhanced_dir, file_id+'*.npz'))
    if cf.rotate_list:
        for file_dir in files_dir:
            file_name = file_dir.split('/')[-1].replace('.npz','')
            image = np.load(file_dir)['data']
            for angle in cf.rotate_list:
                new_image = rotate_image(image, angle)
                np.savez_compressed(os.path.join(cf.enhanced_dir, file_name+'_rotate%s.npz'%(angle)), data = new_image)
    
    # gaussian blur
    files_dir = glob.glob(os.path.join(cf.enhanced_dir, file_id+'*.npz'))
    if cf.gaussian_sigma_list:
        for file_dir in files_dir:
            file_name = file_dir.split('/')[-1].replace('.npz','')
            image = np.load(file_dir)['data']
            for sigma in cf.gaussian_sigma_list:
                new_image = gaussian(image, sigma = sigma)
                np.savez_compressed(os.path.join(cf.enhanced_dir, file_name+'_gaussBlur%s.npz'%(sigma)), data = new_image)
                '''
                # for debug
                for i,img in enumerate(image):
                    plt.imshow(img)
                    plt.savefig('ori_%s.png'%(i))
                for i,img in enumerate(new_image):
                    plt.imshow(img)
                    plt.savefig('gauss_%s.png'%(i))
                '''

def flip_z_axis(image): # 按H (Z轴)图层顺序调换
    new_image = []
    for img in image:
        new_image.append(img)
    new_image = new_image[::-1]
    new_image = np.stack(new_image, axis=0)
    return new_image

def flip_horizontal(image): # 单层图片水平翻转
    new_image = []
    for img in image: 
        new_img = cv.flip(img,1,dst=None)
        new_image.append(new_img)
    new_image = np.stack(new_image, axis=0)
    return new_image

def flip_vertical(image): # 单层图片垂直翻转
    new_image = []
    for img in image: 
        new_img = cv.flip(img,0,dst=None)
        new_image.append(new_img)
    new_image = np.stack(new_image, axis=0)
    return new_image

def rotate_image(image, angle):# 单层图片按照角度旋转
    new_image = []
    cval = np.min(image)
    for img in image: 
        new_img = rotate(img, angle = angle, axes = (1,0), reshape= False, cval = cval)
        new_image.append(new_img)
    new_image = np.stack(new_image, axis=0)
    # for debug
    '''
    for i,img in enumerate(new_image):
        plt.imshow(img)
        plt.savefig('%s.png'%(str(i)))
    '''
    return new_image



if __name__ == '__main__':
    pass