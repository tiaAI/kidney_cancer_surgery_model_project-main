'''
image shape 64*64*64 (H,Y,X)
Using PCA to reduce dimension into 64*32 (H, feature)
reture a dict(.pkl) file
'''

import os
import numpy as np
import pandas as pd
import pickle
import glob
from multiprocessing import Pool
from functools import partial
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE

def pca_reduce_dimension(file_dir):
    assert os.path.exists(file_dir),'Can not find file in %s'%(file_dir)
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('pca process %s'%file_id)
    image = np.load(file_dir)['data']
    image = np.reshape(image, (64,-1))
    # pca
    pca = PCA(4).fit(image)
    image_feature = pca.transform(image)
    image_feature = np.reshape(image_feature, (1,-1))
    return {file_id:image_feature}

def svd_reduce_dimension(file_dir):
    assert os.path.exists(file_dir),'Can not find file in %s'%(file_dir)
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('svd process %s'%file_id)
    image = np.load(file_dir)['data']
    image = np.reshape(image, (64,-1))
    # pca
    U, S, Vh = np.linalg.svd(image)
    image_feature = np.reshape(S, (1,-1))
    return  {file_id:image_feature}

def isomap_reduce_dimension(file_dir):
    assert os.path.exists(file_dir),'Can not find file in %s'%(file_dir)
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('isomap process %s'%file_id)
    image = np.load(file_dir)['data']
    image = np.reshape(image, (64,-1))
    # isomap
    isomap = Isomap(n_components=4)
    image_feature = isomap.fit_transform(image)
    image_feature = np.reshape(image_feature, (1,-1))
    return  {file_id:image_feature}

def ica_reduce_dimension(file_dir):
    assert os.path.exists(file_dir),'Can not find file in %s'%(file_dir)
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('ica process %s'%file_id)
    image = np.load(file_dir)['data']
    image = np.reshape(image, (24,-1))
    # isomap
    ica = FastICA(n_components=3)
    image_feature = ica.fit_transform(image)
    image_feature = np.reshape(image_feature, (1,-1))
    return  {file_id:image_feature}

def lle_reduce_dimension(file_dir):
    assert os.path.exists(file_dir),'Can not find file in %s'%(file_dir)
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('lle process %s'%file_id)
    image = np.load(file_dir)['data']
    image = np.reshape(image, (24,-1))
    # isomap
    lle = LocallyLinearEmbedding(n_components=4)
    image_feature = lle.fit_transform(image)
    image_feature = np.reshape(image_feature, (1,-1))
    return  {file_id:image_feature}

def tsne_reduce_dimension(file_dir):
    assert os.path.exists(file_dir),'Can not find file in %s'%(file_dir)
    file_id = file_dir.split('/')[-1].replace('.npz','')
    print('TSNE process %s'%file_id)
    image = np.load(file_dir)['data']
    image = np.reshape(image, (64,-1))
    # isomap
    tsne = TSNE(n_components=3,verbose=1)
    image_feature = tsne.fit_transform(image)
    image_feature = np.reshape(image_feature, (1,-1))
    return  {file_id:image_feature}

def pca_experiments(file_dir,seed):
    assert os.path.exists(file_dir),'Can not find file in %s'%(file_dir)
    image = np.load(file_dir)['data']
    image = np.reshape(image, (64,-1))
    # pca
    pca = PCA(4, random_state= seed, svd_solver='full').fit(image)
    image_feature = pca.transform(image)
    image_feature = np.reshape(image_feature, (1,-1))
    pc1, pc2,pc3 = image_feature[0][0], image_feature[0][1],image_feature[0][2]
    print(pc1, pc2, pc3)
    return pc1, pc2,pc3


if __name__ == '__main__':
    file_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/CT_images/images_cropped/case00209.npz'
    pca_reduce_dimension(file_dir)
    pca_res_table = pd.DataFrame(columns=['PC1','PC2','PC3'])
    for seed in range(1,1001):
        pc1, pc2,pc3 = pca_experiments(file_dir, seed)
        print(pc1, pc2, pc3)
        #pca_res_table = pd.concat((pca_res_table,pd.DataFrame([[pc1,pc2,pc3]], columns=['PC1','PC2','PC3'])), axis=0)

    #pca_res_table.to_csv('/home/wukai/Desktop/RCC_classification/kidney_cancer_project/pca_randomness_test.csv', index=0)
    