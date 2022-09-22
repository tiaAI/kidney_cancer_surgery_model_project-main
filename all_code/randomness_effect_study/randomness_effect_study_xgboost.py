#!/usr/bin/python
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

def pca_reduce_dimension(file_dir, seed):
    assert os.path.exists(file_dir),'Can not find file in %s'%(file_dir)
    #print('pca process %s'%file_id)
    image = np.load(file_dir)['data']
    image = np.reshape(image, (64,-1))
    # pca
    pca = PCA(4,random_state=seed).fit(image)
    image_feature = pca.transform(image)
    image_feature = np.reshape(image_feature, (1,-1)).squeeze()
    return image_feature


if __name__ == '__main__':
    base_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/randomness_effect_study'
    train_feature = pd.read_csv(os.path.join(base_dir, 'train_features_pyradiomics_images.csv'), header = 0, index_col=0)
    test_feature = pd.read_csv(os.path.join(base_dir, 'test_features_pyradiomics_images.csv'), header = 0, index_col=0)

    train_x = train_feature.iloc[:,2:]
    train_y = train_feature.iloc[:,0]
    test_x = test_feature.iloc[:,2:]
    test_y = test_feature.iloc[:,0]

    with open(os.path.join(base_dir, 'randomness_check_res.txt'),'w') as OUT:
        OUT.write('Random_stat\tAUC\n')
        for seed in range(1,1001):
            # set different random state to PCA feature extraction
            for file_id in train_x.index:
                file_dir = os.path.join(base_dir, 'trainset/images_cropped',file_id+'.npz')
                train_x.loc[file_id][200:456] = pca_reduce_dimension(file_dir, seed)
                
            for file_id in test_x.index:
                file_dir = os.path.join(base_dir, 'testset/images_cropped',file_id+'.npz')
                test_x.loc[file_id][200:456] = pca_reduce_dimension(file_dir, seed)

            # for tumor subtype cls
            model = XGBClassifier(scale_pos_weight = 0.80, # the best
                                min_child_weight = 0.9,
                                #learning_rate = 0.06,
                                n_estimators=300,
                                num_boost_round = 300,
                                gamma = 0.04,
                                #max_depth = 6,
                                reg_lambda = 1.1,
                                subsample = 0.8, # the best
                                colsample_bytree = 1 # the best
                                    )

            eval_set = [(test_x, test_y)]
            model.fit(train_x, train_y, early_stopping_rounds=200, eval_metric=['auc','logloss','error'], eval_set=eval_set, verbose=True)
            #
            pred_y = model.predict_proba(test_x)
            pred_y_proba = pred_y[:,1] 
            pred_y_class = [0 if proba < 0.5 else 1 for proba in pred_y_proba]
            accuracy = accuracy_score(np.array(test_y), pred_y_class)
            Auc_score = roc_auc_score(np.array(test_y), pred_y_proba)
            print('Subtype Accuracy:%.4f AUC:%.4f' %(accuracy, Auc_score))
            OUT.write(str(seed)+'\t'+str(Auc_score)+'\n')

    
