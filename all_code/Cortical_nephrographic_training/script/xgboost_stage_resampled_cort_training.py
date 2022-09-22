#!/usr/bin/python
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from mlxtend.evaluate import mcnemar_table, mcnemar
from collections import Counter
from imblearn.combine import SMOTEENN,SMOTETomek

if __name__ == '__main__':
    base_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training'
    train_feature = pd.read_csv(os.path.join(base_dir, 'train_feat_cort.csv'), header = 0, index_col=0)
    #test_feature1 = pd.read_csv(os.path.join(base_dir, 'test_feat_cort.csv'), header = 0, index_col=0)
    test_feature1 = pd.read_csv(os.path.join(base_dir, 'test_feat_cort_pred_cort.csv'), header = 0, index_col=0)
    test_feature2 = pd.read_csv(os.path.join(base_dir, 'test_feat_cort_pred_nephro.csv'), header = 0, index_col=0)
    test_feature1 = test_feature1.dropna(axis=0)
    test_feature2 = test_feature2.dropna(axis=0)

    train_x = train_feature.iloc[:,2:]
    train_y = train_feature.iloc[:,1]
    test_x = test_feature2.iloc[:,2:]
    test_y = test_feature2.iloc[:,1]
    # resample
    smote_enn = SMOTEENN(random_state=0)
    smote = SMOTETomek(random_state=0)
    train_x,train_y = smote.fit_resample(train_x,train_y)
    test_x,test_y = smote.fit_resample(test_x,test_y )
    # for tumor subtype cls

    model = XGBClassifier(scale_pos_weight = 1.1, # the best 1.2
                        min_child_weight = 1,
                        #learning_rate = 0.01,
                        n_estimators=300,
                        num_boost_round = 300,
                        gamma = 0.15,
                        #max_depth = 10,
                        reg_lambda = 1.1, # the best 1.1
                        subsample = 1, # the best 1
                        colsample_bytree = 1) # the best)
    #model = XGBClassifier()
    eval_set = [(test_x, test_y)]
    model.fit(train_x,train_y, early_stopping_rounds=100, eval_metric=['auc','logloss','error'], eval_set=eval_set, verbose=True)
    #'logloss',,'error'
    test_y_pred = model.predict_proba(test_x)
    test_y_pred_proba = test_y_pred[:,1] 
    test_y_pred = [0 if proba < 0.5 else 1 for proba in test_y_pred_proba ]
    #print(y_pred_proba)
    #print(test_y_pred)
    #print(test_y)
    accuracy = accuracy_score(test_y, test_y_pred)
    print("Subtype class accuracy: %.2f%%" % (accuracy * 100.0))
    Precision = precision_score(test_y, test_y_pred, average='binary')
    print("Subtype class precision: %.2f%%" % (Precision * 100.0))
    F1_score = f1_score(test_y, test_y_pred, average='binary') 
    print('Subtype F1 score:%.2f%%' %(F1_score*100.0))
    Recall_score = recall_score(test_y, test_y_pred, average='binary')
    print('Subtype Recall score label:%.2f%%' %(Recall_score*100.0))
    Auc_score = roc_auc_score(test_y, test_y_pred_proba)
    print('Subtype auc score label:%.2f%%' %(Auc_score*100.0))
    print(confusion_matrix(test_y,test_y_pred ))
    print('ok')
    test_pred_table = pd.concat((test_y, pd.DataFrame(test_y_pred_proba)), axis=1)
    test_pred_table.columns = ['truth','proba']
    test_pred_table.to_csv('stage_proba_cort_pred_nephro.csv', index=0)

