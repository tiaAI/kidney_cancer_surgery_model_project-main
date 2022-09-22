import pandas as pd
import os

if __name__ == '__main__':
    base_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training'
    ori_train_table = pd.read_csv('/media/wukai/Data01/RCC_classification/kidney_cancer_project/xgboost/train_features_pyradiomics_images.csv',header = 0, index_col=0)
    ori_test_table = pd.read_csv('/media/wukai/Data01/RCC_classification/kidney_cancer_project/xgboost/test_features_pyradiomics_images.csv',header = 0, index_col=0)
    cort_pred_cort_test_table = pd.read_csv(os.path.join(base_dir, 'Cortical','total_features_cort_pred_cort.csv'),header=0, index_col=0)
    nephro_pred_cort_test_table = pd.read_csv(os.path.join(base_dir,'Cortical','total_features_nephro_pred_cort.csv'),header=0, index_col=0)
    
    train_id = pd.read_csv(os.path.join(base_dir, 'Cortical','phase2_train_id.txt'),header=None)
    train_id = [file_name.replace('_0000.nii.gz','') for file_name in train_id[0]]
    test_id = pd.read_csv(os.path.join(base_dir,'Cortical', 'phase2_test_id.txt'),header=None)
    test_id = [file_name.replace('_0000.nii.gz','') for file_name in test_id[0]]



    test_table = ori_test_table.loc[test_id,:]
    test_table.to_csv(os.path.join(base_dir,'test_feat_cort.csv'))
    # change test table
    test_table = test_table.iloc[:,:2]
    test_table1 = pd.concat((test_table, cort_pred_cort_test_table),axis=1)
    test_table1.to_csv(os.path.join(base_dir,'test_feat_cort_pred_cort.csv'))
    test_table2 = pd.concat((test_table, nephro_pred_cort_test_table),axis=1)
    test_table2.to_csv(os.path.join(base_dir,'test_feat_nephro_pred_cort.csv'))

    # for train set
    train_table = ori_train_table.loc[train_id,:]
    train_table.columns = test_table1.columns
    train_table.to_csv(os.path.join(base_dir,'train_feat_cort.csv'))

    print('ok')