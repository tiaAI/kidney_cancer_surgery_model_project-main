import pandas as pd
import os

if __name__ == '__main__':
    base_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training'
    ori_train_table = pd.read_csv('/media/wukai/Data01/RCC_classification/kidney_cancer_project/xgboost/train_features_pyradiomics_images.csv',header = 0, index_col=0)
    ori_test_table = pd.read_csv('/media/wukai/Data01/RCC_classification/kidney_cancer_project/xgboost/test_features_pyradiomics_images.csv',header = 0, index_col=0)
    test_table1 = pd.read_csv(os.path.join(base_dir, 'Nephro','total_features_cort_pred_nephro.csv'),header=0, index_col=0)
    test_table2 = pd.read_csv(os.path.join(base_dir,'Nephro','total_features_nephro_pred_nephro.csv'),header=0, index_col=0)
    
    train_id = pd.read_csv(os.path.join(base_dir, 'Nephro','phase3_train_id.txt'),header=None)
    train_id = [file_name.replace('_0000.nii.gz','') for file_name in train_id[0]]
    test_id = pd.read_csv(os.path.join(base_dir,'Nephro', 'phase3_test_id.txt'),header=None)
    test_id = [file_name.replace('_0000.nii.gz','') for file_name in test_id[0]]


    test_table = ori_test_table.loc[test_id,:]
    # change test table
    test_table = test_table.iloc[:,:2]
    test_table1 = pd.concat((test_table, test_table1),axis=1)
    test_table1.to_csv(os.path.join(base_dir,'test_feat_cort_pred_nephro.csv'))
    test_table2 = pd.concat((test_table, test_table2),axis=1)
    test_table2.to_csv(os.path.join(base_dir,'test_feat_nephro_pred_nephro.csv'))

    train_table = ori_train_table.loc[train_id,:]
    train_table.columns = test_table1.columns
    train_table.to_csv(os.path.join(base_dir,'train_feat_nephro.csv'))


    print('ok')