import os
import pandas as pd



if __name__ == '__main__':
    base_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training/Cortical'
    features1_file = os.path.join(base_dir, 'nephro_pred_cort_testset_radiomics_label1.csv')
    features2_file = os.path.join(base_dir, 'nephro_pred_cort_testset_radiomics_label2.csv')
    features1_table = pd.read_csv(features1_file)
    features2_table = pd.read_csv(features2_file)
    files_id = [image_dir.split('/')[-1].replace('_0000.nii.gz','') for image_dir in features1_table['Image']]

    feature_names = [feature for feature in features1_table.columns.tolist() if 'original' in feature and 'diagnostics' not in feature]
    features1_table = features1_table[feature_names]
    features2_table = features2_table[feature_names]
    features1_table.columns = [name.replace('original','kidney') for name in feature_names]
    features2_table.columns = [name.replace('original','tumor') for name in feature_names]
    features_table = pd.concat((features1_table, features2_table), axis=1)
    features_table.index = files_id
    features_table.to_csv(os.path.join(base_dir, 'nephro_pred_cort_testset_radiomics.csv'))
    print('OK')