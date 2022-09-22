'''
Author: your name
Date: 2021-10-08 21:30:13
LastEditTime: 2022-03-01 14:20:33
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \kidney_cancer_project-main\pyradiomics\pyradiomics_output_process.py
'''
import os
import pandas as pd
def pyradiomics_out_process(cf):
    features1_file = cf.radiomics_features1
    features2_file = cf.radiomics_features2
    features1_table = pd.read_csv(features1_file)
    features2_table = pd.read_csv(features2_file)
    files_id = [image_dir.split('/')[-1].replace('_0000.nii.gz','') for image_dir in features1_table['Image']]
#提取后的影像组学特征改名
    feature_names = [feature for feature in features1_table.columns.tolist() if 'original' in feature and 'diagnostics' not in feature]
    features1_table = features1_table[feature_names]
    features1_table.columns = [name.replace('original', 'renal') for name in features1_table.columns]
    features2_table = features2_table[feature_names]
    features2_table.columns = [name.replace('original', 'tumor') for name in features2_table.columns]
    features_table = pd.concat((features1_table, features2_table),axis=1)
    features_table.index = files_id
    return features_table
if __name__ == '__main__':
    pass
