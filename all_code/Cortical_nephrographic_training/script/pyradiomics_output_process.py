import os
import pandas as pd
import json


if __name__ == '__main__':
    features1_file = '/share/Data01/wukai/rcc_classify/xgboost/kidney_tumor_testset_features_label_1.csv'
    features2_file = '/share/Data01/wukai/rcc_classify/xgboost/kidney_tumor_testset_features_label_2.csv'
    features1_table = pd.read_csv(features1_file)
    features2_table = pd.read_csv(features2_file)
    files_id = [image_dir.split('/')[-1].replace('_0000.nii.gz','') for image_dir in features1_table['Image']]

    feature_names = [feature for feature in features1_table.columns.tolist() if 'original' in feature and 'diagnostics' not in feature]
    features1_table = features1_table[feature_names]
    features2_table = features2_table[feature_names]
    results = {}
    results['feature_names'] = feature_names
    for i,file_id in enumerate(files_id):
        results[file_id] = {'tumor':features1_table.iloc[i,:].tolist(),
                            'kidney':features2_table.iloc[i,:].tolist()}
    with open('/share/Data01/wukai/rcc_classify/xgboost/pyradiomics_testset_result.json', 'w') as OUT:
        json.dump(results, OUT)

    print('OK')