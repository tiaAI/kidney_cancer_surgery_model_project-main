import os
import pandas as pd

if __name__ == '__main__':
    base_dir = '/mnt/d/AI_diagnosis/Renal_cancer_project/Cortical_nephrographic_training/Nephro'
    
    total_case_info = pd.read_csv(os.path.join(base_dir, 'clinical_all_rcc_dict.csv'), header=0)
    train_id = pd.read_table(os.path.join(base_dir, 'phase3_train_id.txt'), header=None)
    test_id = pd.read_table(os.path.join(base_dir, 'phase3_test_id.txt'), header=None)
    files_id = pd.concat((train_id, test_id), axis=0)
    clin_info = pd.DataFrame([file_id.replace('_0000.nii.gz','') for file_id in files_id.iloc[:,0]])
    clin_info.columns = ['file_id']
    clin_info['case_id'] = [file_id.split('_')[0] for file_id in clin_info['file_id']]
    clin_info = pd.merge(clin_info, total_case_info, how='left', on = 'case_id')
    clin_info = clin_info.iloc[:,[0,1,2,3,5]]
    clin_info['stage'] = [0 if stage <= 1 else 1 for stage in clin_info['stage']]
    clin_info.to_csv(os.path.join(base_dir, 'nephro_classes_label.csv'),index=None)
    print('ok')
