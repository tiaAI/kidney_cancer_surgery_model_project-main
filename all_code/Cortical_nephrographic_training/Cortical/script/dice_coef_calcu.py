import os
import glob
import numpy as np
import SimpleITK as sitk
import pandas as pd
def dice_coeff_calculate(file_id, true_labels_dir, pred_labels_dir):
    label_true_dir = os.path.join(true_labels_dir, file_id+'.nii.gz')
    assert os.path.join(label_true_dir), 'Can not find file %s'%(label_true_dir)
    label_pred_dir = os.path.join(pred_labels_dir, file_id+'.nii.gz')
    assert os.path.join(label_pred_dir), 'Can not find file %s'%(label_pred_dir)
    print('process %s'%file_id)
    dice_result = [file_id]

    truth_sitk = sitk.ReadImage(label_true_dir)
    truth_image = sitk.GetArrayFromImage(truth_sitk)
    pred_sitk = sitk.ReadImage(label_pred_dir)
    pred_image = sitk.GetArrayFromImage(pred_sitk)

    dice_coeff = two_label_dice_coeff(pred_image, truth_image)
    dice_result += (dice_coeff)
    return dice_result


def two_label_dice_coeff(predict, target):
    dice = 0.
    total_dice = 0.
    smooth = 1.
    dice_coeff = []
    for i in [1,2]:
        i = int(i)
        
        zeros = np.zeros_like(predict)
        zeros[predict == i] =1
        m1 = zeros.flatten()

        zeros = np.zeros_like(target)
        zeros[target == i] =1
        m2 = zeros.flatten()

        dice = (2*(m1*m2).sum() + smooth) / (m1.sum() + m2.sum() + smooth)

        dice_coeff.append(dice)
    return dice_coeff


if __name__ == '__main__':
    base_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training/Cortical'
    true_labels_dir = '/media/wukai/Data01/nnUNet_raw_data_base/nnUNet_raw_data/Task107_KidneyPhase2/labelsTs'
    pred_labels_dir = '/media/wukai/Data01/nnUNet_raw_data_base/nnUNet_raw_data/Task107_KidneyPhase2/labelsTrp_phase3'

    files_dir = glob.glob(os.path.join(true_labels_dir, '*nii.gz'))
    files_id = [file_dir.split('/')[-1].replace('.nii.gz','') for file_dir in files_dir]

    dice_coeff_result = []
    for file_id in files_id:
        dice_coeff_result.append(dice_coeff_calculate(file_id, true_labels_dir, pred_labels_dir))
    
    dice_coeff_result = pd.DataFrame(dice_coeff_result)
    dice_coeff_result.columns = ['file_id', 'kidney_dice', 'tumor_dice']
    dice_coeff_result.to_csv(os.path.join(base_dir, 'nephro_model_pred_cort_set_dice_results.csv'),sep = ',', index = False)
    print('ok')
