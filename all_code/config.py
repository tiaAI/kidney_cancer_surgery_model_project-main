class config():
    def __init__(self):
        self.base_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training/Nephro'
        self.image_dir = '/media/wukai/Data01/nnUNet_raw_data_base/nnUNet_raw_data/Task108_KidneyPhase3/imagesTs'
        self.mask_dir = '/media/wukai/Data01/nnUNet_raw_data_base/nnUNet_raw_data/Task108_KidneyPhase3/labelsTrp_phase2'
        
        self.resample_dir = '/media/wukai/Data01/nnUNet_raw_data_base/nnUNet_raw_data/Task108_KidneyPhase3/imagesTs_resample_cort_pred'
        self.normalized_dir = '/media/wukai/Data01/nnUNet_raw_data_base/nnUNet_raw_data/Task108_KidneyPhase3/imagesTs_normalized_cort_pred'
        self.cropped_dir = '/media/wukai/Data01/nnUNet_raw_data_base/nnUNet_raw_data/Task108_KidneyPhase3/imagesTs_cropped_cort_pred'

        self.clinical_file = '/share/Data01/wukai/rcc_classify/clinical_all_rcc_dict.pkl'

        # radiomics
        self.radiomics_feature_table = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training/Nephro/cort_pred_nephro_testset_radiomics.csv'
        self.radiomics_dir = '/home/wukai/Desktop/RCC_classification/kidney_cancer_project/pyradiomics'
        self.param1_dir = '/home/wukai/Desktop/RCC_classification/kidney_cancer_project/pyradiomics/PyradiomicsSettings/Params.yaml'
        self.param2_dir = '/home/wukai/Desktop/RCC_classification/kidney_cancer_project/pyradiomics/PyradiomicsSettings/Params_label2.yaml'

        self.voxels_meta_dir = '/media/wukai/Data01/RCC_classification/kidney_cancer_project/Cortical_nephrographic_training/voxels_meta_dict.pkl'
        self.process_num = 8
        self.target_spacing = [2.5,2.5,2.5] # [H,Y,X]
        self.tumor_size_threshold = 48*24*24
        self.crop_size = [64,64,64]

