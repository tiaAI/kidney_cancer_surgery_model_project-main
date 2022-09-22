# kidney_cancer_project
The code script used in the kidney cancer project
The end_to_end framework are designed for kidney cancer histologic subtype (ccRCC vs. non-ccRCC)
and tumor stage (T1&T2 vs. T3&T4) prediction.
By using raw enhanced CT images as input, the framework can realize automatic semantic segmentation, feature extraction, tumor subtype&stage prediction.

## END-TO-END predict
1. Install the requirement in the code.
2. Modify the directory in the file 'config.py'. Input CT images should be NITFI format.
3. Run the code 'end_to_end_predict.py'
