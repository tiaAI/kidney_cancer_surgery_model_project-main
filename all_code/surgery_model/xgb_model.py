'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-09-22 09:22:08
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-09-22 09:43:48
FilePath: \code\surgical_procedure_model_3\fold_02.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from regex import R
import xgboost as xgb
#from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import scale
import pandas as pd 
import numpy as np 

feature=pd.read_csv('/media/lizhe/yhc/luohu/model/model_3/surgical_procedure_input/all.feature.csv')
validation_feature = pd.read_csv('/media/lizhe/yhc/luohu/model/model_3/surgical_procedure_input/val_feature.csv')

df=np.array(feature)
y = df[:,2:3].squeeze()
x = df[:,3:]
x = scale(x)
val = np.array(validation_feature)
val_y = val[:,2:3].squeeze().astype(np.int)
#val_y = val_y.astype(float)
val_x= val[:,3:]
val_x=scale(val_x)

ss = StratifiedShuffleSplit(n_splits=5, test_size = 0.2,random_state =42)
idx_list= []

for train_index,test_index in ss.split(x,y):
    idx_list.append([train_index,test_index])

train_index = idx_list[1][0]
test_index = idx_list[1][1]
x_train,x_test = x[train_index],x[test_index]
y_train,y_test = y[train_index],y[test_index]

model = xgb.XGBClassifier(scale_pos_weight =3,
                           reg_lambda =30,
                            n_estimators=150,
                            max_depth = 50,
                            subsample = 0.85,
                            num_boost_round = 200,
                            min_child_weight =50,
                            colsample_bytree=1)
# model.fit(x_train,y_train)

eval_set = [(x_test, y_test)]
model.fit(x_train,y_train, early_stopping_rounds=300, eval_metric=['auc'], eval_set=eval_set, verbose=True)

y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)
p_train=model.predict_proba(x_train)[:,1]
p_test=model.predict_proba(x_test)
test_y_pred_proba = p_test[:,1] 
test_y_pred = [0 if proba < 0.55 else 1 for proba in test_y_pred_proba ]
y_test=np.array(y_test,dtype=np.int)
auc = roc_auc_score(y_true = y_test, y_score = test_y_pred_proba)
acc = accuracy_score(test_y_pred,y_test)
a=auc 
b=acc
z = {'V':y_test,'P':test_y_pred_proba,'auc':a,'acc':b}
new = pd.DataFrame(z)
new.to_csv(path_or_buf='/media/lizhe/yhc/luohu/model/model_3/surgical_procedure_output/surgical_procedure.csv',index=False) 

y_val = model.predict_proba(val_x)
p_val=model.predict_proba(val_x)
val_y_pred_proba = p_val[:,1] 
val_y_pred = [0 if proba < 0.5 else 1 for proba in val_y_pred_proba ]
val_y=np.array(val_y,dtype=np.int)
val_auc = roc_auc_score(val_y,val_y_pred_proba)
val_acc = accuracy_score(val_y,val_y_pred)
c=val_auc
d=val_acc
r = {'V':val_y,'P':val_y_pred_proba,'auc':c,'acc':d}
new = pd.DataFrame(r)
new.to_csv(path_or_buf='/media/lizhe/yhc/luohu/model/model_3/surgical_procedure_output/val.csv',index=False) 



