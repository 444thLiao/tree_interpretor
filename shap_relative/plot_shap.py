import shap
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
import xgboost as xgb
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
os.chdir('/home/liaoth/data2/16s/shandong/SD_codes/ML_haokui')
import load_data

### load data and preprocessing
########################################################################################################################
# X_train, y_train = load_data.get_data_by_libs(libs=['MF2', 'MF1'], cohort='Qingdao', source='Gut', target='host_status')
# X_train, y_train = load_data.get_data_by_libs(libs=['MF2'], cohort='Qingdao', source='Gut', target='host_status')
X_train, y_train = load_data.get_data_by_source(source='Gut', target='host_status')


X_train_health = X_train[y_train=='Health']
y_train_health = y_train[y_train=='Health']
X_train_T2D = X_train[y_train=='Type 2 diabetes']
y_train_T2D = y_train[y_train=='Type 2 diabetes']
print(X_train.shape, y_train.shape)
print(X_train_health.shape, y_train_health.shape)
print(X_train_T2D.shape, y_train_T2D.shape)

n_health = X_train_health.shape[0]
n_T2D = X_train_T2D.shape[0]
n_iteration = 1000
sample_size = 100

bst = xgb.Booster({'nthread':7}) #init model
bst.load_model("model_save")

shap_values = bst.predict(xgb.DMatrix(X_train), pred_contribs=True)
interaction_values = bst.predict(xgb.DMatrix(X_train), pred_interactions=True)
import plotly
import plotly.graph_objs as go

def out_sample_indenpendence(fea1,fea2):
    idxs_f1 = list(X_train.columns).index(fea1)
    idxs_f2 = list(X_train.columns).index(fea2)

    shap_f1 = list(shap_values[:,idxs_f1])
    shap_f2 = list(shap_values[:, idxs_f2])
    ori_v_f1 = list(X_train.iloc[:,idxs_f1])
    ori_v_f2 = list(X_train.iloc[:, idxs_f2])

    draw_data = []
    draw_data.append(go.Scatter(x=ori_v_f1,y=shap_f1,name=fea1,mode='markers'))
    draw_data.append(go.Scatter(x=ori_v_f2, y=shap_f2, name=fea2, mode='markers'))

    plotly.offline.plot(draw_data,filename='./interaction_shap.html')

def out_sample_interaction(X_train,interaction_values,fea1,fea2):
    idxs_f1 = list(X_train.columns).index(fea1)
    idxs_f2 = list(X_train.columns).index(fea2)

    shap_f1 = list(interaction_values[:,idxs_f1,idxs_f2])
    ori_v_f1 = np.log(list(X_train.iloc[:,idxs_f1]))
    ori_v_f2 = np.log(list(X_train.iloc[:, idxs_f2]))

    draw_data = []
    draw_data.append(go.Scatter(x=ori_v_f1,y=ori_v_f2,marker=dict(color=shap_f1,showscale=True),name=fea1,mode='markers'))

    layout = dict(xaxis=dict(title=fea1),
                  yaxis=dict(title=fea2))
    plotly.offline.plot(dict(data=draw_data,layout=layout),filename='./interaction_shap.html')

def out_feature_imp(ori_data,shap_values,num_feas=10):
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])[::-1]
    colors_vals = ori_data.iloc[:,feature_order[:num_feas]]
    x_vals = shap_values[:,feature_order[:num_feas]]
    x_vals = pd.DataFrame(x_vals,index=ori_data.index,columns=colors_vals.columns)
    new_vals = []
    for fea in range(x_vals.shape[1]):
        tmp = x_vals.iloc[:,[fea]]
        tmp.loc[:,'fea_name'] = x_vals.columns[fea]
        tmp.loc[:, 'ori_val'] = colors_vals.iloc[:,fea]
        tmp.columns = ['shap','fea_name','ori_val']
        new_vals.append(tmp)
    final_data = pd.concat(new_vals,axis=0)
    sns.violinplot(x='shap', y='fea_name', data=final_data, inner=None,color='.8')
    sns.stripplot(x='shap',y='fea_name',jitter=True,data=final_data,alpha=.5)
out_sample_interaction('OTU65','OTU299')
out_sample_indenpendence('OTU65','OTU75')