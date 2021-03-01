#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import lightgbm as lgb


# In[2]:


train_data = pd.read_csv('/Users/zhangwanyu/Desktop/KKB/AI大赛/week1/train.csv')
test_data = pd.read_csv('/Users/zhangwanyu/Desktop/KKB/AI大赛/week1/test1.csv')
train_data.columns


# In[3]:


# train_data.info()
# 去掉缺失值比较多的列 ['lan']; 去掉Unnamed列
train_data.isnull().sum()
train_data = train_data.drop(['lan','Unnamed: 0'],axis=1)
test_data = test_data.drop(['lan','Unnamed: 0'],axis=1)


# In[4]:


i = 0
for column in train_data.columns:
    print(column,train_data[column].nunique())
    i += 1
print('There are {} features'.format(i))


# In[5]:


# 去掉唯一值个数大于100 ['android_id','dev_height','dev_width','media_id','osv','package','sid','timestamp','fea_hash','location','fea1_hash']
# 保留唯一值个数小于100 ['apptype','carrier','dev_ppi','ntt','os','version','cus_type']
# 去掉label ['label']
# 特征补充：'osv‘ --> 'media_id' --> 'location' --> 'timestamp'--> 'dev_height','dev_width' --> 'fea1_hash'
features = ['apptype','carrier','dev_ppi','ntt','os','version','cus_type','osv','media_id','timestamp','dev_height','dev_width','fea1_hash']
label = train_data['label']
train_data = train_data[features]
test_data = test_data[features]


# In[7]:


# 去掉唯一值的列['os']
train_data = train_data.drop(['os'],axis=1)
test_data = test_data.drop(['os'],axis=1)


# In[8]:


# 清洗列['version']，去空白字符串
# test_data中有一条数据为’20‘，是train_data中没有的，忽略
train_data['version'] = train_data['version'].apply(lambda x:x.strip())
test_data['version'] = test_data['version'].apply(lambda x:x.strip()) 

train_version = set(train_data['version'].unique().tolist())
test_version = set(test_data['version'].unique().tolist())
all_version = list(train_version | test_version)
print(all_version)

train_data['version'] = train_data['version'].apply(lambda x:all_version.index(x))
test_data['version'] = test_data['version'].apply(lambda x:all_version.index(x)) 


# In[9]:


# 清洗列['osv'],去掉前缀+encode
test_data['osv'] = test_data['osv'].apply(lambda x:str(x).strip('Android_'))
train_data['osv'] = train_data['osv'].apply(lambda x:str(x).strip('Android_'))

osv_unique_test = set(test_data['osv'].tolist())
osv_unique_train = set(train_data['osv'].tolist())
osv_unique = list(osv_unique_test | osv_unique_train)

test_data['osv'] = test_data['osv'].apply(lambda x:osv_unique.index(x))
train_data['osv'] = train_data['osv'].apply(lambda x:osv_unique.index(x))


# In[10]:


# 清洗列['timestamp']
import time
train_data['date_hour'] = train_data['timestamp'].apply(lambda x: int(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x/1000))[11:13]))
test_data['date_hour'] = test_data['timestamp'].apply(lambda x: int(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x/1000))[11:13]))

train_data = train_data.drop(['timestamp'],axis=1)
test_data = test_data.drop(['timestamp'],axis=1)


# In[12]:


# 训练

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def kf_model(clf, train_data, label, test_data, cate_features):
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
    prob = []
    for k,(train_index,val_index) in enumerate(kf.split(train_data,label)):
        train_x = train_data.iloc[train_index]
        train_y = label.iloc[train_index]
        val_x = train_data.iloc[val_index]
        val_y = label.iloc[val_index]
        
        clf.fit(train_x,train_y,eval_metric = 'error')
        val_y_predict = clf.predict(val_x)
        acc = accuracy_score(val_y,val_y_predict)
        print('{} accuracy_score: {}'.format(k+1,acc))
        y_predict = clf.predict_proba(test_data)[:,-1]
        prob.append(y_predict)
    mean_prob = sum(prob)/5
    return mean_prob
        


# In[15]:


cate_features = ['apptype','carrier','ntt','version','cus_type','osv','media_id','date_hour']
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.08, max_delta_step=0, max_depth=8,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=800, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0.5, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

# clf.fit(train_data,label,eval_metric = 'error')
# predicts1 = clf.predict(test_data)


# param_dist = {
#         'n_estimators':range(700,900,100),
#         'max_depth':range(6,9,1)
#         }

# grid = GridSearchCV(clf,param_dist,cv = 3,scoring = 'roc_auc',n_jobs = -1)
# grid.fit(train_data,label)
# best_estimator = grid.best_estimator_
# best_estimator
predicts1 = kf_model(clf,  train_data, label, test_data, cate_features)


# In[17]:


# 预测
test_data_2 = pd.read_csv('/Users/zhangwanyu/Desktop/KKB/AI大赛/week1/test1.csv')
result = pd.DataFrame(test_data_2['sid'])
result['label'] = predicts1
result['label'] = result['label'].apply(lambda x:1 if x>=0.5 else 0)


# In[19]:


result.to_csv('version_13_6.csv',index=None)






