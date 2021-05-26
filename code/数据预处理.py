#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

hr = pd.read_csv('HR_comma_seq.csv')

#将salary转换成数值
hr['salary'] = hr['salary'].map({'low':1, 'medium':2, 'high':3})

#数值归一化
cols = ['number_project','average_montly_hours', 'time_spend_company','salary']
for col in cols:
    hr[col] = (hr[col] - hr[col].min())/(hr[col].max() - hr[col].min())

#对sales进行one-hot编码
hr = pd.get_dummies(hr)

#生成训练集和测试集
y = hr['left']
x = hr.drop(columns='left')
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=8)
train_x['left'] = train_y
test_x['left'] = test_y

#保存训练集和测试集
test_x.to_csv('hr_test.csv', index=None)
train_x.to_csv('hr_train.csv', index=None)

#决策树数据集
y = hr['left']
x = hr.drop(columns='left')

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=8)
train_x['left'] = train_y
test_x['left'] = test_y
test_x.to_csv('hr_test_tree.csv', index=None)
train_x.to_csv('hr_train_tree.csv', index=None)


# In[ ]:




