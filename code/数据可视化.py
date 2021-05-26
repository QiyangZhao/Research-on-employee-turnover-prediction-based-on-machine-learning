#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei']
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


hr = pd.read_csv('HR_comma_seq.csv')    #读取数据集
hr['salary'] = hr['salary'].map({'low':1, 'medium':2, 'high':3})   #将salary中的字符型变量转换成数值型
hr_left = hr[hr['left']==1]   #离职员工数据集
hr_noleft = hr[hr['left']==0]  #在职员工数据集


# In[7]:


hr_left.describe()   #离职员工统计分析


# In[8]:


hr_noleft.describe()   #在职员工统计分析


# In[3]:


#绘制工作满意度分布直方图
plt.rcParams.update({"font.size":13})
plt.figure(figsize=(10,6))
plt.hist(hr.loc[(hr['left']==0), 'satisfaction_level'], bins=20, alpha=0.5, label='在职',range=(0,1))
plt.hist(hr.loc[(hr['left']==1), 'satisfaction_level'], bins=20, alpha=0.5, label='离职',range=(0,1))
plt.legend(loc='upper left')
plt.xticks(np.linspace(0,1,21))
plt.xticks(rotation=45,fontsize=13)
plt.xlabel('工作满意度')
plt.ylabel('人数')
plt.show()


# In[9]:


#绘制上一次绩效评估分布直方图
plt.rcParams.update({"font.size":13})
plt.figure(figsize=(10,6))
plt.hist(hr.loc[(hr['left']==0), 'last_evaluation'], bins=20, alpha=0.5, label='在职',range=(0,1))
plt.hist(hr.loc[(hr['left']==1), 'last_evaluation'], bins=20, alpha=0.5, label='离职',range=(0,1))
plt.legend(loc='upper left')
plt.xticks(np.linspace(0,1,21))
plt.xticks(rotation=45,fontsize=13)
plt.xlabel('上一次绩效评估')
plt.ylabel('人数')
plt.show()


# In[10]:


#绘制项目数量分布直方图
plt.rcParams.update({'font.size':15})
plt.hist(hr.loc[(hr['left']==0), 'number_project'],bins=6,range=(2,8), alpha=0.5, label='在职')
plt.hist(hr.loc[(hr['left']==1), 'number_project'],bins=6,range=(2,8), alpha=0.5, label='离职')
plt.legend()
plt.xticks(np.arange(2,9))
plt.xlabel('项目数量')
plt.ylabel('人数')
plt.show()


# In[11]:


#绘制平均月工作时长分布直方图
plt.rcParams.update({'font.size':15})
plt.hist(hr.loc[(hr['left']==0),'average_montly_hours'], bins=11, range=(90,310), alpha=0.5, label='在职')
plt.hist(hr.loc[(hr['left']==1),'average_montly_hours'], bins=11, range=(90,310), alpha=0.5, label='离职')
plt.legend()
plt.xticks(np.linspace(100,300,9))
plt.xlabel('平均月工作时长')
plt.ylabel('人数')
plt.show()


# In[25]:


#上一次绩效评估、项目数和平均月工作时长的相关矩阵图
hr_LNA_left = hr.loc[(hr['left']==1),['last_evaluation','number_project','average_montly_hours']]
hr_LNA = hr[['last_evaluation','number_project','average_montly_hours']]
hr_LNA_noleft = hr.loc[(hr['left']==0),['last_evaluation','number_project','average_montly_hours']]


# In[26]:


#全部员工的上一次绩效评估、项目数和平均月工作时长的相关矩阵图
sns.heatmap(hr_LNA.corr(), annot=True)


# In[27]:


#离职员工的上一次绩效评估、项目数和平均月工作时长的相关矩阵图
sns.heatmap(hr_LNA_left.corr(), annot=True)


# In[28]:


#在职员工的上一次绩效评估、项目数和平均月工作时长的相关矩阵图
sns.heatmap(hr_LNA_noleft.corr(), annot=True) 


# In[29]:


#上一次绩效评估与平均月工作时长分布散点图
fig=plt.figure(figsize=(10,5))
sns.scatterplot(data=hr,x='last_evaluation',y='average_montly_hours',hue='left', alpha=0.5)


# In[30]:


#工作满意度与上一次绩效评估分布散点图
fig=plt.figure(figsize=(10,8))
sns.scatterplot(data=hr,x='satisfaction_level',y='last_evaluation',hue='left', alpha=0.5)
plt.xticks(np.linspace(0,1,11))
plt.grid()
plt.show()


# In[35]:


#区域一
hr_1 = hr[(hr['satisfaction_level']<0.2)&(hr['last_evaluation']>0.7)]
#区域二
hr_2 = hr[(hr['satisfaction_level']<0.5)&(hr['satisfaction_level']>0.3)&(hr['last_evaluation']>0.4)&(hr['last_evaluation']<0.6)]
#区域三
hr_3 = hr[(hr['satisfaction_level']>0.7)&(hr['last_evaluation']>0.8)]


# In[42]:


#上一次绩效评估大于0.75的员工数据对比
hr_075 = hr[hr['last_evaluation']>0.75]
hr_075.groupby('left').mean()


# In[36]:


#区域一和区域三离职人员数据对比
print("区域一：\n",hr_1[hr_1['left']==1].mean())
print("区域三：\n",hr_3[hr_3['left']==1].mean())


# In[43]:


#区域二员工数据对比
hr_2.groupby('left').mean()


# In[38]:


#绘制在公司工作年数分布直方图
plt.rcParams.update({'font.size':15})
plt.hist(hr.loc[(hr['left']==0),'time_spend_company'],bins=9,range=(2,11),label='在职',alpha=0.5)
plt.hist(hr.loc[(hr['left']==1),'time_spend_company'],bins=9,range=(2,11),label='离职',alpha=0.5)
plt.legend()
plt.xticks(np.arange(2,12))
plt.xlabel('在公司工作年数')
plt.ylabel('人数')
plt.show()


# In[40]:


#在公司工作年数2到6年的工作满意度和上一次绩效评估散点图
for i in [2,3,4,5,6]:
    plt.rcParams.update({"font.size":15})
    fig = plt.figure(figsize=(6,4))
    sns.scatterplot(data=hr.loc[(hr['time_spend_company']==i)],x='satisfaction_level',y='last_evaluation',hue='left', alpha=0.5)
    plt.title('time_spend_company='+str(i))
    plt.show()


# In[44]:


#工作事故
hr.groupby('Work_accident').mean()


# In[45]:


#近五年晋升情况
hr.groupby('promotion_last_5years').mean()


# In[46]:


#工作部门
hr.groupby('sales').mean()


# In[47]:


#相对薪资等级
hr.groupby('salary').mean()


# In[ ]:




