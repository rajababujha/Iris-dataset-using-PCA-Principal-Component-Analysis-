#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


# In[6]:


df = pd.read_csv('iris.csv') #Importing Dataset


# In[9]:


df.info()


# In the above results non-null exists. So no need of dropna aur fillna command

# In[10]:


df.describe()


# In[11]:


df.head(5)


# # Convetring Species into Digits

# In[12]:


df.Species.replace({'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2},inplace=True)


# In[13]:


df.head(5)

# Early Insights :
150 rows
4 Independent variables to act as factors
All have same units of measurement (cm)
No missing data
Three unique target classes namely : 'Iris-setosa', 'Iris-versicolor' and 'Iris-virginica'
No class imbalance, all target classes have equal number of rows (50 each).
# In[16]:


sns.pairplot(df, hue = 'Species')
plt.show()

In the above results, we can say easily detect the PetalLengthCm and PetalWidthCm are of our business. So we only use these both features columns.
# # Without PCA

# In[20]:


X = df.drop(['Species'],axis=1)
y = df.Species


# In[21]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X=scaler.fit_transform(X)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=20, stratify=y)
knn = KNeighborsClassifier(10)
knn.fit(X_train,y_train)


# In[25]:


print("Train score before PCA",knn.score(X_train,y_train),"%")
print("Test score before PCA",knn.score(X_test,y_test),"%")


# # With PCA

# In[26]:


from sklearn.decomposition import PCA
pca = PCA()
X_new = pca.fit_transform(X)


# In[27]:


pca.get_covariance()


# In[28]:


explained_variance=pca.explained_variance_ratio_
explained_variance


# In[29]:


pca=PCA(n_components=3)
X_new=pca.fit_transform(X)


# In[30]:


X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, y, test_size = 0.3, random_state=20, stratify=y)


# In[32]:


knn_pca = KNeighborsClassifier(10)
knn_pca.fit(X_train_new,y_train)


# In[33]:


print("Train score after PCA",knn_pca.score(X_train_new,y_train),"%")
print("Test score after PCA",knn_pca.score(X_test_new,y_test),"%")

