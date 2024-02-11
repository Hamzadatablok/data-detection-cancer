#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[3]:


datacancer=pd.read_csv(r'C:\Users\medtr\Downloads\archive (1)\data.csv')


# IDE **pre processing**
# EXPLORE DATA 

# 1-ANALYSE EXPLORE 

# In[4]:


datacancer.head()


# In[17]:


datacancer.head(10)


# In[18]:


datacancer.tail()


# In[19]:


datacancer.info()


# In[20]:


datacancer.shape


# In[21]:


datacancer.isnull().sum() 


# In[22]:


datacancer['diagnosis'].value_counts()


# In[27]:


datacancer.columns


# In[28]:


sns.countplot(x='diagnosis', data=datacancer)


# In[24]:


datacancer.columns


# In[30]:


datacancer['diagnosis'].replace({'M': '1', 'B': '0'}, inplace=True)


# In[31]:


datacancer.head()


# In[32]:


datacancer.tail()


# In[34]:


sns.countplot(data=datacancer, x='diagnosis')


# In[35]:


datacancer['diagnosis'].value_counts()


# In[36]:


datacancer.describe()


# In[37]:


datacancer.describe().T


# In[39]:


sns.pairplot(datacancer ,hue ='diagnosis')


# In[40]:


datacancer.corr()


# In[42]:


plt.figure(figsize=(20, 20))
sns.heatmap(datacancer.corr(), annot=True, cmap='mako')
plt.show()


# creation MODEL 

# MACHINE LEARNING 

# In[5]:


x = datacancer.drop(['diagnosis'], axis=1)
y = datacancer['diagnosis']


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)


# In[9]:


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[17]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)


# In[62]:


pip install --upgrade scikit-learn


# In[80]:


from sklearn.metrics import classification_report , accuracy_score
from sklearn.metrics import confusion_matrix 


# In[ ]:


pip install --upgrade scikit-learn


# In[ ]:





# In[86]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# استخراج مصفوفة التباين
cm = confusion_matrix(y_test, model.predict(x_test))

# رسم مصفوفة التباين بدون تبويس
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

# مصفوفة التباين المعيارية
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# رسم مصفوفة التباين المعيارية
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Normalized Confusion Matrix")
plt.show()


# In[84]:


from sklearn.metrics import confusion_matrix, classification_reportه
# hiya tawa9o3AT predit
# tkon 3alamat h9i9a hiya men x_test 
conf_matrix = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(conf_matrix)

# khsk tstakhdm tawa9o3at w machi predict
classification_rep = classification_report(y_test, pred)
print("Classification Report:")
print(classification_rep)


# In[18]:


accuracy_score = accuracy_score(y_test, pred)
print("accuracy_score:")
print(accuracy_score)


# In[19]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# hiya tawa9o3AT predit
# tkon 3alamat h9i9a hiya men x_test
conf_matrix = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(conf_matrix)

# khsk tstakhdm tawa9o3at w machi predict 
classification_rep = classification_report(y_test, pred)
print("Classification Report:")
print(classification_rep)

# NDAKHAL TAWA9O3 DYAL DI9A DYAL PRED 
accuracy_score = accuracy_score(y_test, pred)
print("accuracy_score:")
print(accuracy_score)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
#creation Knn classifier model 
knn =KNeighborsClassifier(n_neighbors=2) # k try 2,3,4
# fitting training data 
knn.fit(x_test,y_test)
# prediction data 
knn.predict(x_test)


# In[10]:


from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# إنشاء KNN classifier model
knn = KNeighborsClassifier(n_neighbors=2)  # قم بتغيير عدد الجيران حسب الحاجة
# تدريب البيانات
knn.fit(x_train, y_train)  # قم بتغيير x_train و y_train حسب البيانات الخاصة بك

# حساب مصفوفة التباين
cm = confusion_matrix(y_test, knn.predict(x_test))

# رسم مصفوفة التباين بدون تبويس
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

# حساب مصفوفة التباين المعيارية
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# رسم مصفوفة التباين المعيارية
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Normalized Confusion Matrix")
plt.show()


# In[ ]:




