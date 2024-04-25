#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


X=df.iloc[:,:-1].to_numpy()
y=df.iloc[:,-1].to_numpy()


# In[8]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[10]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)


# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[14]:


clf.set_params(max_depth=3)


# In[15]:


clf.fit(X_train,y_train)
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show


# In[16]:


predictions=clf.predict(X_test)


# In[17]:


clf.predict([[90,20],[200,30]])


# In[19]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X_train,y_train,cv=5,scoring='accuracy')
accuracy=scores.mean()
accuracy


# In[21]:


from sklearn import metrics
cf=metrics.confusion_matrix(y_test,predictions)
cf


# In[23]:


tp=cf[1][1]
tn=cf[0][0]
fp=cf[0][1]
fn=cf[1][0]
print(f"tp:{tp},tn:{tn},fp:{fp},fn:{fn}")


# In[24]:


print("accuarcy",metrics.accuracy_score(y_test,predictions))


# In[25]:


print("Precision",metrics.precision_score(y_test,predictions))


# In[27]:


print("Recall",metrics.recall_score(y_test,predictions))


# In[30]:


feature_importances = clf.feature_importances_
print("feature importances:",feature_importances)


# In[ ]:




