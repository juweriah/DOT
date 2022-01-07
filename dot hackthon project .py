#!/usr/bin/env python
# coding: utf-8

# ### Things Done:
# 1. Data has been cleaned and empty values are replaced with mean of tht column
# 2. Normalizing and Splitting of Data 
# 3. Naive-Bayes
# 4. Knn
# 5. Random-Forest
# 6. SVM
# 
# ### Note:
# 1. We got rid of PCA and the plotting part as they were redundant
# 

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


# In[36]:


data = pd.read_excel('./data.xlsx')
data.head(25)


# In[37]:


data.shape


# In[38]:


data.describe()


# In[39]:


df = pd.DataFrame(data)
features=df.iloc[:,[2,7,8,12,13,15]]
labels=df.iloc[:,-1]
features


# In[40]:


# Replace zeroes with mean of the particular column
zero_not_accepted = ['age','net_yearly_income','no_of_days_employed','yearly_debt_payments','credit_limit','credit_score']

for column in zero_not_accepted:
    features[column] = features[column].replace(np.NaN,0)
    mean = float(features[column].mean(skipna=True))
    features[column] = features[column].replace(0, mean)
    
features.describe()
# ignore the below warning


# In[41]:


labels


# In[42]:


# splitting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.25)
print(len(X_train),len(X_test))


# In[43]:


# normalizing all the values so that every value can have equal contribution factor to the prediction model
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train) 
X_test = sc_x.transform(X_test)


# In[44]:


# training a Naive Bayes classifier
gnb = GaussianNB().fit(X_train, Y_train)
gnb_predictions = gnb.predict(X_test)
accuracy = gnb.score(X_test, Y_test)
print(accuracy)
cm = confusion_matrix(Y_test, gnb_predictions)
print(cm)


# In[45]:


# training a KNN model
Classifier = KNeighborsClassifier(n_neighbors=128)
Classifier.fit(X_train, Y_train)
ypredicted = Classifier.predict(X_test)
accuracy=Classifier.score(X_test,Y_test)
print(accuracy)
cm = confusion_matrix(Y_test, ypredicted)
print(cm)


# In[46]:


# Training a Random Forest Classifier
RF = RandomForestClassifier(n_estimators = 100) 
RF.fit(X_train, Y_train)

y_pred = RF.predict(X_test)
accuracy = RF.score(X_test,Y_test)
print(accuracy)
cm = confusion_matrix(Y_test, y_pred)
print(cm)


# In[47]:


from sklearn.model_selection import train_test_split
# Building a Support Vector Machine on train data
svc_model = svm.SVC(C= .1, kernel='linear', gamma= 1)
svc_model.fit(X_train, Y_train)
y_prediction = svc_model.predict(X_test)
accuracy = svc_model.score(X_test,Y_test)
print(accuracy)
cm = confusion_matrix(Y_test, y_prediction)
print(cm)


# In[ ]:





# In[ ]:




