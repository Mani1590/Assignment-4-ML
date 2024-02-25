#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


import numpy as np
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)


# In[3]:


clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(y_pred)


# In[4]:


accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy * 100:2f}%")


# In[5]:


conf_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix is: ")
print(conf_matrix)


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

labels = iris.target_names
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

