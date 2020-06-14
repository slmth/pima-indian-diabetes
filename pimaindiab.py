# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 04:16:29 2020

@author: Havva
"""

#libraries for dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn import neighbors
from matplotlib.colors import ListedColormap

#upload dataset

pimaindiab=pd.read_csv("pimaindiab.csv")

X=pimaindiab.iloc[:,0:8]
y=pimaindiab.iloc[:,8]
pimaindiab.head()

#change your meaningless column names

pimaindiab_colnames = ['pregtime', '2hourplasma', 'bloodpl', 'skinfold', '2hourserumins', 'bodyindex', 'pedigree', 'age','class']
pimaindiab.columns = pimaindiab_colnames
pimaindiab.info()
pimaindiab.head()

#relationship with classes

def plotFeatures(col_list,title):
    plt.figure(figsize=(10,8))
    i = 0
    print(len(col_list))
    for col in col_list:
        i+=1
        plt.subplot(7,2,i)
        plt.plot(pimaindiab[col],pimaindiab["class"],marker='.',linestyle='none')
        plt.title(title % (col))   
        plt.tight_layout()
        
colnames = ['pregtime', '2hourplasma', 'bloodpl', 'skinfold', '2hourserumins', 'bodyindex', 'pedigree', 'age']
plotFeatures(colnames,"Relationship %s and classes")

#corr
plt.figure(figsize=(8,8))
sns.heatmap(pimaindiab.corr(),annot=True,fmt='.2f')
plt.show()

#how many data has got 0 or 1 classes?
pimaindiab['class'].value_counts()

#plot data
sns.pairplot(pimaindiab, hue='class', size=2.5)

#train-test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=0)

#SVM ALGORITHM 

from sklearn.svm import SVC
svm=SVC(kernel="linear",gamma=1,C=4)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print ("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

from sklearn.model_selection import cross_val_score
cvs=cross_val_score(estimator=svm,X=X_train,y=y_train,cv=4)
print("Cross Val.Score:" , cvs.mean())

#parameter optimization for svm 

from sklearn.model_selection import GridSearchCV
param=[{'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[1,0.5,0.1,0.01,0.001,0.20,0.020]},
    {'C':[1,2,3,4,5],'kernel':['linear'],'gamma':[1,0.5,0.1,0.01,0.001,0.20,0.020]}]
gs=GridSearchCV(estimator=svm,param_grid=param,scoring='accuracy',cv=4,n_jobs=-1)
       
grid_search=gs.fit(X_train,y_train)
bestresult=grid_search.best_score_
bestparameters=grid_search.best_params_
print(bestresult,"this is best cross val score")
print(bestparameters,"these are best parameters")

#KNN ALGORITHM
 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4,metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print ("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

from sklearn.model_selection import cross_val_score
cvs=cross_val_score(estimator=knn,X=X_train,y=y_train,cv=4)
print("Cross Val.Score:" , cvs.mean())

#NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
print ("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

from sklearn.model_selection import cross_val_score
cvs=cross_val_score(estimator=gnb,X=X_train,y=y_train,cv=4)
print("Cross Val.Score:" , cvs.mean())

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(solver='lbfgs',C=0.1)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
print ("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

from sklearn.model_selection import cross_val_score
cvs=cross_val_score(estimator=logr,X=X_train,y=y_train,cv=4)
print("Cross Val.Score:" , cvs.mean())

#parameter optimization for Logistic Regression

from sklearn.model_selection import GridSearchCV
param=[{'solver':['lbfgs', 'sag', 'saga'],'C':[1,2,0.05,0.10,5,10,50,100]},
      {'penalty':['none', 'l2'],'solver':['newton-cg'],'C':[1,2,0.05,0.10,5,10,50,100]}]
gs=GridSearchCV(estimator=logr,param_grid=param,scoring='accuracy',cv=4,n_jobs=-1)
       
grid_search=gs.fit(X_train,y_train)
bestresult=grid_search.best_score_
bestparameters=grid_search.best_params_
print(bestresult,"this is best cross val score")
print(bestparameters,"these are best parameters")
