# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:00:28 2017

@author: Devesh
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import arff


# Data Import

data=arff.load(open('D:\Documents\Coding\Machine Learning\Python\DR\messidor_featuresNEW.arff'))
dataset=pd.read_fwf(data['data'])


# Data Reorganize

mylist=[]

for i in range(1,20):
	mylist.append(str(i))

mylist.append("Label")
dataset.columns=[mylist]


#Preprocessng and scaling

from sklearn import preprocessing

mylist.remove("Label")
for col in mylist:
	dataset[col]=preprocessing.scale(dataset[col])


# Seperating Data

target=dataset["Label"]
train=dataset
train.drop("Label",axis=1,inplace=True)



#ML Algos

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import RandomizedSearchCV
import sklearn as sk


	# Defined algos
clf1=RandomForestClassifier(n_estimators=20,max_features="sqrt",random_state=1)
clf2=LogisticRegression(random_state=1,C=2.8)
clf3=svm.SVC(probability=True)
clf4=GaussianNB()
clf5=GradientBoostingClassifier(random_state=1)

	#training and cross validation
for clf in [clf1,clf2,clf3,clf4,clf5]:
	   scores=cross_val_score(clf, train, target, cv=3, scoring='accuracy')
	   print(clf)
	   print(scores.mean())



# 1000 epoch Neural Network
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

	#define model architecture
ann=Sequential()
ann.add(Dense(12,input_dim=19,activation='relu'))
ann.add(Dense(8,activation='relu'))
ann.add(Dense(5,activation='relu'))
ann.add(Dense(1,activation='sigmoid'))

ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	#conversion to numpyarray
nptrain=train.as_matrix()
nptarget=target.as_matrix()

	#model fitted
ann.fit(nptrain,nptarget,nb_epoch=1000,verbose=1)
scores=ann.evaluate(nptrain,nptarget)
"""