# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:31:57 2017

@author: Devesh
"""


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import arff
import matplotlib.pyplot as plt
import seaborn as sns


data=arff.load(open('D:\Documents\Coding\Machine Learning\Python\DR\messidor_featuresNEW.arff'))
dataset=pd.read_fwf(data['data'])

mylist=[]

for i in range(1,20):
	mylist.append(str(i))

mylist.append("Label")

dataset.columns=[mylist]


# ---x--- Data work done

"""
#Bar plot between means
x=mylist
y=[dataset[i].mean() for i in mylist]
sns.barplot(x,y,palette="Blues_d")
"""

"""
#Pairplot between features 9,10,11
sel=['9','10','11','Label']
nds=dataset[sel]

sns.pairplot(nds,hue="Label")
"""

"""
#Histogram between classes of feature 1
plt.hist(dataset["1"][dataset["Label"]==1],rwidth=0.8,color="r")
plt.hist(dataset["1"][dataset["Label"]==0],rwidth=0.5,color="b")

#Histogram between classes of feature 2
plt.hist(dataset["2"][dataset["Label"]==1],rwidth=0.8,color="r")
plt.hist(dataset["2"][dataset["Label"]==0],rwidth=0.5,color="b")
"""