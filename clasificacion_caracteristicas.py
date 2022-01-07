# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 19:00:24 2022

@author: franc
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import support_functions as sp
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split
#carga de datos
X,Y,Xl,Yl = sp.aviris_data_load()

#%%
X_train, X_test, y_train, y_test = train_test_split(Xl, Yl,random_state=1)

extra_tree = ExtraTreeClassifier(random_state=0, )

clf = extra_tree.fit(X_train, y_train)

score = clf.score(X_test, y_test)

features = clf.feature_importances_



#%%

plt.plot(features)
plt.xlabel('Nº de bandas')
plt.ylabel('Importancia')

