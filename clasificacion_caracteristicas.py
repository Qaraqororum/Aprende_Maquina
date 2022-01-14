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
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  confusion_matrix , cohen_kappa_score
#%% Preparación de los datos
X,Y,Xl,Yl = sp.aviris_data_load()

n_tags = max(Yl)
X_reduced, Y_reduced = sp.data_reducer()

#%% Entrenamiento del modelo para conseguir un ranking de características

X_train, X_test, y_train, y_test = train_test_split(Xl, Yl,random_state=1)

extra_tree = ExtraTreeClassifier(random_state=0, splitter ='best' )

clf = extra_tree.fit(X_train, y_train)

score = clf.score(X_test, y_test)

features = clf.feature_importances_

plt.plot(features)
plt.xlabel('Nº de bandas')
plt.ylabel('Importancia')


#%% Reducción de bandas o características

#Hemos escogido este criterio de 0.01 
indice= np.where(features< 0.01)

X_reduced2 = np.delete(X_reduced, indice, axis=1)




#%% Random forest (ensembles)

criterion = 'gini'
max_depth = None
report = confusion_matrix
report2 = cohen_kappa_score


Xtr, Xts, ytr, yts = train_test_split(X_reduced2, Y_reduced, test_size=0.3)
# Train RF ensemble
rf = RandomForestClassifier(n_estimators=200, criterion=criterion, random_state=100)
rf.fit(Xtr, ytr)
yb = rf.predict(Xts)

# Evaluate and compare results
print('Confusion Matrix:  ', report(yts, yb), 'Cohen-kappa',report2(yts, yb))

Conf_matrix_RF = report(yts, yb)
Kappa_RF = report2(yts, yb)

#Observamos como con el conjunto reducido de bandas, dejando solo 16 bandas, el ensemble funciona
#bastante bien con un cohen-kappa de ~0.74
tag_list = np.arange(1,17)
score = rf.predict_proba(X_test)
sp.draw_ROC(yts,score,tag_list)

sp.draw_ConfusionM(Conf_matrix_RF,tag_list)

X_ = np.delete(X.reshape([145*145,220]), indice, axis=1)
sp.draw_image(rf.predict(X_),"Random Forest ")
