# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:34:34 2022

@author: morte
"""

#%% Reducción de datos
import support_functions as sp
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier


#carga de datos

#%% Preparación de los datos
X_reduced, Y_reduced = sp.data_reducer()

X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y_reduced, stratify=Y_reduced,random_state=1)
#%% KNN
tag_list = np.arange(1,17)

modelo = KNeighborsClassifier(n_neighbors=4,
                              weights='distance',
                              algorithm='auto',
                              leaf_size=40, p=2,
                              metric='cosine',
                              metric_params=None,
                              n_jobs=-1).fit(X_train,y_train)

    
pred = modelo.predict(X_test)
score = modelo.predict_proba(X_test)

conf_mat= confusion_matrix(y_test,pred)
kappa= cohen_kappa_score(y_test,pred)
print(kappa)
    
sp.draw_ROC(y_test,score,tag_list)

sp.draw_ConfusionM(conf_mat,tag_list)

#%% Imagen cualitativa
X,a,a,a = sp.aviris_data_load()
image = modelo.predict(X.reshape([145*145,220]))
sp.draw_image(image,"4NN")
