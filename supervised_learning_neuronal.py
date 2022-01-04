# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:50:43 2022

@author: morte
"""

#%% Reducción de datos
import support_functions as sp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,SpectralClustering
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix , cohen_kappa_score, roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn import metrics

#carga de datos

#%%
X,Y,Xl,Yl = sp.aviris_data_load()

n_tags = max(Yl)
ratio = 5000/Xl.shape[0]
#separamos según etiqueta
tag_list = np.arange(1,n_tags+1)#lista de etiquetas
tag_index_list = [np.where(Yl == i) for i in tag_list]#lista con las posiciones de cada etiqueta
Y_index_sample = [Yl[indx] for indx in tag_index_list]#subconjuntos de Yl y Xl de cada etiqueta
X_index_sample= [Xl[indx,:] for indx in tag_index_list]


#preparamos los vectores reducidos
X_reduced = []
Y_reduced = []

#Llenamos los libros reducidos
for i in range(len(tag_list)):
    
    data = X_index_sample[i][0,:,:]
    n_points_reduced = int(np.ceil(data.shape[0]*ratio))
    cluster = KMeans(n_points_reduced).fit(data)
    newdata = cluster.cluster_centers_.squeeze()
    
    X_reduced.append(newdata)
    Y_reduced.append(np.ones(n_points_reduced)*tag_list[i])

X_reduced = np.concatenate(X_reduced)
Y_reduced = np.concatenate(Y_reduced)

#%% Uso de red neuronal para clasificar
X_train, X_test, y_train, y_test = train_test_split(Xl, Yl, stratify=Yl,random_state=1)

brain = MLPClassifier(random_state = 100,max_iter = 300).fit(X_train,y_train)
pred = brain.predict_proba(X_test)

pred_tag = [np.where(i == (np.max(i)))[0][0] for i in pred]
conf_mat= metrics.confusion_matrix(y_test,pred_tag)
kappa= metrics.cohen_kappa_score(y_test,pred_tag)
