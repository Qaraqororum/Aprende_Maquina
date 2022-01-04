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

#carga de datos
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