# -*- coding: utf-8 -*-
## Clustering branch ##

#esto es una prueba a ver si se sube
import support_functions as sp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,DBSCAN,MeanShift
from sklearn.metrics import silhouette_samples, silhouette_score

# X e Y son matrices, XL y Yl están vectorizadas y sin etiquetas 0
X,Y,Xl,Yl = sp.aviris_data_load()

#Las etiquetas van de 1 a 16
n_clusters = np.arange(2,20)
X_reshape = X.reshape([145*145,220])
#%% Kmeans
# Entrenamos una serie de algoritmos kmeans con diferente número de clusters
for i in n_clusters:
    clstrs = KMeans(n_clusters = i).fit(Xl)
    
    
    clstrs_predictions = clstrs.predict(X_reshape)
    clstrs_Y = clstrs_predictions.reshape([145,145,1])
    #class_map es una función en support functions que da una imagen 
    #del suelo donde aparece la clase ( o el cluster ) de cada pixel
    #Obtenemos las gráficas de Shilouette
    sp.class_map(X_reshape,clstrs_Y)




#Obtenemos las puntuaciones de Silhouette (Tarda un buen rato, aviso)
#clstrs_shilouette = [silhouette_score(X, pred.reshape([145*145]), metric='euclidean') for pred in clstrs_Y]
#Obtenemos las gráficas de Shilouette
#[sp.draw_silhouette(X,pred.reshape([145*145])) for pred in clstrs_Y]
#%% DBSCANfor i in [0.1,1,10,100]:#n_clusters:
for i in [720,730,740,750]:
    for j in [50,60,70]:
        clstrs = DBSCAN(eps=i,min_samples=j).fit(X_reshape).labels_
        max(clstrs)
        sum(clstrs == -1)
        
        clstrs_Y = clstrs.reshape([145,145])
        #class_map es una función en support functions que da una imagen 
        #del suelo donde aparece la clase ( o el cluster ) de cada pixel
        #Obtenemos las gráficas de Shilouette
        sp.class_map(X_reshape,clstrs_Y,tag = ["DBSCAN",i,j])
#%% Mean shift
clstrs = MeanShift().fit_predict(Xl)
clstrs_Y = clstrs.reshape([145,145])
sp.class_map(X_reshape,clstrs_Y)