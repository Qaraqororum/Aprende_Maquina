# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:34:34 2022

@author: morte
"""

#%% Reducción de datos
import support_functions as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix , cohen_kappa_score, roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from pylab import *

#carga de datos

#%% Preparación de los datos
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
    centers = cluster.cluster_centers_.squeeze()
    
    newdata = list()
    
    #Seleccionamos el punto más cercano al centroide para evitar distorisionar la distribución de los puntos
    for z in range(centers.shape[0]):
        d = [np.linalg.norm(data[j,:]-centers[z,:]) for j in range(data.shape[0])]
        indx = d.index(min(d))
        newdata.append(data[indx])
        
        
    X_reduced.append(newdata)
    Y_reduced.append(np.ones(n_points_reduced)*tag_list[i])

X_reduced = np.concatenate(X_reduced)
Y_reduced = np.concatenate(Y_reduced)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y_reduced, stratify=Y_reduced,random_state=1)
#%% KNN

modelo = KNeighborsClassifier(n_neighbors=n_tags,
                              weights='distance',
                              algorithm='auto',
                              leaf_size=40, p=2,
                              metric='cosine',
                              metric_params=None,
                              n_jobs=None).fit(X_train,y_train)

pred = modelo.predict(X_test)
score = modelo.predict_proba(X_test)

conf_mat= metrics.confusion_matrix(y_test,pred)
kappa= metrics.cohen_kappa_score(y_test,pred)
sp.draw_ROC(y_test,score,tag_list)

ax = sns.heatmap(conf_mat, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(tag_list)
ax.yaxis.set_ticklabels(tag_list)

## Display the visualization of the Confusion Matrix.
plt.show()
print(kappa)

#%% Imagen cualitativa
image = modelo.predict(X.reshape([145*145,220])).reshape([145,145])
cmap = cm.get_cmap('tab20c', 15) 

fig, (ax1) = plt.subplots()
plt.imshow(image,cmap = cmap)
plt.colorbar()
plt.title("Red neuronal "+str(16)+" clases")