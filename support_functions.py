# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 13:07:55 2021

@author: morte
"""


# coding: utf-8

import numpy as np
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt
from pylab import *


from sklearn.metrics import silhouette_samples, silhouette_score

def aviris_data_load():
        
    # Lectura de la imagen de fichero de Matlab .mat
    mat_file =  "datasetB1.mat"
    mat = matlab.loadmat(mat_file,squeeze_me=True) #devuelve un dictionary
    list(mat.keys()) #variables almacenadas
    
    
    # Lectura de los datos
    X = mat["X"]   #imagen (hipercubo 3D: filas x columnas x variables) 
    Xl = mat["Xl"]   #muestras etiquetadas (muestas x variables) 
    Yl = mat["Yl"]   #etiquetas de clases (muestras x 1, 0=sin clase)   
    del mat
    Yl.shape
    
    
    # Reshape del Ground Truth como una imagen
    Y = np.reshape(Yl, (X.shape[0], X.shape[1]),order="F")
    Y.shape
    
    
    # Filter background: eliminamos la clase 0 de los datos etiquetados
    Nc=Yl.max()-Yl.min()+1
    if Nc>2:
        Xl = Xl[Yl != 0,:];
        Yl = Yl[Yl != 0];
    
    return([X,Y,Xl,Yl])

    
def class_map(X_,image):
    cluster_labels = image.reshape([145*145])
    n_clusters = np.max(cluster_labels)+1
    
    #Parte donde se dibuja la imagen
    cmap = cm.get_cmap('tab20c', n_clusters) 
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    plt.imshow(image,cmap = cmap)
    plt.colorbar()
    plt.title("Kmeans "+str(n_clusters)+" clusters")
    
    #Galimatías donde se dibujan los silhouette
    samples = silhouette_samples(X_,cluster_labels)
    y_lower = 10
    
    silhouette_avg = silhouette_score(X_, cluster_labels)
    for i in np.arange(0,n_clusters):
        it_sample = samples[cluster_labels == i]
        it_sample.sort()
        
        size_it_sample = it_sample.shape[0]
        y_upper = y_lower + size_it_sample
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            it_sample,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,)
    
        ax1.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
        
        y_lower = y_upper+10
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")   
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    return(0)

def draw_silhouette(X_,cluster_labels):
    n_clusters = np.max(cluster_labels.reshape([145*145]))+1
    samples = silhouette_samples(X_,cluster_labels)
    y_lower = 10
    
    fig,ax=plt.subplots()
    silhouette_avg = silhouette_score(X_, cluster_labels)
    for i in np.arange(2,n_clusters):
        it_sample = samples[cluster_labels == i]
        it_sample.sort()
        
        size_it_sample = it_sample.shape[0]
        y_upper = y_lower + size_it_sample
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            it_sample,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,)
    
        ax.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
        
        y_lower = y_upper+10
    
    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")   
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])   