# -*- coding: utf-8 -*-
## Clustering branch ##

#esto es una prueba a ver si se sube
import support_functions as sp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,DBSCAN,MeanShift,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.ndimage import median_filter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
# X e Y son matrices, XL y Yl están vectorizadas y sin etiquetas 0
X,Y,Xl,Yl = sp.aviris_data_load()

#Las etiquetas van de 1 a 16
n_clusters = np.arange(2,20)
X_reshape = X.reshape([145*145,220])
#%% Kmeans
# Entrenamos una serie de algoritmos kmeans con diferente número de clusters
for i in n_clusters:
    clstrs = KMeans(n_clusters = 9,random_state = 100 ).fit(X_reshape)
    
    
    clstrs_predictions = clstrs.predict(X_reshape)
    clstrs_Y = clstrs_predictions.reshape([145,145])
    clstrs_Y_m = median_filter(clstrs_Y,3)
    #class_map es una función en support functions que da una imagen 
    #del suelo donde aparece la clase ( o el cluster ) de cada pixel
    #Obtenemos las gráficas de Shilouette
    sp.class_map(X_reshape,clstrs_Y)
    sp.class_map(X_reshape,clstrs_Y_m)



#Obtenemos las puntuaciones de Silhouette (Tarda un buen rato, aviso)
#clstrs_shilouette = [silhouette_score(X, pred.reshape([145*145]), metric='euclidean') for pred in clstrs_Y]
#Obtenemos las gráficas de Shilouette
#[sp.draw_silhouette(X,pred.reshape([145*145])) for pred in clstrs_Y]
#%% DBSCANfor i in [0.1,1,10,100]:#n_clusters:
#for i in [710,730,740,750]:
#    for j in [50,60,70]:
#        i,j=[700,20]
#        clstrs = DBSCAN(eps=i,min_samples=j).fit(X_reshape).labels_
#        
#        clstrs_Y = clstrs.reshape([145,145])
#        #class_map es una función en support functions que da una imagen 
#        #del suelo donde aparece la clase ( o el cluster ) de cada pixel
 #       #Obtenemos las gráficas de Shilouette
#        sp.class_map(X_reshape,clstrs_Y,tag = ["DBSCAN",i,j])
        
#%% Mean shift, no usar es muy lento
#clstrs = MeanShift().fit_predict(X_reshape)
#clstrs_Y = clstrs.reshape([145,145])
#sp.class_map(X_reshape,clstrs_Y)
#%% Spectral clustering
#for i in n_clusters:
#    clstrs =AgglomerativeClustering(n_clusters = 9).fit(X_reshape).labels_#
#
#    clstrs_Y = clstrs.reshape([145,145])
#    clstrs_Y_m = median_filter(clstrs_Y,3)
#    #class_map es una función en support functions que da una imagen 
#    #del suelo donde aparece la clase ( o el cluster ) de cada pixel
#    #Obtenemos las gráficas de Shilouette
#    sp.class_map(X_reshape,clstrs_Y)
#    sp.class_map(X_reshape,clstrs_Y_m)

#%% Gaussian Mixture‘full’, ‘tied’, ‘diag’, ‘spherical’
aic_list = list()
bic_list = list()

for i in n_clusters:
    for j in ["full","tied","diag","spherical"]:
        clstrs =GaussianMixture(n_components = i,covariance_type = j)
        
        
        
        clstrs_Y = clstrs.fit_predict(X_reshape).reshape([145,145])
        clstrs_Y_m = median_filter(clstrs_Y,3)
        #class_map es una función en support functions que da una imagen 
        #del suelo donde aparece la clase ( o el cluster ) de cada pixel
        #Obtenemos las gráf icas de Shilouette
        sp.class_map(X_reshape,clstrs_Y,tag = ["Gaussianas",i,j])
        #sp.class_map(X_reshape,clstrs_Y_m)
        aic_list.append(clstrs.aic(X_reshape))
        bic_list.append(clstrs.bic(X_reshape))

fig,(aic,bic) = plt.subplots(1,2)
aic.plot(n_clusters[0:13],aic_list[0:len(aic_list):4],label = "full")
aic.plot(n_clusters[0:13],aic_list[1:len(aic_list):4],label = "tied")
aic.plot(n_clusters[0:13],aic_list[2:len(aic_list):4],label = "diag")
aic.plot(n_clusters[0:13],aic_list[3:len(aic_list):4],label = "spherical")
aic.set_xlabel("AIC")
aic.set_ylabel("Cluster num")
aic.legend()

bic.plot(n_clusters[0:13],bic_list[0:len(aic_list):4],label = "full")
bic.plot(n_clusters[0:13],bic_list[1:len(aic_list):4],label = "tied")
bic.plot(n_clusters[0:13],bic_list[2:len(aic_list):4],label = "diag")
bic.plot(n_clusters[0:13],bic_list[3:len(aic_list):4],label = "spherical")
bic.legend()
plt.show
#%% hierarchical clustering
clstrs = linkage(X_reshape,"ward")
for i in n_clusters:
    
    clusters = fcluster(clstrs, i, criterion='maxclust')
    clstrs_Y = clusters.reshape([145,145])
    clstrs_Y_m = median_filter(clstrs_Y,3)
    #class_map es una función en support functions que da una imagen 
    #del suelo donde aparece la clase ( o el cluster ) de cada pixel
    #Obtenemos las gráf icas de Shilouette
    sp.class_map(X_reshape,clstrs_Y,tag = ["Jerárquico",i, ])

#%% Definitivos
clstrs = KMeans(n_clusters = 9,random_state = 100 ).fit(X_reshape)
clstrs_predictions = clstrs.predict(X_reshape)
clstrs_Y_K9 = clstrs_predictions.reshape([145,145])
clstrs_Y_m_K9 = median_filter(clstrs_Y_K9,2)

clstrs =GaussianMixture(n_components = 8,covariance_type = "spherical")
clstrs_Y_G8 = clstrs.fit_predict(X_reshape).reshape([145,145])
clstrs_Y_m_G8 = median_filter(clstrs_Y_G8,2)

clstrs =GaussianMixture(n_components = 18,covariance_type = "spherical")
clstrs_Y_G18 = clstrs.fit_predict(X_reshape).reshape([145,145])
clstrs_Y_m_G18 = median_filter(clstrs_Y_G18,2)

n_clusters = 9


#Parte donde se dibuja la imagen
cmap = cm.get_cmap('tab20c', n_clusters) 
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(clstrs_Y_K9,cmap = cmap)
ax1.set_title("KMeans 9 clases")
ax2.imshow(clstrs_Y_m_K9,cmap = cmap)
ax2.set_title("Mapa de clases filtrado")

#class_map es una función en support functions que da una imagen 
#del suelo donde aparece la clase ( o el cluster ) de cada pixel
#Obtenemos las gráficas de Shilouette

n_clusters = 8
cmap = cm.get_cmap('tab20c', n_clusters) 
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(clstrs_Y_G8,cmap = cmap)
ax2.imshow(clstrs_Y_m_G8,cmap = cmap)
ax1.set_title("Mixtura Gaussiana 8 clases")
ax2.set_title("Mapa de clases filtrado")

n_clusters = 18
cmap = cm.get_cmap('tab20c', n_clusters) 
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(clstrs_Y_G18,cmap = cmap)
ax2.imshow(clstrs_Y_m_G18,cmap = cmap)
ax1.set_title("Mixtura Gaussiana 19 clases")
ax2.set_title("Mapa de clases filtrado")



fig, (sh1,sh2,sh3) = plt.subplots(1,3)

samples = silhouette_samples(X_reshape,clstrs_Y_K9.reshape(145*145))
y_lower = 10
n_clusters = 9
silhouette_avg = silhouette_score(X_reshape,clstrs_Y_K9.reshape(145*145))
print(silhouette_avg)
for i in np.arange(0,n_clusters):
    it_sample = samples[clstrs_Y_G8.reshape(145*145) == i]
    it_sample.sort()
    
    size_it_sample = it_sample.shape[0]
    y_upper = y_lower + size_it_sample
    
    color = cm.nipy_spectral(float(i) / n_clusters)
    sh1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        it_sample,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,)

    sh1.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
    
    y_lower = y_upper+10

sh1.set_title("KMeans")
#sh1.set_xlabel("The silhouette coefficient values")
sh1.set_ylabel("Cluster label")
sh1.axvline(x=silhouette_avg, color="red", linestyle="--")   
sh1.set_yticks([])  # Clear the yaxis labels / ticks
sh1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

samples = silhouette_samples(X_reshape,clstrs_Y_G8.reshape(145*145))
y_lower = 10
n_clusters = 8
silhouette_avg = silhouette_score(X_reshape,clstrs_Y_G8.reshape(145*145))
print(silhouette_avg)
for i in np.arange(0,n_clusters):
    it_sample = samples[clstrs_Y_G8.reshape(145*145) == i]
    it_sample.sort()
    
    size_it_sample = it_sample.shape[0]
    y_upper = y_lower + size_it_sample
    
    color = cm.nipy_spectral(float(i) / n_clusters)
    sh2.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        it_sample,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,)

    sh2.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
    
    y_lower = y_upper+10

sh2.set_title("Gaussian Mixture")
sh2.set_xlabel("Silhouette coefficients")
sh2.set_ylabel("Cluster label")
sh2.axvline(x=silhouette_avg, color="red", linestyle="--")   
sh2.set_yticks([])  # Clear the yaxis labels / ticks
sh2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

n_clusters = 18
samples = silhouette_samples(X_reshape,clstrs_Y_G8.reshape(145*145))
y_lower = 10
n_clusters = 9
silhouette_avg = silhouette_score(X_reshape,clstrs_Y_G8.reshape(145*145))
print(silhouette_avg)
for i in np.arange(0,n_clusters):
    it_sample = samples[clstrs_Y_G8.reshape(145*145) == i]
    it_sample.sort()
    
    size_it_sample = it_sample.shape[0]
    y_upper = y_lower + size_it_sample
    
    color = cm.nipy_spectral(float(i) / n_clusters)
    sh3.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        it_sample,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,)

    sh3.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
    
    y_lower = y_upper+10

sh3.set_title("Gaussian Mixture")
#sh3.set_xlabel("The silhouette coefficient values")
sh3.set_ylabel("Cluster label")
sh3.axvline(x=silhouette_avg, color="red", linestyle="--")   
sh3.set_yticks([])  # Clear the yaxis labels / ticks
sh3.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])