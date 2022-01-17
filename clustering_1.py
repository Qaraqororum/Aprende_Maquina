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


#%% Definitivos
clstrs = KMeans(n_clusters = 9,random_state = 100 ).fit(X_reshape)
clstrs_predictions = clstrs.predict(X_reshape)
clstrs_Y_K9 = clstrs_predictions

clstrs =GaussianMixture(n_components = 8,covariance_type = "spherical")
clstrs_Y_G8 = clstrs.fit_predict(X_reshape)

clstrs =GaussianMixture(n_components = 18,covariance_type = "spherical")
clstrs_Y_G18 = clstrs.fit_predict(X_reshape)


sp.draw_image(clstrs_Y_K9+1,"Kmeans ")
sp.draw_image(clstrs_Y_G8+1,"Mixtura Gaussiana ")
sp.draw_image(clstrs_Y_G18+1,"Mixtura Gaussiana ")

sp.draw_silhouette(X_reshape, clstrs_Y_K9, clstrs_Y_G8, clstrs_Y_G18)



