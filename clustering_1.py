# -*- coding: utf-8 -*-
## Clustering branch ##

#esto es una prueba a ver si se sube
import support_functions as sp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
print('Hello world')

# X e Y son matrices, XL y Yl están vectorizadas y sin etiquetas 0
X,Y,Xl,Yl = sp.aviris_data_load()

#Las etiquetas van de 1 a 16
n_clusters = np.arange(2,20)

clstrs = [KMeans(n_clusters = i) for i in n_clusters]
clstrs_fit = [model.fit(Xl) for model in clstrs]
clstrs_score = [model.score(Xl) for model in clstrs_fit]

figure, splot = plt.subplots()
splot.plot(n_clusters,clstrs_score)
