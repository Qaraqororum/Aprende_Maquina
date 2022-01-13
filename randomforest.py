# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:27:08 2022

@author: franc
"""

import numpy as np
import matplotlib.pyplot as plt
import support_functions as sp

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import  confusion_matrix , cohen_kappa_score
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



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
    newdata = cluster.cluster_centers_.squeeze()
    
    X_reduced.append(newdata)
    Y_reduced.append(np.ones(n_points_reduced)*tag_list[i])

X_reduced = np.concatenate(X_reduced)
Y_reduced = np.concatenate(Y_reduced)



#%% 1 Sólo árbol

criterion = 'gini'
max_depth = None
report = metrics.confusion_matrix
report2 = metrics.cohen_kappa_score

Xtr, Xts, ytr, yts = train_test_split(X_reduced, Y_reduced, test_size=0.3)


tree = DecisionTreeClassifier(criterion=criterion)
tree.fit(Xtr, ytr)
ytree = tree.predict(Xts)

# Evaluate and compare results
print('Confusion Matrix:  ', report(yts, ytree),'Cohen-kappa', report2(yts, ytree))

Conf_matrix_tree = report(yts, ytree)
Kappa_tree = report2(yts, ytree)



#%% Plot and save confusion matrix
metrics = [Conf_matrix_tree, Conf_matrix_RF]
for i in metrics:

    ax = sns.heatmap(i, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    

#%% SVM
gammas = np.logspace(-2, 2, 10)
Cs = np.logspace(-2, 2, 10)
tuned_parameters = { 'gamma': gammas,'C': Cs}    


SVM = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=5,n_jobs=-1,verbose=1)

SVM.fit(Xtr,ytr)
SVM=SVM.best_estimator_

#%%
print(SVM)
print('OA train %0.2f' % SVM.score(Xtr, ytr)) 
print('OA test %0.2f' % SVM.score(Xts, yts))
preds_test = SVM.predict(Xts)
print('Kappa test %0.2f' % metrics.cohen_kappa_score(yts,preds_test))