# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 19:00:24 2022

@author: franc
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import support_functions as sp
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#carga de datos
X,Y,Xl,Yl = sp.aviris_data_load()

#%%
X_train, X_test, y_train, y_test = train_test_split(Xl, Yl,random_state=1)

extra_tree = ExtraTreeClassifier(random_state=0, splitter ='best' )

clf = extra_tree.fit(X_train, y_train)

score = clf.score(X_test, y_test)

features = clf.feature_importances_



#%%

plt.plot(features)
plt.xlabel('Nº de bandas')
plt.ylabel('Importancia')


#%% SVM
gammas = np.logspace(-2, 2, 10)
Cs = np.logspace(-2, 2, 10)
tuned_parameters = { 'gamma': gammas,'C': Cs}    


clf2 = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=5,n_jobs=-1,verbose=0)

clf2.fit(X_train,y_train)
clf2=clf.best_estimator_

#%%
print(clf2)
print('OA train %0.2f' % clf2.score(X_train, y_train)) 
print('OA test %0.2f' % clf2.score(X_test, y_test))
preds_test = clf2.predict(X_test)
print('Kappa test %0.2f' % metrics.cohen_kappa_score(y_test,preds_test))