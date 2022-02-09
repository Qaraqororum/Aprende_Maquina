# Ejercicio B1
Miguel Hortelano Busto y Francisco Olayo González Rodríguez 

Como para estos clústerings resultantes, al no tener un conjunto de etiquetas verdaderas "Ground Truth", utilizamos métricas que no necesiten de ella. En nuestro caso optaremos por los coeficientes de siluteta (silhouette). También es oportuno

Carga de paquetes a utilizar en el entregable.


```python
import numpy as np
import matplotlib.pyplot as plt
import support_functions as sp
import seaborn as sns
import pandas as pd
import matplotlib as mpl

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import  confusion_matrix , cohen_kappa_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.mixture import GaussianMixture
```

### Ejercicio 1
Usar algún algoritmo de clustering sobre todos los datos sin emplear las etiquetas para obtener una primera clasificación. Definir una estrategia para clasificar nuevos ejemplos y obtener el mapa de clasificación final de toda la imagen. Obviamente, al no haber empleado las etiquetas de las clases puede que nuestro mapa de clasificación basado en clustering no tenga mucha relación con las clases predefinidas.

Al no tener un conjunto de etiquetas con las que comparar los resultados de nuestro clustering recurriremos a métodos que evaluen la cohesión de nuestros clústers, como es el método silhouette. Obtenemos una puntuación entre 1 y -1, cuanto más alto sea este valor más acorde con el clúster asignado estará. 

Tras probar varíos métodos de clústering llegamos a las siguientes conclusiones:
-  Los métodos probados con mejores puntuaciones de silhouette son el kmeans y la mixtura de gaussianas con covarianza esférica. Esto tiene sentido, pues ambos métodos funcionan bien con distribuciones de puntos que se agrupan en hiperesferas gaussianas. Lo que por otro lado nos da una idea de como se estructuran los datos.

-  Para pocas clases ambos métodos dan clases similares y puntuaciones algo mejores para kmeans. Mientras si utilizamos números altos de clústers, alrededor del número de clases que tiene el conjunto original, las mixturas gaussianas sobresalen mucho más (aún incluso teniendo peores puntuaciones que los casos con menos clústers). 

-  Las puntuaciones del índice de silhouette medio son bajas. Este ha sido calculado tomando la distáncia euclidea. Si damos un vistazo a los datos vemos que tenemos randiancias espectrales con valores altos y sin normalizar, por lo que el gran número de datos y su rango de valores puede influir en la interpretación de las distancias. Tal vez estas mismas clases teniendo en cuenta la emisión de la superfície en vez de la radiancia, menos características o con otras unidades dieran mejores medidas de coherencia.

-  Aumentar el número de clústers empeora su cohesión, icluso alcanzando a obtener más puntuaciones negativas que positivas en algunos clústers, esto es indicativo de un número de clústers incorrecto. Por ello sería ideal quedarnos con 8 o 9 clústers que dibujan una imagen más o menos interpretable en la que se ven parcelas de diferentes etiquetas y se distingue también el espacio que las separa.


```python
# X e Y son matrices, XL y Yl están vectorizadas y sin etiquetas 0
X,Y,Xl,Yl = sp.aviris_data_load()

#Las etiquetas van de 1 a 16
n_clusters = np.arange(2,20)
X_reshape = X.reshape([145*145,220])

#%% Definitivos
clstrs = KMeans(n_clusters = 9,random_state = 100 ).fit(X_reshape)#Kmeans 9 clusters
clstrs_predictions = clstrs.predict(X_reshape)
clstrs_Y_K9 = clstrs_predictions

clstrs =GaussianMixture(n_components = 8,covariance_type = "spherical")#Mixtura 8 clusters
clstrs_Y_G8 = clstrs.fit_predict(X_reshape)

clstrs =GaussianMixture(n_components = 18,covariance_type = "spherical")#Mixtura 18 clusters
clstrs_Y_G18 = clstrs.fit_predict(X_reshape)

sp.draw_image(clstrs_Y_K9,"Kmeans")#función definida que pinta un mapa de las clases
sp.draw_image(clstrs_Y_G8,"Gaussian Mixture")
sp.draw_image(clstrs_Y_G18,"Gaussian Mixture")

sp.draw_silhouette_plus(X_reshape,clstrs_Y_K9,clstrs_Y_G8,clstrs_Y_G18)#función definida que dibuja los silhuette de los 3 casos
```


    
![png](Entregable_final_ML_files/Entregable_final_ML_6_0.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_6_1.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_6_2.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_6_3.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_6_4.png)
    


### Ejercicio 2
Para aplicar algoritmos de aprendizaje supervisado simularemos la escasez de muestras etiquetadas empleando solo 5000 muestras con sus etiquetas. Sin embargo, la selección de estas muestras no la vamos a hacer de forma aleatoria sino que vamos a emplear algoritmos de clustering para reducir el número de muestras pero preservando la distribución de los datos de entrada y de las clases definidas.  Primero, separar los datos en función de su etiqueta de clase. En cada subconjunto, aplicar algún algoritmo de clustering y definir una estrategia para obtener un subconjunto reducido que sea representativo del conjunto inicial.

Con intención de reducir los datos manteniendo la estructura de las clases separaremos primero los datos en subconjuntos de píxeles según su etiqueta. Luego a cada subconjunto le aplicaremos un kmeans para obtener agrupaciones de las que obtendremos los nuevos puntos de datos. La elección del Kmeans se debe a que agrupa los datos en clústers con un número de puntos parecidos de forma que mantiene la coherencia de distribución de puntos. Utilizaremos el número de clústers que mantendrá la proporción de píxeles en el conjunto original y en el nuevo. Extraemos los centroides de estos clústers, pero no los usaremos directamente, pues podemos obtener puntos que se alejen de la distribución original (si la distribución fuera una circunferencia obtendríamos un punto en el centro, donde no hay nada). En su lugar seleccionamos el punto más cercano al centroide.

Repitiendo el proceso en todas las clases  y agrupamos los puntos resultantes en el nuevo conjunto que mantiene la proporción de puntos de cada clase y la distribución de los puntos.


```python
def data_reducer():
    X,Y,Xl,Yl = sp.aviris_data_load()

    n_tags = max(Yl)
    ratio = 5000/Xl.shape[0]
    #separamos según etiqueta
    tag_list = np.arange(1,n_tags+1)#lista de etiquetas
    tag_index_list = [np.where(Yl == i) for i in tag_list]#lista con las posiciones de cada etiqueta
    #subconjuntos de Yl y Xl de cada etiqueta
    X_index_sample= [Xl[indx,:] for indx in tag_index_list]
    
    
    #preparamos los vectores reducidos
    X_reduced = []
    Y_reduced = []
    
    #Llenamos los libros reducidos
    for i in range(len(tag_list)):
        
        #hacemos clústering del subconjunto con etiqueta i
        data = X_index_sample[i][0,:,:]
        n_points_reduced = int(np.ceil(data.shape[0]*ratio))#aseguramos que se mantenga la proporción de puntos por clase
        cluster = KMeans(n_points_reduced).fit(data)
        centers = cluster.cluster_centers_.squeeze()#obtenemos los centros
        
        newdata = list()
        
        #Seleccionamos el punto más cercano al centroide para evitar distorisionar la distribución de los puntos
        for z in range(centers.shape[0]):
            d = [np.linalg.norm(data[j,:]-centers[z,:]) for j in range(data.shape[0])]
            indx = d.index(min(d))
            newdata.append(data[indx])
            
            
        X_reduced.append(newdata)
        Y_reduced.append(np.ones(n_points_reduced)*tag_list[i])
    
    #Pegamos los resultados uno tras otro
    X_reduced = np.concatenate(X_reduced)
    Y_reduced = np.concatenate(Y_reduced)
    
    return((X_reduced,Y_reduced))

X_reduced, Y_reduced = data_reducer()
```

### Ejercicio 3 
Utilizar algoritmos supervisados sobre el conjunto reducido de entrenamiento para obtener la clasificación de la imagen. Se debe obtener una aproximación del error de clasificación en test utilizando una partición del conjunto de entrenamiento. Se pueden obtener también resultados cualitativos representado las imágenes de clasificación.

En este apartado, hemos probado varios clasificadores supervisados, y al final hemos decidido utilizar un árbol clasificador, y un KNN (vecinos próximos).
El algoritmo de KNN es el que mejores resultados ofrece en principio, teniendo en cuenta el índice de Cohen-Kappa e imágenes de clasificación. En cambio el algoritmo de árbol de clasificación, no funciona demasiado bien, pero nos sirve para ir viendo una mejora al reducir, posteriormente, las bandas o características del dataset o utilizar ensembles como en el último apartado.

Para evaluar los resultados de los modelos y poder comparar con otros hemos utilizado el índice cohen-kappa, que es una métrica más robusta que solo evaluar el porcentaje de acierto, ya que tiene en cuenta las posibles clasificaciones por azar, en caso de tener clasificadores que ofrezcan etiquetas de manera aleatoria.

También hemos escogido matrices de confusión, que son herramientas muy utilizadas en Machine Learning, para poder visualizar rapidamente lo bien o mal que clasifica un modelo, e identificar que clases están siendo conflictivas a la hora de agrupar.
Además de estas dos anteriores también hemos representado curvas ROC, para apoyar los resultados de las matrices de confusión, que al igual que estas tiene en cuenta los datos bien clasificados, pero también falsos positivos y negativos.

En el diagnóstico del modelo del árbol, mirando con detenimiento la matriz de confusión vemos que no hay un comportamiento uniforme. Las clases del 10 al 12, siendo las más numerosas de la imagen, tienden a estar favorecidas, quitando puntos sobretodo a las clases del 1 al 5. También vemos que algunas clases tienen demasiados pocos valores en test como para observar bien su clasificación, es el caso de la 7, 9 o 1. Ninguna de ellas supera los 5 píxels. Esto se ve mejor en las curvas roc, estas dibujan un perfil muy parecido para todas las clases excepto para un pequeño grupo con el mismo problema de las anteriormente citadas. En estas se ve como estas clases tienden a tener muy pocos falsos positivos, por lo que parece que son ideales, cuando lo que pasa es que los falsos negativos están enmascarados en otras curvas con muchos más puntos disponibles. El resto de curvas tienen un comportamiento esperable y no demasiado satisfactorio (aunque correcto distan de ser ideales).


```python
## Tree classifier ##

tag_list = np.arange(1,17)

criterion = 'gini'
max_depth = None

Xtr, Xts, ytr, yts = train_test_split(X_reduced, Y_reduced, test_size=0.3, random_state=100)


tree = DecisionTreeClassifier(criterion=criterion, random_state = 100, max_depth=7)
tree.fit(Xtr, ytr)
ytree = tree.predict(Xts)

# Evaluate and compare results
print('Cohen-kappa', cohen_kappa_score(yts, ytree))

Conf_matrix_tree = confusion_matrix(yts, ytree)
Kappa_tree = cohen_kappa_score(yts, ytree)
sp.draw_ROC(yts,tree.predict_proba(Xts),tag_list)
sp.draw_ConfusionM(Conf_matrix_tree,tag_list)


X,a,a,a = sp.aviris_data_load()
image = tree.predict(X.reshape([145*145,220]))
sp.draw_image(image,"Árbol de clasificación")
```

    Cohen-kappa 0.5540410385042788
    


    
![png](Entregable_final_ML_files/Entregable_final_ML_13_1.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_13_2.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_13_3.png)
    


Se puede observar como el índice Cohen-kappa es moderadamente bajo (0.50), y tanto en la matriz de confusión como en la imágen de la clasificación, se ve claramente como hay clases que están mal caracterizadas y producen bastante ruido en la clasificación.

Hemos elegido max_depth = 7 tras evaluar los resultados de precisión global (overall accuracy) para conjunto de entrenamiento y test, y hemos escogido el punto de parada para que no estuviese sobreentrenado.

Para probar un método algo más complejo y con mejores resultados hemos utilizado un Knn con distancia de coseno (la que mejor resultados daba) y sin pesos (para no sobreentrenar). Justo para evitar este sobreentrenamiento vemos en el siguiente bloque una evaluación del sobreajuste con el número de vecinos empleados. Hemos seleccionado de manera subjetiva el número de vecinos para el que la distancia entre el kappa de entrenamiento y test no es demasiado grande sin empeorar el resultado final.


```python
Xtr, Xts, ytr, yts = train_test_split(X_reduced, Y_reduced, test_size=0.3,random_state = 150)
#%% KNN
tag_list = np.arange(1,17)

kappa_train = []
kappa_test = []
nn = np.arange(2,20)

# entrenamos modelos para cada numero de vecinos y preparamos una gráfica comparativa
for i in nn:
    modelo = KNeighborsClassifier(n_neighbors=i,
                                  weights='uniform',
                                  algorithm='auto',
                                  leaf_size=40, p=2,
                                  metric='cosine',
                                  metric_params=None,
                                  n_jobs=-1).fit(Xtr,ytr)
    pred = modelo.predict(Xts)
    kappa_test.append(cohen_kappa_score(yts,pred))
    pred = modelo.predict(Xtr)
    kappa_train.append(cohen_kappa_score(ytr,pred))


figure, (kppa) = plt.subplots(1)
kppa.plot(nn,kappa_train,label = "Entrenamiento")
kppa.plot(nn,kappa_test,label = "testeo")
kppa.axvline(x=15, color="red", linestyle="--")
kppa.set_ylabel("Kappa")
kppa.set_xlabel("Número de vecinos")
kppa.set_title("")
plt.legend()
```




    <matplotlib.legend.Legend at 0x1f600676f40>




    
![png](Entregable_final_ML_files/Entregable_final_ML_16_1.png)
    


El resultado es mucho mejor al obtenido con el árbol, notándose en el perfil de las curvas ROC más cercanas al ideal, un mayor kappa y clases menos confundidas (aunque sigue siendo favorecida la clase 11). La imagen final también se ve mejorada, con clases más uniformes y parcelas mejor definidas que en el ejemplo anterior. Aún así persisten los problemas de balanceo de clases.


```python
modelo = KNeighborsClassifier(n_neighbors=12,
                                  weights='uniform',
                                  algorithm='auto',
                                  leaf_size=40, p=2,
                                  metric='cosine',
                                  metric_params=None,
                                  n_jobs=-1).fit(Xtr,ytr)
pred = modelo.predict(Xts)
score = modelo.predict_proba(Xts)

conf_mat= confusion_matrix(yts,pred)
kappa= cohen_kappa_score(yts,pred)
print('Cohen-kappa', kappa)
    
sp.draw_ROC(yts,score,tag_list)

sp.draw_ConfusionM(conf_mat,tag_list)

pred = modelo.predict(Xtr)
kappa = cohen_kappa_score(ytr,pred)

#%% Imagen cualitativa
X,a,a,a = sp.aviris_data_load()
image = modelo.predict(X.reshape([145*145,220]))
sp.draw_image(image,"12NN")
```

    Cohen-kappa 0.7048682498736587
    


    
![png](Entregable_final_ML_files/Entregable_final_ML_18_1.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_18_2.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_18_3.png)
    


### Ejercicio 4
La imagen puede contener bandas con datos erróneos, o con bandas que tengan poca relevancia en el resultado de la clasificación. Utilizando métodos de clasificación que proporcionen un 'ranking' de características, intentar detectar y eliminar aquellas bandas que empeoran (o no mejoran) el resultado de la clasificación. De nuevo, dar un resultado cuantitativo del error en test, y cualitativo mediante la representación de mapas de clasificación.


```python
## Entrenamiento del modelo para conseguir un ranking de características ##


X_train, X_test, y_train, y_test = train_test_split(Xl, Yl, random_state=100)

extra_tree = DecisionTreeClassifier(random_state=0, criterion = criterion, splitter ='best' )

clf = extra_tree.fit(X_train, y_train)

score = clf.score(X_test, y_test)

features = clf.feature_importances_

plt.plot(features)
plt.xlabel('Nº de bandas')
plt.ylabel('Importancia')


# Reducción de bandas o características


indice= np.where(features< 0.01)

X_reduced2 = np.delete(X_reduced, indice, axis=1)
X_2 = np.delete(X.reshape([145*145,220]), indice, axis=1)

```


    
![png](Entregable_final_ML_files/Entregable_final_ML_20_0.png)
    


Para reducir las bandas o características menos importantes, hemos elegido un modelo de clasificador de árbol ya que lleva incluido el método feature_importances, que es bastante útil a la hora de ver qué bandas y cómo de importante son.
Por ello hemos decidido utilizar un umbral de 0.01, tras varias pruebas, reduciendo el dataset de 220 bandas a 16. 
A continuación procedemos a evaluar los resultados de la reducción de bandas, comparando un árbol con el del ejercicio 3.


```python
## Tree classifier ##

tag_list = np.arange(1,17)

criterion = 'gini'
max_depth = None

Xtr, Xts, ytr, yts = train_test_split(X_reduced2, Y_reduced, test_size=0.3, random_state=100)


tree = DecisionTreeClassifier(criterion=criterion, max_depth = 8)
tree.fit(Xtr, ytr)
ytree = tree.predict(Xts)

# Evaluate and compare results
print('Cohen-kappa', cohen_kappa_score(yts, ytree))

Conf_matrix_tree = confusion_matrix(yts, ytree)
Kappa_tree = cohen_kappa_score(yts, ytree)
sp.draw_ROC(yts,tree.predict_proba(Xts),tag_list)
sp.draw_ConfusionM(Conf_matrix_tree,tag_list)


image = tree.predict(X_2)
sp.draw_image(image,"Árbol de clasificación")

```

    Cohen-kappa 0.5462387444109111
    


    
![png](Entregable_final_ML_files/Entregable_final_ML_22_1.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_22_2.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_22_3.png)
    


Tras evaluar los resultados de un árbol clasificador, con el dataset reducido en valores y bandas, observamos como practicamente no mejora los resultados obtenidos en el Ejercicio 3, pero al reducir mucho la dimensión conseguimos reducir la carga computacional de los modelos.

### Ejercicio 5
Por último, empleando solo el conjunto reducido de muestras etiquetadas y las bandas seleccionadas, utilizar alguna estrategia de combinación de clasificadores (métodos 'ensemble') para intentar mejorar los resultados de la clasificación.


```python
## Random forest (ensembles) ##

criterion = 'gini'
max_depth = None


Xtr, Xts, ytr, yts = train_test_split(X_reduced2, Y_reduced, test_size=0.3)
# Train RF ensemble
rf = RandomForestClassifier(n_estimators=100, criterion=criterion, random_state=100, max_depth= 8)
rf.fit(Xtr, ytr)
yb = rf.predict(Xts)

# Evaluate and compare results
print('Cohen-kappa',cohen_kappa_score(yts, yb))

Conf_matrix_RF = confusion_matrix(yts, yb)
Kappa_RF = cohen_kappa_score(yts, yb)

# Observamos como con el conjunto reducido de bandas, dejando solo 16 bandas, el ensemble funciona bastante bien con un
# cohen-kappa de ~0.74
tag_list = np.arange(1,17)
score = rf.predict_proba(Xts)
sp.draw_ROC(yts,score,tag_list)

sp.draw_ConfusionM(Conf_matrix_RF,tag_list)

X_ = np.delete(X.reshape([145*145,220]), indice, axis=1)
sp.draw_image(rf.predict(X_2),"Random Forest ")

```

    Cohen-kappa 0.6170260245405439
    


    
![png](Entregable_final_ML_files/Entregable_final_ML_25_1.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_25_2.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_25_3.png)
    


En este último apartado, hemos probado a hacer un RandomForest y un modelo Adaboosting, para comparar resultados, y observamos como el RandomForest mejor bastante los resultados de los ejercicios anteriores (árboles), menos a KNN que sigue siendo mejor, por ello hemos probado otros métodos ensembles para tratar de llegar a mejores conclusiones.

Con mejores resultados tenemos el adaboost. Pese a los persistentes problemas de balanceo de clases da unas curvas ROC aceptables que repiten el comportamiento de los anteriores apartados así como la matriz de confusión, y se acerca bastante al modelo KNN, por lo que podríamos utilizar cualquiera de los dos.


```python
## Adaboosting (ensembles) ##

Xtr, Xts, ytr, yts = train_test_split(X_reduced2, Y_reduced, test_size=0.3)

ada = AdaBoostClassifier(n_estimators=200, random_state=100)
ada.fit(Xtr, ytr)
yb = rf.predict(Xts)

# Evaluate and compare results
print('Cohen-kappa',cohen_kappa_score(yts, yb))

Conf_matrix_RF = confusion_matrix(yts, yb)
Kappa_RF = cohen_kappa_score(yts, yb)

tag_list = np.arange(1,17)
score = rf.predict_proba(Xts)
sp.draw_ROC(yts,score,tag_list)

sp.draw_ConfusionM(Conf_matrix_RF,tag_list)

X_ = np.delete(X.reshape([145*145,220]), indice, axis=1)
sp.draw_image(rf.predict(X_2),"Ada Boost ")
```

    Cohen-kappa 0.6849302035266522
    


    
![png](Entregable_final_ML_files/Entregable_final_ML_27_1.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_27_2.png)
    



    
![png](Entregable_final_ML_files/Entregable_final_ML_27_3.png)
    

