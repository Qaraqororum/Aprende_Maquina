# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 13:07:55 2021

@author: morte
"""


# coding: utf-8

import numpy as np
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt

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

    
