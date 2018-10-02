#!/usr/bin/env python

#se importan datos de un archivo .csv y se divide en n columnas cada una de las columnas originales. 
#se normalizan los datos y posteriormente se exportan a un archivo .csv 

import numpy as np
import pandas as pd

def datos(archivo, columnas, nombre_columna):
    file = pd.read_csv(archivo) #se importa el archivo.
    n = len(file['0']) #se calcula la longitud de los datos. 
    Col = file[nombre_columna]
    en_blanco = np.zeros((n-columnas,columnas))
    for k in np.arange(columnas): 
        en_blanco[:,k] = Col[k:n-columnas+k] #se rellenan los datos vacíos con los originales.
    en_blanco=((en_blanco.T - np.mean(en_blanco,axis=1))/np.std(en_blanco,axis=1)).T # se trasponen debido a que numpy acepta únicamente operaciones matriciales-vectoriales en columnas.
    return(en_blanco)
    
archivo = 'AC.csv'
nombre_columna = 'Close'
#nombre_columna = 'Volume'
close = datos(archivo, 5, nombre_columna)
print(file)
file2 = pd.DataFrame(data=file)
file2.to_csv(nombre_columna + archivo)

