#!/usr/bin/env python

#se importan datos de un archivo .csv y se divide en n columnas cada una de las columnas originales. 
#se normalizan los datos y posteriormente se exportan a un archivo .csv 

import numpy as np
import pandas as pd

def datos(archivo, columnas):
    file2 = [] #se crea un 'archivo' en blaco, este tendrá las columnas originales redimensionadas. 
    file = pd.read_csv(archivo) #se importa el archivo.
    n = len(file['0']) #se calcula la longitud de los datos. 
    Col_n = list(file.columns.values) #se enlistan las columnas disponibles en el Dataframe
    for i in Col_n[1:]:
        columna = file[i] #se accede cada una de las columnas
        en_blanco = np.zeros((n-columnas,columnas)) #se crea un array en blanco de dimensiones(n-columna * colummna)
        for k in np.arange(columnas): 
            en_blanco[:,k] = columna[k:n-columnas+k] #se rellenan los datos vacíos con los originales.
        file2.append(en_blanco) #se escriben las funciones
    for i in range(len(file2)): #se normalizan las columnas. 
    	file2[i]=((file2[i].T - np.mean(file2[i],axis=1))/np.std(file2[i],axis=1)).T # se trasponen debido a que numpy acepta únicamente operaciones matriciales-vectoriales en columnas.
    file3 = np.concatenate(file2[0:len(file2)], axis=1) #Se concatenan en el eje'x' los 6 arreglos.
    return(file3)
    
archivo = 'AC.csv'
file = datos(archivo, 5)
print(file)
file2 = pd.DataFrame(data=file)
file2.to_csv('normalizado' + archivo)
