#!/usr/bin/env python

#se importan datos de un archivo .csv y se divide en n columnas cada una de las columnas originales. 

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
    return(file2)
file = datos('AC.csv',5)
print(file)
