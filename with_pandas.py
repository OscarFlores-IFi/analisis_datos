#!/usr/bin/env python


import numpy as np
import pandas as pd

def datos(archivo, columnas):
    file = pd.read_csv(archivo)
    n = len(file['0'])
    Open = file['Open']
    m_open = np.zeros((n-columnas,columnas))
    for k in np.arange(columnas):
        m_open[:,k] = Open[k:n-columnas+k] 
    
    
    return(file)
#    retur1n(m_open)
file = datos('AC.csv',5)
print(file)

    
    
    
 



