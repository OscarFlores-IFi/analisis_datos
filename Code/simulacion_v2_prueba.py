#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:53:21 2019

@author: fh
"""

#Librerias a usar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from mylib import mylib
portafolio_sim = mylib.portafolio_sim


#%% Cargar resultados anteriores

# Cargar resultados obtenidos del algoritmo genetico
[p,a,m,hist,m_hist] = pickle.load(open('genetico.sav','rb'))

# Obtención del super padre
m0 = m.mean(axis=0) 
lim = .35
m0[np.abs(m0)<=lim] = 0
m0[m0< -lim] = -1
m0[m0>lim] = 1

# Cargar el modelo de clasificacion (KMeans) ya enetrenado
model_close = pickle.load(open('model_close.sav','rb'))


#%% Cargar datos y calcular las situaciones del archivo deseado

# Cargar los datos
archivo = 'WALMEXn'
data = pd.read_csv('../Data/'+archivo+'.csv', index_col=0)

# Crear la ventanas para el analisis
ndias = [5,20,40,125]
vent = []
for i in ndias:
    vent.append(mylib.crear_ventanas(data['Close'],i))


# Normalizar las ventas
cont = len(ndias)    
for i in range(cont):
    vent[i] = np.transpose((vent[i].transpose()-vent[i].mean(axis=1))/vent[i].std(axis=1))
    
# Clasificacion de las ventanas usando el modelo KMeans entrenado
clasif_close = []
for i in range(cont):
    clasif_close.append(model_close[i].predict(vent[i]))

# Elegir el mismo numero de clasificaciones para las diferentes ventanas
for i in range(cont):
    clasif_close[i]=clasif_close[i][len(vent[i])-len(vent[-1]):]
    
# Crear el ventor de situaciones
sit = np.zeros(len(clasif_close[0]))
for i in range(cont):
    sit += clasif_close[i]*4**i

# Tomar los precios que se utilizaran para la simulacion
precio = data.Close[-len(sit):]

#%% Realizar la simulacion de cada padres y el super padre
# Probar los 16 vectores en una sola acción. 
# Es requerido haber corrido Simulacion_v2 con return(Vp)

n = 16 # número de gráficas
Vp = np.zeros((n,len(sit)))
for i in np.arange(len(m)):
    Vp[i,:] = portafolio_sim(precio,sit,m[i])
Vp_m0 = portafolio_sim(precio,sit,m0)
#%% Realizar la grafica de los resultados de todas las simulaciones
Fig = plt.figure(figsize=(20,7))
cmap = plt.cm.plasma # también se puede plt.get_cmap('plasma')
colors = cmap(np.linspace(0,1,n))
for i in np.arange(n):
    plt.plot(Vp[i,:], c=colors[i,:],label='padre%d'%i)
plt.plot(Vp_m0,'k-',linewidth=4,label='Super padre')
plt.legend(loc=1,bbox_to_anchor=(1.1, 1))
plt.vlines(1129,Vp.min(),Vp.max())
plt.xlim(0,len(Vp_m0))
plt.title(archivo)
plt.xlabel('Time (days)')
plt.ylabel('Vp ($)')
plt.show()

Fig.savefig('../Data/'+archivo+'.png',bbox_inches='tight')









