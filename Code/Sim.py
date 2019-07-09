# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:51:08 2019

@author: Oscar Flores
"""
#%% Importar Librerías. 
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulacion import Optimizacion
from Simulacion import Graficos
from Simulacion import Genetico
from Simulacion import Kclusters


optimizacion = Optimizacion.simulacion
grafico = Graficos.simulacion
genetico = Genetico.genetico
k_clusters = Kclusters.k_clusters
#%%############################################################################
############################  Crear model_close ###############################
###############################################################################

#%% Datos en csv
csv = ['AC','ALFAA','ALPEKA','ALSEA','ELEKTRA','IENOVA','MEXCHEM','PE&OLES','PINFRA','WALMEX']
for i in np.arange(len(csv)):
    csv[i] = '../Data/%sn.csv'%csv[i]

#%% Guardamos model_close en model_close3.sav
ndias = [3,5,8,13,21,34,55,89,144]
n_clusters = 2
nombre = 'model_close3'

k_clusters(csv, ndias, n_clusters, nombre)

#%%############################################################################
######################  Simulación para optimización ##########################
###############################################################################

#%% Seleccionamos los csv con los datos a simular.
csv = ['AC','ALFAA','ALPEKA','ALSEA','ELEKTRA','IENOVA','MEXCHEM','PE&OLES','PINFRA','WALMEX']
for i in np.arange(len(csv)):
    csv[i] = '../Data/%sn.csv'%csv[i]
#%% Simulamos

ndias = [3,5,8,13,21,34,55,89,144]
model_close = pickle.load(open('model_close3.sav','rb'))
Ud = np.random.randint(-1,2,2**9)

Vp = optimizacion(csv,ndias,model_close,Ud)



#%%############################################################################
#########################  Simulación para Graficar ###########################
###############################################################################

#%% Seleccionamos los csv con los datos a simular.
csv = ['AC','ALFAA','ALPEKA','ALSEA','ELEKTRA','IENOVA','MEXCHEM','PE&OLES','PINFRA','WALMEX']
for i in np.arange(len(csv)):
    csv[i] = '../Data/%s.csv'%csv[i]
#%% Simulamos

ndias = [3,5,8,13,21,34,55,89,144]
model_close = pickle.load(open('model_close3.sav','rb'))
Ud = np.random.randint(-1,2,2**9)

Sim = grafico(csv,ndias,model_close,Ud)

#%%############################################################################
###################  Optimización por algorigmo genético ######################
###############################################################################

#%% Seleccionamos los csv con los datos que serán utilizados para la optimización.
csv = ['AC','ALFAA','ALPEKA','ALSEA','ELEKTRA','IENOVA','MEXCHEM','PE&OLES','PINFRA','WALMEX']
for i in np.arange(len(csv)):
    csv[i] = '../Data/%sn.csv'%csv[i]
#%% Simulamos

ndias = [3,5,8,13,21,34,55,89,144]
model_close = pickle.load(open('model_close3.sav','rb'))
l_vec = 512 # longitud de cada vector de toma de decisiones
n_vec = 16 # cantidad de vectores de toma de decisiones por generacion
iteraciones = 20
C = 0 
nombre = 'prueba'
#genetico(func,csv,ndias,model_close,l_vec,l_dec,iteraciones,C)
genetico(optimizacion,csv,ndias,model_close,l_vec,n_vec,iteraciones,C,nombre)

#%%
[p,a,m,hist_m,hist_s,hist_a,m_hist] = pickle.load(open(nombre + '.sav','rb'))

