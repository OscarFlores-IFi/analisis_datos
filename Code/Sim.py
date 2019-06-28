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
from Simulacion import Optimizacion as sim_opt
from Simulacion import Graficos as sim_graph


optimizacion = sim_opt.simulacion
grafico = sim_graph.simulacion

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
