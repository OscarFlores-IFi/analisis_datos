# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:51:08 2019

@author: Oscar Flores
"""
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulacion import Optimizacion as SIM
#from Simulacion import Graficos as SIM



simulacion = SIM.simulacion


#%% Seleccionamos los csv con los datos a simular.
csv = ['AC','ALFAA','ALPEKA','ALSEA','ELEKTRA','IENOVA','MEXCHEM','PE&OLES','PINFRA','WALMEX']
for i in np.arange(len(csv)):
    csv[i] = '../Data/%sn.csv'%csv[i]
#%%



ndias = [3,5,8,13,21,34,55,89,144]
model_close = pickle.load(open('model_close3.sav','rb'))
Ud = np.random.randint(-1,2,ndata)


simulacion(csv,ndias,model_close,Ud)



