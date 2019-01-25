# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import pandas as pd
import numpy as np


data = pd.read_csv('Data/AC.csv', index_col='0')
data = data.drop(columns=['Volume','Adj Close'])
close = data['Close']
#%%
Min = data['Low']
Min_mat = np.zeros(data.shape)
for i in range(data.shape[1]):
    Min_mat[:,i] = Min.values
    
Var = data-Min_mat
    
#%%
Max = Var['High']
Max_mat = np.zeros(data.shape)
for i in range(data.shape[1]):
    Max_mat[:,i] = Max.values
#Var[0:200].plot(figsize=(20,12))

#%%
rel = Var/Max_mat
dias = 10

p_movil = rel.rolling(dias).mean()


#%%
r1 = 0
r2 = 180

rel[r1:r2].plot(figsize=(20,5),grid=True)

p_movil[r1:r2].plot(figsize=(20,5),grid=True)
#%%
#close[r1:r2].plot(figsize=(20,5),grid=True)
data[dias:r2].plot(figsize=(20,5),grid=True)

