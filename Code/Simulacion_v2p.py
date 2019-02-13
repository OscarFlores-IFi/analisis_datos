# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:21:31 2019

@author: if715029
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time
import pickle 

#%%

csv = ['AC','ALFAA','ALPEKA','ALSEA','ELEKTRA','IENOVA','MEXCHEM','PE&OLES','PINFRA','WALMEX']
data = []
t1 = time()

for i in csv: 
#    data.append(pd.read_csv('../Data/%s.csv'%i, index_col='0'))
    data.append(pd.read_csv('Data/%s.csv'%i, index_col=0))
#%%
def crear_ventanas(data,n_ventana):
    n_data = len(data)
    dat_new = np.zeros((n_data-n_ventana+1,n_ventana))
    for k in np.arange(n_ventana):
        dat_new[:,k] = data[k:(n_data-n_ventana+1)+k]
    return dat_new

#%%
ndias = [5,20,40,125]
vent = []
for j in data: 
    ven = []
    for i in ndias:
        ven.append(crear_ventanas(j['Close'],i))
    vent.append(ven)
    
#%%
cont = len(ndias)    

norm = []
for j in vent:
    for i in range(cont):
        j[i] = np.transpose((j[i].transpose()-j[i].mean(axis=1))/j[i].std(axis=1))
    norm.append(j)
        
#%%
model_close = pickle.load(open('model_close.sav','rb'))

#%%    
clasif_close = []

for i in range(cont):
    clasif_close.append(model_close[i].predict(norm[i]))
    
#%%  
for i in range(cont):
    clasif_close[i]=clasif_close[i][:-len(clasif_close[-1])-1:-1]
    
#%%    
sit = np.zeros(len(clasif_close[0]))
for i in range(cont):
    sit += clasif_close[i]*4**i
    

#%% Funcion del modelo del portafolio
def portafolio(x,u,p,rcom):
    x_1 = x;
    vp = x[0]+p*x[1] #Valor presente del portafolios
    x_1[0] = x[0]-p*u-rcom*p*abs(u) #Dinero disponible
    x_1[1] = x[1]+u #Acciones disponibles
    return vp,x_1


#%% Función para realizar la simulación del portafolio
def portafolio_sim(precio,sit,Ud):
    T = np.arange(len(precio))
        
    Vp = np.zeros(T.shape)
    X  = np.zeros((T.shape[0]+1,2)) 
    u = np.zeros(T.shape)
    X[0][0] = 10000
    rcom = 0.0025
    
    for t in T:
        
        u_max = np.floor(X[t][0]/((1+rcom)*precio[t])) # Numero maximo de la operacion
        u_min  = X[t][1] # Numero minimo de la operacion
        
        #AC (operacion matricial)
        if Ud[int(sit[t])]>0:
            u[t] = u_max*Ud[int(sit[t])]
        else:
            u[t] = u_min*Ud[int(sit[t])]
        
        Vp[t],X[t+1]=portafolio(X[t],u[t],precio[t],rcom)
    
    return T,Vp,X,u

#%% Ejecucion de la funcion de simulacion
ndata = int(max(sit))+1
precio = data.Close[-len(sit):]
#Ud = np.random.randint(-1,2,ndata)
Ud = [ 1,  0,  0, -1, -1,  1,  1,  0,  1,  1,  1, -1, -1,  0, -1,  1, -1,
        0, -1,  1,  1,  1,  0,  1, -1,  1,  1,  1,  0,  1,  1, -1, -1,  0,
       -1, -1,  0,  1, -1,  0,  0, -1, -1,  0, -1,  1,  1,  0,  0,  0,  1,
        0,  1,  0,  0,  0, -1,  1,  0, -1,  0, -1, -1, -1, -1, -1,  1, -1,
        0,  0,  0, -1, -1,  1,  0,  1, -1,  1,  1, -1,  1,  0,  0,  0,  1,
        1, -1,  1,  0,  1,  1,  0, -1,  0,  0, -1,  0,  1,  0,  1,  0,  1,
        0, -1,  0,  1,  1,  0, -1,  1,  1, -1,  0, -1,  0,  0, -1,  1,  1,
        1,  1,  0, -1,  0,  0, -1, -1, -1,  1,  0, -1,  1, -1,  0,  1,  0,
        1,  0,  0,  1, -1,  0, -1,  1,  0,  0, -1,  0,  0,  0, -1, -1,  1,
        0, -1,  1, -1,  0,  1, -1,  1,  1,  1,  1,  0, -1, -1,  1, -1,  0,
       -1,  1,  1,  1,  0,  0,  0,  1, -1,  1, -1, -1,  1,  0,  1, -1, -1,
       -1, -1, -1, -1,  0,  1, -1, -1, -1, -1, -1, -1,  1,  0, -1,  1,  0,
       -1, -1, -1, -1,  1,  1,  0,  1, -1, -1,  1,  0,  1,  0, -1, -1,  0,
        1,  1,  1,  1,  0,  0, -1,  1,  1,  0, -1,  1, -1, -1,  1,  0,  1,
       -1, -1,  1, -1,  1,  1,  0,  0, -1, -1, -1, -1,  0, -1,  0, -1,  0,
        1]
T,Vp,X,u = portafolio_sim(precio,sit,Ud)
#%% 
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
plt.plot(T,precio)
plt.ylabel('p(t)')
plt.grid()

plt.subplot(3,1,2)
plt.plot( T,Vp)
plt.ylabel('vp(t)')
plt.xlabel('time')
plt.grid()

plt.subplot(3,1,3)
plt.plot( T,u)
plt.ylabel('u(t)')
plt.grid()
plt.show()

print(time()-t1)