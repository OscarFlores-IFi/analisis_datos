# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:21:31 2019

@author: if715029
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from time import time
import pickle 

#%%

csv = ['AC','ALFAA','ALPEKA','ALSEA','ELEKTRA','IENOVA','MEXCHEM','PE&OLES','PINFRA','WALMEX']
data = []
t1 = time()

for i in csv: 
#    data.append(pd.read_csv('../Data/%sn.csv'%i, index_col=0))
    data.append(pd.read_csv('../Data/%s.csv'%i, index_col=0))
    
#%%
def crear_ventanas(data,n_ventana):
    n_data = len(data)
    dat_new = np.zeros((n_data-n_ventana+1,n_ventana))
    for k in np.arange(n_ventana):
        dat_new[:,k] = data[k:(n_data-n_ventana+1)+k]
    return dat_new

#%% Se crean ventanas para los datos en distinta longitud de días. 
#ndias = [5,20,40,125]
ndias = [3,5,8,13,21,34,55,89,144]
vent = []
for j in data: 
    ven = []
    for i in ndias:
        ven.append(crear_ventanas(j['Close'],i))
    vent.append(ven)
    
#%% Se estandarizan los datos
cont = len(ndias)    

norm = []
for j in vent:
    for i in range(cont):
        j[i] = np.transpose((j[i].transpose()-j[i].mean(axis=1))/j[i].std(axis=1))
    norm.append(j)
        
#%% Se importa el modelo pre-diseñado
model_close = pickle.load(open('model_close3.sav','rb'))

#%% Se clasifica la situación de los precios en cada cluster de k-means.

clasif_close = []

for norm in norm:
    tmp = []
    for i in range(cont):
        tmp.append(model_close[i].predict(norm[i]))
    clasif_close.append(tmp)   
    
#%% Cortar la longitud de las clasificaciones para que tengan la misma longitud
for j in clasif_close:
    for i in range(cont):
        j[i]=j[i][len(norm[0][i])-len(vent[0][-1]):]
    
#%% 
sit = []
for j in clasif_close:
    s1 = np.zeros(len(j[0]))
    for i in range(cont):
        s1 += j[i]*2**i
    sit.append(s1)

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
    
    return T,Vp,X,u # para graficar 
#    return Vp # para optimizar en genético. 

#%% para optimizar
#def portafolios_sim(data,sit,Ud):
#    Sim = np.zeros((len(data),len(sit[0])))
#    for i in range(len(data)):
#        Sim[i] = portafolio_sim(data[i].Close[-len(sit[0]):],sit[i],Ud)
#        
#    return(Sim)
#    
#%% para graficar
    
def portafolios_sim(data,sit,Ud):
    Sim = []
    for i in range(len(data)):
        Sim.append(portafolio_sim(data[i].Close[-len(sit[0]):],sit[i],Ud))
        
    return(np.array(Sim))
    
#%% Ejecucion de la funcion de simulacion
    
ndata = 4**cont
Ud = np.random.randint(-1,2,ndata)
#Ud = [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  1.,
#        0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,
#        1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,
#       -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,
#        0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0., -1.,  1.,  0., -1., -1.,  0.,  0.,  0.,  0.,
#        0.,  1.,  0.,  0.,  0., -1.,  0.,  0., -1.,  0.,  0., -1., -1.,
#        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#        0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#        0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
#        0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.] # Anterior máximo. Considera únicamente valor final y no cambios porcentuales.

#Ud = m0 # resultantedespues de un caso representativo del genético

Sim = portafolios_sim(data,sit,Ud)

#%% Grafica los resultados
for i in range(len(Sim)): 
    plt.figure(figsize=(8,6))
    plt.subplot(3,1,1)
    plt.plot(Sim[i][0],data[i].Close[-len(sit[0]):])
#    plt.vlines(1129,data[i].min(),data[i].max())
    plt.ylabel('p(t)')
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.plot(Sim[i][0],Sim[i][1])
    plt.ylabel('vp(t)')
#    plt.vlines(1129,Sim[i][1].min(),Sim[i][1].max())
    plt.xlabel('time')
    plt.grid()
    
    plt.subplot(3,1,3)
    plt.plot(Sim[i][0],Sim[i][3])
    plt.ylabel('u(t)')
    plt.grid()
    plt.show()
