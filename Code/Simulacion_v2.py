# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:16:35 2019

@author: if715029
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from time import time
import pickle 


#%%

t1 = time()
data = pd.read_csv('../Data/AMX.csv', index_col=0)
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
for i in ndias:
    vent.append(crear_ventanas(data['Close'],i))
#%%
cont = len(ndias)    
    
for i in range(cont):
    vent[i] = np.transpose((vent[i].transpose()-vent[i].mean(axis=1))/vent[i].std(axis=1))
#%% Función para la gráfica de codo
#def grafica_codo_kmeans(data,n_centroides):
#    inercias = np.zeros(n_centroides.shape)
#    for k in n_centroides:
#        model = KMeans(n_clusters=k,init='k-means++').fit(data)
#        inercias[k-1] = model.inertia_
#    # Grafica de codo
#    plt.plot(n_centroides,inercias)
#    plt.xlabel('Num grupos')
#    plt.ylabel('Inercia')
#    plt.show()
#    return n_centroides,inercias
#%%
#for i in range(cont):
#    grafica_codo_kmeans(vent[i],np.arange(1,16))
#%%
#model_close = []
#for i in range(cont):
#    model_close.append(KMeans(n_clusters=4,init='k-means++').fit(vent[i]))

#%% Función para dibujar los centroides del modelo
#def ver_centroides(centroides):
#    n_subfig = np.ceil(np.sqrt(centroides.shape[0]))
#    for k in np.arange(centroides.shape[0]):
#        plt.subplot(n_subfig,n_subfig,k+1)
#        plt.plot(centroides[k,:])
#        plt.ylabel('Grupo %d'%k)
#    plt.show()
#%%
#for i in range(cont):
#    ver_centroides(model_close[i].cluster_centers_)

#%%
model_close = pickle.load(open('model_close.sav','rb'))

#%%    
clasif_close = []
for i in range(cont):
    clasif_close.append(model_close[i].predict(vent[i]))
    
#%%  
for i in range(cont):
    clasif_close[i]=clasif_close[i][len(vent[i])-len(vent[-1]):]
    
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
ndata = 4**cont
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
#Ud = m0
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