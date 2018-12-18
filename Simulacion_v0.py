# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:19:51 2018

@author: RIEMANNRUIZ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime
from sklearn.cluster import KMeans
import pickle

#%% Importar los datos para la simulacion
csv = ['AC','ALFAA','ALPEKA','ALSEA','ELEKTRA','IENOVA','MEXCHEM','PE&OLES','PINFRA','WALMEX']
data = []
for i in csv:   
    data.append(pd.read_csv('Data/%s.csv'%i, index_col=0))
#data = pd.read_csv('analisis_datos-master/GRUMAB.csv', index_col=0)


#%% Función apra crear la base de datos en ventanas
def crear_ventanas(data,n_ventana):
    n_data = len(data)
    dat_new = np.zeros((n_data-n_ventana+1,n_ventana))
    for k in np.arange(n_ventana):
        dat_new[:,k] = data[k:(n_data-n_ventana+1)+k]
    return dat_new
#%% Crear la base de datos en ventanas de n dias
ndias = 5
close_v = crear_ventanas(data[0]['Close'],ndias)
volum_v = crear_ventanas(data[0]['Volume'],ndias)
for i in range(1,len(data)):
    close_v = np.concatenate((close_v, crear_ventanas(data[i]['Close'],ndias)))
    volum_v = np.concatenate((volum_v, crear_ventanas(data[i]['Volume'],ndias)))
#%%
# Normalizar los datos
close_v = np.transpose((close_v.transpose()-close_v.mean(axis=1))/close_v.std(axis=1))
volum_v = np.transpose((volum_v.transpose()-volum_v.mean(axis=1))/volum_v.std(axis=1))

#%% Función para la gráfica de codo
def grafica_codo_kmeans(data,n_centroides):
    inercias = np.zeros(n_centroides.shape)
    for k in n_centroides:
        model = KMeans(n_clusters=k,init='k-means++').fit(data)
        inercias[k-1] = model.inertia_
    # Grafica de codo
    plt.plot(n_centroides,inercias)
    plt.xlabel('Num grupos')
    plt.ylabel('Inercia')
    plt.show()
    return n_centroides,inercias

#%% Graficas de codo de los datos
grafica_codo_kmeans(close_v,np.arange(1,16))
grafica_codo_kmeans(volum_v,np.arange(1,16))

#%% Aplicar el clustering con el numero de grupos determinado
model_close = KMeans(n_clusters=4,init='k-means++').fit(close_v)
model_volum = KMeans(n_clusters=5,init='k-means++').fit(volum_v)

#%% Guardar los modelos
pickle.dump(model_close,open('model_close.sav','wb'))
pickle.dump(model_volum,open('model_volum.sav','wb'))

#%% Leer los modelos
model_close = pickle.load(open('model_close.sav','rb'))
model_volum = pickle.load(open('model_volum.sav','rb'))

#%% Función para dibujar los centroides del modelo
def ver_centroides(centroides):
    n_subfig = np.ceil(np.sqrt(centroides.shape[0]))
    for k in np.arange(centroides.shape[0]):
        plt.subplot(n_subfig,n_subfig,k+1)
        plt.plot(centroides[k,:])
        plt.ylabel('Grupo %d'%k)
    plt.show()

#%% Ver centroides
ver_centroides(model_close.cluster_centers_)
ver_centroides(model_volum.cluster_centers_)

#%% Clasificar las ventanas en el tiempo
clasif_close = model_close.predict(close_v)
clasif_volum = model_volum.predict(volum_v)

#%% Función para calculo de la matriz de Markov
def crear_Markov(clusters):
    ns = 2
    n_pat = len(clusters)
    sec_clust = np.zeros((n_pat-ns+1,ns))
    sec_clust[:,0] = clusters[0:n_pat-ns+1]
    sec_clust[:,1] = clusters[1:n_pat-ns+2]
    nclusters = np.unique(clusters)
    M = np.zeros((len(nclusters),len(nclusters)))
    for k in nclusters:
        for l in nclusters:
            index = (sec_clust[:,0]==k)&(sec_clust[:,1]==l)
            M[l,k] = np.sum(index)
    M = M/np.sum(M,axis=0)
    return M

#%% Obteer matriz de Markov
M_close = crear_Markov(clasif_close)
M_volum = crear_Markov(clasif_volum)


#%% Funcion del modelo del portafolio
def portafolio(x,u,p,rcom):
    x_1 = x;
    vp = x[0]+p*x[1] #Valor presente del portafolios
    x_1[0] = x[0]-p*u-rcom*p*abs(u) #Dinero disponible
    x_1[1] = x[1]+u #Acciones disponibles
    return vp,x_1

#%% Simulacion del sistema con las reglas basicas de compra y venta
precios = [i['Close'] for i in data]
T = np.arange(len(precios[0]))

Vp = np.zeros(T.shape)
X  = np.zeros((T.shape[0]+1,2))
u = np.zeros(T.shape)
X[5][0] = 10000
rcom = 0.0025

# Creación de la matriz de decisiones
Ud = np.reshape([-1,1,1,-1,-1,1,1,-1,-1,1,0,-1,-1,1,1,0],M_close.shape).T

#%%
for precio in precios: 
    for t in T[ndias:]:
        
        
        u_max = np.floor(X[t][0]/((1+rcom)*precio[t])) # Numero maximo de la operacion
        u_min  = -X[t][1] # Numero minimo de la operacion
        
    #    #AC
    #    X0 = np.zeros((4,1))
    #    X0[clasif_close[t-ndias]] = 1
    #    X1 =  np.array(np.matrix(M_close)*np.matrix(X0))
    #    if clasif_close[t-ndias] == 0:
    #        u[t] = X1[0][0]*u_min+X1[1][0]*u_max+X1[2][0]*u_max+X1[3][0]*u_min
    #    elif clasif_close[t-ndias] == 1:
    #        u[t] = X1[0][0]*u_min+X1[1][0]*u_max+X1[2][0]*u_max+X1[3][0]*u_min
    #    elif clasif_close[t-ndias] == 2:
    #        u[t] = X1[0][0]*u_min+X1[1][0]*u_max+X1[2][0]*0+X1[3][0]*u_min
    #    elif clasif_close[t-ndias] == 3:
    #        u[t] = X1[0][0]*u_min+X1[1][0]*u_max+X1[2][0]*u_max+X1[3][0]*0
            
        #AC (operacion matricial)
        Um = Ud.copy()
        Um[Ud==1]=u_max
        Um[Ud==-1]=u_min
        u[t] = np.sum(Um[:,clasif_close[t-ndias]]*M_close[:,clasif_close[t-ndias]])
        
        Vp[t],X[t+1]=portafolio(X[t],u[t],precio[t],rcom)
    
#%% Visualizar los resultados de la última simulación.
        
plt.figure(figsize=(8,8))
plt.subplot(3,1,1)
plt.plot(T,precio)
plt.ylabel('p(t)')
plt.grid()

plt.subplot(3,1,2)
plt.plot(T[ndias:],Vp[ndias:])
plt.ylabel('vp(t)')
plt.xlabel('time')
plt.grid()

plt.subplot(3,1,3)
plt.plot(T[ndias:],u[ndias:])
plt.ylabel('u(t)')
plt.grid()
plt.show()


#%% Función para realizar la simulación del portafolio
def portafolio_sim(precio,Markov,Ud):
    T = np.arange(len(precio))
    
    Vp = np.zeros(T.shape)
    X  = np.zeros((T.shape[0]+1,2))
    u = np.zeros(T.shape)
    X[5][0] = 10000
    rcom = 0.0025
    
    # Creación de la matriz de decisiones
    Ud = np.reshape(Ud,Markov.shape).T
    
    for t in T[ndias:]:
        
        
        u_max = np.floor(X[t][0]/((1+rcom)*precio[t])) # Numero maximo de la operacion
        u_min  = -X[t][1] # Numero minimo de la operacion
        
        #AC (operacion matricial)
        Um = Ud.copy()
        Um[Ud==1]=u_max
        Um[Ud==-1]=u_min
        u[t] = np.sum(Um[:,clasif_close[t-ndias]]*Markov[:,clasif_close[t-ndias]])
        
        Vp[t],X[t+1]=portafolio(X[t],u[t],precio[t],rcom)
    
    return T,Vp,X,u

#%% Simulación que se requiere iterar con el algoritmo genético
def portafolios_sim(data,Markov,Ud):
    Sim = []
    for i in range(len(data)):
        Sim.append(portafolio_sim(data[i]['Close'],M_close,Ud))
        
    return(Sim)

        
#%% Simulación que se requiere iterar con el algoritmo genético        
Ud = [ 0,  0,  1,  0,  0,  0,  0,  -1,  0,  0,  0,  0,  0,  0,  0,  0]
#T,Vp,X,u = portafolio_sim(data[6]['Close'],M_close,Ud)
Sim = portafolios_sim(data,M_close,Ud) #lista de simulaciones, contiene; T,Vp,X,u de cada una. 

#%% Grafica los resultados
for i in range(len(Sim)): 
    plt.figure(figsize=(8,8))
    plt.subplot(3,1,1)
    plt.plot(Sim[i][0],data[i]['Close'])
    plt.ylabel('p(t)')
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.plot(Sim[i][0][ndias:],Sim[i][1][ndias:])
    plt.ylabel('vp(t)')
    plt.xlabel('time')
    plt.grid()
    
    plt.subplot(3,1,3)
    plt.plot(Sim[i][0][ndias:],Sim[i][3][ndias:])
    plt.ylabel('u(t)')
    plt.grid()
    plt.show()