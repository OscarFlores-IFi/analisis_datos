#!/usr/bin/env python

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def compra(acciones, dinero, precio, cantidad, comision):
    return acciones+cantidad, dinero-(precio*cantidad)-np.abs(comision*cantidad*precio)

def simulacion(datos, decisiones, c_clusters, m_clusters):
    Sim = []
    
    dinero = 1000000
    comision = 0.0025
    acciones = 0
    
    close = datos['Close']
    vol = datos['Volume']
    magn = (vol-vol.mean())/vol.std()
    
    situacion = [acciones, dinero]  
    
    for i in range(len(datos)-5):
        C = [(close[i:i+5]-close[i:i+5].mean())/close[i:i+5].std()]
        CP = c_clusters.predict(C)
        MP = m_clusters.predict(magn[i+5])
        
        c_vec = np.zeros((1,c_clusters.n_clusters)) # genera un vector de dimensiones 1,20 
        c_vec[0][CP] = 1 # el valor indicado será 1 para que al ser multiplicado por la matriz de probabilidades de Markov de la situación. 

        m_vec = np.zeros((1,m_clusters.n_clusters))
        m_vec[0][MP] = 1
    
        Val = decisiones*np.concatenate((c_vec,m_vec),axis=1)

        
        cant = .10 ## para evitar que se compre todo y se venda todo lo que se tiene, se limita a comprar un porcentaje de lo que puede comprar y vender. 
        if Val.sum() > 0 and situacion[1] > 0:
            situacion = compra(situacion[0],situacion[1],close[i+5],cant*dinero//close[i+5],comision) ## se compra un porcentaje de la capacidad que se tiene. no permite compras sin dinero. 
        elif Val.sum() < 0 and situacion[0] > 0: 
            situacion = compra(situacion[0],situacion[1],close[i+5],-situacion[0]*cant,comision) ## se vende un porcentaje de las acciones que tiene. no permite ventas en corto.
    
        Sim.append([situacion[0],situacion[1],situacion[1]+close[i+5]*situacion[0]])
    return (Sim) ## Regresa el balance general


###########################
c_clusters = pickle.load(open('close_model.sav', 'rb'))
m_clusters = pickle.load(open('magn_model.sav', 'rb'))
###########################

datos = pd.read_csv('AC.csv') ## lee los valores cierre del csv original


###############################################################################
# Se corre la simulación con vectores de decisiones genéticos

l_vec = 9
l_dec = 10
### Se otorgan 3 opciones a la toma de decisiones
#decisiones = [[np.random.randint(0,3)-1 for i in range(l_vec)] for i in range(l_dec)] # Inicial. 
decisiones = [[1, 0, 1, 0, 1, 1, 0, 1, 0]]

for cic in range(1):
    a = []
    m = []
    
    for i in decisiones: ## se suman todos vectores de decisión para escoger el que de la suma mayor
        plt.plot(simulacion(datos,i,c_clusters,m_clusters))
#        a.append(simulacion(datos,i,c_clusters,m_clusters))
#    
#    for i in range(3): ## se escojen los mejores resultados
#        m.append(decisiones[a.index(max(a))])
#        a.pop(a.index(max(a)))
#    
#    m = np.array(m) ## hacemos l_vec nuevos vectores derivados únicamente de los 3 mejores anteriores.
#    decisiones = [[np.random.choice(m.T[i]) for i in range(l_vec)] for i in range(l_dec)]
#    for k in range(l_dec): ## mutamos un tercio de los dígitos de los l_vec vectores que tenemos. 
#        for i in range(int(l_dec//3)):
#            decisiones[i][np.random.randint(0,l_vec)] = np.random.randint(0,3)-1
#    [decisiones.append(i) for i in m] ## agregamos los 'padres' de las nuevas generaciones a la lista. 
#    
print(decisiones[-3:])