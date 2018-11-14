#!/usr/bin/env python

import numpy as np
import pickle
import pandas as pd

def compra(acciones, dinero, precio, cantidad, comision):
    return acciones+cantidad, dinero-(precio*cantidad)-np.abs(comision*cantidad*precio)

def simulacion(datos, decisiones, clusters):
    
    dinero = 1000000
    comision = 0.0025
    acciones = 0
    
    close = datos
    
    situacion = [acciones, dinero]  
    
    for i in range(len(datos)-5):
        C = [(close[i:i+5]-close[i:i+5].mean())/close[i:i+5].std()]
        CP = clusters.predict(C)
        
        vec = np.zeros((1,clusters.n_clusters)) # genera un vector de dimensiones 1,20 
        vec[0][CP] = 1 # el valor indicado será 1 para que al ser multiplicado por la matriz de probabilidades de Markov de la situación. 

        Val = decisiones*vec
        
        if Val.sum() > 0:
            situacion = compra(situacion[0],situacion[1],close[i+5],1000,comision)
        elif Val.sum() < 0: 
            situacion = compra(situacion[0],situacion[1],close[i+5],1000,comision)
            
    return (situacion[1]+close[i+5]*situacion[0]) ## Regresa el balance general


###########################
clusters = pickle.load(open('gen.sav', 'rb'))
###########################

close = pd.read_csv('AC.csv')['Close'].values ## lee los valores cierre del csv original

dec = np.ones((1,16))
dinero_final = simulacion(close, dec, clusters)


"""
l_vec = 16
l_dec = 10
### Se otorgan 3 opciones a la toma de decisiones
decisiones = [[np.random.randint(0,3)-1 for i in range(l_vec)] for i in range(l_dec)] # Inicial. 

for cic in range(200):
    a = []
    m = []
    
    for i in decisiones: ## se suman todos vectores de decisión para escoger el que de la suma mayor
        a.append(np.sum(i))
    
    for i in range(3): ## se escojen los mejores resultados
        m.append(decisiones[a.index(max(a))])
        a.pop(a.index(max(a)))
    
    m = np.array(m) ## hacemos l_vec nuevos vectores derivados únicamente de los 3 mejores anteriores.
    decisiones = [[np.random.choice(m.T[i]) for i in range(l_vec)] for i in range(l_dec)]
    for k in range(l_dec): ## mutamos un tercio de los dígitos de los l_vec vectores que tenemos. 
        for i in range(int(l_dec//3)):
            decisiones[i][np.random.randint(0,l_vec)] = np.random.randint(0,3)-1
    [decisiones.append(i) for i in m] ## agregamos los 'padres' de las nuevas generaciones a la lista. 

print(decisiones[-3])"""