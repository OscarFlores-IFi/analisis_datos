#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## se prueban los vectores generados por el algorítmo genético. 
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def compra(acciones, dinero, precio, cantidad, comision):
    return acciones+cantidad, dinero-(precio*cantidad)-np.abs(comision*cantidad*precio)

def simulacion(datos, decisiones, clusters):
    
    dinero = 1000000
    comision = 0.0025
    acciones = 0
    
    close = datos
    
    situacion = [acciones, dinero]  
    hist = []
    
    for i in range(len(datos)-5):
        C = [(close[i:i+5]-close[i:i+5].mean())/close[i:i+5].std()]
        CP = clusters.predict(C)
        
        vec = np.zeros((1,clusters.n_clusters)) # genera un vector de dimensiones 1,20 
        vec[0][CP] = 1 # el valor indicado será 1 para que al ser multiplicado por la matriz de probabilidades de Markov de la situación. 

        Val = decisiones*vec
        
        cant = .10 ## para evitar que se compre todo y se venda todo lo que se tiene, se limita a comprar un porcentaje de lo que puede comprar y vender. 
        if Val.sum() > 0 and situacion[1] > 0:
            situacion = compra(situacion[0],situacion[1],close[i+5],cant*dinero//close[i+5],comision) ## se compra un porcentaje de la capacidad que se tiene. no permite compras sin dinero. 
        elif Val.sum() < 0 and situacion[0] > 0: 
            situacion = compra(situacion[0],situacion[1],close[i+5],-situacion[0]*cant,comision) ## se vende un porcentaje de las acciones que tiene. no permite ventas en corto.
        hist.append([situacion[0],situacion[1],situacion[1]+close[i+5]*situacion[0]])
    return (np.array(hist)) ## Regresa el balance general



###########################
#clusters = pickle.load(open('gen.sav', 'rb'))
clusters = pickle.load(open('close_model.sav', 'rb'))
###########################

close = pd.read_csv('AC.MX.csv')['Close'].values ## lee los valores cierre del csv de prueba.
decision0 = [ 0,  1,  1,  1,  1,  0,  0, 0,  1, -1, -1,  0, 1,  -1,  1,  1]
decision1 = [ 0,  1,  1,  1,  0,  0,  0, -1,  0, -1, -1,  0, -1,  0,  0,  0]
decision2 = [0, 0, 0, 1]

hist = simulacion(close, decision2, clusters)
plt.plot(hist)



