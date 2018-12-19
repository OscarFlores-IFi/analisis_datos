#!/usr/bin/env python

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Se corre la simulación con vectores de decisiones genéticos

l_vec = 16
l_dec = 15

### Se otorgan 3 opciones a la toma de decisiones
decisiones = [[np.random.randint(0,3)-1 for i in range(l_vec)] for i in range(l_dec)] # Inicial. 

for cic in range(1):
    a = []
    m = []
    
    for i in decisiones: ## se suman todos vectores de decisión para escoger el que de la suma mayor
        
        #######################################################################
        Sim = portafolios_sim(data,M_close,Ud) ################################
        a.append(portafolios_sim(data,M_close,i))) ############################
        #######################################################################
    
    for i in range(5): ## se escojen los mejores resultados
        m.append(decisiones[a.index(max(a))])
        a.pop(a.index(max(a)))
    
    m = np.array(m) ## hacemos l_vec nuevos vectores derivados únicamente de los 3 mejores anteriores.
    decisiones = [[np.random.choice(m.T[i]) for i in range(l_vec)] for i in range(l_dec)]
    for k in range(l_dec): ## mutamos un tercio de los dígitos de los l_vec vectores que tenemos. 
        for i in range(int(l_dec//2)):
            decisiones[i][np.random.randint(0,l_vec)] = np.random.randint(0,3)-1
    [decisiones.append(i) for i in m] ## agregamos los 'padres' de las nuevas generaciones a la lista. 

print(decisiones[-5:])