#!/usr/bin/env python

import numpy as np
from time import time
import pickle

###############################################################################
# Se corre la simulación con vectores de decisiones genéticos
t1 = time()


l_vec = 256 #longitud del vector de toma de decisiones
l_dec = 64 #Cantidad de vectores de toma de decisiones     *** en potencias de 2 (2**n) ***

#### Se otorgan 3 opciones a la toma de decisiones

decisiones = np.random.randint(-1,2,(l_dec,l_vec)) # Inicial. 

iteraciones = 500
hist = np.zeros((iteraciones,l_dec//4)) # no se sobre-escribe


p = np.zeros(l_dec//4) # calificaciones padres, se sobre-escribe en cada ciclo
a = np.zeros(l_dec//4*5) # puntuaciones de hijos, se sobre-escribe en cada ciclo
m = np.zeros((l_dec//4,l_vec)) # padres, se sobre-escribe en cada ciclo

for cic in range(iteraciones):
    
    for i in np.arange(l_dec): ## se simulan todos vectores de decisión para escoger el que de la suma mayor
        
#        #######################################################################
#        T,Vp,X,u = portafolio_sim(precio,sit,decisiones[i]) ###################
#        pct = Vp[1:]/Vp[:-1]-1 ################################################
#        a[i] = (pct) ########################################################## 1 empresa
#        #######################################################################
        
        
        #######################################################################
        Sim = portafolios_sim(data,sit,decisiones[i]) #########################
        pct = Sim[:,1:]/Sim[:,:-1]-1 ##########################################
        a[i] = pct.mean() ##################################################### todas las empresas
        #######################################################################
        
        
    decisiones = np.concatenate((decisiones,m)) # agregamos los 'padres' de las nuevas generaciones a la lista. 
    a[-int(l_dec//4):] = p # se juntan las puntuaciones de los hijos con la de los padres
    m = decisiones[np.argsort(a)[-int(l_dec//4):]] # se escojen los padres
    p = np.sort(a)[-int(l_dec//4):] # se guardan las calificaciones de los nuevos padres  
    
    hist[cic,:] = p #se almacena el promedio de los padres para observar avance generacional
    
    decisiones = np.array([[np.random.choice(m.T[i]) for i in range(l_vec)] for i in range(l_dec)])
    for k in range(l_dec): ## mutamos la cuarta parte de los dígitos de los l_dec vectores que tenemos. 
        for i in range(int(l_vec//4)):
            decisiones[k][np.random.randint(0,l_vec)] = np.random.randint(0,3)-1
        
    print(np.ceil((1+cic)/iteraciones*1000)/10)

pickle.dump(m,open('m.sav','wb'))
pickle.dump(hist,open('hist.sav','wb'))

print(m, time()-t1)















