#!/usr/bin/env python

import numpy as np
from time import time
import pickle

#%%
###############################################################################
# Se corre la simulación con vectores de decisiones genéticos
t1 = time()


l_vec = np.int(np.max(sit))+1 #longitud del vector de toma de decisiones
l_dec = 64 #Cantidad de vectores de toma de decisiones     *** en potencias de 2 (2**n) ***

#### Se otorgan 3 opciones a la toma de decisiones

decisiones = np.random.randint(-1,2,(l_dec,l_vec)) # Inicial. 
iteraciones = 500



hist_m = np.zeros((iteraciones,l_dec//4*5)) # historial de media
hist_s = np.zeros((iteraciones,l_dec//4*5)) # historial de desviación estandar
hist_a = np.zeros((iteraciones,l_dec//4*5)) # historial de calificaciones
m_hist = []

p = np.zeros(l_dec//4) # calificaciones padres, se sobre-escribe en cada ciclo
a = np.zeros(l_dec//4*5) # puntuaciones de hijos, se sobre-escribe en cada ciclo
m = np.zeros((l_dec//4,l_vec)) # padres, se sobre-escribe en cada ciclo

#Para castigar y premiar baja desviación de rendimientos. 
C = 1/10 # Multiplicador arbitrario de castigo por desviación estándar. 
pct_mean = np.zeros(a.shape)
pct_std = np.zeros(a.shape)


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
        pct_mean[i] = pct.mean() ########################################## todas las empresas
        pct_std[i] = pct.std() ############################################
        #######################################################################
    
    # Se da una calificación a cada vector de toma de decisiones.
    pmr = pct_mean # pct_mean no estandarizado se respalda
    psr = pct_std # pct_std no estandarizado se respalda
    pct_mean = (pct_mean-pct_mean.mean())/pct_mean.std() # pct_mean estandarizado 
    pct_std = (pct_std-pct_std.mean())/pct_std.std() # pct_std estandarizado
    a = pct_mean-pct_std*C # Se le da una calificación 
    
    # Se escogen los padres.
    decisiones = np.concatenate((decisiones,m)) # agregamos los 'padres' de las nuevas generaciones a la lista. 
    m = decisiones[np.argsort(a)[-int(l_dec//4):]] # se escojen los padres
    pct_mean[-int(l_dec//4):] = pmr[np.argsort(a)[-int(l_dec//4):]] # se guarda la media que obtuvieron los padres  
    pct_std[-int(l_dec//4):] = psr[np.argsort(a)[-int(l_dec//4):]] # se guarda la desviación que obtuvieron los padres 
    
    
    hist_m[cic,:] = pmr #se almacena el promedio de los padres para observar avance generacional
    hist_s[cic,:] = psr
    hist_a[cic,:] = a
    
    
    # Se mutan los vectores de toma de decisiones
    decisiones = np.array([[np.random.choice(m.T[i]) for i in range(l_vec)] for i in range(l_dec)])
    for k in range(l_dec): ## mutamos la cuarta parte de los dígitos de los l_dec vectores que tenemos. 
        for i in range(int(l_vec//4)):
            decisiones[k][np.random.randint(0,l_vec)] = np.random.randint(0,3)-1
        
        
        
    # Para imprimir el proceso del algoritmo genérico en relación al total por simular.    
    print(np.ceil((1+cic)/iteraciones*1000)/10)

    # Cada 10 iteraciones se guardan los resultados de las simulaciones en un respaldo. 
    if cic % 10 == 0: 
        m_hist.append(m)
        pickle.dump([m,hist_m,hist_s,hist_a,m_hist],open('tmp.sav','wb'))
    
print(m, time()-t1)

pickle.dump([p,a,m,hist_m,hist_s,hist_a,m_hist],open('genetico3.sav','wb')) # guarda las variables más importantes al finalizar. 

#%% para abrir el .sav
[p,a,m,hist_m,hist_s,hist_a,m_hist] = pickle.load(open('genetico3.sav','rb'))

#%% generar un vector de toma de decisiones representativo. 

m0 = m.mean(axis=0) 
#%% rango : [ -1 , -.35 , .35 , 1 ]
lim = .35

m0[np.abs(m0)<=lim] = 0
m0[m0< -lim] = -1
m0[m0>lim] = 1



