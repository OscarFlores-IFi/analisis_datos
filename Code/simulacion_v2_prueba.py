#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:53:21 2019

@author: fh
"""

# Probar los 16 vectores en una sola acción. 
# Es requerido haber corrido Simulacion_v2 con return(Vp)


n = 16 # número de gráficas
Vp = np.zeros((n,len(sit)))
for i in np.arange(len(m)):
    Vp[i,:] = portafolio_sim(precio,sit,m[i])
Vp_m0 = portafolio_sim(precio,sit,m0)
#%%    
Fig = plt.figure(figsize=(20,7))
cmap = plt.cm.plasma # también se puede plt.get_cmap('plasma')
colors = cmap(np.linspace(0,1,n))
for i in np.arange(n):
    plt.plot(Vp[i,:], c=colors[i,:])
plt.plot(Vp_m0,'k-',linewidth=4)
plt.vlines(1129,Vp.min(),Vp.max())
plt.show()

Fig.savefig('../Data/'+archivo+'.pgn')









