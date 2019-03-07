#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:53:21 2019

@author: fh
"""


Vp = np.zeros((16,len(sit)))
for i in np.arange(len(m)):
    Vp[i,:] = portafolio_sim(precio,sit,m[i])
    
#%%    
plt.figure(figsize=(20,7))
for j in Vp:
    plt.plot(j)
plt.vlines(1129,Vp.min(),Vp.max())
plt.show()