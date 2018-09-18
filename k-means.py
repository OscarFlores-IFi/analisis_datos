#El algoritmo importa un archivo.csv, encuentra de 1 a 10 centroides y almacena los datos en listas.
# Sigue con error de no converger en algunas situaciones. como resultado arroja un NaN en total_inercia.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def datos(archivo):
    ### extraer los datos ###
    datos = pd.read_csv(archivo,index_col=0)
    datos = np.array(datos)
    return (datos)

    
def calc_inercias(semanas,centroides,k):
    inercias = [np.sqrt(((semanas-centroides[l])**2).sum(axis=1)) for l in range(len(k))]
    return(inercias)

def calc_grupos(semanas,inercias,k):
    minimos = np.argmin(inercias,axis=0)
    grupos = [semanas[minimos==l] for l in range(len(k))]
    return(grupos)

def calc_centroides(grupos,k):
    centroides = [grupos[l].sum(axis=0)/len(grupos[l]) for l in range(len(k))]
    return(centroides)

def k_means(clusters):
    k = np.ones(clusters)    
        
    semanas = datos("normalizadoAC.csv")
    rand = np.random.randint(len(semanas), size=len(k))
    centroides = [semanas[l] for l in rand]
    
    m = 0
    total_inercia = 1
    
    while m != total_inercia:
        
        inercias = calc_inercias(semanas,centroides,k)      
        grupos = calc_grupos(semanas,inercias,k)
        centroides = calc_centroides(grupos,k)
        
        m = total_inercia
        total_inercia = np.min(inercias,axis=0).sum()
        print(total_inercia)
    return(semanas,centroides,grupos,inercias,total_inercia)

### Se ejecuta el algoritmo de k-means en los datos. Posteriormente se guardan los resultados en listas. 
x = 10
semanas,centroides,grupos,inercias,total_inercia = [],[],[],[],[]
for i in range(1,11):
    print('\n Total_inercia con %i centroides'%i)
    k_values = k_means(i)
    semanas.append(k_values[0])
    centroides.append(k_values[1])
    grupos.append(k_values[2])
    inercias.append(k_values[3])
    total_inercia.append(k_values[4])
    

### se grafican las inercias globales (total_inercia) para conocer el 'codo'
    
plt.plot(total_inercia)
plt.xlabel('iteraciones')
plt.ylabel('inercia global')
plt.show()
