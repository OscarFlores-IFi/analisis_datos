

import numpy as np

def datos(archivo):
    ### extraer los datos ###
    doc = open(archivo, 'r')
    lines = doc.readlines()
    doc.close()
    lines.pop(0) #elimina el titulo de cada uno de las listas
    semanas = [l.split(",") for l in lines] #recibe los valores en una lista. 
    for a in range(len(lines)):
        semanas[a][6]=semanas[a][6][:-1] #elimina el caractér de salto de linea.
        semanas[a] = [int(l) for l in semanas[a]] #hace número cada uno de los valores de la lista
    sem = [a[b] for a in [semanas[a:a+7]  for a in range(len(semanas)-7)] for b in range(7)] #toma de 7 en 7 conjuntos de datos y los hace una sola matriz
    sem = np.array(sem).T #se hace un arreglo (en forma de matriz) de sem en numpy
    return (sem)
#    return(np.array(semanas).T)
    
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
        
    semanas = datos("moda2.csv").T
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

semanas,centroides,grupos,inercias,total_inercia = k_means(4)



