# Esta versión hace exactamente lo mismo que k-means0; en vez de transformar cada uno de los valores a texto y viceversa utiliza la librería numpy
# Obtuvo un tiempo record de 0.008729 segundos en converger.

from random import choice as ch
import numpy as np

from time import time
t1 = time()


def inercia_ind(ref,semanas):
    ### calculo de inercias ###
    diferencias = semanas-ref #se calculan las diferencias entre las semanas y las referencias
    cuadrados = diferencias**2 #se eleva cada una de las diferencias al cuadrado
    sumas_parciales = np.sqrt(cuadrados.sum(axis=1)) #se suman los cuadrados y se les saca la raiz. 
    return (sumas_parciales)

def inercia_tot(inercia_ind):
    suma = inercia_ind.sum()/len(inercia_ind) #se suma cada una de las inercias haciendo una inercia total
    return(suma)
    
    
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
    #semanas = [[semanas[a][b] for a in range(len(semanas))] for b in range(len(semanas[0]))] #este procedimiento toma los 104 valores en vez de la agrupación de 7 de estos.
    sem = [a[b] for a in [semanas[a:a+7]  for a in range(len(semanas)-7)] for b in range(7)] #toma de 7 en 7 conjuntos de datos y los hace una sola matriz
    sem = np.array(sem) #se hace un arreglo (en forma de matriz) de sem en numpy
    return (sem)

def grupos(semanas, inercia1, inercia2):
    # En base a las inercias entre los puntos y las semanas elige en que grupo deberìa estar cada uno de los datos. 
    # Por el momento solo le es posible comparar entre dos grupos.
    grupo = np.array([semanas[l] for l in range(len(inercia1)) if inercia1[l] < inercia2[l]]) 
    return (grupo)

def puntos(grupo):
    # Calcula nuevos puntos (centroides) de los grupos. 
    # suma las columnas y las divide entre la cantidad de grupos totales para conseguir el punto.
    punto = grupo.sum(axis=0)/len(grupo)
    return (punto)

semanas = datos("moda2.csv")

punto1 = ch(semanas)
punto2 = ch(semanas)
p = np.zeros(len(semanas[0]))
while punto1.sum() != p.sum(): #debido a que no se pueden comparar arreglos se comparan las sumas de sus dígitos.
    p = punto1
    inercia1 = inercia_ind(punto1,semanas)
    inercia2 = inercia_ind(punto2,semanas)
    grupo1 = grupos(semanas, inercia1, inercia2)
    grupo2 = grupos(semanas, inercia2, inercia1)
    punto1 = puntos(grupo1)
    punto2 = puntos(grupo2)
    print(inercia_tot(inercia1),inercia_tot(inercia2))

print(time()-t1)
