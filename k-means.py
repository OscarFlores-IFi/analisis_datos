from math import sqrt
from random import choice as ch

def diferencia(ref,semanas):
    ### calculo de diferencias ###
    diferencias = [[semanas[a][b] - ref[b] for b in range(len(ref))] for a in range(len(semanas))]
        ###diferencias = [[semanas[a][b] - ref[b] for a in range(len(semanas))] for b in range(len(ref))] #cambia filas por columnas.
        ###diferencias = [semanas[a][b] - ref[b] for b in range(len(ref)) for a in range(len(semanas))] #almacena todos los datos en una sola lista 
    return (diferencias)


def inercia_ind(diferencias):
    ### calculo de inercias ###
    cuadrados = [[diferencias[a][b]**2 for b in range(len(diferencias[0]))] for a in range(len(diferencias))]
    sumas_parciales = [sqrt(eval("+".join([[str(l[a])  for a in range(len(cuadrados[0]))] for l in cuadrados][m]))) for m in range(len(cuadrados))] #se convierte en str cada uno de los valores; se suman con eval y "+".join()
    return (sumas_parciales)

def inercia_tot(inercia_ind):
    suma = eval("+".join([str(l) for l in inercia_ind]))/len(inercia_ind)
    return(suma)
    
    
def datos(archivo):
    ### extraer los datos ###
    doc = open(archivo, 'r')
    lines = doc.readlines()
    doc.close()
    lines.pop(0)
    semanas = [l.split(",") for l in lines] #recibe los valores en una lista. 
    for a in range(len(lines)):
        semanas[a][6]=semanas[a][6][:-1] #elimina el caractér de salto de linea.
        semanas[a] = [int(l) for l in semanas[a]] #hace número cada uno de los valores de la lista
    return (semanas)
       
def grupos(semanas, inercia1, inercia2):
    grupo = [semanas[inercia1.index(inercia1[a])] for a in range(len(semanas)) if inercia1[a] < inercia2[a]]
    return (grupo)

def puntos(grupo):
    punto = [eval("+".join([[str(grupo[l][j]) for l in range(len(grupo))] for j in range(len(grupo[0]))][m]))/len(grupo) for m in range(len(grupo[0]))]
    return (punto)

semanas = datos("moda2.csv")
punto1 = ch(semanas)
punto2 = ch(semanas)
p = []
while punto1 != p:
    p = punto1
    inercia1 = inercia_ind(diferencia(punto1,semanas))
    inercia2 = inercia_ind(diferencia(punto2,semanas))
    grupo1 = grupos(semanas, inercia1, inercia2)
    grupo2 = grupos(semanas, inercia2, inercia1)
    punto1 = puntos(grupo1)
    punto2 = puntos(grupo2)
    print(inercia_tot(inercia1),inercia_tot(inercia2))
        
