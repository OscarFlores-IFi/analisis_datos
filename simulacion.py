import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def compra(acciones, dinero, precio, cantidad, comision):
    return acciones+cantidad, dinero-(precio*cantidad)-np.abs(comision*cantidad*precio)


################### se importaron los modelos de k-means, falta integrarlos en la simulación con la toma de decisiones. 
close_model = pickle.load(open('close_model.sav', 'rb'))
qc = len(close_model.cluster_centers_) # número de centroides en close
volume_model = pickle.load(open('volume_model.sav', 'rb'))
qv = len(volume_model.cluster_centers_) # número de centroides en volume
Markov = pickle.load(open('Markov.sav', 'rb'))
Dic = pickle.load(open('Dictionary.sav', 'rb'))
###################

close = pd.read_csv('AC.csv')['Close'].values ## lee los valores cierre del csv original
volume = pd.read_csv('AC.csv')['Volume'].values ## lee los valores de volumen del csv original 

dinero = 1000000
comision = 0.05
acciones = 0

historial = []
situacion = [acciones, dinero]


Decisiones = []
for i in range(len(close)-5):
    C = [(close[i:i+5]-close[i:i+5].mean())/close[i:i+5].std()] ## normaliza los datos de cierre en grupos de 5. 
    V = [(volume[i:i+5]-volume[i:i+5].mean())/volume[i:i+5].std()] ## normaliza los datos de volumen en grupos de 5. 
    CP, VP = close_model.predict(C), volume_model.predict(V) ## almacena en variables la predicción del modelo para cierre y volumen.
    loc = Dic.index([CP,VP]) ## encuentra en que lugar del diccionario está almacenado el par CP,VP
    
    vec = np.zeros((1,len(Dic))) # genera un vector de dimensiones 1,20 
    vec[0][loc] = 1 # el valor indicado será 1 para que al ser multiplicado por la matriz de probabilidades de Markov de la situación. 
    vec = (vec*Markov).sum(axis=1)
        
    matrix = np.zeros((qc, qv)) ##se hace una matriz en blanco de dimension (cierre,volumen)
    for i in range(qc):  ## se llenan los datos en la matriz
        matrix[i,:] = vec[qv*i:(i+1)*qv] ##Grupos de decisiones a tomar; si sube o baja se toma decisión, sino no se hace nada. 
    decision = matrix.sum(axis=1) ## reduce la matriz a cuatro opciones (cambios de direccion, sube o baja) cada una de estas esta expresada con un número y una posición para cada caso.
    decision = (decision[0]-decision[3])/matrix.sum() ## En el caso de muestra decision[0] era la probabilidad de que el siguiente patron fuera de subida, decision[1] era la probabilidad de que el siguiente patrón fuera de bajada. Por ello se restan.
    Decisiones.append(decision)
    
    if decision > .6:
        decision = 1
    elif decision < -.19: 
        decision = -1
    else: 
        decision = 0
    situacion = compra(situacion[0],situacion[1],close[i+5],decision*1000,comision)
    historial.append((situacion[0], situacion[1],situacion[1]+close[i+5]*situacion[0]))

        
    
print(situacion)

historial = np.array(historial)

plt.plot(historial)
plt.legend()

