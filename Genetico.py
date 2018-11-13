# Algoritmo genético. 
import numpy as np


decisiones = [[np.random.randint(0,3)-1 for i in range(25)] for i in range(10)] # Inicial. 

for cic in range(200):
    a = []
    m = []
    
    for i in decisiones: ## se suman todos vectores de decisión para escoger el que de la suma mayor
        a.append(np.sum(i))
    
    for i in range(3): ## se escojen losejores resultados
        m.append(decisiones[a.index(max(a))])
        a.pop(a.index(max(a)))
    
    m = np.array(m) ## hacemos 10 nuevos vectores derivados únicamente de los 3 mejores anteriores.
    decisiones = [[np.random.choice(m.T[i]) for i in range(25)] for i in range(10)]
    for i in range(10): ## mutamos 5 dígitos de los 10 vectores que tenemos. 
        for i in range(5):
            decisiones[i][np.random.randint(0,25)] = np.random.randint(0,3)-1
    [decisiones.append(i) for i in m] ## agregamos los 'padres' de las nuevas generaciones a la lista. 

print(decisiones)