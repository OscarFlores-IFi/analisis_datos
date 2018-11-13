# Algoritmo genético. 
import numpy as np

l_vec = 16
l_dec = 10
### Se otorgan 3 opciones a la toma de decisiones
decisiones = [[np.random.randint(0,3)-1 for i in range(l_vec)] for i in range(l_dec)] # Inicial. 

for cic in range(200):
    a = []
    m = []
    
    for i in decisiones: ## se suman todos vectores de decisión para escoger el que de la suma mayor
        a.append(np.sum(i))
    
    for i in range(3): ## se escojen los mejores resultados
        m.append(decisiones[a.index(max(a))])
        a.pop(a.index(max(a)))
    
    m = np.array(m) ## hacemos l_vec nuevos vectores derivados únicamente de los 3 mejores anteriores.
    decisiones = [[np.random.choice(m.T[i]) for i in range(l_vec)] for i in range(l_dec)]
    for k in range(l_dec): ## mutamos un tercio de los dígitos de los l_vec vectores que tenemos. 
        for i in range(int(l_dec//3)):
            decisiones[i][np.random.randint(0,l_vec)] = np.random.randint(0,3)-1
    [decisiones.append(i) for i in m] ## agregamos los 'padres' de las nuevas generaciones a la lista. 

print(decisiones[-3])