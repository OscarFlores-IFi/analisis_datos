import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compra(acciones, dinero, precio, cantidad, comision, historial):
    historial.append((cantidad, precio, precio*cantidad,dinero+acciones*precio))
    return acciones + cantidad, dinero-(precio*cantidad)*(1+comision)

def venta(acciones, dinero, precio, cantidad, comision, historial):
    historial.append((-cantidad, precio, -precio*cantidad,dinero+acciones*precio))
    return acciones - cantidad, dinero+(precio*cantidad)*(1-comision)


cierre = pd.read_csv('AC.csv')['Close']

dinero = 1000000
comision = 0.05
acciones = 0

historial = []
situacion = [acciones, dinero]

for i in range(100):
    if i%2 == 0:
        situacion = compra(situacion[0], situacion[1],cierre[i],np.ceil(np.random.randint(0,dinero//cierre[i])),comision, historial)
    if i%2 == 1:
        situacion = venta(situacion[0], situacion[1],cierre[i],np.ceil(np.random.randint(0,situacion[0])),comision, historial)
print(situacion)

historial = np.array(historial)

plt.plot(historial)
