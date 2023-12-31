# -*- coding: utf-8 -*-
"""Descenso del Gradiente.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1njaqdlG6cO26UklhOClsW9Kjhe3J6038
"""

import matplotlib.pyplot as plt
import math

#ajuste de parametros
x_inicial = 5
alpha = 0.25
n_iteraciones = 15
def funcion(x):
  return x**2+1

x1_inicial =2
x2_inicial =3
def mi_funcion(x1, x2):
    return 10 - math.exp(-(x1**2 + 3*x2**2))

#Inicio
x = x_inicial
iteraciones = []
y=[]
for i in range(n_iteraciones):
  #print("------------------------------------------");
  #print("N° Iteracion: "+str(i+1))
  #calculo gradiente
  gradiente = 2*x
  #actualizacion de x usando gradiente descendente
  x = x-alpha*gradiente
  #almacenamos el numero de la iteracuion y el valor del mismo
  y.append(funcion(x))
  iteraciones.append(i+1)
  #mostramos resultados
  #print("x=",str(x),", y=",str(funcion(x)))

#Graficar Resultados
plt.subplot(1,2,1)
plt.plot(iteraciones,y)
plt.xlabel('Iteracion')
plt.ylabel('y')
plt.show()

#graficar funcion
x = range(-5, 5)
plt.ylim(-1, 15)
plt.plot(x, [funcion(i) for i in x])
plt.show()

x = [x1_inicial, x2_inicial]  # Inicializamos las variables x1 y x2
iteraciones = []
y = []

for i in range(n_iteraciones):
    # Calculamos el gradiente
    gradiente = [2*x[0], 6*x[1]]

    # Actualizamos x usando el gradiente descendente
    x[0] = x[0] - alpha*gradiente[0]
    x[1] = x[1] - alpha*gradiente[1]

    # Almacenamos el número de la iteración y el valor de la función
    y.append(mi_funcion(x[0], x[1]))
    iteraciones.append(i+1)

    # Mostramos resultados
    print(f"Iteración {i+1}: x1 = {x[0]}, x2 = {x[1]}, y = {mi_funcion(x[0], x[1])}")
#Graficar Resultados
plt.subplot(1,2,1)
plt.plot(iteraciones,y)
plt.xlabel('Iteracion')
plt.ylabel('valor')
plt.show()