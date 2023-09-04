import pandas as pd
import matplotlib as plt
import numpy as np

#funcion de activacion
def sigmoid(x):
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))

def relu(x):
    return np.maximum(0, x)

#Creacion de nuestro perceptron
def Perceptron_entrenamiento(entrada1_train,entrada2_train,salida_train, aprendizaje, epocas):
  pesos = np.random.rand(2)
  sesgo = np.random.rand()
  for i in range(epocas):
    for l in range(len(entrada1_train)):
      #Calcular la salida
      salida = sigmoid(np.dot([entrada1_train[l], entrada2_train[l]], [pesos[0],pesos[1]])+sesgo)
      #Calcula el error
      error = salida_train[l]- salida
      #Cambiar pesos
      pesos[0] = pesos[0] + aprendizaje*error*entrada1_train[l]
      pesos[1] = pesos[1] + aprendizaje*error*entrada2_train[l]
      sesgo += aprendizaje * error
  return pesos, sesgo

#Reemplazar valores
def sustituir(x):
  if x<0:
    return 0
  else:
    return 1

#Informacion de entrenamiento
Datos_entrenamiento = pd.read_csv("/content/XOR_trn.csv", header=None)
#Datos_entrenamiento = pd.read_csv("/content/OR_trn.csv", header=None)
Datos_entrenamiento = Datos_entrenamiento.applymap(sustituir)
Datos_entrenamiento.columns=['X','Y','S']
entrada1 = Datos_entrenamiento['X']
entrada2 = Datos_entrenamiento['Y']
salidas = Datos_entrenamiento['S']
Valor_aprendizaje=0.2
Epocas = 50
Pesos_entrenados, sesgo_entrenado = Perceptron_entrenamiento(entrada1,entrada2,salidas,Valor_aprendizaje,Epocas)

#Informacion de test
Datos_test = pd.read_csv("/content/XOR_tst.csv", header=None)
#Datos_test = pd.read_csv("/content/OR_tst.csv", header=None)
Datos_test = Datos_test.applymap(sustituir)
Datos_test.columns=['X','Y','S']
entrada3 = Datos_test['X']
entrada4 = Datos_test['Y']
salidas2 = Datos_test['S']
def predecir(x,y):
    return sigmoid(np.dot([x, y], Pesos_entrenados)+sesgo_entrenado)
# Ejemplos de predicción
salida_pro = []
for i in range(len(entrada3)):
    prediccion = predecir(entrada3[i], entrada4[i])
    salida_pro.append(prediccion)
# Obtener las coordenadas x e y por separado
#ejemplos = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
ejemplos = np.array([[0,0], [0, 1], [1,0], [1, 1]])
x = ejemplos[:, 0]
y = ejemplos[:, 1]
#Grafica de puntos
plt.scatter(x,y,color="black")
plt.plot(salida_pro,color="g")
plt.xlim(-0.02,1.02)
plt.ylim(-0.02,1.02)
plt.show()
