import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

# Función de activación sigmoide
def sigmoid(x):
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))

def sigmoid_derivada(x):
    clipped_x = np.clip(x, -500, 500)
    return clipped_x * (1 - clipped_x)

#Reemplazar valores
def sustituir(x):
  if x<0:
    return 0
  else:
    return 1

def MLP_entrenamiento(entrada_train, salida_train, neuronas_capas_ocultas, aprendizaje, epocas):
    num_entradas = len(entrada_train[0])
    salida_train = salida_train.tolist()
    num_salidas = len(salida_train)
    num_muestras = len(entrada_train)
    num_capas_ocultas = len(neuronas_capas_ocultas)

    # Inicializar pesos y sesgos
    w_ocultas = [np.random.rand(num_entradas, neuronas_capas_ocultas[0])]
    b_ocultas = [np.random.rand(neuronas_capas_ocultas[0])]
    w_salida = [np.random.rand(neuronas_capas_ocultas[-1], num_salidas)]
    b_salida = [np.random.rand(num_salidas)]

    for i in range(1, num_capas_ocultas):
        w_ocultas.append(np.random.rand(neuronas_capas_ocultas[i-1], neuronas_capas_ocultas[i]))
        b_ocultas.append(np.random.rand(neuronas_capas_ocultas[i]))

    for _ in range(epocas):
        for l in range(num_muestras):
            # Propagación hacia adelante
            salida_oculta = sigmoid(np.dot(entrada_train[l], w_ocultas[0]) + b_ocultas[0])
            for i in range(1, num_capas_ocultas):
                salida_oculta = sigmoid(np.dot(salida_oculta, w_ocultas[i]) + b_ocultas[i])
            salida = sigmoid(np.dot(salida_oculta, w_salida[-1]) + b_salida[-1])

            # Calcular el error
            error = salida_train[l] - salida

            # Retropropagación para la capa de salida
            delta_salida = error * sigmoid_derivada(salida)
            w_salida[-1] += aprendizaje * np.outer(salida_oculta, delta_salida)
            b_salida[-1] += aprendizaje * delta_salida

            # Retropropagación para las capas ocultas
            delta_oculta = np.dot(delta_salida, w_salida[-1].T) * sigmoid_derivada(salida_oculta)
            for i in range(num_capas_ocultas-1, -1, -1):
                if i == 0:
                    entrada = entrada_train[l]
                else:
                    entrada = salida_oculta
                w_ocultas[i] += aprendizaje * np.outer(entrada, delta_oculta)
                b_ocultas[i] += aprendizaje * delta_oculta

    return w_ocultas, b_ocultas, w_salida, b_salida

def MLP_entrenamiento_adam(entrada_train, salida_train, neuronas_capas_ocultas, epocas, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    num_entradas = len(entrada_train[0])
    salida_train = salida_train.tolist()
    num_salidas = len(salida_train)
    num_muestras = len(entrada_train)
    num_capas_ocultas = len(neuronas_capas_ocultas)

    # Inicializar pesos y sesgos
    w_ocultas = [np.random.rand(num_entradas, neuronas_capas_ocultas[0])]
    b_ocultas = [np.random.rand(neuronas_capas_ocultas[0])]
    w_salida = [np.random.rand(neuronas_capas_ocultas[-1], num_salidas)]
    b_salida = [np.random.rand(num_salidas)]

    for i in range(1, num_capas_ocultas):
        w_ocultas.append(np.random.rand(neuronas_capas_ocultas[i-1], neuronas_capas_ocultas[i]))
        b_ocultas.append(np.random.rand(neuronas_capas_ocultas[i]))

    m_w_ocultas, m_b_ocultas, m_w_salida, m_b_salida = [np.zeros_like(w) for w in w_ocultas], [np.zeros_like(b) for b in b_ocultas], [np.zeros_like(w) for w in w_salida], [np.zeros_like(b) for b in b_salida]
    v_w_ocultas, v_b_ocultas, v_w_salida, v_b_salida = [np.zeros_like(w) for w in w_ocultas], [np.zeros_like(b) for b in b_ocultas], [np.zeros_like(w) for w in w_salida], [np.zeros_like(b) for b in b_salida]

    for _ in range(int(epocas)):
        for l in range(num_muestras):
            # Propagación hacia adelante
            salida_oculta = sigmoid(np.dot(entrada_train[l], w_ocultas[0]) + b_ocultas[0])
            for i in range(1, num_capas_ocultas):
                salida_oculta = sigmoid(np.dot(salida_oculta, w_ocultas[i]) + b_ocultas[i])
            salida = sigmoid(np.dot(salida_oculta, w_salida[-1]) + b_salida[-1])

            # Calcular el error
            error = salida_train[l] - salida

            # Retropropagación para la capa de salida
            delta_salida = error * sigmoid_derivada(salida)
            m_w_salida[-1] = beta1 * m_w_salida[-1] + (1 - beta1) * np.outer(salida_oculta, delta_salida)
            m_b_salida[-1] = beta1 * m_b_salida[-1] + (1 - beta1) * delta_salida
            v_w_salida[-1] = beta2 * v_w_salida[-1] + (1 - beta2) * np.outer(salida_oculta, delta_salida)**2
            v_b_salida[-1] = beta2 * v_b_salida[-1] + (1 - beta2) * delta_salida**2
            m_w_hat = m_w_salida[-1] / (1 - beta1**(l+1))
            m_b_hat = m_b_salida[-1] / (1 - beta1**(l+1))
            v_w_hat = v_w_salida[-1] / (1 - beta2**(l+1))
            v_b_hat = v_b_salida[-1] / (1 - beta2**(l+1))
            w_salida[-1] += alpha * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            b_salida[-1] += alpha * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            # Retropropagación para las capas ocultas
            delta_oculta = np.dot(delta_salida, w_salida[-1].T) * sigmoid_derivada(salida_oculta)
            for i in range(num_capas_ocultas-1, -1, -1):
                if i == 0:
                    entrada = entrada_train[l]
                else:
                    entrada = salida_oculta
                m_w_ocultas[i] = beta1 * m_w_ocultas[i] + (1 - beta1) * np.outer(entrada, delta_oculta)
                m_b_ocultas[i] = beta1 * m_b_ocultas[i] + (1 - beta1) * delta_oculta
                v_w_ocultas[i] = beta2 * v_w_ocultas[i] + (1 - beta2) * np.outer(entrada, delta_oculta)**2
                v_b_ocultas[i] = beta2 * v_b_ocultas[i] + (1 - beta2) * delta_oculta**2
                m_w_hat = m_w_ocultas[i] / (1 - beta1**(l+1))
                m_b_hat = m_b_ocultas[i] / (1 - beta1**(l+1))
                v_w_hat = v_w_ocultas[i] / (1 - beta2**(l+1))
                v_b_hat = v_b_ocultas[i] / (1 - beta2**(l+1))
                w_ocultas[i] += alpha * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
                b_ocultas[i] += alpha * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return w_ocultas, b_ocultas, w_salida, b_salida

datos = pd.read_csv("/content/concentlite.csv", header=None)
# Obtener el 80% de los datos
datos_80_porcentaje = datos.sample(frac=0.8)
# Obtener el índice de las filas que están en datos_80_porcentaje
indices_80_porcentaje = datos_80_porcentaje.index
# Obtener el 20% restante
datos_20_porcentaje = datos.drop(indices_80_porcentaje)
#Datos_test = Datos_test.applymap(sustituir)
datos_80_porcentaje.columns=['X','Y','S']
entrada1_tr = datos_80_porcentaje['X']
entrada2_tr = datos_80_porcentaje['Y']
entradas = np.column_stack((entrada1_tr, entrada2_tr))
salidas_esperada = datos_80_porcentaje['S'].values

aprendizaje = 0.1
epocas = 100
neuronas_capas_ocultas = [4, 4]
w_ocultas, b_ocultas, w_salida, b_salida = MLP_entrenamiento_adam(entradas, salidas_esperada, neuronas_capas_ocultas, aprendizaje, epocas)

datos_20_porcentaje.columns=['X','Y','S']
entrada1_tr = datos_20_porcentaje['X']
entrada2_tr = datos_20_porcentaje['Y']
entradas = np.column_stack((entrada1_tr, entrada2_tr))
salidas_esperada = datos_20_porcentaje['S'].values
# Función para hacer predicciones
def predecir(entrada, w_ocultas, b_ocultas, w_salida, b_salida):
    salida_oculta = sigmoid(np.dot(entrada, w_ocultas[0]) + b_ocultas[0])
    for i in range(1, len(w_ocultas)):
        salida_oculta = sigmoid(np.dot(salida_oculta, w_ocultas[i]) + b_ocultas[i])
    salida = sigmoid(np.dot(salida_oculta, w_salida[-1]) + b_salida[-1])
    return np.round(salida)


# Graficar las salidas
plt.figure(figsize=(10, 6))
plt.scatter(entradas[:, 0], entradas[:, 1], c=salidas_esperada)
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')
plt.title('Distribución de Clases')
plt.show()
