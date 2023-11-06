#LIBRERIAS EN GENERAL
import pandas as pd
import statsmodels.api as sm
import numpy as np
import tensorflow as tf
import warnings
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
# Suprimir todos los warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('/content/winequality-white.csv')
df.hist()

"""# *REGRESION LOGISTICA*

---

"""

# DataFrame de auto
df = pd.read_csv('/content/AutoInsurSweden.csv')
X = df[['x','y']]
y = df[['z']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regression_model = LogisticRegression()
# Entrenar el modelo
regression_model.fit(X_train, y_train)
# Realizar predicciones
pred = regression_model.predict(X_test)

# Evaluar el rendimiento en el conjunto de entrenamiento y prueba
print('R2 score on training set: {:.2f}'.format(regression_model.score(X_train, y_train)))
print('R2 score on test set: {:.2f}'.format(regression_model.score(X_test, y_test)))
#print(df.keys())
#df.describe()
#df.hist()

#DataFrame de Diabetes
df = pd.read_csv('/content/Pima Indians Diabetes Dataset.csv')
df = df.dropna(subset=['Class'])
X = df.drop('Class',axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regression_model = LogisticRegression()
# Entrenar el modelo
regression_model.fit(X_train, y_train)
# Realizar predicciones
pred = regression_model.predict(X_test)


# Evaluar el rendimiento en el conjunto de entrenamiento y prueba
print('R2 score on training set: {:.2f}'.format(regression_model.score(X_train, y_train)))
print('R2 score on test set: {:.2f}'.format(regression_model.score(X_test, y_test)))
#df.hist()

# DataFrame Vino
df = pd.read_csv('/content/winequality-white.csv')
df = df.dropna(subset=['quality'])
X = df.drop('quality',axis=1)
y = df['quality']
#df.hist()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regression_model = LogisticRegression()
# Entrenar el modelo
regression_model.fit(X_train, y_train)
# Realizar predicciones
pred = regression_model.predict(X_test)


# Evaluar el rendimiento en el conjunto de entrenamiento y prueba
print('R2 score on training set: {:.2f}'.format(regression_model.score(X_train, y_train)))
print('R2 score on test set: {:.2f}'.format(regression_model.score(X_test, y_test)))

"""# *K-Vecinos Cercanos*

---

"""

# DataFrame de auto
df = pd.read_csv('/content/AutoInsurSweden.csv')
X = df[['x','y']]
y = df[['z']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

#df.describe()
#df.hist()
#print(confusion_matrix(y_test, pred))
#print(classification_report(y_test, pred))

#DataFrame de Diabetes
df = pd.read_csv('/content/Pima Indians Diabetes Dataset.csv')
df = df.dropna(subset=['Class'])
X = df.drop('Class',axis=1)
y = df['Class']
#print(df.groupby('Class').size())
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
#print(confusion_matrix(y_test, pred))
#print(classification_report(y_test, pred))

# DataFrame Vino
df = pd.read_csv('/content/winequality-white.csv')
df = df.dropna(subset=['quality'])
X = df.drop('quality',axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
#print(confusion_matrix(y_test, pred))
#print(classification_report(y_test, pred))

"""# *Máquinas de Vectores de Soporte*

---

"""

# DataFrame de auto
df = pd.read_csv('/content/AutoInsurSweden.csv')
X = df[['x','y']]
y = df[['z']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Otras métricas (opcional)
#print(df.keys())
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))

#DataFrame de Diabetes
df = pd.read_csv('/content/Pima Indians Diabetes Dataset.csv')
df = df.dropna(subset=['Class'])
X = df.drop('Class',axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Otras métricas (opcional)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# DataFrame Vino
df = pd.read_csv('/content/winequality-white.csv')
df = df.dropna(subset=['quality'])
X = df.drop('quality',axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Otras métricas (opcional)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#df.hist()

"""# *Naive Bayes*

---

"""

# DataFrame de auto
df = pd.read_csv('/content/AutoInsurSweden.csv')
X = df[['x','y']]
y = df[['z']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = GaussianNB()
model.fit(X, y);
yprob = model.predict_proba(X_test)
y_pred = model.predict(X_test)


print('Accuracy of NB classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
#print(df.keys())

#DataFrame de Diabetes
df = pd.read_csv('/content/Pima Indians Diabetes Dataset.csv')
df = df.dropna(subset=['Class'])
X = df.drop('Class',axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = GaussianNB()
model.fit(X, y);
yprob = model.predict_proba(X_test)
y_pred = model.predict(X_test)


print('Accuracy of NB classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# DataFrame Vino
df = pd.read_csv('/content/winequality-white.csv')
df = df.dropna(subset=['quality'])
X = df.drop('quality',axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = GaussianNB()
model.fit(X, y);
yprob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

print('Accuracy of NB classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

"""# *Red Neuronal con Tensorflow*

---
"""

# DataFrame de auto
df = pd.read_csv('/content/AutoInsurSweden.csv')
X = df[['x','y']]
y = df[['z']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Puedes ajustar los hiperparámetros según tus necesidades
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp.fit(X_train, y_train)
# Hacer predicciones
y_pred = mlp.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
#print(df.keys())
#df.hist()

# DataFrame Diabetes
df = pd.read_csv('/content/Pima Indians Diabetes Dataset.csv')
df = df.dropna(subset=['Class'])
X = df.drop('Class',axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Puedes ajustar los hiperparámetros según tus necesidades
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp.fit(X_train, y_train)
# Hacer predicciones
y_pred = mlp.predict(X_test)


# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# DataFrame Vino
df = pd.read_csv('/content/winequality-white.csv')
df = df.dropna(subset=['quality'])
X = df.drop('quality',axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Puedes ajustar los hiperparámetros según tus necesidades
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp.fit(X_train, y_train)
# Hacer predicciones
y_pred = mlp.predict(X_test)
# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#df.hist()
