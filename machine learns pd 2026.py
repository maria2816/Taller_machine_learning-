import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Cargar el dataset Iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 1. Descripcion y  estadisticas basicas
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nEstadisticas descriptivas:")
print(df.describe())

# 2. Visualizacion de distribuciones (pairplot)
sns.pairplot(df, hue='target', palette='viridis')
plt.suptitle("Distribucion y Relaciones de las características de Iris", y=1.02)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Dividir variables
X = df.drop(columns=['target'])
y = df['target']

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar  características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#Modelo 1: Arbol de Desicion
clf_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_tree.fit(X_train_scaled, y_train)

#Modelo 2: k-NN
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Predicciones
y_pred_tree = clf_tree.predict(X_test_scaled)
y_pred_knn = clf_knn.predict(X_test_scaled)


print("---ARBOL DE DECISION---")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("F1-Score (Macro):", f1_score(y_test, y_pred_tree, average='macro'))
print("Matriz de Confusion:\n", confusion_matrix(y_test, y_pred_tree))


print("\n---K-NEAREST NEIGHBORS---")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("F1-Score (Macro):", f1_score(y_test, y_pred_knn, average='macro'))
print("Matriz de Confusion:\n", confusion_matrix(y_test, y_pred_knn))

