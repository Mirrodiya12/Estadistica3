import dask.dataframe as dd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Definir los dtypes para las columnas problemáticas
dtype_dict = {
    'ESTU_ACTIVIDADREFUERZOAREAS': 'object',
    'ESTU_ACTIVIDADREFUERZOGENERIC': 'object',
    'ESTU_CURSODOCENTESIES': 'object',
    'ESTU_CURSOIESAPOYOEXTERNO': 'object',
    'ESTU_CURSOIESEXTERNA': 'object',
    'ESTU_PRESENTACIONCASA': 'object',
    'ESTU_SEMESTRECURSA': 'object',
    'ESTU_SIMULACROTIPOICFES': 'object'
}

# Cargar los datos
training_df = dd.read_csv('training_pruebas.csv', dtype=dtype_dict, low_memory=False)
test_df = dd.read_csv('test_pruebas.csv', dtype=dtype_dict, low_memory=False)

# Asegurarse de que los nombres de las columnas no tengan espacios
training_df.columns = training_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Definir la columna objetivo
target_column = 'PUNT_GLOBAL'

# Separar las características y la columna objetivo en el conjunto de entrenamiento
X_train = training_df.drop(target_column, axis=1)
y_train = training_df[target_column]

# Separar las características en el conjunto de prueba (sin y_test)
X_test = test_df  # No hay columna objetivo en el conjunto de prueba

# Convertir columnas de tipo objeto a categórico
X_train = X_train.categorize()  # Usar categorize en el conjunto de entrenamiento
X_test = X_test.categorize()      # Usar categorize en el conjunto de prueba

# Convertir las columnas categóricas a dummies
X_train = dd.get_dummies(X_train, drop_first=True).compute()
X_test = dd.get_dummies(X_test, drop_first=True).compute()

# Asegurarse de que los conjuntos de entrenamiento y prueba tengan las mismas columnas
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

# Convertir las series de entrenamiento a numpy arrays
y_train = y_train.compute()

# Lista de valores de K a probar
k_values = [5, 10, 20, 30]
mse_results = []

# Inicializar una lista para almacenar las predicciones
predicciones = []

for k in k_values:
    # Crear y ajustar el modelo
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = knn.predict(X_test)

    # Almacenar las predicciones
    predicciones.append((k, y_pred))

    # Si tuvieras una forma de calcular el MSE (por ejemplo, con un conjunto de validación)
    # puedes agregarlo aquí. En este caso, simplemente mostramos las predicciones.
    print(f'K={k}, Predicciones: {y_pred}')

# Si tienes una manera de evaluar las predicciones más adelante, puedes usar mse_results
# Para calcular el MSE, necesitarías las etiquetas reales de otro conjunto (por ejemplo, un conjunto de validación)
