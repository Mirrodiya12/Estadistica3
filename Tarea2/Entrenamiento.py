import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos con la opción low_memory=False para evitar advertencias
training_data = pd.read_csv('training_pruebas_transformados.csv', low_memory=False)
test_data = pd.read_csv('test_pruebas_transformados.csv', low_memory=False)

# Convertir las columnas problemáticas a numéricas (forzando errores a NaN)
training_data = training_data.apply(pd.to_numeric, errors='coerce', downcast='float')
test_data = test_data.apply(pd.to_numeric, errors='coerce', downcast='float')

# Preparar los datos: Eliminar cualquier columna que aún contenga NaN tras la conversión
X = training_data.drop(columns=['PUNT_GLOBAL']).dropna(axis=1, how='any')  # Características
y = training_data['PUNT_GLOBAL']

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de validación
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error cuadrático medio (MSE): {mse:.2f}')
print(f'Coeficiente de determinación (R²): {r2:.4f}')

# Guardar las predicciones en un archivo CSV
X_test['PUNT_GLOBAL_PRED'] = y_pred
X_test.to_csv('predicciones_test.csv', index=False)

print("Entrenamiento y predicciones completados.")
