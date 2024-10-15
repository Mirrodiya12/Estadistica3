import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Cargar los datos de entrenamiento y prueba con dtype especificado
dtype_dict = {
    'Col49': str,  # Reemplaza 'Col49' con el nombre real de la columna 49
    # Puedes agregar otras columnas que puedan tener tipos mixtos aquí
}

train_data = pd.read_csv('training_pruebas.csv', dtype=dtype_dict, low_memory=False)
test_data = pd.read_csv('test_pruebas.csv', dtype=dtype_dict, low_memory=False)

# Inspeccionar las primeras filas para identificar problemas
print(train_data.head())
print(train_data.info())  # Para ver los tipos de datos

# Separar las características y la variable objetivo en el conjunto de entrenamiento
X_train = train_data.drop('PUNT_GLOBAL', axis=1, errors='ignore')  # Características
y_train = train_data['PUNT_GLOBAL']  # Objetivo

# Cargar el conjunto de prueba
X_test = test_data  # Asumiendo que el conjunto de prueba no tiene 'PUNT_GLOBAL'

# Dividir el conjunto de entrenamiento (opcional, si deseas validación)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convertir variables categóricas en variables dummy si es necesario
X_train_split = pd.get_dummies(X_train_split, drop_first=True)
X_val_split = pd.get_dummies(X_val_split, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Alinear las columnas de los conjuntos de validación y prueba con el conjunto de entrenamiento
X_val_split, X_test = X_val_split.align(X_train_split, join='left', axis=1, fill_value=0)
X_test = X_test.reindex(columns=X_train_split.columns, fill_value=0)

# Crear y entrenar el modelo GBM
gbm = GradientBoostingRegressor()
gbm.fit(X_train_split, y_train_split)

# Hacer predicciones para el conjunto de entrenamiento (valores de entrenamiento)
y_train_pred = gbm.predict(X_train_split)

# Hacer predicciones para el conjunto de prueba
y_test_pred = gbm.predict(X_test)

# Calcular MSE y MAPE para el conjunto de entrenamiento
mse_train = mean_squared_error(y_train_split, y_train_pred)
mape_train = mean_absolute_percentage_error(y_train_split, y_train_pred)

# Guardar las predicciones en un DataFrame para exportar o evaluar
predictions_df = pd.DataFrame({
    'ID': test_data['ID'],  # Suponiendo que hay una columna 'ID' en test_pruebas.csv
    'PREDICCION': y_test_pred
})

# Guardar las predicciones en un archivo CSV
predictions_df.to_csv('predicciones.csv', index=False)

# Imprimir los resultados
print(f'Train MSE: {mse_train:.4f}')
print(f'Train MAPE: {mape_train:.4f}')
