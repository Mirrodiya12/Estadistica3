import dask.dataframe as dd
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Definir tipos de datos manualmente
dtypes = {
    'ESTU_ACTIVIDADREFUERZOAREAS': 'object',
    'ESTU_ACTIVIDADREFUERZOGENERIC': 'object',
    'ESTU_CURSODOCENTESIES': 'object',
    'ESTU_CURSOIESAPOYOEXTERNO': 'object',
    'ESTU_CURSOIESEXTERNA': 'object',
    'ESTU_PRESENTACIONCASA': 'object',
    'ESTU_SEMESTRECURSA': 'object',
    'ESTU_SIMULACROTIPOICFES': 'object'
}

# Cargar los datos con Dask especificando los dtypes
training_data = dd.read_csv('D:\\Juegos\\Tarea2\\training_pruebas_transformados.csv', dtype=dtypes)
test_data = dd.read_csv('D:\\Juegos\\Tarea2\\test_pruebas_transformados.csv', dtype=dtypes)

# Convertir la columna de fecha de nacimiento en edad
def calcular_edad(fecha_nacimiento):
    fecha_nacimiento = dd.to_datetime(fecha_nacimiento, errors='coerce')  # Convertir a datetime
    edad = (pd.Timestamp.now() - fecha_nacimiento) // pd.Timedelta(days=365.25)  # Calcular edad en años
    edad = edad.fillna(-1).astype(int)  # Reemplazar NaT con -1 y convertir a int
    return edad

# Aplicar la función a las columnas de los datasets
training_data['EDAD'] = calcular_edad(training_data['ESTU_FECHANACIMIENTO'])
test_data['EDAD'] = calcular_edad(test_data['ESTU_FECHANACIMIENTO'])

# Eliminar la columna de fecha de nacimiento si no es necesaria
X_train = training_data.drop(['PUNT_GLOBAL', 'ESTU_FECHANACIMIENTO'], axis=1)
y_train = training_data['PUNT_GLOBAL']

# Verificar si PUNT_GLOBAL está presente en el conjunto de prueba
if 'PUNT_GLOBAL' in test_data.columns:
    X_test = test_data.drop(['PUNT_GLOBAL', 'ESTU_FECHANACIMIENTO'], axis=1)
    y_test = test_data['PUNT_GLOBAL']
else:
    X_test = test_data
    y_test = None  # No se proporcionan valores reales de prueba

# Convertir columnas categóricas a tipo 'category'
def convert_to_category(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category')
    return df

X_train = X_train.map_partitions(convert_to_category, meta=X_train)
X_test = X_test.map_partitions(convert_to_category, meta=X_test)

# Categorizar todas las columnas categóricas
X_train = X_train.categorize()
X_test = X_test.categorize()

# Obtener dummies en el conjunto de entrenamiento
X_train = dd.get_dummies(X_train, drop_first=True)

# Obtener dummies en el conjunto de prueba usando las mismas columnas que X_train
X_test = dd.get_dummies(X_test, drop_first=True)

# Alinear columnas de entrenamiento y prueba, llenando con ceros donde sea necesario
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Asegurarse de que todos los datos sean numéricos antes de entrenar el modelo
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train.compute(), y_train.compute())

# Realizar predicciones
y_pred = modelo.predict(X_test.compute())

# Evaluar el modelo si los valores reales están disponibles
if y_test is not None:
    r2 = r2_score(y_test.compute(), y_pred)
    mse = mean_squared_error(y_test.compute(), y_pred)

    # Filtrar casos donde PUNT_GLOBAL no sea 0 para el cálculo de MAPE
    valid_indices = y_test.compute() != 0
    y_test_filtered = y_test[valid_indices].compute()
    y_pred_filtered = y_pred[valid_indices]

    # Calcular MAPE
    if len(y_test_filtered) > 0:  # Verificar que hay datos válidos para el cálculo
        mape = np.mean(np.abs((y_test_filtered - y_pred_filtered) / y_test_filtered)) * 100
    else:
        mape = np.nan  # Asignar NaN si no hay datos válidos

    print("\nEjemplos de valores reales y predichos:")
    for real, predicho in zip(y_test[:10], y_pred[:10]):
        print(f"Real: {real}, Predicho: {predicho}")

    print(f"\nR² (R-squared): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
else:
    print("\nNo se proporcionaron valores reales de PUNT_GLOBAL para evaluar el modelo.")

# Guardar las predicciones en un archivo CSV
X_test['PUNT_GLOBAL_PRED'] = y_pred
X_test.to_csv('D:/Juegos/Tarea2/predicciones_test.csv', index=False)

print("\nEntrenamiento, evaluación y guardado de predicciones completados.")
