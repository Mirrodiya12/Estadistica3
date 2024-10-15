import dask.dataframe as dd

# Especificar los tipos de datos de las columnas problemáticas
dtypes = {
    'ESTU_ACTIVIDADREFUERZOAREAS': 'object',
    'ESTU_ACTIVIDADREFUERZOGENERIC': 'object',
    'ESTU_CURSODOCENTESIES': 'object',
    'ESTU_CURSOIESAPOYOEXTERNO': 'object',
    'ESTU_CURSOIESEXTERNA': 'object',
    'ESTU_PRESENTACIONCASA': 'object',
    'ESTU_SEMESTRECURSA': 'object',  # Cambiado a 'object' para evitar problemas
    'ESTU_SIMULACROTIPOICFES': 'object'
}

# Cargar los datos con tipos de datos especificados
training_df = dd.read_csv('training_pruebas.csv', dtype=dtypes, low_memory=False)
test_df = dd.read_csv('test_pruebas.csv', dtype=dtypes, low_memory=False)

# Verificar los nombres de las columnas
print("Nombres de columnas en el conjunto de entrenamiento:")
print(training_df.columns.tolist())

print("\nNombres de columnas en el conjunto de prueba:")
print(test_df.columns.tolist())

# Mostrar las primeras filas
print("\nPrimeras filas del conjunto de entrenamiento:")
print(training_df.head(5))  # No se necesita compute() aquí

print("\nPrimeras filas del conjunto de prueba:")
print(test_df.head(5))  # No se necesita compute() aquí
