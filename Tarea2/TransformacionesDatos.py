import pandas as pd
from datetime import datetime

# Cargar los archivos CSV
training_data = pd.read_csv('training_pruebas.csv', low_memory=False)
test_data = pd.read_csv('test_pruebas.csv', low_memory=False)

# Función para calcular la edad
def calcular_edad(fecha_nacimiento):
    if pd.isnull(fecha_nacimiento):
        return None
    # Intentar convertir la fecha y manejar errores
    fecha_nacimiento = pd.to_datetime(fecha_nacimiento, errors='coerce', format='%d/%m/%Y')
    if pd.isnull(fecha_nacimiento):
        return None  # Devuelve None si la conversión falla
    edad = (datetime.now() - fecha_nacimiento).days // 365
    return edad

# Aplicar la función de calcular edad
for data in [training_data, test_data]:
    data['EDAD'] = data['ESTU_FECHANACIMIENTO'].apply(calcular_edad)

# Agrupar variables categóricas con alta cardinalidad
def agrupar_categoria(df, columna, nuevo_nombre):
    if columna in df.columns:
        # Crear un mapeo a categorías menos numerosas (personaliza según tus datos)
        mapeo = {
            'CC': 'Identificación',
            'CE': 'Identificación',
            'TI': 'Identificación',
            # Agrega más mapeos según tu criterio
        }
        df[nuevo_nombre] = df[columna].map(mapeo).fillna('Otros')
        df.drop(columns=[columna], inplace=True)

# Agrupar variables categóricas en ambos conjuntos
categorias_a_agrupar = ['ESTU_TIPODOCUMENTO', 'ESTU_NACIONALIDAD', 'FAMI_OCUPACIONPADRE', 'FAMI_OCUPACIONMADRE']
for data in [training_data, test_data]:
    for categoria in categorias_a_agrupar:
        agrupar_categoria(data, categoria, categoria + '_AGRUPADA')

# Manejar valores nulos (llenando con la mediana para columnas numéricas)
for data in [training_data, test_data]:
    # Filtrar solo las columnas numéricas
    num_cols = data.select_dtypes(include=['number']).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# Guardar los datos transformados si es necesario
training_data.to_csv('training_pruebas_transformados.csv', index=False)
test_data.to_csv('test_pruebas_transformados.csv', index=False)

print("Transformaciones completadas.")
