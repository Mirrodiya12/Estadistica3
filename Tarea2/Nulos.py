import pandas as pd

# Cargar el archivo Excel
ruta_archivo = 'training_pruebas.xlsx'
df = pd.read_excel(ruta_archivo)

# 1. Verificar si existen valores nulos en el DataFrame
nulos_totales = df.isnull().sum().sum()
print(f"\nNúmero total de valores nulos: {nulos_totales}")

if nulos_totales > 0:
    # Mostrar el número de nulos por columna
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # 2. Sugerencias de imputación
    print("\nSugerencias de imputación:")

    for columna in df.columns:
        nulos_columna = df[columna].isnull().sum()

        if nulos_columna > 0:
            print(f"\nColumna: {columna}")
            if pd.api.types.is_numeric_dtype(df[columna]):
                # Para datos numéricos, sugerimos imputar con la media o mediana
                media = df[columna].mean()
                mediana = df[columna].median()
                print(f" - Tipo: Numérico")
                print(f" - Imputar con media: {media:.2f} o mediana: {mediana:.2f}")
            elif pd.api.types.is_string_dtype(df[columna]):
                # Para datos categóricos, sugerimos imputar con la moda
                moda = df[columna].mode()[0]
                print(f" - Tipo: Categórico")
                print(f" - Imputar con la moda: {moda}")
            else:
                print(f" - Tipo desconocido. Revisar manualmente.")

    # 3. Imputación automática
    df_imputado = df.fillna({
        col: df[col].mean() if pd.api.types.is_numeric_dtype(df[col])
        else df[col].mode()[0] for col in df.columns if df[col].isnull().sum() > 0
    })

    print("\nValores imputados. Revisa el DataFrame imputado:")
    print(df_imputado.head())

    # Guardar el DataFrame imputado en un archivo CSV
    df_imputado.to_csv('archivo_imputado.csv', index=False, encoding='utf-8')
    print("\nArchivo imputado guardado como 'archivo_imputado.csv'.")
else:
    print("No se encontraron valores nulos.")

