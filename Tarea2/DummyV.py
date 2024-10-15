import dask.dataframe as dd

def main():
    # Especificar los tipos de datos al cargar el CSV
    dtypes = {
        'ESTU_SEMESTRECURSA': 'object',  # Asegúrate de que sea un tipo adecuado
        # Agrega aquí otros tipos de datos si es necesario
    }

    # Cargar el dataset con Dask
    df = dd.read_csv('archivo_imputado.csv', dtype=dtypes)

    # Variable objetivo
    variable_objetivo = 'PUNT_GLOBAL'

    # Crear una lista de columnas categóricas
    columnas_categoricas = [
        'ESTU_TIPODOCUMENTO',
        'ESTU_NACIONALIDAD',
        'ESTU_GENERO',
        'ESTU_EXTERIOR',
        'ESTU_PAIS_RESIDE',
        'ESTU_DEPTO_RESIDE',
        'ESTU_ESTADOCIVIL',
        'ESTU_TITULOOBTENIDOBACHILLER',
        'FAMI_EDUCACIONPADRE',
        'FAMI_EDUCACIONMADRE',
        'FAMI_OCUPACIONPADRE',
        'FAMI_OCUPACIONMADRE',
        'FAMI_ESTRATOVIVIENDA',
        'FAMI_TIENESERVICIOTV',
        'FAMI_TIENECOMPUTADOR',
        'FAMI_TIENELAVADORA',
        'FAMI_TIENEAUTOMOVIL',
        'FAMI_TIENEMOTOCICLETA',
        'FAMI_TIENECONSOLAVIDEOJUEGOS',
        'ESTU_COMOCAPACITOEXAMENSB11',
        'ESTU_ACTIVIDADREFUERZOAREAS',
        'ESTU_CURSOIESAPOYOEXTERNO',
        'ESTU_CURSOIESEXTERNA',
        'FAMI_TRABAJOLABORPADRE',
        'FAMI_TRABAJOLABORMADRE',
    ]

    # Convertir columnas categóricas a tipo categórico y establecer categorías conocidas
    for col in columnas_categoricas:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.as_known()

    # Crear dummy variables
    df_dummies = dd.get_dummies(df, columns=columnas_categoricas)

    # Unir las dummy variables al dataframe original
    df_final = df_dummies.persist()
    df_final = df_final.drop(columns=columnas_categoricas)  # Eliminar columnas originales

    # Calcular la correlación
    correlacion = df_final.corr().compute()

    # Imprimir la correlación
    print("Matriz de correlación:")
    print(correlacion)

    # Correlaciones con la variable objetivo
    correlaciones_punt_global = correlacion[variable_objetivo].sort_values(ascending=False)

    print("Correlaciones con PUNT_GLOBAL:")
    print(correlaciones_punt_global)

    # Efectos positivos y negativos
    positivos = correlaciones_punt_global[correlaciones_punt_global > 0]
    negativos = correlaciones_punt_global[correlaciones_punt_global < 0]

    print("\nVariables con efecto positivo en PUNT_GLOBAL:")
    print(positivos)

    print("\nVariables con efecto negativo en PUNT_GLOBAL:")
    print(negativos)

if __name__ == "__main__":
    main()
