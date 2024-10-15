import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta al archivo Excel
ruta_archivo = 'training_pruebas.xlsx'

# Cargar el archivo Excel
df = pd.read_excel(ruta_archivo, engine='openpyxl')

# Ver las primeras filas del DataFrame
print(df.head())

# Seleccionar solo columnas numéricas para el análisis
df_numerico = df.select_dtypes(include=['number'])

# Verificar las columnas seleccionadas
print("Columnas numéricas seleccionadas:", df_numerico.columns)

# Calcular la correlación entre las variables numéricas
correlacion = df_numerico.corr()
print("Matriz de correlación:\n", correlacion)

# Graficar la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

# Histograma para cada columna numérica
df_numerico.hist(bins=20, figsize=(15, 10))
plt.suptitle('Histogramas de Variables Numéricas')
plt.show()

# Boxplots para cada columna numérica
for col in df_numerico.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_numerico[col])
    plt.title(f'Boxplot de {col}')
    plt.show()

# Scatter plots entre PUNT_GLOBAL y otras variables numéricas
for col in df_numerico.columns:
    if col != 'PUNT_GLOBAL':
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df_numerico[col], y=df_numerico['PUNT_GLOBAL'])
        plt.title(f'Scatter Plot: {col} vs PUNT_GLOBAL')
        plt.show()
