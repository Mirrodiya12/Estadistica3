import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Ruta al archivo Excel
ruta_archivo = 'training_pruebas.xlsx'

# Cargar el archivo Excel
df = pd.read_excel(ruta_archivo, engine='openpyxl')

# Ver las primeras filas del DataFrame
print(df.head())

# Seleccionar solo columnas numéricas y eliminar filas con valores faltantes
df_numerico = df.select_dtypes(include=['number']).dropna()

# Verificar las columnas numéricas seleccionadas
print("Columnas numéricas seleccionadas:", df_numerico.columns)

# Calcular la correlación con la variable objetivo (PUNT_GLOBAL)
correlacion = df_numerico.corr()
print("Matriz de correlación:\n", correlacion)

# Visualizar la correlación de las variables con PUNT_GLOBAL
plt.figure(figsize=(10, 8))
sns.heatmap(correlacion[['PUNT_GLOBAL']].sort_values(by='PUNT_GLOBAL', ascending=False),
            annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlación con PUNT_GLOBAL')
plt.show()

# Preparar los datos para el modelo de regresión lineal
X = df_numerico.drop(columns=['PUNT_GLOBAL'])
y = df_numerico['PUNT_GLOBAL']

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo de regresión lineal
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)

# Mostrar los coeficientes de la regresión lineal
coeficientes = pd.Series(modelo_lineal.coef_, index=X.columns).sort_values(ascending=False)
print("Coeficientes de la Regresión Lineal:\n", coeficientes)

# Visualizar los coeficientes en un gráfico de barras
plt.figure(figsize=(10, 6))
coeficientes.plot(kind='bar')
plt.title('Importancia de las Variables según Regresión Lineal')
plt.show()

# Entrenar un modelo de Random Forest para analizar la importancia de las variables
modelo_rf = RandomForestRegressor(random_state=42)
modelo_rf.fit(X_train, y_train)

# Mostrar la importancia de las variables según el Random Forest
importancia_rf = pd.Series(modelo_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Importancia de las Variables según Random Forest:\n", importancia_rf)

# Visualizar la importancia de las variables en un gráfico de barras
plt.figure(figsize=(10, 6))
importancia_rf.plot(kind='bar', color='skyblue')
plt.title('Importancia de las Variables según Random Forest')
plt.show()
