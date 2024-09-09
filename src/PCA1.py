import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Datos completos de la tabla
ciudades = [
    ["City", "GDP (USD Billion)", "Population (Millions)", "Unemployment Rate (%)", "Average Age", "Women (%)", "Men (%)", "Budget (USD Billion)"],
    ["Bogotá", 103.5, 7.18, 10.5, 32, 52, 48, 18],
    ["Medellín", 44.1, 2.57, 11.2, 31, 53, 47, 7.5],
    ["Cali", 22.4, 2.23, 13.8, 30, 52, 48, 4.2],
    ["Barranquilla", 16.8, 1.23, 12.4, 29, 51, 49, 3.1],
    ["Cartagena", 10.5, 1.03, 10.9, 30, 51, 49, 2.8],
    ["Bucaramanga (test)", 7.3, 0.58, 9.2, 33, 52, 48, 1.5],
    ["Pereira", 6.2, 0.48, 12, 32, 52, 48, 1.3],
    ["Cúcuta (test)", 5.1, 0.76, 16.3, 28, 51, 49, 1.2],
    ["Ibagué (test)", 4.8, 0.53, 13.4, 31, 52, 48, 1.1],
    ["Santa Marta", 4, 0.52, 11.6, 29, 51, 49, 0.9],
    ["Manizales", 3.8, 0.43, 10.7, 32, 53, 47, 0.8],
    ["Villavicencio", 3.5, 0.5, 13, 30, 51, 49, 0.8],
    ["Pasto", 3.2, 0.45, 12.9, 31, 52, 48, 0.7],
    ["Montería", 3, 0.49, 13.5, 29, 51, 49, 0.7],
    ["Valledupar", 2.8, 0.47, 14.8, 28, 51, 49, 0.6],
    ["Neiva", 2.5, 0.35, 14.1, 30, 52, 48, 0.6],
    ["Popayán", 2.3, 0.33, 15.2, 31, 52, 48, 0.5],
    ["Armenia", 2.1, 0.3, 13.3, 32, 53, 47, 0.5],
    ["Sincelejo", 2, 0.28, 16.5, 29, 51, 49, 0.5],
    ["Tunja", 1.8, 0.25, 10, 31, 52, 48, 0.4],
    ["Florencia", 1.7, 0.2, 17.5, 28, 51, 49, 0.4],
    ["Riohacha", 1.5, 0.22, 15.7, 27, 51, 49, 0.3],
    ["Quibdó", 1.3, 0.13, 18.2, 26, 52, 48, 0.3],
    ["San Andrés", 1.2, 0.08, 14, 27, 50, 50, 0.2],
    ["Yopal", 1.1, 0.15, 11.5, 29, 51, 49, 0.2],
    ["Leticia", 1, 0.05, 13.6, 26, 51, 49, 0.1],
    ["Arauca (test)", 0.9, 0.08, 12.2, 29, 51, 49, 0.1],
    ["Mocoa (test)", 0.8, 0.04, 15, 28, 52, 48, 0.1],
    ["Mitú", 0.7, 0.01, 20, 25, 51, 49, 0.05],
    ["Puerto Carreño (test)", 0.6, 0.01, 22, 24, 50, 50, 0.05]
]

# Extraer los datos necesarios
data = {
    "City": [row[0] for row in ciudades[1:]],
    "GDP (USD Billion)": [row[1] for row in ciudades[1:]],
    "Population (Millions)": [row[2] for row in ciudades[1:]]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Seleccionar las columnas para PCA
X = df[["GDP (USD Billion)", "Population (Millions)"]]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Matriz de covarianza
cov_matrix = np.cov(X_scaled.T)

# Eigenvalues y eigenvectors
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# Varianza explicada por el eigenvalue
explained_variance_ratio = pca.explained_variance_ratio_

# Matriz proyectada
X_projected = X_pca

# Error o diferencia entre la matriz proyectada y los datos originales
# Proyección inversa (reconstrucción)
X_projected_back = X_projected @ eigenvectors
X_projected_back = scaler.inverse_transform(X_projected_back)
error = np.linalg.norm(X_scaled - X_projected_back)

# Imprimir resultados
print("Matriz de covarianza:")
print(cov_matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nVarianza explicada por el eigenvalue:")
print(explained_variance_ratio)
print("\nEigenvector:")
print(eigenvectors)
print("\nMatriz proyectada:")
print(X_projected)
print("\nError entre la matriz proyectada y los datos originales:")
print(error)

# Pintar todas las ciudades en 1 dimensión
plt.figure(figsize=(10, 2))
plt.scatter(X_pca, np.zeros_like(X_pca), c='blue', label='Cities')
for i, city in enumerate(df["City"]):
    plt.text(X_pca[i], 0, city, fontsize=9)
plt.xlabel('Principal Component 1')
plt.title('Cities in 1 Dimension')
plt.yticks([])
plt.grid(True)
plt.show()

# Pintar todas las ciudades en 2 dimensiones
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', label='Cities')
for i, city in enumerate(df["City"]):
    plt.text(X_scaled[i, 0], X_scaled[i, 1], city, fontsize=9)
plt.xlabel('GDP (USD Billion)')
plt.ylabel('Population (Millions)')
plt.title('Cities in 2 Dimensions')
plt.grid(True)
plt.show()
