import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Datos de las 24 ciudades (entrenamiento)
datos_entrenamiento = np.array([
    [7.18, 103.5, 7.18, 10.5, 32, 52, 48, 18],  # Bogotá
    [2.57, 44.1, 2.57, 11.2, 31, 53, 47, 7.5],  # Medellín
    [2.23, 22.4, 2.23, 13.8, 30, 52, 48, 4.2],  # Cali
    [1.23, 16.8, 1.23, 12.4, 29, 51, 49, 3.1],  # Barranquilla
    [1.03, 10.5, 1.03, 10.9, 30, 51, 49, 2.8],  # Cartagena
    [0.48, 6.2, 0.48, 12, 32, 52, 48, 1.3],  # Pereira
    [0.52, 4, 0.52, 11.6, 29, 51, 49, 0.9],  # Santa Marta
    [0.43, 3.8, 0.43, 10.7, 32, 53, 47, 0.8],  # Manizales
    [0.5, 3.5, 0.5, 13, 30, 51, 49, 0.8],  # Villavicencio
    [0.45, 3.2, 0.45, 12.9, 31, 52, 48, 0.7],  # Pasto
    [0.49, 3, 0.49, 13.5, 29, 51, 49, 0.7],  # Montería
    [0.47, 2.8, 0.47, 14.8, 28, 51, 49, 0.6],  # Valledupar
    [0.35, 2.5, 0.35, 14.1, 30, 52, 48, 0.6],  # Neiva
    [0.33, 2.3, 0.33, 15.2, 31, 52, 48, 0.5],  # Popayán
    [0.3, 2.1, 0.3, 16.5, 29, 51, 49, 0.5],  # Armenia
    [0.28, 2, 0.28, 17.8, 28, 52, 48, 0.5],  # Sincelejo
    [0.25, 1.8, 0.25, 11.6, 31, 52, 48, 0.4],  # Tunja
    [0.2, 1.7, 0.2, 18, 29, 51, 49, 0.4],  # Florencia
    [0.22, 1.5, 0.22, 16.5, 27, 51, 49, 0.3],  # Riohacha
    [0.13, 1.3, 0.13, 14.2, 26, 52, 48, 0.3],  # Quibdó
    [0.08, 1.2, 0.08, 19, 27, 50, 50, 0.2],  # San Andrés
    [0.15, 1.1, 0.15, 12.7, 29, 51, 49, 0.2],  # Yopal
    [0.05, 1, 0.05, 13.4, 26, 51, 49, 0.1],  # Leticia
    [0.01, 0.7, 0.01, 20, 25, 51, 49, 0.05]  # Mitú
])

# Datos de prueba
datos_test = np.array([
    [0.76, 5.1, 0.76, 16.3, 28, 51, 49, 1.2],  # Cúcuta
    [0.01, 0.6, 0.01, 22, 24, 50, 50, 0.05],  # Puerto Carreño
    [0.58, 7.3, 0.58, 9.2, 33, 52, 48, 1.5],  # Bucaramanga
    [0.08, 0.9, 0.08, 12.2, 29, 51, 49, 0.1],  # Arauca
    [0.53, 4.8, 0.53, 13.4, 31, 52, 48, 1.1],  # Ibagué
    [0.04, 0.8, 0.04, 15, 28, 52, 48, 0.1]  # Mocoa
])

# Combina los datos de entrenamiento y prueba
datos_combinados = np.vstack([datos_entrenamiento, datos_test])

# Calcula la matriz de distancias usando la distancia euclidiana
distancias = sch.distance.pdist(datos_combinados, metric='euclidean')

# Crea el clustering jerárquico usando el enlace completo
enlace = sch.linkage(distancias, method='complete')

# Dibuja el dendograma
plt.figure(figsize=(12, 8))
labels_combinados = [
    'Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena', 'Pereira',
    'Santa Marta', 'Manizales', 'Villavicencio', 'Pasto', 'Montería',
    'Valledupar', 'Neiva', 'Popayán', 'Armenia', 'Sincelejo', 'Tunja',
    'Florencia', 'Riohacha', 'Quibdó', 'San Andrés', 'Yopal', 'Leticia',
    'Mitú', 'Cúcuta', 'Puerto Carreño', 'Bucaramanga', 'Arauca',
    'Ibagué', 'Mocoa'
]
sch.dendrogram(enlace, labels=labels_combinados)
plt.title('Dendograma usando el método de enlace completo')
plt.xlabel('Índice de punto de datos')
plt.ylabel('Distancia')
plt.xticks(rotation=90)
plt.show()
