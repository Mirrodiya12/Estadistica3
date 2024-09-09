import numpy as np

def inicializar_centroides(datos, k):
   #Inicializa los centroides seleccionando aleatoriamente k puntos de los datos.
    np.random.seed(0)  # Para reproducibilidad
    indices = np.random.choice(len(datos), k, replace=False)
    return datos[indices]

def actualizar_centroides(datos, etiquetas, k):
    #Actualiza los centroides calculando la media de los puntos asignados a cada cluster.
    centroides = np.zeros((k, datos.shape[1]))
    for i in range(k):
        puntos_cluster = datos[etiquetas == i]
        if len(puntos_cluster) > 0:
            centroides[i] = np.mean(puntos_cluster, axis=0)
    return centroides

def k_means(datos, k, max_iter=100):
    #Algoritmo de K-means para clustering.
    centroides = inicializar_centroides(datos, k)
    for _ in range(max_iter):
        distancias = np.linalg.norm(datos[:, np.newaxis] - centroides, axis=2)
        etiquetas = np.argmin(distancias, axis=1)
        nuevos_centroides = actualizar_centroides(datos, etiquetas, k)
        if np.all(centroides == nuevos_centroides):
            break
        centroides = nuevos_centroides
    return etiquetas, centroides

def clasificar_test(datos_test, centroides):
    #Clasifica los puntos de test basándose en la distancia a los centroides.
    distancias = np.linalg.norm(datos_test[:, np.newaxis] - centroides, axis=2)
    etiquetas = np.argmin(distancias, axis=1)
    return etiquetas

def encontrar_ciudad_mas_cercana(punto, datos_entrenamiento, ciudades):
   #Encuentra la ciudad más cercana en el conjunto de entrenamiento.
    distancias = np.linalg.norm(datos_entrenamiento - punto, axis=1)
    indice_mas_cercano = np.argmin(distancias)
    ciudad_mas_cercana = ciudades[indice_mas_cercano]
    return ciudad_mas_cercana

# Datos de las 24 ciudades (entrenamiento)
ciudades_entrenamiento = [
    'Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena', 'Pereira',
    'Santa Marta', 'Manizales', 'Villavicencio', 'Pasto', 'Montería',
    'Valledupar', 'Neiva', 'Popayán', 'Armenia', 'Sincelejo', 'Tunja',
    'Florencia', 'Riohacha', 'Quibdó', 'San Andrés', 'Yopal', 'Leticia',
    'Mitú'
]

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
ciudades_test = [
    'Cúcuta', 'Puerto Carreño', 'Bucaramanga', 'Arauca', 'Ibagué', 'Mocoa'
]

datos_test = np.array([
    [0.76, 5.1, 0.76, 16.3, 28, 51, 49, 1.2],  # Cúcuta
    [0.01, 0.6, 0.01, 22, 24, 50, 50, 0.05],  # Puerto Carreño
    [0.58, 7.3, 0.58, 9.2, 33, 52, 48, 1.5],  # Bucaramanga
    [0.08, 0.9, 0.08, 12.2, 29, 51, 49, 0.1],  # Arauca
    [0.53, 4.8, 0.53, 13.4, 31, 52, 48, 1.1],  # Ibagué
    [0.04, 0.8, 0.04, 15, 28, 52, 48, 0.1]  # Mocoa
])

# Aplicar K-means
k = 6
etiquetas, centroides = k_means(datos_entrenamiento, k)

# Clasificar las ciudades de prueba
etiquetas_test = clasificar_test(datos_test, centroides)

# Determinar la ciudad más cercana en el conjunto de entrenamiento para cada ciudad de prueba
for i, punto_test in enumerate(datos_test):
    ciudad_mas_cercana = encontrar_ciudad_mas_cercana(punto_test, datos_entrenamiento, ciudades_entrenamiento)
    print(f"La ciudad de prueba '{ciudades_test[i]}' es más cercana a '{ciudad_mas_cercana}' del conjunto de entrenamiento")
