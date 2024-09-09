import math
from collections import Counter
import plotly.graph_objects as go

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

def media(datos):
    return sum(datos) / len(datos)

def mediana(datos):
    datos_ordenados = sorted(datos)
    n = len(datos_ordenados)
    mitad = n // 2
    if n % 2 == 0:
        return (datos_ordenados[mitad - 1] + datos_ordenados[mitad]) / 2
    else:
        return datos_ordenados[mitad]

def desviacion_estandar(datos):
    media_valor = media(datos)
    varianza = sum((x - media_valor) ** 2 for x in datos) / len(datos)
    return math.sqrt(varianza)

def moda(datos):
    frecuencia = Counter(datos)
    max_freq = max(frecuencia.values())
    return [item for item, freq in frecuencia.items() if freq == max_freq], max_freq

# Extraer datos numéricos
pib = [fila[1] for fila in ciudades[1:]]
poblacion = [fila[2] for fila in ciudades[1:]]
tasa_desempleo = [fila[3] for fila in ciudades[1:]]
edad_promedio = [fila[4] for fila in ciudades[1:]]
porcentaje_mujeres = [fila[5] for fila in ciudades[1:]]
porcentaje_hombres = [fila[6] for fila in ciudades[1:]]
presupuesto = [fila[7] for fila in ciudades[1:]]

# Cálculos para cada conjunto de datos
conjuntos_datos = {
    "GDP (USD Billion)": pib,
    "Population (Millions)": poblacion,
    "Unemployment Rate (%)": tasa_desempleo,
    "Average Age": edad_promedio,
    "Women (%)": porcentaje_mujeres,
    "Men (%)": porcentaje_hombres,
    "Budget (USD Billion)": presupuesto
}

resultados = {}
for nombre, datos in conjuntos_datos.items():
    resultados[nombre] = {
        "Media": media(datos),
        "Mediana": mediana(datos),
        "Desviación Estándar": desviacion_estandar(datos),
        "Moda": moda(datos)
    }

# Mostrar resultados
for nombre, estadisticas in resultados.items():
    print(f"{nombre}:")
    print(f"  Media: {estadisticas['Media']:.2f}")
    print(f"  Mediana: {estadisticas['Mediana']:.2f}")
    print(f"  Desviación Estándar: {estadisticas['Desviación Estándar']:.2f}")
    moda_valores, moda_freq = estadisticas['Moda']
    print(f"  Moda: {moda_valores} con {moda_freq} ocurrencias")
    print()

# Crear el boxplot con plotly
fig = go.Figure()

# Añadir cada conjunto de datos al boxplot
fig.add_trace(go.Box(y=pib, name='PIB'))
fig.add_trace(go.Box(y=poblacion, name='Población'))
fig.add_trace(go.Box(y=tasa_desempleo, name='Tasa de Desempleo'))
fig.add_trace(go.Box(y=edad_promedio, name='Edad Promedio'))
fig.add_trace(go.Box(y=porcentaje_mujeres, name='Porcentaje de Mujeres'))
fig.add_trace(go.Box(y=porcentaje_hombres, name='Porcentaje de Hombres'))
fig.add_trace(go.Box(y=presupuesto, name='Presupuesto'))

# Actualizar el diseño del gráfico
fig.update_layout(
    title="Boxplot de Datos de Ciudades",
    xaxis_title="Categorías",
    yaxis_title="Valores",
    boxmode="group"  # Agrupa las cajas por categoría
)

# Mostrar el gráfico
fig.show()


def covarianza(X1, X2):
    n = len(X1)
    if n != len(X2):
        raise ValueError("Las dos listas deben tener la misma longitud")

    media_X1 = media(X1)
    media_X2 = media(X2)

    cov = sum((X1[i] - media_X1) * (X2[i] - media_X2) for i in range(n)) / (n - 1)
    return cov


# Datos numéricos
pib = [fila[1] for fila in ciudades[1:]]
poblacion = [fila[2] for fila in ciudades[1:]]

# Calcular covarianza
cov_pib_poblacion = covarianza(pib, poblacion)

# Calcular desviación estándar
desv_std_pib = desviacion_estandar(pib)
desv_std_poblacion = desviacion_estandar(poblacion)

# Calcular correlación
correlacion_pib_poblacion = cov_pib_poblacion / (desv_std_pib * desv_std_poblacion)

# Imprimir resultados de covarianza y correlación
print(f"Covarianza entre PIB y Población: {cov_pib_poblacion:.2f}")
print(f"Desviación Estándar PIB: {desv_std_pib:.2f}")
print(f"Desviación Estándar Población: {desv_std_poblacion:.2f}")
print(f"Correlación entre PIB y Población: {correlacion_pib_poblacion:.2f}")

# Explicación de la relación entre covarianza y correlación
print("\nExplicación sobre la relación entre covarianza y correlación:")
print("La covarianza y la correlación miden la relación entre dos variables.")
print("La covarianza indica la dirección de la relación (positiva o negativa), pero su magnitud depende de las unidades de las variables.")
print("La correlación, en cambio, estandariza esta medida dividiendo la covarianza por el producto de las desviaciones estándar de las variables.")
print("Esto transforma la covarianza en una medida de relación que está en el rango de -1 a 1, facilitando la interpretación y comparaciones.")
print("Un valor de correlación cercano a 1 o -1 indica una fuerte relación lineal positiva o negativa, respectivamente, mientras que un valor cercano a 0 indica poca o ninguna relación lineal.")
