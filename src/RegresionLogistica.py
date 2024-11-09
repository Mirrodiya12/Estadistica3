import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, StandardScaler

train_dir = 'uco-animals-vs-plants/train'
test_dir = 'uco-animals-vs-plants/test'

img_height, img_width = 64, 64

def cargar_imagenes_y_etiquetas(directorio):
    data = []
    labels = []

    for label in os.listdir(directorio):
        label_dir = os.path.join(directorio, label)

        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                try:
                    img = imread(img_path, as_gray=True)
                    img_resized = resize(img, (img_height, img_width)).flatten()
                    data.append(img_resized)
                    labels.append(label)
                except Exception as e:
                    print(f"Error al cargar la imagen {img_path}: {e}")

    return np.array(data), np.array(labels)


X, y = cargar_imagenes_y_etiquetas(train_dir)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

modelo = LogisticRegression(max_iter=3000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Precisión en el conjunto de validación: {accuracy:.2f}")

test_data = []
test_names = []

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    try:
        img = imread(img_path, as_gray=True)
        img_resized = resize(img, (img_height, img_width)).flatten()
        img_scaled = scaler.transform([img_resized])
        test_data.append(img_scaled[0])
        test_names.append(img_name)
    except Exception as e:
        print(f"Error al cargar la imagen {img_path}: {e}")

X_test = np.array(test_data)
test_predictions = modelo.predict(X_test)
test_labels = label_encoder.inverse_transform(test_predictions)


resultados = pd.DataFrame({
    'file': test_names,
    'label': test_labels
})
resultados.to_csv('resultados_regresion_logistica.csv', index=False)
print("Archivo CSV de resultados guardado como 'resultados_regresion_logistica.csv'")

