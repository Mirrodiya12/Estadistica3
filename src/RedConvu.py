import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
import pandas as pd


img_height, img_width = 160, 160
batch_size = 8

train_dir = 'D:\\Juegos\\Tarea3\\Estadistica3\\Tarea3\\uco-animals-vs-plants\\train'
test_dir = 'D:\\Juegos\\Tarea3\\Estadistica3\\Tarea3\\uco-animals-vs-plants\\test'

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet")
for layer in base_model.layers[-20:]:  # Descongelar últimas 20 capas para afinado
    layer.trainable = True

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),  # Regularización L2
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model_mobilenet_v2_tuned.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

epochs = 31
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop]
)

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Precisión en el conjunto de validación: {val_accuracy:.2f}")

test_images = []
test_filenames = []

for filename in os.listdir(test_dir):
    filepath = os.path.join(test_dir, filename)
    if os.path.isfile(filepath):
        img = load_img(filepath, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0
        test_images.append(img_array)
        test_filenames.append(filename)

test_images = np.array(test_images)
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
class_labels = list(train_generator.class_indices.keys())
predicted_labels = [class_labels[i] for i in predicted_classes]

resultados = pd.DataFrame({
    'file': test_filenames,
    'label': predicted_labels
})
resultados.to_csv('resultados_cnn.csv', index=False)
print("Archivo CSV de resultados guardado como 'resultados_cnn.csv'")
