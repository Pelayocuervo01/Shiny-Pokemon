import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_dir = r"C:\Users\Mildred\Desktop\Shiny Pokemon\dataset\train"
test_dir = r"C:\Users\Mildred\Desktop\Shiny Pokemon\dataset\test"

train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Cargar datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=4,
    class_mode='sparse'
)

# Cargar datos de prueba
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    batch_size=4,
    class_mode='sparse'
)

# Obtener el número de clases detectadas automáticamente
num_classes = len(train_generator.class_indices)
print(f"Número de clases detectadas: {num_classes}")
print(f"Clases detectadas: {train_generator.class_indices}")

# Definir el modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))  # Cambiado para coincidir con el número de clases

# Compilar el modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Entrenar el modelo
model.fit(train_generator, epochs=50, validation_data=test_generator)

# Evaluar el modelo
loss, accuracy = model.evaluate(test_generator)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Guardar el modelo
model.save("pokemon_classifier_v1.keras")
