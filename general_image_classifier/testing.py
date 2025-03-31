from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar el modelo<
model_path = r"C:\Users\Mildred\Desktop\Shiny Pokemon\general_image_classifier\pokemon_classifier_v2.keras"
model = load_model(model_path)

# Las clases que detectaste en el entrenamiento
class_names = ['poochyena', 'poochyena_shiny', 'wurmple', 'wurmple_shiny', 'zigzagoon', 'zigzagoon_shiny']

# Ruta de la imagen que deseas probar
image_path = r"C:\Users\Mildred\Desktop\Shiny Pokemon\dataset\comprobaciones\testWurmple.jpg"

# Cargar la imagen con OpenCV
img = cv2.imread(image_path)
img = cv2.resize(img, (32, 32))  # Cambiar tamaño a 32x32C:\Users\Mildred\Desktop\Shiny Pokemon\dataset\comprobaciones\testPoochyenaShiny2.png
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
img = img / 255.0  # Normalizar la imagen
img = np.expand_dims(img, axis=0)  # Agregar un eje para que tenga forma (1, 32, 32, 3)

# Mostrar la imagen
plt.imshow(np.squeeze(img))
plt.axis('off')
plt.show()

# Hacer la predicción
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

# Mostrar resultado
print(f"Predicción: {class_names[predicted_class]} (Confianza: {confidence * 100:.2f}%)")
