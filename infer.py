import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. Cargar el modelo
model_path = "models/symbol_cnn.h5"
model = load_model(model_path)
print("Modelo cargado correctamente.")

# 2. Definir clases (igual que en el entrenamiento)
classes = ['0','1','2','3','4','5','6','7','8','9',
           'divide','equals','minus','parentesisA','parentesisC',
           'plus','times','x','y','z']

# 3. Cargar imagen de prueba
img_path = "dataset_processed/train/x/000252.png"  # Cambiar por tu imagen
img = image.load_img(img_path, color_mode='grayscale', target_size=(64,64))  # <- color_mode='grayscale'
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalización
img_array = np.expand_dims(img_array, axis=0)  # Batch de 1


# 4. Hacer predicción
pred = model.predict(img_array)
pred_class = classes[np.argmax(pred)]

print(f"Clase predicha: {pred_class}")
