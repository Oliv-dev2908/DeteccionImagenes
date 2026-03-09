import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ==============================
# CONFIG
# ==============================

MODEL_PATH = "models/symbol_model_efficientnet.keras"

IMG_PATH = "dataset_processed/train/0/0-0000.png"

IMG_SIZE = (96, 96)

CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'divide','equals','minus','parentesisA','parentesisC',
    'plus','times','x','y','z'
]

# ==============================
# Cargar modelo
# ==============================

print("Cargando modelo...")
model = load_model(MODEL_PATH)
print("Modelo cargado correctamente.\n")

# ==============================
# Cargar imagen
# ==============================

img = image.load_img(
    IMG_PATH,
    color_mode='grayscale',
    target_size=IMG_SIZE
)

img_array = image.img_to_array(img)

# normalizar (porque el modelo no tiene Rescaling)
img_array = img_array / 255.0

# agregar dimensión batch
img_array = np.expand_dims(img_array, axis=0)

# ==============================
# Predicción
# ==============================

pred = model.predict(img_array, verbose=0)

pred_index = np.argmax(pred)
pred_class = CLASSES[pred_index]
confidence = float(np.max(pred))

print("=================================")
print(f"Imagen: {IMG_PATH}")
print(f"Clase predicha: {pred_class}")
print(f"Confianza: {confidence:.4f}")

# ==============================
# Top 3 predicciones
# ==============================

top_indices = pred[0].argsort()[-3:][::-1]

print("\nTop 3 predicciones:")

for i in top_indices:
    print(f"{CLASSES[i]} -> {pred[0][i]:.4f}")