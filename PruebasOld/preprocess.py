import cv2
import numpy as np
import os

# ===============================
# CONFIGURACIÓN
# ===============================
FINAL_SIZE = 28  # Puedes cambiar a 28 si quieres estilo MNIST

CLASS_MAP = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',

    'plus': '+',
    'minus': '-',
    'times': '*',
    'divide': '/',
    'equals': '=',

    'parentesisA': '(',
    'parentesisC': ')',

    'x': 'x',
    'y': 'y',
    'z': 'z'
}

# ===============================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar {image_path}")

    # 1️⃣ Binarización simple (fondo blanco digital)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # 2️⃣ Encontrar el contorno más grande (el símbolo)
    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return np.zeros((FINAL_SIZE, FINAL_SIZE), dtype=np.float32)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    symbol = img[y:y+h, x:x+w]

    # 3️⃣ Crear padding cuadrado
    size = max(w, h)
    canvas = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = symbol

    # 4️⃣ Redimensionar
    final_img = cv2.resize(
        canvas,
        (FINAL_SIZE, FINAL_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # 5️⃣ Normalizar 0-1
    final_img = final_img.astype(np.float32) / 255.0

    return final_img


# ===============================
def preprocess_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Crear carpetas de salida
    for folder in CLASS_MAP.keys():
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    for folder in CLASS_MAP.keys():
        input_folder = os.path.join(input_dir, folder)
        output_folder = os.path.join(output_dir, folder)

        if not os.path.exists(input_folder):
            print(f"⚠️ Carpeta faltante: {folder}")
            continue

        for file in os.listdir(input_folder):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            input_path = os.path.join(input_folder, file)

            try:
                processed = preprocess_image(input_path)

                output_path = os.path.join(
                    output_folder,
                    os.path.splitext(file)[0] + ".png"
                )

                cv2.imwrite(
                    output_path,
                    (processed * 255).astype(np.uint8)
                )

            except Exception as e:
                print(f"❌ Error en {input_path}: {e}")

    print("✅ Preprocesamiento COMPLETO")


# ===============================
if __name__ == "__main__":
    preprocess_dataset("dataset", "dataset_processed")