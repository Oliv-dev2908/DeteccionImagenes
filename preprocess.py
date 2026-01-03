import cv2
import numpy as np
import os

# ===============================
# CONFIGURACIÓN
# ===============================
FINAL_SIZE = 64
THRESH_BLOCK_SIZE = 11
THRESH_C = 2

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

    # 1. Blur suave (clave para cuaderno)
    img = cv2.GaussianBlur(img, (5,5), 0)

    # 2. Eliminar líneas horizontales del cuaderno
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_h)
    img = cv2.subtract(img, lines)

    # 3. Normalizar fondo
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # 4. Binarización adaptativa (ajustada)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 3
    )

    # 5. Limpieza de ruido
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # 6. Bounding box
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((FINAL_SIZE, FINAL_SIZE), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    symbol = img[y:y+h, x:x+w]

    # 7. Padding cuadrado
    size = max(w, h)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[
        (size-h)//2:(size-h)//2+h,
        (size-w)//2:(size-w)//2+w
    ] = symbol

    # 8. Resize final
    final_img = cv2.resize(canvas, (FINAL_SIZE, FINAL_SIZE),
                            interpolation=cv2.INTER_AREA)

    return final_img.astype(np.float32) / 255.0


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
                cv2.imwrite(output_path, (processed * 255).astype(np.uint8))
            except Exception as e:
                print(f"❌ Error en {input_path}: {e}")

    print("✅ Preprocesamiento COMPLETO")

# ===============================
if __name__ == "__main__":
    preprocess_dataset("dataset", "dataset_processed")
