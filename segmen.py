import cv2
import numpy as np
import easyocr
import tensorflow as tf

# ================================
# Cargar modelo MNIST preentrenado
# ================================
print("Cargando modelo MNIST...")

modelo = tf.keras.models.load_model("mnist_model.h5")

# EasyOCR
reader = easyocr.Reader(['en'], gpu=False)


# ================================
# Preprocesar imagen
# ================================
def preprocesar(ruta):

    img = cv2.imread(ruta)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _,thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return thresh


# ================================
# Segmentar símbolos
# ================================
def segmentar(img):

    # conectar trazos cercanos
    kernel = np.ones((5,5), np.uint8)
    dilatada = cv2.dilate(img, kernel, iterations=1)

    contornos,_ = cv2.findContours(
        dilatada,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    simbolos = []

    for c in contornos:

        x,y,w,h = cv2.boundingRect(c)

        area = w*h

        # filtrar ruido
        if area > 500 and area < 20000:

            simbolo = img[y:y+h, x:x+w]

            simbolos.append((x,y,simbolo))

    # ordenar primero por línea y luego por x
    simbolos = sorted(simbolos, key=lambda s: (s[1]//50, s[0]))

    return [s[2] for s in simbolos]

# ================================
# Predicción número MNIST
# ================================
def predecir_numero(img):

    img = cv2.resize(img,(28,28))

    img = img / 255.0

    img = img.reshape(1,28,28,1)

    pred = modelo.predict(img, verbose=0)

    numero = np.argmax(pred)

    return str(numero)


# ================================
# Reconocer símbolo con EasyOCR
# ================================
def predecir_simbolo(img):

    resultado = reader.readtext(img, detail=0)

    if len(resultado) > 0:

        texto = resultado[0]

        if texto in ["x","+","-","="]:
            return texto

    return None


# ================================
# Leer ecuación completa
# ================================
def leer_ecuacion(ruta):

    thresh = preprocesar(ruta)

    simbolos = segmentar(thresh)

    ecuacion = ""

    for s in simbolos:

        simbolo = predecir_simbolo(s)

        cv2.imshow("Simbolo", s)

        if simbolo:
            ecuacion += simbolo
        else:
            ecuacion += predecir_numero(s)

    return ecuacion


# ================================
# Ejecutar
# ================================
if __name__ == "__main__":

    ruta = "ecuCom.jpeg"

    ecuacion = leer_ecuacion(ruta)

    print("\nEcuación detectada:")
    print(ecuacion)