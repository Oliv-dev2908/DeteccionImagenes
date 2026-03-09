from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img = cv2.imread("ecuCom6.jpeg")

result = ocr.ocr(img)

pasos = []

for linea in result[0]:   # aquí está el detalle importante
    box = linea[0]
    texto = linea[1][0]

    y = box[0][1]

    pasos.append((y, texto))

# ordenar por posición vertical
pasos.sort(key=lambda x: x[0])

print("\nProcedimiento detectado:\n")

for i, paso in enumerate(pasos, 1):
    print(f"Paso {i}: {paso[1]}")