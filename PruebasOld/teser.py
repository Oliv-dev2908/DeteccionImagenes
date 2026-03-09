import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Leer imagen
img = cv2.imread("image.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# detectar contornos
contours,_ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

boxes = []

# guardar bounding boxes
for c in contours:

    x,y,w,h = cv2.boundingRect(c)

    area = w*h

    if area > 100:   # eliminar ruido
        boxes.append((x,y,w,h))

# ordenar izquierda → derecha
boxes = sorted(boxes, key=lambda b: b[0])

ecuacion = ""

for (x,y,w,h) in boxes:

    symbol = gray[y:y+h, x:x+w]

    symbol = cv2.resize(symbol,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)

    config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789xX+-=*/='

    text = pytesseract.image_to_string(symbol,config=config)

    text = text.strip()

    ecuacion += text

    cv2.imshow("simbolo",symbol)
    cv2.waitKey(200)

print("\nEcuacion detectada:")
print(ecuacion)

cv2.destroyAllWindows()