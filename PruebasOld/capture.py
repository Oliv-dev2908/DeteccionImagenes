# capture.py
import cv2
import os
from pathlib import Path

# Configuración: ajusta clases y ruta
CLASSES = ['0','1','2','3','4','5','6','7','8','9','x','y','z','plus','minus','times','divide','equals']
DATA_DIR = Path("dataset")
DATA_DIR.mkdir(exist_ok=True)

for c in CLASSES:
    (DATA_DIR/c).mkdir(parents=True, exist_ok=True)

# Variables
cam = cv2.VideoCapture(0)
cur_class_idx = 0
count = {c: len(list((DATA_DIR/c).glob("*.png"))) for c in CLASSES}

print("Controles:")
print("  <- / -> : cambiar clase")
print("  s : guardar imagen")
print("  q : salir")

while True:
    ret, frame = cam.read()
    if not ret:
        print("No se detecta cámara")
        break

    h, w = frame.shape[:2]
    # dibujar info
    cv2.putText(frame, f"Clase: {CLASSES[cur_class_idx]} ({count[CLASSES[cur_class_idx]]})", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, "s: guardar, <-/-> cambiar, q: salir", (10,h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Captura - presiona s para guardar", frame)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('s'):
        # recortar centro (opcional) y guardar
        margin = 200
        cx, cy = w//2, h//2
        crop = frame[max(0,cy-margin):min(h,cy+margin), max(0,cx-margin):min(w,cx+margin)]
        filename = DATA_DIR/CLASSES[cur_class_idx]/f"{count[CLASSES[cur_class_idx]]:04d}.png"
        cv2.imwrite(str(filename), crop)
        count[CLASSES[cur_class_idx]] += 1
        print(f"Guardado {filename}")
    elif k == 81 or k == 2424832:  # flecha izquierda
        cur_class_idx = (cur_class_idx - 1) % len(CLASSES)
    elif k == 83 or k == 2555904:  # flecha derecha
        cur_class_idx = (cur_class_idx + 1) % len(CLASSES)

cam.release()
cv2.destroyAllWindows()
