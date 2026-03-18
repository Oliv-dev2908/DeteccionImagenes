from paddleocr import PaddleOCR
import cv2
import re

ocr = PaddleOCR(use_angle_cls=True, lang='en')

img = cv2.imread("ecuCom6.jpeg")
result = ocr.ocr(img)

# =========================
# FUNCIONES
# =========================
def recuperar_variable(paso_actual, paso_anterior):
    if paso_actual["sin_x"] and paso_anterior:
        if 'x' in paso_anterior["reconstruido"]:
            partes = paso_actual["reconstruido"].split("=")
            if len(partes) == 2:
                return "2x=" + partes[1]  # heurística simple
    return paso_actual["reconstruido"]

def reparar_division_oculta(texto):
    # Caso típico: 11-2 → /2
    if re.fullmatch(r'11[-]?\d+', texto):
        num = re.findall(r'\d+', texto)
        if num:
            return "/" + num[-1]
    return texto

def corregir_caracteres(texto):
    texto = texto.replace('S', '5')
    texto = texto.replace('X', 'x')
    texto = texto.replace('÷', '/')
    texto = texto.replace(':', '/')
    texto = texto.replace('//', '/')
    return texto


def limpiar(texto):
    texto = texto.replace(" ", "")
    texto = re.sub(r'[^0-9x+\-=\/]', '', texto)
    return texto


def normalizar_division(texto):
    texto = re.sub(r'(\d)11(\d)', r'\1/\2', texto)
    return texto


def corregir_numeros_raros(texto):
    # SOLO corregir si parece error típico OCR tipo 41/2 (cuando antes había //2)
    
    match = re.fullmatch(r'(\d{2,})/(\d)', texto)
    
    if match:
        num = match.group(1)
        
        # Si el número empieza con 4 y es largo → sospechoso (ej: 41, 42)
        if num.startswith('4') and len(num) == 2:
            return num[0] + '/' + match.group(2)
    
    return texto


def detectar_invalido(texto):
    if '=' not in texto:
        return True
    partes = texto.split('=')
    if len(partes) != 2:
        return True
    if partes[0] == "" or partes[1] == "":
        return True
    return False


def detectar_incertidumbre(texto):
    return bool(re.search(r'(\+\-|\-\+)', texto))


def detectar_sin_variable(texto):
    return 'x' not in texto


def es_linea_division(texto):
    # detecta cosas como /2 o //2
    return bool(re.fullmatch(r'\/?\d+', texto))


def procesar_texto(texto):
    original = texto

    texto_proc = corregir_caracteres(texto)
    texto_proc = limpiar(texto_proc)
    texto_proc = normalizar_division(texto_proc)
    texto_proc = corregir_numeros_raros(texto_proc)

    invalido = detectar_invalido(texto_proc)
    incierto = detectar_incertidumbre(texto_proc)
    sin_x = detectar_sin_variable(texto_proc)

    return {
        "original": original,
        "reconstruido": texto_proc,
        "invalido": invalido,
        "incierto": incierto,
        "sin_x": sin_x
    }


# =========================
# OCR + ORDEN
# =========================

pasos = []

for linea in result[0]:
    box = linea[0]
    texto = linea[1][0]
    y = box[0][1]
    pasos.append((y, texto))

pasos.sort(key=lambda x: x[0])

# =========================
# FUSIÓN INTELIGENTE
# =========================

pasos_finales = []
i = 0

while i < len(pasos):
    _, texto = pasos[i]
    actual = procesar_texto(texto)

    # Aplicar reparación de división
    texto_reparado = reparar_division_oculta(limpiar(texto))
    
    if texto_reparado != limpiar(texto):
        actual["reconstruido"] = texto_reparado
        actual["repair"] = True

    # Intentar fusionar si hay problemas
    if (actual["invalido"] or actual["sin_x"]) and i + 1 < len(pasos):
        _, siguiente = pasos[i + 1]

        siguiente_limpio = limpiar(siguiente)
        siguiente_limpio = reparar_division_oculta(siguiente_limpio)

        combinado = limpiar(texto) + siguiente_limpio
        combinado_proc = procesar_texto(combinado)

        if not combinado_proc["invalido"]:
            combinado_proc["fusion"] = True
            pasos_finales.append(combinado_proc)
            i += 2
            continue

    pasos_finales.append(actual)
    i += 1

for i in range(len(pasos_finales)):
    if i > 0:
        pasos_finales[i]["reconstruido"] = recuperar_variable(
        pasos_finales[i],
        pasos_finales[i-1]
        )

# =========================
# OUTPUT FINAL EN UNA LÍNEA
# =========================

print("\nOCR vs Reconstrucción\n")

for i, paso in enumerate(pasos_finales, 1):

    estado = "OK"

    if paso["invalido"]:
        estado = "⚠️ inválido"
    elif paso["incierto"]:
        estado = "⚠️ dudoso"
    elif paso["sin_x"]:
        estado = "❌ sin variable"
    elif paso.get("fusion"):
        estado = "🔧 fusionado"

    print(f"Paso {i:02d}: {paso['original']:<25} -> {paso['reconstruido']:<25} [{estado}]")