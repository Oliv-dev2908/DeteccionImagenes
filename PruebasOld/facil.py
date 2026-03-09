"""
OCR para ecuaciones manuscritas
Versión simple y robusta - usa easyOCR con correcciones inteligentes
"""

import cv2
import numpy as np
import re

IMAGE_PATH = "ecuCom.jpeg"

# Cargar OCR
print("Inicializando OCR...")
import easyocr
reader = easyocr.Reader(['en'], gpu=False)
print("✓ OCR listo\n")


# ==========================
# DESKEW (corregir rotacion)
# ==========================

def deskew(image):
    """Corrige la rotación de la imagen"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is None:
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# ==========================
# PREPROCESAMIENTO
# ==========================

def preprocess(img):
    """Preprocesa la imagen para mejor OCR"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# ==========================
# PROCESAR RESULTADOS OCR
# ==========================

def extraer_elementos(resultados):
    """
    Extrae elementos de texto con sus posiciones.
    Retorna lista de dicts con información de posición.
    """
    elementos = []
    
    for (bbox, text, prob) in resultados:
        if prob < 0.15:  # Umbral bajo para no perder datos
            continue
        
        # Convertir bbox a array numpy
        coords = np.array(bbox, dtype=np.float32)
        
        # Calcular centros
        y_center = coords[:, 1].mean()
        x_center = coords[:, 0].mean()
        y_min = coords[:, 1].min()
        y_max = coords[:, 1].max()
        altura = y_max - y_min
        
        elementos.append({
            'text': text.strip(),
            'x': x_center,
            'y': y_center,
            'y_min': y_min,
            'y_max': y_max,
            'altura': altura,
            'prob': prob
        })
    
    return elementos


# ==========================
# AGRUPAR POR LINEAS
# ==========================

def agrupar_por_lineas(elementos):
    """
    Agrupa elementos en líneas basado en coordenada Y.
    Usa análisis jerárquico de distancias.
    """
    if not elementos:
        return []
    
    # Ordenar por Y
    elementos = sorted(elementos, key=lambda e: e['y'])
    
    # Agrupar por proximidad en Y
    lineas = []
    current_line = []
    
    for i, elem in enumerate(elementos):
        if not current_line:
            current_line.append(elem)
            continue
        
        # Distancia al elemento anterior
        prev_elem = current_line[-1]
        dist_y = abs(elem['y'] - prev_elem['y'])
        
        # Threshold basado en altura de fuente
        altura_promedio = np.mean([e['altura'] for e in current_line])
        threshold = max(altura_promedio * 1.5, 35)
        
        if dist_y < threshold:
            # Mismo línea
            current_line.append(elem)
        else:
            # Nueva línea
            lineas.append(current_line)
            current_line = [elem]
    
    if current_line:
        lineas.append(current_line)
    
    return lineas


# ==========================
# RECONSTRUIR LINEA
# ==========================

def reconstruir_linea(elementos):
    """
    Reconstruye una línea ordenando por X
    y aplicando correcciones.
    """
    if not elementos:
        return ""
    
    # Ordenar por X (horizontal)
    elementos = sorted(elementos, key=lambda e: e['x'])
    
    # Construir texto
    partes = []
    prev_x_end = None
    
    for elem in elementos:
        text = elem['text']
        x = elem['x']
        
        # Agregar espacio si hay gap
        if prev_x_end is not None:
            gap = x - prev_x_end
            if gap > 40:
                partes.append(" ")
        
        partes.append(text)
        # Aproximar ancho del texto
        prev_x_end = x + len(text) * 6
    
    # Unir
    linea = "".join(partes).strip()
    
    # Aplicar correcciones
    linea = corregir_linea(linea)
    
    return linea


# ==========================
# CORRECCIONES
# ==========================

def corregir_linea(text):
    """Aplica correcciones de caracteres confundibles"""
    
    # Reemplazos específicos
    correcciones = [
        # Caracteres simples
        ('<', 'x'),
        ('|', '1'),
        ('!', '1'),
        ('O', '0'),
        ('l', '1'),  # letra l por número 1
        (';', '='),
        (':', '='),
        
        # Patrones comunes de error
        ('1 5', '15'),
        ('4 5', '15'),  # Confunde 15 como 45
        ('1 0', '10'),
        ('= =', '='),
        ('==', '='),
        ('71', '='),
        ('|1', '1'),
        ('|5', '5'),
    ]
    
    original = text
    for old, new in correcciones:
        text = text.replace(old, new)
    
    # Limpiar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Si empieza con "= ", probablemente falte lado izquierdo
    if text.startswith('= '):
        text = text[2:].strip()
    
    return text.strip()


# ==========================
# MAIN
# ==========================

def main():
    print("=" * 60)
    print("OCR DE ECUACIONES MANUSCRITAS")
    print("=" * 60)
    
    # Cargar imagen
    print("\n1. Cargando imagen...")
    img = cv2.imread(IMAGE_PATH)
    
    if img is None:
        print(f"❌ Error: No se pudo abrir {IMAGE_PATH}")
        return
    
    print(f"   Tamaño: {img.shape[1]}x{img.shape[0]} píxeles")
    
    # Deskew
    print("2. Corrigiendo rotación...")
    img = deskew(img)
    
    # Preprocesar
    print("3. Preprocesando...")
    img_proc = preprocess(img)
    cv2.imwrite("debug_imagen_procesada.png", img_proc)
    print("   ✓ Guardado: debug_imagen_procesada.png")
    
    # OCR
    print("4. Ejecutando OCR...")
    results = reader.readtext(img_proc)
    print(f"   Detectados {len(results)} fragmentos")
    
    # Mostrar detecciones crudas
    print("\n   Fragmentos detectados:")
    for i, (bbox, text, prob) in enumerate(results):
        print(f"      {i+1}. '{text}' (conf: {prob:.2f})")
    
    # Extraer elementos con posiciones
    print("\n5. Extrayendo posiciones...")
    elementos = extraer_elementos(results)
    print(f"   {len(elementos)} elementos con buena confianza")
    
    # Agrupar por líneas
    print("6. Agrupando por líneas...")
    lineas = agrupar_por_lineas(elementos)
    print(f"   {len(lineas)} líneas detectadas")
    
    # Reconstruir
    print("7. Reconstruyendo ecuaciones...")
    ecuaciones = []
    for i, linea in enumerate(lineas, 1):
        texto = reconstruir_linea(linea)
        if texto:
            ecuaciones.append(texto)
            print(f"   Línea {i}: {texto}")
    
    # Salida final
    print("\n" + "=" * 60)
    print("RESULTADO FINAL:")
    print("=" * 60)
    for i, ec in enumerate(ecuaciones, 1):
        print(f"{i}. {ec}")
    print("=" * 60)
    
    return ecuaciones


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()