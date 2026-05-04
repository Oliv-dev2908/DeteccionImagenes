# ============================================================
# AGENTE 1 — OCR + Salida estructurada para Agente 2
# (preprocesamiento mínimo, conserva detección original)
# ============================================================

from paddleocr import PaddleOCR
import cv2
import re
import json

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# =========================
# CORRECCIÓN DE CARACTERES
# =========================
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
    match = re.fullmatch(r'(\d{2,})/(\d)', texto)
    if match:
        num = match.group(1)
        if num.startswith('4') and len(num) == 2:
            return num[0] + '/' + match.group(2)
    return texto

def reparar_division_oculta(texto):
    if re.fullmatch(r'11[-]?\d+', texto):
        num = re.findall(r'\d+', texto)
        if num:
            return "/" + num[-1]
    return texto

# =========================
# ANÁLISIS DE PROBLEMAS
# =========================
def analizar_detallado(texto):
    problemas = []
    sugerencias = []

    if '=' not in texto:
        problemas.append({"tipo": "missing_equals", "detalle": "No se detecta signo '='"})
        sugerencias.append({"tipo": "insert_equals", "confianza": 0.7})
    else:
        partes = texto.split('=')
        if len(partes) != 2:
            problemas.append({"tipo": "multiple_equals", "detalle": "Más de un '=' detectado"})
        else:
            if partes[0] == "" or partes[1] == "":
                problemas.append({"tipo": "incomplete_expression",
                                  "detalle": "Expresión incompleta en un lado del '='"})

    if 'x' not in texto:
        problemas.append({"tipo": "missing_variable", "detalle": "No se detecta variable 'x'"})
        sugerencias.append({"tipo": "insert_variable",
                            "posible": "inicio o junto a número", "confianza": 0.6})

    if re.search(r'(\+\-|\-\+)', texto):
        problemas.append({"tipo": "ambiguous_operator",
                          "detalle": "Secuencia +- o -+ detectada"})

    if re.search(r'\d11\d', texto):
        problemas.append({"tipo": "possible_division_error",
                          "detalle": "Patrón '11' detectado (posible '/' mal reconocido)"})
        sugerencias.append({"tipo": "replace_with_division", "confianza": 0.8})

    if len(texto) <= 2:
        problemas.append({"tipo": "too_short",
                          "detalle": "Texto demasiado corto para ser ecuación"})

    return problemas, sugerencias

def procesar_texto(texto, confianza_ocr=None):
    original = texto
    texto_proc = corregir_caracteres(texto)
    texto_proc = limpiar(texto_proc)
    texto_proc = normalizar_division(texto_proc)
    texto_proc = corregir_numeros_raros(texto_proc)
    problemas, sugerencias = analizar_detallado(texto_proc)

    return {
        "original": original,
        "reconstruido": texto_proc,
        "confianza_ocr": round(confianza_ocr, 4) if confianza_ocr is not None else None,
        "problemas": problemas,
        "sugerencias": sugerencias,
        "fusion": False,
        "repair": False,
        "bbox_y": None
    }

def tiene_problemas_graves(paso):
    tipos = [p["tipo"] for p in paso["problemas"]]
    return ("missing_equals" in tipos or
            "incomplete_expression" in tipos or
            "too_short" in tipos)

def tiene_falta_variable(paso):
    return any(p["tipo"] == "missing_variable" for p in paso["problemas"])

def es_ecuacion_valida(paso):
    tipos = [p["tipo"] for p in paso["problemas"]]
    return not ("missing_equals" in tipos or
                "multiple_equals" in tipos or
                "incomplete_expression" in tipos or
                "too_short" in tipos)

def recuperar_variable(paso_actual, paso_anterior):
    if tiene_falta_variable(paso_actual) and paso_anterior:
        if 'x' in paso_anterior["reconstruido"]:
            partes = paso_actual["reconstruido"].split("=")
            if len(partes) == 2:
                return "2x=" + partes[1]
    return paso_actual["reconstruido"]

# =========================
# OCR + ORDEN VERTICAL
# =========================
def extraer_lineas_ocr(result):
    """Retorna lista vacía si OCR no detectó nada (evita TypeError)."""
    if not result or not result[0]:
        return []
    lineas = []
    for linea in result[0]:
        box = linea[0]
        texto = linea[1][0]
        confianza = linea[1][1]   # ← capturar confianza OCR
        y = box[0][1]
        lineas.append((y, texto, confianza))
    lineas.sort(key=lambda x: x[0])
    return lineas

# =========================
# FUSIÓN INTELIGENTE (tu lógica original)
# =========================
def fusionar_lineas(lineas):
    pasos_finales = []
    i = 0

    while i < len(lineas):
        y, texto, confianza = lineas[i]
        actual = procesar_texto(texto, confianza)
        actual["bbox_y"] = round(y, 2)

        texto_reparado = reparar_division_oculta(limpiar(texto))
        if texto_reparado != limpiar(texto):
            actual["reconstruido"] = texto_reparado
            actual["repair"] = True

        # Solo fusionar si hay problemas GRAVES (tu criterio original)
        if tiene_problemas_graves(actual) and i + 1 < len(lineas):
            _, siguiente_txt, siguiente_conf = lineas[i + 1]
            siguiente_limpio = reparar_division_oculta(limpiar(siguiente_txt))
            combinado = limpiar(texto) + siguiente_limpio
            combinado_proc = procesar_texto(combinado, min(confianza, siguiente_conf))
            combinado_proc["bbox_y"] = round(y, 2)

            if es_ecuacion_valida(combinado_proc):
                combinado_proc["fusion"] = True
                pasos_finales.append(combinado_proc)
                i += 2
                continue

        pasos_finales.append(actual)
        i += 1

    return pasos_finales

def aplicar_recuperacion_contextual(pasos):
    for i in range(1, len(pasos)):
        pasos[i]["reconstruido"] = recuperar_variable(pasos[i], pasos[i - 1])
    return pasos

# =========================
# SALIDA ESTRUCTURADA PARA AGENTE 2
# =========================
def construir_salida_agente2(pasos, ruta_imagen):
    return {
        "version": "1.0",
        "fuente": "agente_ocr",
        "imagen": ruta_imagen,
        "total_lineas": len(pasos),
        "pasos": [
            {
                "numero":       i + 1,
                "original":     paso["original"],
                "reconstruido": paso["reconstruido"],
                "confianza_ocr": paso.get("confianza_ocr"),
                "bbox_y":       paso.get("bbox_y"),
                "fusion":       paso.get("fusion", False),
                "repair":       paso.get("repair", False),
                "problemas":    paso["problemas"],
                "sugerencias":  paso["sugerencias"]
            }
            for i, paso in enumerate(pasos)
        ]
    }

# =========================
# PIPELINE PRINCIPAL
# =========================
def ejecutar_agente1(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {ruta_imagen}")

    result = ocr.ocr(img)

    lineas = extraer_lineas_ocr(result)

    if not lineas:
        print("⚠ OCR no detectó texto. Verificar calidad de imagen.")
        return {
            "version": "1.0",
            "fuente": "agente_ocr",
            "imagen": ruta_imagen,
            "total_lineas": 0,
            "advertencia": "OCR no detectó texto.",
            "pasos": []
        }

    pasos = fusionar_lineas(lineas)
    pasos = aplicar_recuperacion_contextual(pasos)
    return construir_salida_agente2(pasos, ruta_imagen)

# =========================
# EJECUCIÓN + DEBUG
# =========================
if __name__ == "__main__":
    IMAGEN = "ecuCom5.jpeg"
    salida = ejecutar_agente1(IMAGEN)

    print(f"\n{'='*50}")
    print(f"  AGENTE 1 — {salida['imagen']}")
    print(f"  Líneas detectadas: {salida['total_lineas']}")
    print(f"{'='*50}")

    for paso in salida["pasos"]:
        conf_str = f"{paso['confianza_ocr']:.2f}" if paso['confianza_ocr'] else "N/A"
        flags = []
        if paso["fusion"]: flags.append("FUSIÓN")
        if paso["repair"]: flags.append("REPAIR")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"\nPaso {paso['numero']:02d}{flag_str}")
        print(f"  OCR raw:      {paso['original']}  (conf={conf_str})")
        print(f"  Reconstruido: {paso['reconstruido']}")

        if paso["problemas"]:
            print("  Problemas:")
            for p in paso["problemas"]:
                print(f"    ✗ ({p['tipo']}) {p['detalle']}")
        if paso["sugerencias"]:
            print("  Sugerencias:")
            for s in paso["sugerencias"]:
                print(f"    → ({s['tipo']}) confianza={s.get('confianza','?')}")

    print(f"\n{'='*50}")
    print("JSON → Agente 2:")
    print(json.dumps(salida, indent=2, ensure_ascii=False))