import re

# =========================
# PARSER
# =========================

def parsear_entrada(texto):
    pasos = []
    bloques = texto.strip().split("Paso")

    for bloque in bloques:
        if not bloque.strip():
            continue

        lineas = bloque.strip().split("\n")

        paso = {
            "ocr": "",
            "reconstruido": "",
            "problemas": [],
            "sugerencias": []
        }

        for linea in lineas:
            if "OCR:" in linea:
                paso["ocr"] = linea.split("OCR:")[1].strip()
            elif "Reconstruido:" in linea:
                paso["reconstruido"] = linea.split("Reconstruido:")[1].strip()
            elif "(" in linea and ")" in linea:
                if "confianza" in linea:
                    paso["sugerencias"].append(linea.strip())
                else:
                    paso["problemas"].append(linea.strip())

        pasos.append(paso)

    return pasos


# =========================
# OBTENER RESULTADO FINAL
# =========================

def obtener_x_final(pasos):
    for paso in reversed(pasos):
        match = re.match(r'x\s*=\s*(-?\d+)', paso["reconstruido"])
        if match:
            return int(match.group(1))
    return None


# =========================
# VALIDACIÓN MATEMÁTICA
# =========================

def preparar_para_eval(expr):
    # convierte 2x → 2*x
    expr = re.sub(r'(\d)x', r'\1*x', expr)
    expr = re.sub(r'x(\d)', r'x*\1', expr)
    return expr

def validar_ecuacion(eq, x_val):
    try:
        izq, der = eq.split('=')

        izq = preparar_para_eval(izq)
        der = preparar_para_eval(der)

        izq_eval = eval(izq.replace('x', f'({x_val})'))
        der_eval = eval(der)

        return abs(izq_eval - der_eval) < 1e-6
    except:
        return False
    
def insertar_equals_valido(texto):
    candidatos = []

    for i in range(1, len(texto)-1):
        izquierda = texto[:i]
        derecha = texto[i:]

        if izquierda and derecha:
            if not izquierda.endswith(('+', '-', '/')) and not derecha.startswith(('+', '-', '/')):
                candidatos.append(izquierda + "=" + derecha)

    return candidatos


# =========================
# FILTRO SINTÁCTICO
# =========================

def es_forma_valida(eq):
    if eq.count('=') > 1:
        return False
    if re.search(r'[^0-9x+\-=/]', eq):
        return False
    if re.search(r'\d+x\d', eq):  # ejemplo: 2x59
        return False
    return True


# =========================
# GENERACIÓN DE HIPÓTESIS
# =========================

def generar_hipotesis(paso):
    texto = paso["reconstruido"]
    candidatos = set()

    # siempre incluir original
    candidatos.add(texto)

    # ===== insertar '=' =====
    if "missing_equals" in str(paso["problemas"]):
        for i in range(1, len(texto)):
            candidatos.add(texto[:i] + "=" + texto[i:])

    # ===== insertar 'x' =====
    if "missing_variable" in str(paso["problemas"]):
        for match in re.finditer(r'\d+', texto):
            i = match.end()
            candidatos.add(texto[:i] + "x" + texto[i:])

    # ===== corregir tipo 41/2 → 4/2 =====
    if re.search(r'\d{2,}/\d', texto):
        candidatos.add(re.sub(r'(\d)(\d)/', r'\1/', texto))

    # ===== eliminar basura obvia =====
    candidatos = [c for c in candidatos if es_forma_valida(c)]

    return candidatos


# =========================
# SCORE MEJORADO
# =========================

def score_candidato(eq, x_val):
    score = 0

    # ===== estructura =====
    if '=' in eq:
        score += 5
    else:
        return -10  # descartar fuerte

    if 'x' in eq:
        score += 3

    # ===== penalizaciones fuertes =====
    if re.search(r'\d+x\d', eq):  # 2x59
        return -10

    if re.search(r'==', eq):
        return -10

    # ===== validación matemática =====
    if x_val is not None:
        if validar_ecuacion(eq, x_val):
            score += 20
        else:
            score -= 10

    return score


# =========================
# PROCESAMIENTO PRINCIPAL
# =========================

def reconstruir(pasos):
    x_final = obtener_x_final(pasos)

    resultados = []

    for paso in pasos:
        candidatos = generar_hipotesis(paso)

        mejor = None
        mejor_score = -999

        interpretaciones = []

        for c in candidatos:
            s = score_candidato(c, x_final)

            interpretaciones.append({
                "ecuacion": c,
                "score": s
            })

            if s > mejor_score:
                mejor_score = s
                mejor = c

        resultados.append({
            "original": paso["reconstruido"],
            "mejor": mejor,
            "score": mejor_score,
            "interpretaciones": sorted(interpretaciones, key=lambda x: -x["score"])[:3]
        })

    return resultados


# =========================
# TEST
# =========================

if __name__ == "__main__":

    texto = """
Paso 01
OCR: 2+59
Reconstruido: 2+59
Problemas detectados:
  - (missing_equals)
  - (missing_variable)

Paso 02
OCR: 9-5
Reconstruido: 9-5
Problemas detectados:
  - (missing_equals)
  - (missing_variable)

Paso 03
OCR: 2X=
Reconstruido: 2x=
Problemas detectados:
  - (incomplete_expression)

Paso 04
OCR: 2x=41/2
Reconstruido: 2x=41/2

Paso 05
OCR: X=2
Reconstruido: x=2
"""

    pasos = parsear_entrada(texto)
    resultado = reconstruir(pasos)

    for i, r in enumerate(resultado, 1):
        print(f"\nPaso {i}")
        print(f"Original: {r['original']}")
        print(f"Mejor: {r['mejor']} (score={r['score']})")
        print("Top candidatos:")
        for c in r["interpretaciones"]:
            print(f"  - {c['ecuacion']} (score={c['score']})")