import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# =========================
# ESTRUCTURAS DE DATOS
# =========================

@dataclass
class Hipotesis:
    ecuacion: str
    score: float
    confianza: float  # 0.0 - 1.0
    transformaciones: list = field(default_factory=list)  # qué se modificó
    valida_matematicamente: bool = False

@dataclass
class ResultadoPaso:
    numero: int
    ocr_original: str
    reconstruido_agente1: str
    problemas: list
    sugerencias: list
    hipotesis: list  # lista de Hipotesis
    mejor_hipotesis: Optional[str]
    confianza_global: float
    necesita_contexto: bool  # flag para el Agente 3
    tipo_ruido: str  # "ninguno", "leve", "moderado", "severo"

# =========================
# PARSER ROBUSTO (desde dict o texto)
# =========================

def parsear_desde_dict(pasos_raw: list) -> list:
    """Acepta directamente la salida estructurada del Agente 1"""
    pasos = []
    for i, p in enumerate(pasos_raw):
        pasos.append({
            "numero": i + 1,
            "ocr": p.get("original", ""),
            "reconstruido": p.get("reconstruido", ""),
            "problemas": [prob["tipo"] for prob in p.get("problemas", [])],
            "sugerencias": p.get("sugerencias", [])
        })
    return pasos

# agente2.py — reemplazar parsear_desde_texto por esto:

def parsear_desde_agente1(salida_agente1: dict) -> list:

    pasos = []
    for p in salida_agente1["pasos"]:
        pasos.append({
            "numero":       p["numero"],
            "ocr":          p["original"],
            "reconstruido": p["reconstruido"],
            "confianza_ocr": p.get("confianza_ocr"),  # ← NUEVO: usar en score
            "bbox_y":       p.get("bbox_y"),           # ← NUEVO: detectar saltos raros
            "fusion":       p.get("fusion", False),    # ← NUEVO: hipótesis más conservadoras
            "repair":       p.get("repair", False),    # ← NUEVO: ya fue reparado
            "problemas":    [pr["tipo"] for pr in p.get("problemas", [])],
            "sugerencias":  p.get("sugerencias", [])
        })
    return pasos

# =========================
# VALIDACIÓN MATEMÁTICA
# =========================

def preparar_eval(expr: str) -> str:
    expr = re.sub(r'(\d)(x)', r'\1*\2', expr)
    expr = re.sub(r'(x)(\d)', r'\1*\2', expr)
    return expr

def validar_ecuacion(eq: str, x_val) -> bool:
    if x_val is None or '=' not in eq:
        return False
    try:
        izq, der = eq.split('=', 1)
        izq_eval = eval(preparar_eval(izq).replace('x', f'({x_val})'))
        der_eval = eval(preparar_eval(der).replace('x', f'({x_val})'))
        return abs(izq_eval - der_eval) < 1e-6
    except:
        return False

def obtener_x_final(pasos: list):
    for paso in reversed(pasos):
        match = re.match(r'x\s*=\s*(-?\d+(?:\.\d+)?)', paso["reconstruido"])
        if match:
            return float(match.group(1))
    return None

# =========================
# GENERACIÓN DE HIPÓTESIS CON CONTEXTO
# =========================

def es_forma_valida(eq: str) -> bool:
    if eq.count('=') != 1:
        return False
    if re.search(r'[^0-9x+\-=/*.]', eq):
        return False
    if re.search(r'\d{3,}x', eq):  # 123x → sospechoso
        return False
    partes = eq.split('=')
    if any(p.strip() == '' for p in partes):
        return False
    return True

def generar_hipotesis_con_contexto(paso: dict, paso_anterior: dict, paso_siguiente: dict, x_val) -> list:
    texto = paso["reconstruido"]
    problemas = paso["problemas"]
    candidatos = {}  # ecuacion → transformaciones aplicadas

    def agregar(eq, transformacion):
        if eq not in candidatos:
            candidatos[eq] = []
        candidatos[eq].append(transformacion)

    # Siempre incluir original
    agregar(texto, "original")

    # ===== INSERTAR '=' =====
    if "missing_equals" in problemas:
        for i in range(1, len(texto)):
            c = texto[:i] + "=" + texto[i:]
            agregar(c, f"insert_equals_pos_{i}")

    # ===== INSERTAR 'x' =====
    if "missing_variable" in problemas:
        for match in re.finditer(r'\d+', texto):
            for pos in [match.start(), match.end()]:
                c = texto[:pos] + "x" + texto[pos:]
                agregar(c, f"insert_x_pos_{pos}")

    # ===== CORREGIR NÚMERO LARGO ANTES DE '/' =====
    if re.search(r'\d{2,}/', texto):
        corregido = re.sub(r'(\d)(\d+)/', lambda m: m.group(1) + '/', texto)
        agregar(corregido, "fix_ocr_digit_before_slash")

    # ===== CONTEXTO: usar paso anterior para recuperar variable =====
    if "missing_variable" in problemas and paso_anterior:
        prev = paso_anterior["reconstruido"]
        if 'x' in prev:
            # Intentar insertar x al inicio
            c = "x" + texto if not texto.startswith('x') else texto
            agregar(c, "recover_x_from_prev_step")

    # ===== CONTEXTO: usar paso siguiente para inferir lado derecho =====
    if "incomplete_expression" in problemas and paso_siguiente:
        sig = paso_siguiente["reconstruido"]
        if '=' in sig:
            der = sig.split('=')[1]
            if texto.endswith('='):
                c = texto + der
                agregar(c, "recover_rhs_from_next_step")

    # Filtrar y construir objetos Hipotesis
    resultado = []
    for eq, transformaciones in candidatos.items():
        if not es_forma_valida(eq):
            continue
        valida = validar_ecuacion(eq, x_val)
        score = calcular_score(eq, valida, transformaciones, problemas)
        confianza = min(1.0, max(0.0, (score + 10) / 40))

        resultado.append(Hipotesis(
            ecuacion=eq,
            score=score,
            confianza=round(confianza, 3),
            transformaciones=transformaciones,
            valida_matematicamente=valida
        ))

    return sorted(resultado, key=lambda h: -h.score)

# =========================
# SCORING
# =========================

def calcular_score(eq, valida, transformaciones, problemas, paso_meta=None):
    score = 0.0
    if '=' in eq: score += 5
    if 'x' in eq: score += 3
    if valida:    score += 20
    if "original" in transformaciones: score += 2

    # ← NUEVO: penalizar si el OCR ya tenía baja confianza
    if paso_meta and paso_meta.get("confianza_ocr") is not None:
        if paso_meta["confianza_ocr"] < 0.6:
            score -= 3   # OCR dudoso → hipótesis menos confiables

    # ← NUEVO: si ya fue reparado por A1, no volver a transformar
    if paso_meta and paso_meta.get("repair"):
        if "fix_ocr" in str(transformaciones):
            score -= 5

    n_cambios = len([t for t in transformaciones if t != "original"])
    score -= n_cambios * 1.5

    if any("from_prev" in t or "from_next" in t for t in transformaciones):
        score += 4

    return round(score, 2)

# =========================
# CLASIFICAR TIPO DE RUIDO
# =========================

def clasificar_ruido(problemas: list, mejor_score: float) -> str:
    n = len(problemas)
    if n == 0 and mejor_score >= 25:
        return "ninguno"
    elif n <= 1 and mejor_score >= 15:
        return "leve"
    elif n <= 2 or mejor_score >= 5:
        return "moderado"
    else:
        return "severo"

# =========================
# PROCESAMIENTO PRINCIPAL
# =========================

def reconstruir(pasos_raw) -> dict:
    # Aceptar: string JSON, dict completo del Agente 1, o lista de pasos
    if isinstance(pasos_raw, str):
        # Parsear el JSON string primero, luego extraer pasos
        salida_dict = json.loads(pasos_raw)
        pasos = parsear_desde_agente1(salida_dict)
    elif isinstance(pasos_raw, dict):
        # Dict completo del Agente 1 (con clave "pasos")
        pasos = parsear_desde_agente1(pasos_raw)
    else:
        # Lista directa de pasos
        pasos = parsear_desde_dict(pasos_raw)

    x_val = obtener_x_final(pasos)
    resultados = []

    for i, paso in enumerate(pasos):
        anterior = pasos[i - 1] if i > 0 else None
        siguiente = pasos[i + 1] if i < len(pasos) - 1 else None

        hipotesis = generar_hipotesis_con_contexto(paso, anterior, siguiente, x_val)

        mejor = hipotesis[0] if hipotesis else None
        mejor_score = mejor.score if mejor else -999
        tipo_ruido = clasificar_ruido(paso["problemas"], mejor_score)

        resultado = ResultadoPaso(
            numero=paso["numero"],
            ocr_original=paso["ocr"],
            reconstruido_agente1=paso["reconstruido"],
            problemas=paso["problemas"],
            sugerencias=paso.get("sugerencias", []),
            hipotesis=[asdict(h) for h in hipotesis[:5]],  # top 5
            mejor_hipotesis=mejor.ecuacion if mejor else None,
            confianza_global=mejor.confianza if mejor else 0.0,
            necesita_contexto=tipo_ruido in ("moderado", "severo"),
            tipo_ruido=tipo_ruido
        )
        resultados.append(asdict(resultado))

    # ===== SALIDA PARA AGENTE 3 =====
    salida_agente3 = {
        "x_detectada": x_val,
        "total_pasos": len(resultados),
        "pasos_con_ruido": sum(1 for r in resultados if r["tipo_ruido"] != "ninguno"),
        "pasos": resultados,
        "metadata": {
            "version_agente": "2.1",
            "requiere_verificacion_ia": any(r["necesita_contexto"] for r in resultados)
        }
    }

    return salida_agente3

# =========================
# TEST
# =========================

if __name__ == "__main__":
    texto_test = """
{
  "version": "1.0",
  "fuente": "agente_ocr",
  "imagen": "ecuCom5.jpeg",
  "total_lineas": 5,
  "pasos": [
    {
      "numero": 1,
      "original": "2+59",
      "reconstruido": "2+59",
      "confianza_ocr": 0.9233,
      "bbox_y": 192.0,
      "fusion": false,
      "repair": false,
      "problemas": [
        {
          "tipo": "missing_equals",
          "detalle": "No se detecta signo '='"
        },
        {
          "tipo": "missing_variable",
          "detalle": "No se detecta variable 'x'"
        }
      ],
      "sugerencias": [
        {
          "tipo": "insert_equals",
          "confianza": 0.7
        },
        {
          "tipo": "insert_variable",
          "posible": "inicio o junto a número",
          "confianza": 0.6
        }
      ]
    },
    {
      "numero": 2,
      "original": "9-5",
      "reconstruido": "9-5",
      "confianza_ocr": 0.731,
      "bbox_y": 296.0,
      "fusion": false,
      "repair": false,
      "problemas": [
        {
          "tipo": "missing_equals",
          "detalle": "No se detecta signo '='"
        },
        {
          "tipo": "missing_variable",
          "detalle": "No se detecta variable 'x'"
        }
      ],
      "sugerencias": [
        {
          "tipo": "insert_equals",
          "confianza": 0.7
        },
        {
          "tipo": "insert_variable",
          "posible": "inicio o junto a número",
          "confianza": 0.6
        }
      ]
    },
    {
      "numero": 3,
      "original": "2X=",
      "reconstruido": "2x=",
      "confianza_ocr": 0.7055,
      "bbox_y": 333.0,
      "fusion": false,
      "repair": false,
      "problemas": [
        {
          "tipo": "incomplete_expression",
          "detalle": "Expresión incompleta en un lado del '='"
        }
      ],
      "sugerencias": []
    },
    {
      "numero": 4,
      "original": "2x=41/2",
      "reconstruido": "2x=41/2",
      "confianza_ocr": 0.8,
      "bbox_y": 474.0,
      "fusion": false,
      "repair": false,
      "problemas": [],
      "sugerencias": []
    },
    {
      "numero": 5,
      "original": "X=2",
      "reconstruido": "x=2",
      "confianza_ocr": 0.5903,
      "bbox_y": 603.0,
      "fusion": false,
      "repair": false,
      "problemas": [],
      "sugerencias": []
    }
  ]
}
"""

    resultado = reconstruir(texto_test)

    # Imprimir resumen legible
    print(f"\n{'='*50}")
    print(f"x detectada: {resultado['x_detectada']}")
    print(f"Pasos con ruido: {resultado['pasos_con_ruido']}/{resultado['total_pasos']}")
    print(f"Requiere verificación IA: {resultado['metadata']['requiere_verificacion_ia']}")
    print(f"{'='*50}")

    for paso in resultado["pasos"]:
        print(f"\nPaso {paso['numero']:02d} [{paso['tipo_ruido'].upper()}]")
        print(f"  OCR:        {paso['ocr_original']}")
        print(f"  Agente 1:   {paso['reconstruido_agente1']}")
        print(f"  Mejor:      {paso['mejor_hipotesis']} (confianza={paso['confianza_global']})")
        if paso["hipotesis"]:
            print("  Top hipótesis:")
            for h in paso["hipotesis"][:3]:
                print(f"    • {h['ecuacion']} score={h['score']} válida={h['valida_matematicamente']}")

    # JSON listo para Agente 3
    print("\n\n--- JSON para Agente 3 ---")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))