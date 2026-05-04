# Agente3.py
import json
import re
import base64
import openai
from typing import Optional

# =========================
# CONFIGURACIÓN
# =========================

client = openai.OpenAI(
    api_key="s2_de118541a5804ebb880e0a91b19a450c",  # https://apps.abacus.ai/chatllm/admin/profile
    base_url="https://routellm.abacus.ai/v1"
)

# Imagen hardcodeada para pruebas — cambiar por la ruta real
IMAGEN_PRUEBA = "ecuCom5.jpeg"

SYSTEM_PROMPT_TEXTO = """Eres un asistente especializado en recuperar ecuaciones algebraicas 
dañadas por ruido OCR en cuadernos de estudiantes de secundaria.
Son ejercicios de ecuaciones lineales con una incógnita (x).

Errores OCR más comunes:
- '//' en lugar de '/' o ':'
- Dígitos pegados a operadores
- Mayúsculas donde debería haber minúsculas
- '=' o 'x' no detectados

Reglas ESTRICTAS:
- Responde SOLO con la ecuación, sin explicaciones
- Usa solo: 0-9, x, +, -, *, /, =, (, ), .
- La ecuación DEBE tener exactamente un '='
- No inventes pasos nuevos, solo corrige el ruido
"""

SYSTEM_PROMPT_VISION = """Eres un asistente especializado en leer ecuaciones algebraicas 
escritas a mano en cuadernos de estudiantes de secundaria.

Tu tarea: mirar la imagen del cuaderno y recuperar la ecuación exacta que escribió el estudiante.

Reglas ESTRICTAS:
- Responde SOLO con la ecuación, sin explicaciones
- Usa solo: 0-9, x, +, -, *, /, =, (, ), .
- La ecuación DEBE tener exactamente un '='
- Si ves múltiples líneas, enfócate en el paso indicado
"""

# =========================
# DECIDIR SI USAR VISIÓN
# =========================

def necesita_vision(paso: dict) -> bool:
    """Usar imagen solo cuando el texto solo no es suficiente."""
    confianza   = paso.get("confianza_global", 0.0)
    hipotesis   = paso.get("hipotesis", [])
    hay_valida  = any(h.get("valida_matematicamente") for h in hipotesis)
    n_problemas = len(paso.get("problemas", []))

    # Usar visión si:
    # 1. Confianza muy baja Y ninguna hipótesis válida
    # 2. O perdió múltiples elementos simultáneamente (x Y = al mismo tiempo)
    return (confianza < 0.4 and not hay_valida) or n_problemas >= 2

def cargar_imagen_b64(imagen_path: str) -> Optional[str]:
    """Carga imagen y la convierte a base64."""
    try:
        with open(imagen_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"    ✗ No se pudo cargar imagen: {e}")
        return None

# =========================
# LLM SOLO TEXTO
# =========================

def recuperar_con_llm(paso: dict, paso_anterior: Optional[dict], paso_siguiente: Optional[dict]) -> Optional[str]:
    """LLM sin imagen — para ruido leve/moderado con hipótesis disponibles."""

    def get_ec(p):
        if p is None: return None
        return p.get("ecuacion_final") or p.get("mejor_hipotesis") or p.get("reconstruido_agente1", "")

    ctx_anterior  = f"Paso anterior : {get_ec(paso_anterior)}"  if paso_anterior  else "Es el primer paso"
    ctx_siguiente = f"Paso siguiente: {get_ec(paso_siguiente)}" if paso_siguiente else "Es el último paso"

    hipotesis_texto = "\n".join([
        f"  [{i+1}] {h['ecuacion']}  (score={h['score']}, confianza={h['confianza']}, válida={h['valida_matematicamente']})"
        for i, h in enumerate(paso["hipotesis"])
    ])

    prompt = f"""Recupera este paso dañado por OCR:

OCR original (con ruido) : {paso['ocr_original']}
Reconstrucción Agente 1  : {paso['reconstruido_agente1']}
Tipo de ruido            : {paso['tipo_ruido']}
Confianza                : {paso['confianza_global']}

{ctx_anterior}
{ctx_siguiente}

Hipótesis disponibles:
{hipotesis_texto}

Responde ÚNICAMENTE con la ecuación corregida."""

    try:
        response = client.chat.completions.create(
            model="route-llm",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEXTO},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.1,
            max_tokens=60
        )
        raw = response.choices[0].message.content.strip()
        ecuacion = re.sub(r'[^0-9x+\-=/*.()\s]', '', raw).strip()
        if '=' in ecuacion:
            print(f"    ✓ LLM texto recuperó : {ecuacion}")
            return ecuacion
        else:
            print(f"    ✗ LLM texto inválido : '{raw}'")
            return None
    except Exception as e:
        print(f"    ✗ LLM texto ERROR    : {e}")
        return None

# =========================
# LLM CON VISIÓN
# =========================

def recuperar_con_vision(paso: dict, paso_anterior: Optional[dict], paso_siguiente: Optional[dict], imagen_path: str) -> Optional[str]:
    """LLM con imagen — para casos donde el texto solo no alcanza."""

    imagen_b64 = cargar_imagen_b64(imagen_path)
    if not imagen_b64:
        return None

    # Detectar extensión para el mime type
    ext = imagen_path.lower().split('.')[-1]
    mime = "image/png" if ext == "png" else "image/jpeg"

    def get_ec(p):
        if p is None: return None
        return p.get("ecuacion_final") or p.get("mejor_hipotesis") or p.get("reconstruido_agente1", "")

    ctx_anterior  = f"Paso anterior : {get_ec(paso_anterior)}"  if paso_anterior  else "Es el primer paso"
    ctx_siguiente = f"Paso siguiente: {get_ec(paso_siguiente)}" if paso_siguiente else "Es el último paso"

    hipotesis_texto = "\n".join([
        f"  [{i+1}] {h['ecuacion']} (confianza={h['confianza']})"
        for i, h in enumerate(paso["hipotesis"])
    ]) if paso["hipotesis"] else "  (sin hipótesis disponibles)"

    prompt_texto = f"""Mira la imagen del cuaderno del estudiante.

Necesito recuperar el paso {paso['numero']} de un ejercicio de álgebra.
El OCR detectó con ruido: "{paso['ocr_original']}"

{ctx_anterior}
{ctx_siguiente}

Hipótesis generadas automáticamente:
{hipotesis_texto}

Mirando la imagen, ¿cuál es la ecuación exacta del paso {paso['numero']}?
Responde ÚNICAMENTE con la ecuación."""

    try:
        response = client.chat.completions.create(
            model="route-llm",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_VISION},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_texto},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{imagen_b64}",
                        "detail": "low"   # 85 tokens fijos — suficiente para ecuaciones
                    }}
                ]}
            ],
            temperature=0.1,
            max_tokens=60
        )
        raw = response.choices[0].message.content.strip()
        ecuacion = re.sub(r'[^0-9x+\-=/*.()\s]', '', raw).strip()
        if '=' in ecuacion:
            print(f"    ✓ LLM visión recuperó: {ecuacion}")
            return ecuacion
        else:
            print(f"    ✗ LLM visión inválido: '{raw}'")
            return None
    except Exception as e:
        print(f"    ✗ LLM visión ERROR   : {e}")
        return None

# =========================
# FALLBACK LOCAL
# =========================

def recuperar_local(paso: dict) -> str:
    """Sin LLM: mejor hipótesis por score."""
    hipotesis = paso.get("hipotesis", [])
    if not hipotesis:
        return paso.get("reconstruido_agente1", "")
    validas = [h for h in hipotesis if h.get("valida_matematicamente")]
    if validas:
        return max(validas, key=lambda h: (h["score"], h["confianza"]))["ecuacion"]
    return max(hipotesis, key=lambda h: (h["score"], h["confianza"]))["ecuacion"]

# =========================
# PROCESAMIENTO PRINCIPAL
# =========================

def ejecutar_agente3(salida_agente2: dict, usar_llm: bool = True, imagen_path: Optional[str] = None) -> dict:
    pasos_entrada   = salida_agente2["pasos"]
    pasos_resultado = []

    print(f"\n{'='*55}")
    print(f"  AGENTE 3 — Recuperación de información OCR")
    print(f"{'='*55}")
    print(f"  x detectada por A2 : {salida_agente2.get('x_detectada')}")
    print(f"  Pasos con ruido    : {salida_agente2.get('pasos_con_ruido')}/{salida_agente2.get('total_pasos')}")
    print(f"  Modo LLM           : {'Activado' if usar_llm else 'Desactivado'}")
    print(f"  Imagen disponible  : {'Sí → ' + imagen_path if imagen_path else 'No'}")
    print(f"{'='*55}\n")

    for i, paso in enumerate(pasos_entrada):
        anterior  = pasos_resultado[i - 1] if i > 0 else None
        siguiente = pasos_entrada[i + 1]   if i < len(pasos_entrada) - 1 else None

        tipo_ruido          = paso.get("tipo_ruido", "ninguno")
        necesita_recuperacion = tipo_ruido in ("moderado", "severo") or paso.get("necesita_contexto", False)

        if necesita_recuperacion:
            print(f"  Paso {paso['numero']:02d} [{tipo_ruido.upper():8s}] → Recuperando...")

            ecuacion_recuperada = None
            metodo_usado        = "local_score"

            if usar_llm:
                # ¿Necesita visión o solo texto?
                if imagen_path and necesita_vision(paso):
                    print(f"    → Usando visión (confianza={paso.get('confianza_global')}, problemas={len(paso.get('problemas',[]))})")
                    ecuacion_recuperada = recuperar_con_vision(paso, anterior, siguiente, imagen_path)
                    if ecuacion_recuperada:
                        metodo_usado = "llm_vision"

                # Si visión falló o no era necesaria → texto solo
                if not ecuacion_recuperada:
                    ecuacion_recuperada = recuperar_con_llm(paso, anterior, siguiente)
                    if ecuacion_recuperada:
                        metodo_usado = "llm_texto"

            # Fallback local si todo falla
            if not ecuacion_recuperada:
                ecuacion_recuperada = recuperar_local(paso)
                metodo_usado        = "local_score"
                print(f"    → Fallback local     : {ecuacion_recuperada}")

        else:
            ecuacion_recuperada = paso.get("mejor_hipotesis") or paso.get("reconstruido_agente1", "")
            metodo_usado        = "sin_cambio"
            print(f"  Paso {paso['numero']:02d} [LIMPIO  ] → {ecuacion_recuperada}")

        pasos_resultado.append({
            "numero"              : paso["numero"],
            "ocr_original"        : paso["ocr_original"],
            "reconstruido_a1"     : paso["reconstruido_agente1"],
            "ecuacion_final"      : ecuacion_recuperada,
            "metodo_recuperacion" : metodo_usado,
            "tipo_ruido_original" : tipo_ruido,
            "confianza_a2"        : paso.get("confianza_global", 0.0),
            "fue_recuperado"      : metodo_usado not in ("sin_cambio", "local_score") and tipo_ruido != "ninguno",
            "no_recuperable"      : metodo_usado == "local_score" and tipo_ruido in ("moderado", "severo")
        })

    # Detectar x final
    x_final = None
    for paso in reversed(pasos_resultado):
        m = re.match(r'x\s*=\s*(-?\d+(?:\.\d+)?)', paso["ecuacion_final"].strip())
        if m:
            x_final = float(m.group(1))
            break

    salida = {
        "x_detectada"      : x_final,
        "total_pasos"      : len(pasos_resultado),
        "pasos_recuperados": sum(1 for p in pasos_resultado if p["fue_recuperado"]),
        "pasos_no_recuperables": sum(1 for p in pasos_resultado if p["no_recuperable"]),
        "pasos"            : pasos_resultado,
        "metadata": {
            "version_agente" : "3.1",
            "llm_usado"      : usar_llm,
            "vision_usada"   : any(p["metodo_recuperacion"] == "llm_vision" for p in pasos_resultado),
            "imagen_path"    : imagen_path,
            "listo_para_a4"  : True
        }
    }

    print(f"\n{'='*55}")
    print(f"  x final recuperada   : {x_final}")
    print(f"  Recuperados con LLM  : {sum(1 for p in pasos_resultado if 'llm' in p['metodo_recuperacion'])}")
    print(f"  Recuperados con visión: {sum(1 for p in pasos_resultado if p['metodo_recuperacion'] == 'llm_vision')}")
    print(f"  No recuperables      : {salida['pasos_no_recuperables']}")
    print(f"{'='*55}\n")

    return salida

# =========================
# PIPELINE
# =========================

def pipeline(salida_agente2_json, usar_llm: bool = True, imagen_path: Optional[str] = None) -> dict:
    if isinstance(salida_agente2_json, str):
        salida_agente2 = json.loads(salida_agente2_json)
    else:
        salida_agente2 = salida_agente2_json
    return ejecutar_agente3(salida_agente2, usar_llm=usar_llm, imagen_path=imagen_path)

# =========================
# TEST
# =========================

if __name__ == "__main__":
    test_agente2 = {
        "x_detectada": 2.0,
        "total_pasos": 5,
        "pasos_con_ruido": 4,
        "pasos": [
            {
                "numero": 1, "ocr_original": "2+59",
                "reconstruido_agente1": "2+59",
                "problemas": ["missing_equals", "missing_variable"], "sugerencias": [],
                "hipotesis": [
                    {"ecuacion": "2=+59", "score": 3.5, "confianza": 0.338, "transformaciones": ["insert_equals_pos_1"], "valida_matematicamente": False},
                    {"ecuacion": "2+=59", "score": 3.5, "confianza": 0.338, "transformaciones": ["insert_equals_pos_2"], "valida_matematicamente": False},
                    {"ecuacion": "2+5=9", "score": 3.5, "confianza": 0.338, "transformaciones": ["insert_equals_pos_3"], "valida_matematicamente": False},
                ],
                "mejor_hipotesis": "2=+59", "confianza_global": 0.338,
                "necesita_contexto": True, "tipo_ruido": "moderado"
            },
            {
                "numero": 2, "ocr_original": "9-5",
                "reconstruido_agente1": "9-5",
                "problemas": ["missing_equals", "missing_variable"], "sugerencias": [],
                "hipotesis": [
                    {"ecuacion": "9=-5", "score": 3.5, "confianza": 0.338, "transformaciones": ["insert_equals_pos_1"], "valida_matematicamente": False},
                    {"ecuacion": "9-=5", "score": 3.5, "confianza": 0.338, "transformaciones": ["insert_equals_pos_2"], "valida_matematicamente": False},
                ],
                "mejor_hipotesis": "9=-5", "confianza_global": 0.338,
                "necesita_contexto": True, "tipo_ruido": "moderado"
            },
            {
                "numero": 3, "ocr_original": "2X=",
                "reconstruido_agente1": "2x=",
                "problemas": ["incomplete_expression"], "sugerencias": [],
                "hipotesis": [
                    {"ecuacion": "2x=41/2", "score": 10.5, "confianza": 0.512, "transformaciones": ["recover_rhs_from_next_step"], "valida_matematicamente": False},
                ],
                "mejor_hipotesis": "2x=41/2", "confianza_global": 0.512,
                "necesita_contexto": True, "tipo_ruido": "moderado"
            },
            {
                "numero": 4, "ocr_original": "2x=41/2",
                "reconstruido_agente1": "2x=41/2",
                "problemas": [], "sugerencias": [],
                "hipotesis": [
                    {"ecuacion": "2x=41/2", "score": 10.0, "confianza": 0.5,   "transformaciones": ["original"],                   "valida_matematicamente": False},
                    {"ecuacion": "2x=4/2",  "score": 6.5,  "confianza": 0.412, "transformaciones": ["fix_ocr_digit_before_slash"],  "valida_matematicamente": False},
                ],
                "mejor_hipotesis": "2x=41/2", "confianza_global": 0.5,
                "necesita_contexto": True, "tipo_ruido": "moderado"
            },
            {
                "numero": 5, "ocr_original": "X=2",
                "reconstruido_agente1": "x=2",
                "problemas": [], "sugerencias": [],
                "hipotesis": [
                    {"ecuacion": "x=2", "score": 30.0, "confianza": 1.0, "transformaciones": ["original"], "valida_matematicamente": True},
                ],
                "mejor_hipotesis": "x=2", "confianza_global": 1.0,
                "necesita_contexto": False, "tipo_ruido": "ninguno"
            }
        ],
        "metadata": {"version_agente": "2.1", "requiere_verificacion_ia": True}
    }

    # ← Cambiar por la ruta de tu imagen de prueba
    resultado = pipeline(test_agente2, usar_llm=True, imagen_path=IMAGEN_PRUEBA)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))