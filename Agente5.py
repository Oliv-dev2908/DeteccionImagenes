#!/usr/bin/env python3
"""
=============================================================================
  AGENTE 5 — Evaluador Final Inteligente
  Sistema Multi-Agente de Evaluación de Ejercicios Matemáticos
=============================================================================

  Este agente recibe la salida del Agente 4 (verificación con SymPy) y usa
  la API de IA de Abacus AI para dar una evaluación más justa y contextual,
  considerando que saltarse pasos intermedios no es tan grave si la lógica
  matemática y el resultado final son correctos.

  Cadena de agentes:
    Agente 1 → Captura imagen
    Agente 2 → OCR (extrae texto)
    Agente 3 → Extrae pasos matemáticos a JSON
    Agente 4 → Verifica con SymPy y califica  ← ENTRADA de este agente
    Agente 5 → Evaluación final inteligente   ← ESTE SCRIPT
=============================================================================
"""

import json
import os
import sys
from datetime import datetime

try:
    import openai
except ImportError:
    print("ERROR: El paquete 'openai' no está instalado.")
    print("Instálalo con: pip install openai")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

# Cliente OpenAI apuntando a RouteLLM de Abacus AI
OPENAI_API_KEY = "s2_de118541a5804ebb880e0a91b19a450c"  # https://apps.abacus.ai/chatllm/admin/profile
OPENAI_BASE_URL = "https://routellm.abacus.ai/v1"
OPENAI_MODEL = "route-llm"

# Archivo de salida
OUTPUT_FILE = "agente5_salida.json"


# ═══════════════════════════════════════════════════════════════════════════
# 2. JSON HARDCODEADO DEL AGENTE 4 (entrada)
# ═══════════════════════════════════════════════════════════════════════════

AGENTE4_JSON = {
    "agente": "Agente4",
    "ejercicio": "Ecuación lineal: 2x + 3 = 7",
    "resultado_final_estudiante": "x = 2",
    "resultado_correcto": "x = 2",
    "resultado_final_correcto": True,
    "calificacion": 5.8,
    "total_pasos": 5,
    "pasos_correctos": 3,
    "pasos_con_errores": 2,
    "detalle_pasos": [
        {
            "paso": 1,
            "descripcion": "Escribir la ecuación original: 2x + 3 = 7",
            "correcto": True,
            "verificacion_sympy": "Ecuación válida"
        },
        {
            "paso": 2,
            "descripcion": "Restar 3 de ambos lados: 2x = 4",
            "correcto": True,
            "verificacion_sympy": "Operación algebraica correcta"
        },
        {
            "paso": 3,
            "descripcion": "Dividir ambos lados entre 2: x = 2",
            "correcto": True,
            "verificacion_sympy": "Resultado verificado: x = 2"
        },
        {
            "paso": 4,
            "descripcion": "Verificación: 2(2) + 3 = 7 → 4 + 3 = 7 ✓",
            "correcto": False,
            "verificacion_sympy": "Error aritmético detectado: paso de verificación contiene salto lógico",
            "tipo_error": "Se omitió mostrar la sustitución explícita paso a paso"
        },
        {
            "paso": 5,
            "descripcion": "Conclusión: x = 2 es la solución",
            "correcto": False,
            "verificacion_sympy": "Incoherencia secuencial: conclusión sin justificación formal completa",
            "tipo_error": "Falta justificación formal del paso anterior"
        }
    ],
    "observaciones": [
        "El resultado final x=2 es matemáticamente correcto",
        "Se detectaron 2 pasos con errores técnicos de forma",
        "Los errores son de presentación/formalidad, no de lógica matemática",
        "La calificación refleja penalización por falta de rigor formal"
    ],
    "timestamp": "2026-05-11T10:30:00"
}


# ═══════════════════════════════════════════════════════════════════════════
# 3. PROMPT INTELIGENTE PARA LA IA
# ═══════════════════════════════════════════════════════════════════════════

def construir_prompt(agente4_data: dict) -> str:
    """Construye el prompt contextual para enviar a la IA de Abacus."""

    agente4_json_str = json.dumps(agente4_data, indent=2, ensure_ascii=False)

    prompt = f"""Eres un evaluador experto en educación matemática. Formas parte de un sistema 
multi-agente de evaluación de ejercicios matemáticos donde:

- Agente 1: Captura la imagen del ejercicio del estudiante
- Agente 2: Extrae el texto con OCR
- Agente 3: Convierte los pasos matemáticos a formato JSON estructurado
- Agente 4: Verifica cada paso con SymPy (verificación simbólica) y asigna calificación
- Agente 5 (TÚ): Evaluador final inteligente que revisa la calificación del Agente 4

El Agente 4 ha producido la siguiente evaluación:

json
{agente4_json_str}


CONTEXTO IMPORTANTE:
- El Agente 4 asignó una calificación de {agente4_data['calificacion']}/10
- El resultado final del estudiante ({agente4_data['resultado_final_estudiante']}) es CORRECTO
- Los "errores" detectados son principalmente de FORMA (saltar pasos intermedios, falta de 
  formalidad), NO de lógica matemática
- La lógica matemática del estudiante es correcta en todos los pasos esenciales
- El estudiante llegó a la respuesta correcta siguiendo un razonamiento válido

TU TAREA:
Analiza críticamente si la calificación de {agente4_data['calificacion']}/10 es justa, 
considerando que:

1. En educación matemática, lo más importante es la COMPRENSIÓN del concepto y la 
   CORRECCIÓN del resultado
2. Saltarse pasos intermedios de verificación no es un error grave si la lógica es correcta
3. Un estudiante que llega al resultado correcto con lógica válida merece una calificación alta
4. La falta de "rigor formal" en pasos de verificación no debería penalizar tanto como 
   un error conceptual real
5. Los errores detectados por el Agente 4 son de presentación, no de comprensión matemática

RESPONDE ESTRICTAMENTE en el siguiente formato JSON (sin texto adicional antes o después).
Sé CONCISO y DIRECTO: máximo 2 oraciones por campo, sin rodeos ni explicaciones largas.
{{
    "calificacion_ajustada": <número decimal entre 1 y 10>,
    "retroalimentacion_final": "<​1-2 oraciones directas: qué hizo bien y qué debe mejorar>",
    "analisis_agente4": "<​1-2 oraciones: ¿fue justa la nota del Agente 4 y por qué no?>",
    "justificacion_ajuste": "<​1-2 oraciones: razón concreta del ajuste con argumento pedagógico>"
}}"""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# 4. LLAMADA A LA API DE ABACUS AI
# ═══════════════════════════════════════════════════════════════════════════

def llamar_api_abacus(prompt: str) -> str:
    """Envía el prompt a la API OpenAI (RouteLLM de Abacus AI) y retorna la respuesta."""

    print("🔗 Conectando con RouteLLM (Abacus AI)...")
    print(f"   Modelo: {OPENAI_MODEL}")
    print()

    try:
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )

        print("📤 Enviando solicitud de evaluación a la IA...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": ("Eres un evaluador experto en educación matemática. "
                                "Respondes siempre en JSON válido cuando se te solicita.")
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )

        respuesta_texto = response.choices[0].message.content

        print("   ✅ Respuesta recibida de la IA")
        print()

        return respuesta_texto

    except Exception as e:
        print(f"   ❌ Error al llamar a la API: {type(e).__name__}: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# 5. PROCESAMIENTO DE LA RESPUESTA
# ═══════════════════════════════════════════════════════════════════════════

def extraer_json_de_respuesta(respuesta_cruda: str) -> dict:
    """Intenta extraer un objeto JSON de la respuesta de la IA."""

    texto = respuesta_cruda.strip()

    # Intentar parsear directamente
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    # Buscar bloque json ... 
    if "json" in texto:
        inicio = texto.index("json") + len("json")
        fin = texto.index("", inicio)
        bloque = texto[inicio:fin].strip()
        try:
            return json.loads(bloque)
        except json.JSONDecodeError:
            pass

    # Buscar bloque  ... 
    if "" in texto:
        inicio = texto.index("") + 3
        fin = texto.index("", inicio)
        bloque = texto[inicio:fin].strip()
        try:
            return json.loads(bloque)
        except json.JSONDecodeError:
            pass

    # Buscar primer { ... último }
    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio != -1 and fin != -1 and fin > inicio:
        bloque = texto[inicio:fin + 1]
        try:
            return json.loads(bloque)
        except json.JSONDecodeError:
            pass

    # No se pudo parsear — devolver estructura por defecto
    print("⚠️  No se pudo parsear JSON de la respuesta. Usando estructura por defecto.")
    return {
        "calificacion_ajustada": None,
        "retroalimentacion_final": respuesta_cruda,
        "analisis_agente4": "No se pudo extraer análisis estructurado de la respuesta de la IA.",
        "justificacion_ajuste": "Revisar respuesta cruda de la IA."
    }


def construir_salida(datos_ia: dict, agente4_data: dict) -> dict:
    """Construye el JSON final de salida del Agente 5."""

    return {
        "agente": "Agente5",
        "descripcion": "Evaluador Final Inteligente — Revisión con IA de Abacus AI",
        "calificacion_original_agente4": agente4_data["calificacion"],
        "calificacion_ajustada": datos_ia.get("calificacion_ajustada", "N/A"),
        "retroalimentacion_final": datos_ia.get("retroalimentacion_final", "N/A"),
        "analisis_agente4": datos_ia.get("analisis_agente4", "N/A"),
        "justificacion_ajuste": datos_ia.get("justificacion_ajuste", "N/A"),
        "ejercicio_evaluado": agente4_data.get("ejercicio", "N/A"),
        "resultado_estudiante": agente4_data.get("resultado_final_estudiante", "N/A"),
        "resultado_correcto": agente4_data.get("resultado_correcto", "N/A"),
        "modelo_ia": "Abacus AI ChatLLM",
        "timestamp": datetime.now().isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def imprimir_resultado(salida: dict, respuesta_cruda: str = ""):
    """Imprime el resultado de forma legible en consola."""

    separador = "═" * 72

    print()
    print(separador)
    print("  🎓  AGENTE 5 — EVALUACIÓN FINAL INTELIGENTE")
    print(separador)
    print()

    print(f"  📝 Ejercicio evaluado : {salida['ejercicio_evaluado']}")
    print(f"  📌 Respuesta estudiante: {salida['resultado_estudiante']}")
    print(f"  ✅ Respuesta correcta  : {salida['resultado_correcto']}")
    print()

    print("  ─── CALIFICACIONES ───────────────────────────────────────────")
    print(f"  📊 Calificación Agente 4 (original) : {salida['calificacion_original_agente4']}/10")
    print(f"  🎯 Calificación Ajustada (Agente 5) : {salida['calificacion_ajustada']}/10")

    # Calcular diferencia
    try:
        original = float(salida["calificacion_original_agente4"])
        ajustada = float(salida["calificacion_ajustada"])
        diff = ajustada - original
        signo = "+" if diff > 0 else ""
        print(f"  📈 Diferencia                        : {signo}{diff:.1f} puntos")
    except (ValueError, TypeError):
        pass

    print()
    print("  ─── RETROALIMENTACIÓN PARA EL ESTUDIANTE ─────────────────────")
    # Imprimir con word-wrap básico
    retro = salida.get("retroalimentacion_final", "N/A")
    for linea in _wrap_text(retro, 68):
        print(f"  {linea}")

    print()
    print("  ─── ANÁLISIS DEL AGENTE 4 ────────────────────────────────────")
    analisis = salida.get("analisis_agente4", "N/A")
    for linea in _wrap_text(analisis, 68):
        print(f"  {linea}")

    print()
    print("  ─── JUSTIFICACIÓN DEL AJUSTE ─────────────────────────────────")
    justif = salida.get("justificacion_ajuste", "N/A")
    for linea in _wrap_text(justif, 68):
        print(f"  {linea}")

    print()
    print(separador)


def _wrap_text(text: str, width: int) -> list:
    """Divide texto largo en líneas de máximo 'width' caracteres."""
    if not text:
        return ["N/A"]
    palabras = text.split()
    lineas = []
    linea_actual = ""
    for palabra in palabras:
        if len(linea_actual) + len(palabra) + 1 <= width:
            linea_actual += (" " if linea_actual else "") + palabra
        else:
            if linea_actual:
                lineas.append(linea_actual)
            linea_actual = palabra
    if linea_actual:
        lineas.append(linea_actual)
    return lineas if lineas else ["N/A"]


# ═══════════════════════════════════════════════════════════════════════════
# 7. FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     SISTEMA MULTI-AGENTE DE EVALUACIÓN MATEMÁTICA              ║")
    print("║     Agente 5: Evaluador Final Inteligente (Abacus AI)          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # --- Paso 1: Cargar datos del Agente 4 ---
    print("📥 Paso 1: Cargando datos del Agente 4...")
    agente4_data = AGENTE4_JSON
    print(f"   Ejercicio    : {agente4_data['ejercicio']}")
    print(f"   Calificación : {agente4_data['calificacion']}/10")
    print(f"   Pasos correctos: {agente4_data['pasos_correctos']}/{agente4_data['total_pasos']}")
    print(f"   Resultado final correcto: {'Sí ✅' if agente4_data['resultado_final_correcto'] else 'No ❌'}")
    print()

    # --- Paso 2: Construir prompt ---
    print("🧠 Paso 2: Construyendo prompt inteligente para la IA...")
    prompt = construir_prompt(agente4_data)
    print(f"   Longitud del prompt: {len(prompt)} caracteres")
    print()

    # --- Paso 3: Llamar a la API ---
    print("🚀 Paso 3: Enviando a Abacus AI para evaluación inteligente...")
    print()

    try:
        respuesta_cruda = llamar_api_abacus(prompt)
    except Exception as e:
        print()
        print(f"❌ ERROR FATAL: No se pudo obtener respuesta de la IA: {e}")
        print("   Verifica tu API key y conexión a internet.")
        print("   Generando salida con valores por defecto...")
        print()

        # Salida de respaldo
        respuesta_cruda = "Error al conectar con la IA"
        salida = {
            "agente": "Agente5",
            "calificacion_original_agente4": agente4_data["calificacion"],
            "calificacion_ajustada": "ERROR",
            "retroalimentacion_final": "No se pudo generar evaluación por error en la API.",
            "analisis_agente4": "No disponible por error de conexión.",
            "justificacion_ajuste": "N/A",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(salida, f, indent=2, ensure_ascii=False)
        print(f"💾 Archivo de error guardado en: {OUTPUT_FILE}")
        sys.exit(1)

    # --- Paso 4: Procesar respuesta ---
    print("🔍 Paso 4: Procesando respuesta de la IA...")
    datos_ia = extraer_json_de_respuesta(respuesta_cruda)
    print(f"   Calificación ajustada extraída: {datos_ia.get('calificacion_ajustada', 'N/A')}")
    print()

    # --- Paso 5: Construir salida final ---
    print("📦 Paso 5: Construyendo JSON de salida final...")
    salida = construir_salida(datos_ia, agente4_data)

    # --- Paso 6: Guardar archivo ---
    print(f"💾 Paso 6: Guardando resultado en '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(salida, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Archivo guardado exitosamente: {os.path.abspath(OUTPUT_FILE)}")
    except IOError as e:
        print(f"   ❌ Error al guardar archivo: {e}")

    # --- Paso 7: Mostrar resultado ---
    imprimir_resultado(salida, respuesta_cruda)

    print(f"✅ Agente 5 completado exitosamente — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()