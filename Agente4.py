#!/usr/bin/env python3
"""
Agente4.py
----------
Verificador matemático paso a paso para auditar la salida del Agente 3.

Funcionalidades:
- Verificación doble por paso: algebraica + coherencia secuencial
- Detección y clasificación de errores (aritméticos, algebraicos, incoherencias)
- Generación de corrección esperada cuando se detecta error
- Sistema de calificación de 0 a 10 basado en criterios
- Salida JSON estructurada para consumo de Agente 5

Requisitos: Python 3.10+, sympy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sympy import Eq, Symbol, simplify, solve
from sympy.core.expr import Expr
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)


x = Symbol("x")
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


@dataclass
class EcuacionLineal:
    """Representación parseada de una ecuación lineal simple."""

    original: str
    normalizada: str
    lhs: Expr
    rhs: Expr

    @property
    def eq(self) -> Eq:
        return Eq(self.lhs, self.rhs)

    @property
    def expr_cero(self) -> Expr:
        return simplify(self.lhs - self.rhs)

    def soluciones(self) -> List[Expr]:
        try:
            sols = solve(self.eq, x, dict=False)
            if isinstance(sols, list):
                return sols
            return [sols]
        except Exception:
            return []


class VerificadorPasoAPaso:
    """Lógica principal de validación y scoring del Agente 4."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @staticmethod
    def _normalizar_ecuacion(texto: str) -> str:
        """Limpia y normaliza formatos comunes de ecuaciones OCR."""
        s = (texto or "").strip()
        s = s.replace("−", "-").replace("–", "-").replace("—", "-")
        s = s.replace("×", "*").replace("·", "*")
        s = s.replace(" ", "")

        # Limpieza de signos repetidos comunes por OCR
        s = s.replace("++", "+").replace("+-", "-").replace("-+", "-")

        # Si viene con dos '=' por ruido, conserva el primero como separador principal
        if s.count("=") > 1:
            partes = s.split("=")
            s = f"{partes[0]}={partes[-1]}"

        return s

    def _parsear_ecuacion(self, ecuacion: str) -> Tuple[Optional[EcuacionLineal], Optional[str]]:
        """Parsea ecuación de forma robusta. Espera formato con '='."""
        normalizada = self._normalizar_ecuacion(ecuacion)
        if "=" not in normalizada:
            return None, "Formato inválido: la ecuación no contiene '='"

        lhs_raw, rhs_raw = normalizada.split("=", 1)
        if not lhs_raw or not rhs_raw:
            return None, "Formato inválido: lado izquierdo o derecho vacío"

        try:
            lhs = parse_expr(lhs_raw, transformations=TRANSFORMATIONS)
            rhs = parse_expr(rhs_raw, transformations=TRANSFORMATIONS)
            return EcuacionLineal(ecuacion, normalizada, lhs, rhs), None
        except Exception as exc:
            return None, f"No se pudo interpretar la ecuación: {exc}"

    @staticmethod
    def _solo_numerica(expr: Expr) -> bool:
        return expr.free_symbols == set()

    def _verificacion_algebraica(self, actual: EcuacionLineal) -> Tuple[bool, str]:
        """
        Verifica corrección algebraica interna del paso.
        - Si no hay variable x en ambos lados: evalúa igualdad numérica.
        - Si contiene x: valida que sea una ecuación interpretable y lineal simple.
        """
        lhs, rhs = simplify(actual.lhs), simplify(actual.rhs)

        if self._solo_numerica(lhs) and self._solo_numerica(rhs):
            if simplify(lhs - rhs) == 0:
                return True, "Igualdad numérica correcta"
            return False, f"Error aritmético interno: {lhs} ≠ {rhs}"

        # Para lineales simples, el término en x no debe involucrar potencias/funciones no lineales.
        expr = simplify(actual.lhs - actual.rhs)
        if expr.has(x) and expr.as_poly(x) is None:
            return False, "Ecuación no lineal o fuera del formato soportado"

        return True, "Ecuación algebraicamente válida"

    def _equivalentes(self, prev: EcuacionLineal, curr: EcuacionLineal) -> bool:
        """Determina equivalencia por conjunto solución en ecuaciones lineales."""
        p = simplify(prev.lhs - prev.rhs)
        c = simplify(curr.lhs - curr.rhs)

        p_poly = p.as_poly(x)
        c_poly = c.as_poly(x)

        # Si alguno no es polinomio en x, fallback por comparación de soluciones
        if p_poly is None or c_poly is None:
            return self._soluciones_equivalentes(prev, curr)

        a1 = p_poly.coeff_monomial(x)
        b1 = p_poly.coeff_monomial(1)
        a2 = c_poly.coeff_monomial(x)
        b2 = c_poly.coeff_monomial(1)

        # Ambos constantes
        if a1 == 0 and a2 == 0:
            return simplify(b1) == 0 and simplify(b2) == 0

        # Un constante y otro no constante
        if (a1 == 0) != (a2 == 0):
            return False

        # Ecuaciones lineales equivalentes si son múltiplos escalares
        return simplify(a1 * b2 - a2 * b1) == 0

    def _soluciones_equivalentes(self, prev: EcuacionLineal, curr: EcuacionLineal) -> bool:
        s1 = prev.soluciones()
        s2 = curr.soluciones()
        if len(s1) != len(s2):
            return False
        return all(simplify(a - b) == 0 for a, b in zip(s1, s2))

    def _detectar_error_aritmetico_transicion(
        self,
        prev: EcuacionLineal,
        curr: EcuacionLineal,
    ) -> Optional[str]:
        """Detecta patrón típico de simplificación numérica incorrecta entre pasos."""
        # Caso típico: misma izquierda y derecha pasa de expresión numérica a número errado
        if simplify(prev.lhs - curr.lhs) == 0:
            if curr.rhs.free_symbols == set() and prev.rhs.free_symbols == set():
                esperado = simplify(prev.rhs)
                obtenido = simplify(curr.rhs)
                if simplify(esperado - obtenido) != 0:
                    return (
                        "Error aritmético: simplificación numérica incorrecta "
                        f"(se esperaba {esperado}, se obtuvo {obtenido})"
                    )
        return None

    def _detectar_error_signo(
        self,
        prev: EcuacionLineal,
        curr: EcuacionLineal,
    ) -> Optional[str]:
        """Detecta error de signo en paso tipo ax+b=c -> ax=c-b."""
        prev_expr = simplify(prev.lhs - prev.rhs)
        poly = prev_expr.as_poly(x)
        if poly is None:
            return None

        a = poly.coeff_monomial(x)
        b = poly.coeff_monomial(1)

        # Prev en forma ax + b = 0 equivaldría a ax = -b. No siempre coincide con entrada directa.
        # Detección aproximada: si curr mantiene término ax y altera el término independiente con signo opuesto.
        curr_poly = simplify(curr.lhs - curr.rhs).as_poly(x)
        if curr_poly is None:
            return None

        a2 = curr_poly.coeff_monomial(x)
        b2 = curr_poly.coeff_monomial(1)

        if simplify(a - a2) == 0 and simplify(b + b2) == 0 and simplify(b) != 0:
            return "Posible error algebraico de signo al transponer términos"
        return None

    def _generar_correccion_esperada(self, prev: EcuacionLineal) -> str:
        """
        Genera una corrección esperada aproximada para el siguiente paso,
        priorizando operaciones lineales típicas.
        """
        # Intento 1: simplificar solo lados
        lhs_s = simplify(prev.lhs)
        rhs_s = simplify(prev.rhs)
        if simplify(lhs_s - prev.lhs) != 0 or simplify(rhs_s - prev.rhs) != 0:
            return f"{lhs_s}={rhs_s}"

        expr = simplify(prev.lhs - prev.rhs)
        poly = expr.as_poly(x)
        if poly is None:
            return f"{lhs_s}={rhs_s}"

        a = simplify(poly.coeff_monomial(x))
        b = simplify(poly.coeff_monomial(1))

        # ax + b = 0 -> ax = -b
        if a != 0 and b != 0:
            return f"{a}*x={-b}"

        # ax = c -> x = c/a
        if a != 0 and b == 0:
            return "x=0"

        # Caso degenerado
        return f"{lhs_s}={rhs_s}"

    def _verificacion_coherencia(
        self,
        prev: Optional[EcuacionLineal],
        curr: EcuacionLineal,
    ) -> Tuple[bool, str, Optional[str], str]:
        """
        Verifica coherencia secuencial del paso.
        Retorna: (es_coherente, mensaje, tipo_error, correccion_esperada)
        """
        if prev is None:
            return True, "Primer paso: coherencia base aceptada", None, ""

        if self._equivalentes(prev, curr):
            return True, "Paso derivado lógicamente del anterior", None, ""

        err_arit = self._detectar_error_aritmetico_transicion(prev, curr)
        if err_arit:
            return False, err_arit, "aritmetico", self._generar_correccion_esperada(prev)

        err_signo = self._detectar_error_signo(prev, curr)
        if err_signo:
            return False, err_signo, "algebraico", self._generar_correccion_esperada(prev)

        return (
            False,
            "Incoherencia secuencial: el paso no se deriva del anterior",
            "incoherencia",
            self._generar_correccion_esperada(prev),
        )

    def _calcular_calificacion(
        self,
        pasos_verificados: Sequence[Dict[str, Any]],
    ) -> float:
        """
        Sistema de criterios eficiente (0-10):
        - 6 pts: proporción de pasos correctos
        - 2 pts: coherencia secuencial
        - 2 pts: base por completitud, menos penalización por gravedad
        """
        total = len(pasos_verificados)
        if total == 0:
            return 0.0

        correctos = sum(1 for p in pasos_verificados if p["es_correcto"])
        coherentes = sum(
            1
            for p in pasos_verificados
            if p["verificacion_coherencia"].startswith("✅")
        )

        ratio_correctos = correctos / total
        ratio_coherencia = coherentes / total

        errores_mayores = 0
        errores_menores = 0

        for paso in pasos_verificados:
            msg = (paso.get("error_detectado") or "").lower()
            if not msg:
                continue
            if "aritm" in msg or "algebra" in msg or "formato" in msg:
                errores_mayores += 1
            else:
                errores_menores += 1

        penalizacion = min(2.0, errores_mayores * 0.7 + errores_menores * 0.3)

        score = (6.0 * ratio_correctos) + (2.0 * ratio_coherencia) + 2.0 - penalizacion
        return round(max(0.0, min(10.0, score)), 2)

    def _retroalimentacion(
        self,
        pasos_verificados: Sequence[Dict[str, Any]],
        resumen_errores: Sequence[str],
    ) -> List[str]:
        sugerencias: List[str] = []

        if not resumen_errores:
            return [
                "Excelente procedimiento: los pasos son algebraicamente válidos y coherentes.",
                "Mantén el hábito de justificar brevemente cada transformación.",
            ]

        tiene_arit = any("aritm" in e.lower() for e in resumen_errores)
        tiene_alg = any("algebra" in e.lower() or "signo" in e.lower() for e in resumen_errores)
        tiene_incoh = any("incoher" in e.lower() for e in resumen_errores)

        if tiene_arit:
            sugerencias.append(
                "Revisa operaciones numéricas intermedias antes de continuar al siguiente paso."
            )
        if tiene_alg:
            sugerencias.append(
                "Al transponer términos de un lado al otro, verifica el cambio correcto de signo."
            )
        if tiene_incoh:
            sugerencias.append(
                "Asegura que cada paso sea equivalente al anterior (misma solución para x)."
            )

        incorrectos = [p for p in pasos_verificados if not p["es_correcto"]]
        if len(incorrectos) > max(1, len(pasos_verificados) // 3):
            sugerencias.append(
                "Te conviene escribir pasos más pequeños para facilitar la verificación y evitar saltos lógicos."
            )

        return sugerencias or [
            "Hubo inconsistencias puntuales; repasa la justificación algebraica de cada transformación."
        ]

    @staticmethod
    def _extraer_ecuacion_paso(paso: Dict[str, Any]) -> str:
        """Extrae la ecuación del paso, priorizando campos de Agente 3."""
        for key in ("ecuacion_final", "reconstruido_a1", "ocr_original"):
            valor = paso.get(key)
            if isinstance(valor, str) and valor.strip():
                return valor.strip()
        return ""

    def verificar(self, entrada: Dict[str, Any], fuente_entrada: str) -> Dict[str, Any]:
        inicio = time.perf_counter()

        pasos_entrada = entrada.get("pasos", [])
        if not isinstance(pasos_entrada, list):
            pasos_entrada = []

        total_reportado = entrada.get("total_pasos", len(pasos_entrada))
        self.logger.info("[Agente4] Iniciando verificación de %s pasos", len(pasos_entrada))

        pasos_verificados: List[Dict[str, Any]] = []
        resumen_errores: List[str] = []

        prev_ok: Optional[EcuacionLineal] = None

        for idx, paso in enumerate(pasos_entrada, start=1):
            ecuacion = self._extraer_ecuacion_paso(paso)
            self.logger.info("[Agente4] Paso %s/%s | Ecuación: %s", idx, len(pasos_entrada), ecuacion or "<vacía>")

            eq_actual, err_parse = self._parsear_ecuacion(ecuacion)

            if err_parse or eq_actual is None:
                error = err_parse or "Error desconocido de parseo"
                registro = {
                    "numero": idx,
                    "ecuacion": ecuacion,
                    "es_correcto": False,
                    "error_detectado": error,
                    "correccion_esperada": "Corregir formato a una ecuación válida del tipo ax+b=c",
                    "verificacion_algebraica": f"❌ {error}",
                    "verificacion_coherencia": "❌ No evaluable por error de formato",
                }
                pasos_verificados.append(registro)
                resumen_errores.append(f"Paso {idx}: {error}")
                self.logger.info("[Agente4] ❌ Paso %s inválido: %s", idx, error)
                continue

            ok_alg, msg_alg = self._verificacion_algebraica(eq_actual)
            ok_coh, msg_coh, _tipo_error, correccion = self._verificacion_coherencia(prev_ok, eq_actual)

            es_correcto = ok_alg and ok_coh
            error_detectado = ""
            correccion_esperada = ""

            if not es_correcto:
                if not ok_alg and not ok_coh:
                    error_detectado = f"{msg_alg}; {msg_coh}"
                elif not ok_alg:
                    error_detectado = msg_alg
                else:
                    error_detectado = msg_coh

                correccion_esperada = correccion or self._generar_correccion_esperada(prev_ok or eq_actual)
                resumen_errores.append(f"Paso {idx}: {error_detectado}")
                self.logger.info("[Agente4] ❌ Error en paso %s: %s", idx, error_detectado)
            else:
                self.logger.info("[Agente4] ✅ Paso %s correcto", idx)

            registro = {
                "numero": idx,
                "ecuacion": ecuacion,
                "es_correcto": es_correcto,
                "error_detectado": error_detectado,
                "correccion_esperada": correccion_esperada,
                "verificacion_algebraica": f"{'✅' if ok_alg else '❌'} {msg_alg}",
                "verificacion_coherencia": f"{'✅' if ok_coh else '❌'} {msg_coh}",
            }
            pasos_verificados.append(registro)

            # Mantener último paso parseable para comparar secuencia completa.
            prev_ok = eq_actual

        calificacion = self._calcular_calificacion(pasos_verificados)
        retro = self._retroalimentacion(pasos_verificados, resumen_errores)

        duracion = round(time.perf_counter() - inicio, 4)
        salida = {
            "calificacion": calificacion,
            "pasos_verificados": pasos_verificados,
            "resumen_errores": resumen_errores,
            "retroalimentacion": retro,
            "metadata": {
                "agente": "Agente4",
                "version": "1.0.0",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "fuente_entrada": fuente_entrada,
                "total_pasos_entrada": len(pasos_entrada),
                "total_pasos_reportado_agente3": total_reportado,
                "duracion_segundos": duracion,
            },
        }

        self.logger.info("[Agente4] Verificación finalizada | Calificación: %.2f/10", calificacion)
        return salida


def configurar_logger(nivel: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("Agente4")
    logger.setLevel(getattr(logging, nivel.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def cargar_entrada(path_entrada: Optional[str], logger: logging.Logger) -> Tuple[Dict[str, Any], str]:
    if path_entrada:
        p = Path(path_entrada)
        if not p.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {p}")
        logger.info("[Agente4] Leyendo entrada desde archivo: %s", p)
        contenido = p.read_text(encoding="utf-8")
        return json.loads(contenido), str(p)

    logger.info("[Agente4] Leyendo entrada JSON desde stdin...")
    data = sys.stdin.read().strip()
    if not data:
        raise ValueError("No se recibió contenido JSON por stdin")
    return json.loads(data), "stdin"


def guardar_salida(path_salida: str, salida: Dict[str, Any], logger: logging.Logger) -> None:
    p = Path(path_salida)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[Agente4] JSON guardado en: %s", p)


def construir_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agente4 - Verificador matemático paso a paso (auditor de Agente 3)",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        default=None,
        help="Ruta al JSON de entrada generado por Agente 3. Si se omite, lee desde stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default="agente4_salida.json",
        help="Ruta donde guardar el JSON de salida (default: agente4_salida.json)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nivel de logging",
    )
    return parser


def main() -> int:
    parser = construir_argparser()
    args = parser.parse_args()

    logger = configurar_logger(args.log_level)

    try:
        entrada, fuente = cargar_entrada(args.input_path, logger)
    except Exception as exc:
        logger.error("[Agente4] Error al cargar entrada: %s", exc)
        return 1

    verificador = VerificadorPasoAPaso(logger)

    try:
        resultado = verificador.verificar(entrada, fuente)
        guardar_salida(args.output_path, resultado, logger)

        logger.info("[Agente4] Resultado JSON:\n%s", json.dumps(resultado, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        logger.exception("[Agente4] Error durante verificación: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
