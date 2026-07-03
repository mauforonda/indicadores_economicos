#!/usr/bin/env python3

"""Construye una tabla compacta de tickers para el dashboard ejecutivo."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable


RUTA_BASE = Path(__file__).resolve().parent.parent
RUTA_SALIDA = RUTA_BASE / "tickers" / "tickers.csv"


@dataclass(frozen=True)
class ResultadoTicker:
    ultimo_valor: float
    valor_comparacion: float
    periodos: int


def leer_filas_csv(ruta_relativa: str) -> list[dict[str, str]]:
    """Lee un CSV del repositorio y devuelve sus filas."""

    ruta = RUTA_BASE / ruta_relativa
    with ruta.open(newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def parsear_fecha_iso(valor: str) -> datetime:
    """Parsea fechas ISO simples o con zona horaria."""

    return datetime.fromisoformat(valor)


def parsear_numero(valor: str) -> float:
    """Convierte un texto numerico a float."""

    return float(valor)


def formatear_numero(valor: float) -> str:
    """Escribe enteros sin decimales y flotantes con hasta 3 decimales."""

    if valor.is_integer():
        return str(int(valor))
    return f"{valor:.3f}".rstrip("0").rstrip(".")


def obtener_fila_comparacion_por_dias(
    filas: list[dict[str, str]],
    columna_fecha: str,
    fecha_referencia: datetime,
    dias_atras: int,
) -> tuple[dict[str, str], int]:
    """Busca la fila mas reciente con fecha calendario menor o igual al objetivo."""

    fecha_objetivo = fecha_referencia.date() - timedelta(days=dias_atras)
    filas_ordenadas = sorted(
        filas,
        key=lambda fila: parsear_fecha_iso(fila[columna_fecha]),
    )

    candidata: dict[str, str] | None = None
    for fila in filas_ordenadas:
        fecha = parsear_fecha_iso(fila[columna_fecha])
        if fecha.date() <= fecha_objetivo:
            candidata = fila
        else:
            break

    if candidata is None:
        raise ValueError(
            f"No existe fila con {columna_fecha} <= {fecha_objetivo.isoformat()}."
        )

    fecha_candidata = parsear_fecha_iso(candidata[columna_fecha])
    return candidata, (fecha_referencia.date() - fecha_candidata.date()).days


def obtener_fila_anterior_disponible(
    filas: list[dict[str, str]],
    columna_fecha: str,
) -> tuple[dict[str, str], int]:
    """Devuelve la fila previa a la ultima disponible."""

    filas_ordenadas = sorted(
        filas,
        key=lambda fila: parsear_fecha_iso(fila[columna_fecha]),
    )
    if len(filas_ordenadas) < 2:
        raise ValueError("Se requieren al menos dos filas para comparar.")

    ultima = filas_ordenadas[-1]
    previa = filas_ordenadas[-2]
    fecha_ultima = parsear_fecha_iso(ultima[columna_fecha])
    fecha_previa = parsear_fecha_iso(previa[columna_fecha])
    return previa, (fecha_ultima.date() - fecha_previa.date()).days


def construir_bloqueos() -> ResultadoTicker:
    """Construye el ticker de bloqueos activos."""

    filas = leer_filas_csv("conflictos/datos/bloqueos_diarios.csv")
    ultima = filas[-1]
    fecha_ultima = parsear_fecha_iso(ultima["fecha"])
    comparacion, periodos = obtener_fila_comparacion_por_dias(
        filas=filas,
        columna_fecha="fecha",
        fecha_referencia=fecha_ultima,
        dias_atras=1,
    )
    return ResultadoTicker(
        ultimo_valor=parsear_numero(ultima["bloqueos"]),
        valor_comparacion=parsear_numero(comparacion["bloqueos"]),
        periodos=periodos,
    )


def construir_dolar_usdt_compra() -> ResultadoTicker:
    """Construye el ticker de USDT compra competitivo."""

    filas = leer_filas_csv("tipodecambio/datos/binance_buy_precio_competitivo.csv")
    ultima = filas[-1]
    fecha_ultima = parsear_fecha_iso(ultima["fecha"])
    comparacion, periodos = obtener_fila_comparacion_por_dias(
        filas=filas,
        columna_fecha="fecha",
        fecha_referencia=fecha_ultima,
        dias_atras=1,
    )
    return ResultadoTicker(
        ultimo_valor=parsear_numero(ultima["valor"]),
        valor_comparacion=parsear_numero(comparacion["valor"]),
        periodos=periodos,
    )


def construir_dolar_referencial_compra() -> ResultadoTicker:
    """Construye el ticker de dolar referencial compra."""

    filas = leer_filas_csv("tipodecambio/datos/oficial_bcb.csv")
    ultima = filas[-1]
    comparacion, periodos = obtener_fila_anterior_disponible(
        filas=filas,
        columna_fecha="fecha",
    )
    return ResultadoTicker(
        ultimo_valor=parsear_numero(ultima["valor"]),
        valor_comparacion=parsear_numero(comparacion["valor"]),
        periodos=periodos,
    )


CATALOGO_TICKERS: list[dict[str, str | Callable[[], ResultadoTicker]]] = [
    {
        "nombre": "Bloqueos",
        "funcion": construir_bloqueos,
        "unidad": "bloqueos",
        "periodos_unidad": "días",
    },
    {
        "nombre": "USDT compra",
        "funcion": construir_dolar_usdt_compra,
        "unidad": "Bs/USDT",
        "periodos_unidad": "días",
    },
    {
        "nombre": "Dolar Referencial compra",
        "funcion": construir_dolar_referencial_compra,
        "unidad": "Bs/USD",
        "periodos_unidad": "días",
    },
]


def construir_filas_tickers() -> list[dict[str, str]]:
    """Resuelve el catalogo completo de tickers."""

    filas_salida: list[dict[str, str]] = []
    for ticker in CATALOGO_TICKERS:
        funcion = ticker["funcion"]
        if not callable(funcion):
            raise TypeError(f"funcion invalida para ticker {ticker['nombre']}.")
        resultado = funcion()
        filas_salida.append(
            {
                "nombre": str(ticker["nombre"]),
                "ultimo_valor": formatear_numero(resultado.ultimo_valor),
                "valor_comparacion": formatear_numero(resultado.valor_comparacion),
                "unidad": str(ticker["unidad"]),
                "periodos": str(resultado.periodos),
                "periodos_unidad": str(ticker["periodos_unidad"]),
            }
        )
    return filas_salida


def escribir_csv(filas: list[dict[str, str]]) -> None:
    """Escribe el CSV final de tickers."""

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    columnas = [
        "nombre",
        "ultimo_valor",
        "valor_comparacion",
        "unidad",
        "periodos",
        "periodos_unidad",
    ]
    with RUTA_SALIDA.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def main() -> None:
    """Construye y exporta el CSV de tickers."""

    escribir_csv(construir_filas_tickers())
    print(RUTA_SALIDA)


if __name__ == "__main__":
    main()
