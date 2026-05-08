#!/usr/bin/env python3

"""Actualiza las series cortas del tipo de cambio referencial del BCB.

El flujo es:
1. Descarga las series publicadas en el repo publico `mauforonda/dolares`.
2. Toma como referencia la fecha mas temprana de una serie de Binance local.
3. Filtra compra y venta del BCB desde esa fecha en adelante.
4. Renombra columnas a `fecha,valor` y guarda dos CSV para dashboard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = RUTA_BASE / "datos"
RUTA_SERIE_BINANCE_REFERENCIA = (
    DIRECTORIO_SALIDA / "binance_buy_precio_profundidad.csv"
)
FUENTES = {
    "compra": "https://raw.githubusercontent.com/mauforonda/dolares/main/buy_oficial.csv",
    "venta": "https://raw.githubusercontent.com/mauforonda/dolares/main/sell_oficial.csv",
}


def descargar_serie(url: str) -> pd.DataFrame:
    """Descarga una serie fuente y la ordena por fecha."""
    tabla = pd.read_csv(url)
    tabla["timestamp"] = pd.to_datetime(tabla["timestamp"])
    return tabla.sort_values("timestamp").reset_index(drop=True)


def obtener_fecha_inicio_binance() -> pd.Timestamp:
    """Lee la fecha mas temprana disponible en una serie local de Binance."""
    tabla = pd.read_csv(RUTA_SERIE_BINANCE_REFERENCIA)
    tabla["fecha"] = pd.to_datetime(tabla["fecha"])
    return tabla["fecha"].min().normalize()


def construir_serie_corta(tabla: pd.DataFrame, fecha_inicio: pd.Timestamp) -> pd.DataFrame:
    """Filtra la serie desde la fecha inicial de Binance y normaliza columnas."""
    serie = tabla.loc[tabla["timestamp"] >= fecha_inicio].copy()
    serie["timestamp"] = serie["timestamp"].dt.strftime("%Y-%m-%d")
    return serie.rename(columns={"timestamp": "fecha", "value": "valor"})[
        ["fecha", "valor"]
    ]


def exportar_serie(tabla: pd.DataFrame, lado: str) -> Path:
    """Guarda la serie corta normalizada para el dashboard."""
    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = DIRECTORIO_SALIDA / f"referencial_bcb_{lado}.csv"
    tabla.to_csv(ruta, float_format="%.3f", index=False)
    return ruta


def main() -> None:
    """Ejecuta la actualizacion completa de compra y venta."""
    fecha_inicio = obtener_fecha_inicio_binance()
    rutas_exportadas: list[Path] = []

    for lado, url in FUENTES.items():
        serie_fuente = descargar_serie(url)
        serie_corta = construir_serie_corta(serie_fuente, fecha_inicio)
        rutas_exportadas.append(exportar_serie(serie_corta, lado))

    for ruta in rutas_exportadas:
        print(ruta)


if __name__ == "__main__":
    main()
