#!/usr/bin/env python3

"""Actualiza la serie desagregada del tipo de cambio referencial compra por banco.

El flujo es:
1. Descarga precios y montos publicados en el repo publico `mauforonda/dolares`.
2. Toma como referencia la fecha mas temprana de una serie local de Binance.
3. Desagrega ambas fuentes por banco, excluyendo la columna agregada `value`.
4. Une precio y monto por fecha y banco, normaliza nombres legibles y exporta
   `fecha,banco,monto,valor`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = RUTA_BASE / "datos"
RUTA_SERIE_BINANCE_REFERENCIA = (
    DIRECTORIO_SALIDA / "binance_buy_precio_profundidad.csv"
)
RUTA_SALIDA = DIRECTORIO_SALIDA / "referencial_bcb_compra_desagregado.csv"
FUENTE_VALOR = (
    "https://raw.githubusercontent.com/mauforonda/dolares/refs/heads/main/"
    "buy_oficial_completo.csv"
)
FUENTE_MONTO = (
    "https://raw.githubusercontent.com/mauforonda/dolares/refs/heads/main/"
    "buy_oficial_monto.csv"
)
MAPEO_BANCOS = {
    "banco_bisa": "Banco BISA",
    "banco_de_credito": "Banco de Crédito",
    "banco_de_la_nacion_argentina": "Banco de la Nación Argentina",
    "banco_economico": "Banco Económico",
    "banco_fie": "Banco FIE",
    "banco_fortaleza": "Banco Fortaleza",
    "banco_ganadero": "Banco Ganadero",
    "banco_mercantil_santa_cruz": "Banco Mercantil Santa Cruz",
    "banco_nacional_de_bolivia": "Banco Nacional de Bolivia",
    "banco_prodem": "Banco Prodem",
    "banco_pyme_de_la_comunidad": "Banco PYME de la Comunidad",
    "banco_pyme_ecofuturo": "Banco PYME Ecofuturo",
    "banco_solidario": "Banco Solidario",
    "banco_union": "Banco Unión",
}


def descargar_serie(url: str) -> pd.DataFrame:
    """Descarga una serie fuente y la ordena por fecha."""
    tabla = pd.read_csv(url)
    tabla["timestamp"] = pd.to_datetime(tabla["timestamp"])
    return tabla.sort_values("timestamp").reset_index(drop=True)


def obtener_fecha_inicio_binance() -> pd.Timestamp:
    """Lee la fecha más temprana disponible en una serie local de Binance."""
    tabla = pd.read_csv(RUTA_SERIE_BINANCE_REFERENCIA)
    tabla["fecha"] = pd.to_datetime(tabla["fecha"])
    return tabla["fecha"].dt.tz_localize(None).min().normalize()


def obtener_columnas_banco(tabla: pd.DataFrame) -> list[str]:
    """Devuelve las columnas bancarias, excluyendo timestamp y agregado value."""
    return [columna for columna in tabla.columns if columna not in {"timestamp", "value"}]


def desagregar(tabla: pd.DataFrame, nombre_valor: str) -> pd.DataFrame:
    """Transforma la tabla ancha en una serie larga por banco."""
    columnas_banco = obtener_columnas_banco(tabla)
    salida = tabla.melt(
        id_vars="timestamp",
        value_vars=columnas_banco,
        var_name="banco",
        value_name=nombre_valor,
    )
    salida = salida.dropna(subset=[nombre_valor]).copy()
    salida["banco"] = salida["banco"].map(MAPEO_BANCOS)
    return salida.sort_values(["timestamp", "banco"]).reset_index(drop=True)


def validar_fuentes(
    tabla_valor: pd.DataFrame,
    tabla_monto: pd.DataFrame,
) -> None:
    """Verifica que ambas fuentes compartan exactamente los mismos bancos."""
    bancos_valor = set(obtener_columnas_banco(tabla_valor))
    bancos_monto = set(obtener_columnas_banco(tabla_monto))
    if bancos_valor != bancos_monto:
        solo_valor = sorted(bancos_valor - bancos_monto)
        solo_monto = sorted(bancos_monto - bancos_valor)
        raise SystemExit(
            "Las fuentes de valor y monto no comparten el mismo conjunto de bancos. "
            f"Solo en valor: {solo_valor}. Solo en monto: {solo_monto}."
        )
    faltantes = sorted(banco for banco in bancos_valor if banco not in MAPEO_BANCOS)
    if faltantes:
        raise SystemExit(f"Falta mapear nombres legibles para: {faltantes}")


def construir_serie_desagregada(
    tabla_valor: pd.DataFrame,
    tabla_monto: pd.DataFrame,
    fecha_inicio: pd.Timestamp,
) -> pd.DataFrame:
    """Filtra por cobertura local y une valor con monto por fecha y banco."""
    valor = desagregar(tabla_valor, "valor")
    monto = desagregar(tabla_monto, "monto")

    serie = valor.merge(
        monto,
        on=["timestamp", "banco"],
        how="inner",
        validate="one_to_one",
    )
    serie = serie.loc[serie["timestamp"].dt.normalize() >= fecha_inicio].copy()
    serie["fecha"] = serie["timestamp"].dt.strftime("%Y-%m-%d")
    salida = serie[["fecha", "banco", "monto", "valor"]].sort_values(
        ["fecha", "banco"]
    )
    salida["monto"] = salida["monto"].astype(int)
    return salida.reset_index(drop=True)


def main() -> None:
    """Ejecuta la actualización completa de la serie desagregada de compra."""
    fecha_inicio = obtener_fecha_inicio_binance()
    tabla_valor = descargar_serie(FUENTE_VALOR)
    tabla_monto = descargar_serie(FUENTE_MONTO)
    validar_fuentes(tabla_valor, tabla_monto)

    serie = construir_serie_desagregada(tabla_valor, tabla_monto, fecha_inicio)
    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)
    serie.to_csv(RUTA_SALIDA, float_format="%.2f", index=False)
    print(RUTA_SALIDA)


if __name__ == "__main__":
    main()
