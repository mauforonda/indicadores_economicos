#!/usr/bin/env python3

"""Construye series compactas de precios diarios del INE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


URL = "https://servicioswm.ine.gob.bo/canastita/dashboard/reporte2"
FECHA_INICIO = pd.Timestamp("2025-08-08")
DEPARTAMENTOS = ["La Paz", "Cochabamba", "Santa Cruz"]
UMBRAL_COBERTURA = 0.5
DECIMALES_PRECIO = 3
RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = RUTA_BASE / "datos"
RUTA_SERIE = DIRECTORIO_SALIDA / "ine_diario.csv"
RUTA_DICCIONARIO = DIRECTORIO_SALIDA / "ine_diario_diccionario.csv"


def descargar_fuente(url: str) -> pd.DataFrame:
    """Descarga la respuesta JSON del INE y normaliza columnas básicas."""

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    raw = response.json()
    df = pd.DataFrame(raw)
    df["fecha"] = pd.to_datetime(
        df["dia"].astype(str).str.strip() + " " + df["gestion"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    df["precio"] = pd.to_numeric(df["precio_mercado"], errors="coerce")
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df["unidad"] = df["unidad_madre"].astype(str).str.strip().str.lower()
    df["producto"] = df["producto"].astype(str).str.strip().str.lower()
    df["departamento"] = df["departamento"].astype(str).str.strip()
    return df


def construir_base(tabla: pd.DataFrame) -> pd.DataFrame:
    """Filtra la base diaria y colapsa duplicados por identidad de producto."""

    filtrada = tabla.loc[
        tabla["fecha"].notna()
        & (tabla["fecha"] >= FECHA_INICIO)
        & tabla["departamento"].isin(DEPARTAMENTOS)
        & (tabla["precio"] > 0),
        ["fecha", "departamento", "producto", "unidad", "cantidad", "precio"],
    ].copy()
    filtrada["cantidad"] = filtrada["cantidad"].round(6)

    return (
        filtrada.groupby(
            ["fecha", "departamento", "producto", "unidad", "cantidad"],
            as_index=False,
        )["precio"]
        .mean()
        .sort_values(["producto", "unidad", "cantidad", "departamento", "fecha"])
        .reset_index(drop=True)
    )


def cargar_serie_existente(ruta: Path) -> pd.DataFrame:
    """Carga la serie existente si ya fue generada previamente."""

    if not ruta.exists() or not RUTA_DICCIONARIO.exists():
        return pd.DataFrame(
            columns=["fecha", "departamento", "producto", "unidad", "cantidad", "precio"]
        )

    serie = pd.read_csv(ruta)
    diccionario = pd.read_csv(RUTA_DICCIONARIO)
    serie["fecha"] = pd.to_datetime(serie["fecha"], errors="coerce")
    serie["precio"] = pd.to_numeric(serie["precio"], errors="coerce")
    serie["departamento"] = serie["departamento"].astype(str).str.strip()

    diccionario["producto_id"] = pd.to_numeric(diccionario["producto_id"], errors="coerce")
    diccionario["cantidad"] = pd.to_numeric(diccionario["cantidad"], errors="coerce")
    diccionario["unidad"] = diccionario["unidad"].astype(str).str.strip().str.lower()
    diccionario["producto"] = diccionario["producto"].astype(str).str.strip().str.lower()

    tabla = serie.merge(
        diccionario,
        left_on="id_producto",
        right_on="producto_id",
        how="left",
        validate="many_to_one",
    )
    return tabla[
        ["fecha", "departamento", "producto", "unidad", "cantidad", "precio"]
    ].copy()


def consolidar_base_existente(tabla_existente: pd.DataFrame, tabla_nueva: pd.DataFrame) -> pd.DataFrame:
    """Combina serie existente con descarga nueva, priorizando la observación reciente."""

    consolidada = pd.concat([tabla_existente, tabla_nueva], ignore_index=True)
    consolidada = consolidada.drop_duplicates(
        subset=["fecha", "departamento", "producto", "unidad", "cantidad"],
        keep="last",
    )
    return consolidada.sort_values(
        ["producto", "unidad", "cantidad", "departamento", "fecha"]
    ).reset_index(drop=True)


def filtrar_por_cobertura(tabla: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo productos con presencia en más del 50% de los días."""

    dias_totales = tabla["fecha"].nunique()
    if dias_totales == 0:
        return tabla.iloc[0:0].copy()

    cobertura = (
        tabla.groupby(["producto", "unidad", "cantidad"])["fecha"]
        .nunique()
        .div(dias_totales)
        .reset_index(name="cobertura")
    )
    elegibles = cobertura.loc[cobertura["cobertura"] > UMBRAL_COBERTURA].copy()
    return tabla.merge(
        elegibles[["producto", "unidad", "cantidad"]],
        on=["producto", "unidad", "cantidad"],
        how="inner",
    )


def construir_diccionario(tabla: pd.DataFrame) -> pd.DataFrame:
    """Construye un diccionario determinístico y liviano de productos."""

    diccionario = (
        tabla[["producto", "unidad", "cantidad"]]
        .drop_duplicates()
        .sort_values(["producto", "unidad", "cantidad"])
        .reset_index(drop=True)
    )
    diccionario.insert(0, "producto_id", range(1, len(diccionario) + 1))
    return diccionario


def construir_serie(tabla: pd.DataFrame, diccionario: pd.DataFrame) -> pd.DataFrame:
    """Asigna ids compactos y produce la serie final."""

    salida = tabla.merge(
        diccionario,
        on=["producto", "unidad", "cantidad"],
        how="left",
        validate="many_to_one",
    )
    salida["precio"] = salida["precio"].round(DECIMALES_PRECIO)
    salida = salida.rename(columns={"producto_id": "id_producto"})
    return salida[["fecha", "departamento", "id_producto", "precio"]].sort_values(
        ["fecha", "departamento", "id_producto"]
    ).reset_index(drop=True)


def main() -> None:
    """Genera la serie compacta y el diccionario de productos."""

    base_nueva = construir_base(descargar_fuente(URL))
    base_existente = cargar_serie_existente(RUTA_SERIE)
    base = consolidar_base_existente(base_existente, base_nueva)
    base = filtrar_por_cobertura(base)
    diccionario = construir_diccionario(base)
    serie = construir_serie(base, diccionario)
    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)

    serie.to_csv(RUTA_SERIE, index=False, float_format=f"%.{DECIMALES_PRECIO}f")
    diccionario.to_csv(RUTA_DICCIONARIO, index=False)

    print(RUTA_SERIE)
    print(RUTA_DICCIONARIO)


if __name__ == "__main__":
    main()
