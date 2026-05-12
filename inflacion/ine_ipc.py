#!/usr/bin/env python3

"""Construye series compactas del IPC mensual del INE.

Salidas:
- `ine_ipc_general.csv`: fecha,mensual,12_m
- `ine_ipc_divisiones.csv`: fecha,division,mensual,12_m
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


FECHA_INICIO = pd.Timestamp("2022-01-01")
DECIMALES = 3
RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = RUTA_BASE / "datos"
RUTA_SALIDA_GENERAL = DIRECTORIO_SALIDA / "ine_ipc_general.csv"
RUTA_SALIDA_DIVISIONES = DIRECTORIO_SALIDA / "ine_ipc_divisiones.csv"
URL_SERIE_HISTORICA = "https://www.ine.gob.bo/index.php/serie-historica-empalmada/"
LINK_GENERAL = "Índice General, Variación Mensual, Acumulada y a 12 Meses"
LINK_DIVISIONES = "Índice General, Variación Mensual, Acumulada y a 12 Meses por División"


def descargar_excel(url: str, link_texto: str, ruta_salida: Path) -> None:
    """Descarga un Excel oficial a una ruta temporal."""

    respuesta = requests.get(url, timeout=60)
    respuesta.raise_for_status()
    html = BeautifulSoup(respuesta.text, "html.parser")
    href = [a for a in html.select("#main a") if a.get_text(strip=True) == link_texto][0][
        "href"
    ]
    contenido = requests.get(href, timeout=120)
    contenido.raise_for_status()
    ruta_salida.write_bytes(contenido.content)


def extraer_indice(xl: pd.ExcelFile, sheet_query: str, nombre: str) -> pd.Series:
    """Extrae una serie nacional mensual desde un sheet del Excel."""

    meses = {
        "Enero": 1,
        "Febrero": 2,
        "Marzo": 3,
        "Abril": 4,
        "Mayo": 5,
        "Junio": 6,
        "Julio": 7,
        "Agosto": 8,
        "Septiembre": 9,
        "Octubre": 10,
        "Noviembre": 11,
        "Diciembre": 12,
    }
    sheet = [s for s in xl.sheet_names if sheet_query in s.lower()][0]
    df = pd.read_excel(xl, sheet, skiprows=4)
    df = df[df.MES.isin(meses.keys())]
    df = df.set_index("MES").stack().reset_index()
    df.columns = ["mes", "año", nombre]
    df["fecha"] = pd.to_datetime(
        df.apply(lambda fila: dt.date(int(fila["año"]), meses[fila["mes"]], 1), axis=1)
    )
    df = df[["fecha", nombre]]
    df[nombre] = pd.to_numeric(df[nombre], errors="coerce")
    df = df.dropna(subset=[nombre]).copy()
    return df.set_index("fecha")[nombre].sort_index()


def extraer_nacional_division(
    xl: pd.ExcelFile,
    sheet_query: str,
    nombre: str,
) -> pd.DataFrame:
    """Extrae la serie mensual nacional por división."""

    def drop_footer_rows(raw: pd.DataFrame, min_non_nan: int = 2) -> pd.DataFrame:
        non_nan = raw.notna().sum(axis=1)
        skip = 0
        for valor in reversed(non_nan.tolist()):
            if valor <= min_non_nan:
                skip += 1
            else:
                break
        return raw.iloc[:-skip] if skip > 0 else raw

    meses = [
        "ENERO",
        "FEBRERO",
        "MARZO",
        "ABRIL",
        "MAYO",
        "JUNIO",
        "JULIO",
        "AGOSTO",
        "SEPTIEMBRE",
        "OCTUBRE",
        "NOVIEMBRE",
        "DICIEMBRE",
    ]
    meses_map = {mes: i + 1 for i, mes in enumerate(meses)}
    sheet = [s for s in xl.sheet_names if sheet_query in s.lower()][0]
    raw = pd.read_excel(xl, sheet, skiprows=4, header=None)
    raw = drop_footer_rows(raw, min_non_nan=2)

    with pd.option_context("future.no_silent_downcasting", True):
        years = raw.iloc[0, 2:].ffill().infer_objects(copy=False)
    months = raw.iloc[1, 2:]
    data = raw.iloc[3:].copy()
    data.columns = ["categoria_codigo", "categoria"] + list(range(2, raw.shape[1]))

    date_cols = pd.MultiIndex.from_arrays([years, months], names=["year", "month"])
    table = data.iloc[:, 2:]
    table.columns = date_cols
    table.index = pd.MultiIndex.from_frame(data.iloc[:, :2])

    vertical = (
        table.stack([0, 1], future_stack=True)
        .reset_index(name=nombre)
        .dropna(subset=["year", "month", nombre])
    )
    vertical["year"] = vertical["year"].astype(int)
    vertical["month"] = vertical["month"].astype(str).str.strip().map(meses_map)
    vertical.insert(
        0,
        "fecha",
        vertical[["year", "month"]].apply(
            lambda fila: f"{fila['year']}-{fila['month']}-1",
            axis=1,
        ),
    )
    vertical["fecha"] = pd.to_datetime(vertical["fecha"])
    vertical = vertical[["fecha", "categoria_codigo", "categoria", nombre]]
    vertical = vertical[vertical["categoria_codigo"] != 0].copy()
    vertical[nombre] = vertical[nombre].astype(float)
    return vertical.set_index(["fecha", "categoria_codigo", "categoria"])


def cargar_indice_nacional() -> pd.DataFrame:
    """Descarga y estructura el IPC nacional mensual."""

    ruta_temporal = RUTA_BASE / "indice.xlsx"
    descargar_excel(URL_SERIE_HISTORICA, LINK_GENERAL, ruta_temporal)
    try:
        xl = pd.ExcelFile(ruta_temporal)
        df = pd.concat(
            [
                extraer_indice(xl, sheet, nombre)
                for sheet, nombre in zip(
                    ["ndice mensual", "var mensual", "var acumulada", "12 meses"],
                    [
                        "indice_mensual",
                        "variacion_mensual",
                        "variacion_acumulada",
                        "variacion_12_meses",
                    ],
                )
            ],
            axis=1,
        ).reset_index()
    finally:
        ruta_temporal.unlink(missing_ok=True)

    return df.loc[df["fecha"] >= FECHA_INICIO].copy()


def cargar_indice_divisiones() -> pd.DataFrame:
    """Descarga y estructura el IPC nacional por división."""

    ruta_temporal = RUTA_BASE / "indice_divisiones.xlsx"
    descargar_excel(URL_SERIE_HISTORICA, LINK_DIVISIONES, ruta_temporal)
    try:
        xl = pd.ExcelFile(ruta_temporal)
        df = pd.concat(
            [
                extraer_nacional_division(xl, sheet, nombre)
                for sheet, nombre in zip(
                    ["ndice", "var mensual", "var acumulada", "12 meses"],
                    [
                        "indice_mensual",
                        "variacion_mensual",
                        "variacion_acumulada",
                        "variacion_12_meses",
                    ],
                )
            ],
            axis=1,
        ).reset_index()
    finally:
        ruta_temporal.unlink(missing_ok=True)

    return df.loc[df["fecha"] >= FECHA_INICIO].copy()


def construir_serie_general(tabla: pd.DataFrame) -> pd.DataFrame:
    """Reduce la tabla nacional a las columnas compactas requeridas."""

    salida = tabla[["fecha", "variacion_mensual", "variacion_12_meses"]].rename(
        columns={
            "variacion_mensual": "mensual",
            "variacion_12_meses": "12_m",
        }
    )
    salida = salida.dropna(subset=["mensual", "12_m"], how="all").copy()
    salida[["mensual", "12_m"]] = salida[["mensual", "12_m"]].round(DECIMALES)
    return salida.sort_values("fecha").reset_index(drop=True)


def construir_serie_divisiones(tabla: pd.DataFrame) -> pd.DataFrame:
    """Reduce la tabla por divisiones a la forma compacta requerida."""

    salida = tabla[["fecha", "categoria", "variacion_mensual", "variacion_12_meses"]].rename(
        columns={
            "categoria": "division",
            "variacion_mensual": "mensual",
            "variacion_12_meses": "12_m",
        }
    )
    salida = salida.dropna(subset=["mensual", "12_m"], how="all").copy()
    salida[["mensual", "12_m"]] = salida[["mensual", "12_m"]].round(DECIMALES)
    return salida.sort_values(["fecha", "division"]).reset_index(drop=True)


def main() -> None:
    """Genera ambos CSV compactos del IPC mensual."""

    general = construir_serie_general(cargar_indice_nacional())
    divisiones = construir_serie_divisiones(cargar_indice_divisiones())
    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)

    general.to_csv(RUTA_SALIDA_GENERAL, index=False, float_format=f"%.{DECIMALES}f")
    divisiones.to_csv(
        RUTA_SALIDA_DIVISIONES,
        index=False,
        float_format=f"%.{DECIMALES}f",
    )

    print(RUTA_SALIDA_GENERAL)
    print(RUTA_SALIDA_DIVISIONES)


if __name__ == "__main__":
    main()
