#!/usr/bin/env python3

"""Construye indicadores diarios y geográficos de bloqueos por conflictos."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


URL_FUENTE = (
    "https://raw.githubusercontent.com/mauforonda/transitabilidad-bolivia/"
    "refs/heads/master/data.csv"
)
TZ_BOLIVIA = ZoneInfo("America/La_Paz")
VENTANA_DIAS = 90
RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_DATOS = RUTA_BASE / "datos"
RUTA_BLOQUEOS_DIARIOS = DIRECTORIO_DATOS / "bloqueos_diarios.csv"
RUTA_BLOQUEOS_MENSUALES = DIRECTORIO_DATOS / "bloqueos_mensuales.csv"
RUTA_BLOQUEOS_PUNTOS = DIRECTORIO_DATOS / "bloqueos_puntos.csv"
COLUMNAS_FECHA = ["fecha_consulta", "fecha_reporte", "fecha_fin"]
FECHA_INICIO_MENSUAL = pd.Timestamp("2020-09-01", tz=TZ_BOLIVIA)


def descargar_fuente(url: str) -> pd.DataFrame:
    """Descarga la fuente y normaliza sus columnas básicas."""

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    tabla = pd.read_csv(
        pd.io.common.StringIO(response.text),
        parse_dates=COLUMNAS_FECHA,
    )
    for columna in COLUMNAS_FECHA:
        tabla[columna] = pd.to_datetime(tabla[columna], errors="coerce")
        tabla[columna] = tabla[columna].dt.tz_localize(TZ_BOLIVIA)

    tabla["estado"] = tabla["estado"].astype(str)
    tabla["latitud"] = pd.to_numeric(tabla["latitud"], errors="coerce")
    tabla["longitud"] = pd.to_numeric(tabla["longitud"], errors="coerce")
    return tabla


def filtrar_conflictos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo eventos vinculados a conflictos con fechas válidas."""

    filtrada = tabla.loc[
        tabla["estado"].str.contains("conflictos", case=False, na=False)
        & tabla["fecha_reporte"].notna(),
        ["fecha_reporte", "fecha_fin", "latitud", "longitud"],
    ].copy()
    return filtrada.sort_values(["fecha_reporte", "fecha_fin"]).reset_index(drop=True)


def conflictos_activos_en_instante(tabla: pd.DataFrame, instante: pd.Timestamp) -> pd.DataFrame:
    """Devuelve conflictos activos en un instante dado."""

    mascara = (tabla["fecha_reporte"] <= instante) & (
        tabla["fecha_fin"].isna() | (instante <= tabla["fecha_fin"])
    )
    return tabla.loc[mascara].copy()


def conflictos_activos_en_ventana(
    tabla: pd.DataFrame,
    inicio: pd.Timestamp,
    fin: pd.Timestamp,
) -> pd.DataFrame:
    """Devuelve conflictos activos en algún momento de la ventana."""

    mascara = (tabla["fecha_reporte"] <= fin) & (
        tabla["fecha_fin"].isna() | (tabla["fecha_fin"] >= inicio)
    )
    return tabla.loc[mascara].copy()


def construir_bloqueos_diarios(
    tabla: pd.DataFrame,
    ahora: pd.Timestamp,
) -> pd.DataFrame:
    """Construye el conteo diario al mediodía y agrega el valor actual."""

    dias = pd.date_range(
        end=ahora.normalize(),
        periods=VENTANA_DIAS,
        freq="D",
        tz=TZ_BOLIVIA,
    )
    mediodias = dias + pd.Timedelta(hours=12)
    mediodias = mediodias[mediodias <= ahora]

    filas = []
    for instante in mediodias:
        filas.append(
            {
                "fecha": instante.isoformat(),
                "bloqueos": int(len(conflictos_activos_en_instante(tabla, instante))),
            }
        )

    if not filas or filas[-1]["fecha"] != ahora.isoformat():
        filas.append(
            {
                "fecha": ahora.isoformat(),
                "bloqueos": int(len(conflictos_activos_en_instante(tabla, ahora))),
            }
        )

    return pd.DataFrame(filas)


def construir_bloqueos_mensuales(
    tabla: pd.DataFrame,
    ahora: pd.Timestamp,
) -> pd.DataFrame:
    """Construye el conteo mensual de conflictos activos en algún momento del mes."""

    meses = pd.date_range(
        start=FECHA_INICIO_MENSUAL,
        end=ahora.normalize(),
        freq="MS",
        tz=TZ_BOLIVIA,
    )

    filas = []
    for inicio_mes in meses:
        siguiente_mes = inicio_mes + pd.offsets.MonthBegin(1)
        fin_mes = min(siguiente_mes - pd.Timedelta(seconds=1), ahora)
        activos = conflictos_activos_en_ventana(tabla, inicio_mes, fin_mes)
        filas.append(
            {
                "fecha": inicio_mes.isoformat(),
                "bloqueos": int(len(activos)),
            }
        )

    return pd.DataFrame(filas)


def construir_bloqueos_puntos(
    tabla: pd.DataFrame,
    ahora: pd.Timestamp,
) -> pd.DataFrame:
    """Construye puntos únicos con prioridad por recencia 0 > 3 > 7."""

    ventanas = [
        (
            0,
            conflictos_activos_en_instante(tabla, ahora),
        ),
        (
            3,
            conflictos_activos_en_ventana(tabla, ahora - pd.Timedelta(days=3), ahora),
        ),
        (
            7,
            conflictos_activos_en_ventana(tabla, ahora - pd.Timedelta(days=7), ahora),
        ),
    ]

    frames = []
    for hace_dias, conflictos in ventanas:
        puntos = (
            conflictos.loc[
                conflictos["latitud"].notna() & conflictos["longitud"].notna(),
                ["longitud", "latitud"],
            ]
            .drop_duplicates()
            .rename(columns={"longitud": "x", "latitud": "y"})
        )
        if puntos.empty:
            continue
        puntos["hace_dias"] = hace_dias
        frames.append(puntos)

    if not frames:
        return pd.DataFrame(columns=["x", "y", "hace_dias"])

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["x", "y"], keep="first")
        .sort_values(["hace_dias", "y", "x"])
        .reset_index(drop=True)
    )


def main() -> None:
    """Genera ambos indicadores de bloqueos."""

    ahora = pd.Timestamp.now(tz=TZ_BOLIVIA).floor("s")
    conflictos = filtrar_conflictos(descargar_fuente(URL_FUENTE))
    bloqueos_diarios = construir_bloqueos_diarios(conflictos, ahora)
    bloqueos_mensuales = construir_bloqueos_mensuales(conflictos, ahora)
    bloqueos_puntos = construir_bloqueos_puntos(conflictos, ahora)

    DIRECTORIO_DATOS.mkdir(parents=True, exist_ok=True)
    bloqueos_diarios.to_csv(RUTA_BLOQUEOS_DIARIOS, index=False)
    bloqueos_mensuales.to_csv(RUTA_BLOQUEOS_MENSUALES, index=False)
    bloqueos_puntos.to_csv(RUTA_BLOQUEOS_PUNTOS, index=False, float_format="%.5f")

    print(RUTA_BLOQUEOS_DIARIOS)
    print(RUTA_BLOQUEOS_MENSUALES)
    print(RUTA_BLOQUEOS_PUNTOS)


if __name__ == "__main__":
    main()
