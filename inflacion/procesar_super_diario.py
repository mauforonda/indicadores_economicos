#!/usr/bin/env python3

"""Procesa la serie diaria de inflación de supermercado a un CSV compacto.

Entradas:
- `inflacion/datos/super_diario.csv` con columnas
  `departamento,fecha,componente,inflacion_28d`.

Salidas:
- `inflacion/datos/super_diario_compacto.csv` con columnas
  `departamento,fecha,componente,valor`.
- `inflacion/datos/super_diario_compacto_diccionario.csv` con el mapeo de ids.

Compresión aplicada:
- `departamento` y `componente` se reemplazan por ids enteros secuenciales.
- `fecha` se codifica como número entero de días desde `2025-11-08`.
  Por ejemplo, `0` corresponde a `2025-11-08`, `1` a `2025-11-09`, etc.

Para interpretar los valores codificados de `departamento` y `componente`,
consultar `super_diario_compacto_diccionario.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FECHA_INICIO = pd.Timestamp("2025-11-08")
DECIMALES = 3
RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_DATOS = RUTA_BASE / "datos"
RUTA_FUENTE = DIRECTORIO_DATOS / "super_diario.csv"
RUTA_SALIDA = DIRECTORIO_DATOS / "super_diario_compacto.csv"
RUTA_DICCIONARIO = DIRECTORIO_DATOS / "super_diario_compacto_diccionario.csv"
MAPEO_DEPARTAMENTOS = {
    "la_paz": "La Paz",
    "cochabamba": "Cochabamba",
    "santa_cruz": "Santa Cruz",
}


def cargar_fuente(ruta: Path) -> pd.DataFrame:
    """Carga la serie fuente y normaliza tipos básicos."""

    tabla = pd.read_csv(ruta)
    tabla["fecha"] = pd.to_datetime(tabla["fecha"], errors="coerce")
    tabla["inflacion_28d"] = pd.to_numeric(tabla["inflacion_28d"], errors="coerce")
    tabla["departamento"] = tabla["departamento"].astype(str).str.strip()
    tabla["componente"] = tabla["componente"].astype(str).str.strip()
    return tabla


def construir_serie_compacta(tabla: pd.DataFrame) -> pd.DataFrame:
    """Filtra y normaliza la serie base para construir ids compactos."""

    salida = tabla.loc[
        tabla["fecha"].notna()
        & (tabla["fecha"] >= FECHA_INICIO)
        & tabla["departamento"].isin(MAPEO_DEPARTAMENTOS),
        ["departamento", "fecha", "componente", "inflacion_28d"],
    ].copy()
    salida["departamento"] = salida["departamento"].map(MAPEO_DEPARTAMENTOS)
    return salida.rename(columns={"inflacion_28d": "valor"}).reset_index(drop=True)


def filtrar_componentes_validos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo componentes presentes en los 3 departamentos y no nulos en valor."""

    cobertura = (
        tabla.groupby("componente")
        .agg(
            departamentos=("departamento", "nunique"),
            suma_abs=("valor", lambda serie: serie.abs().sum()),
        )
        .reset_index()
    )
    validos = cobertura.loc[
        (cobertura["departamentos"] == len(MAPEO_DEPARTAMENTOS))
        & (cobertura["suma_abs"] > 0),
        "componente",
    ]
    return tabla.loc[tabla["componente"].isin(validos)].copy()


def construir_diccionarios(
    tabla: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye ids determinísticos para departamentos y componentes."""

    departamentos = pd.DataFrame(
        {
            "id": range(1, len(MAPEO_DEPARTAMENTOS) + 1),
            "valor": list(MAPEO_DEPARTAMENTOS.values()),
        }
    )
    componentes_unicos = sorted(tabla["componente"].dropna().unique().tolist())
    if "Compuesto" in componentes_unicos:
        componentes_unicos = ["Compuesto"] + [
            valor for valor in componentes_unicos if valor != "Compuesto"
        ]
    componentes = pd.DataFrame(
        {
            "id": range(1, len(componentes_unicos) + 1),
            "valor": componentes_unicos,
        }
    )

    diccionario = pd.concat(
        [
            departamentos.assign(tipo="departamento"),
            componentes.assign(tipo="componente"),
        ],
        ignore_index=True,
    )[["tipo", "id", "valor"]]

    return departamentos, componentes, diccionario


def asignar_ids(
    tabla: pd.DataFrame,
    departamentos: pd.DataFrame,
    componentes: pd.DataFrame,
) -> pd.DataFrame:
    """Reemplaza nombres por ids enteros secuenciales y formatea la salida."""

    con_departamentos = tabla.merge(
        departamentos.rename(
            columns={"id": "departamento_id", "valor": "departamento_nombre"}
        ),
        left_on="departamento",
        right_on="departamento_nombre",
        how="left",
        validate="many_to_one",
    ).drop(columns="departamento_nombre")
    salida = con_departamentos.merge(
        componentes.rename(
            columns={"id": "componente_id", "valor": "componente_nombre"}
        ),
        left_on="componente",
        right_on="componente_nombre",
        how="left",
        validate="many_to_one",
    ).drop(columns="componente_nombre")
    salida["valor"] = salida["valor"].round(DECIMALES)
    salida["fecha"] = (
        salida["fecha"].dt.normalize().sub(FECHA_INICIO).dt.days.astype(int)
    )
    salida = salida[
        ["departamento_id", "fecha", "componente_id", "valor"]
    ].rename(
        columns={
            "departamento_id": "departamento",
            "componente_id": "componente",
        }
    )
    return salida.sort_values(
        ["departamento", "fecha", "componente"]
    ).reset_index(drop=True)


def main() -> None:
    """Genera el CSV compacto desde la serie diaria de supermercado."""

    base = construir_serie_compacta(cargar_fuente(RUTA_FUENTE))
    base = filtrar_componentes_validos(base)
    departamentos, componentes, diccionario = construir_diccionarios(base)
    serie = asignar_ids(base, departamentos, componentes)
    DIRECTORIO_DATOS.mkdir(parents=True, exist_ok=True)
    serie.to_csv(RUTA_SALIDA, index=False, float_format=f"%.{DECIMALES}f")
    diccionario.to_csv(RUTA_DICCIONARIO, index=False)
    print(RUTA_SALIDA)
    print(RUTA_DICCIONARIO)


if __name__ == "__main__":
    main()
