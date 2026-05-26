#!/usr/bin/env python3

"""Construye indicadores diarios y geográficos de bloqueos por conflictos.

Procesamiento de la fuente:
- La fuente es `data.csv` del repositorio `transitabilidad-bolivia`.
- Cada fila representa una observación reportada por el sistema de transitabilidad
  para un punto y un intervalo de vigencia (`fecha_reporte` a `fecha_fin`).
- `fecha_fin` no viene de la fuente oficial original: en el scraper externo se
  completa cuando una observación deja de reaparecer en scrapes posteriores.
- Por eso, varias filas pueden corresponder al mismo bloqueo real si el mismo
  punto vuelve a ser reportado repetidamente a lo largo del tiempo.

Reconstrucción de episodios:
- Este script conserva solo filas cuyo `estado` contiene `conflictos`.
- Luego consolida filas en episodios de bloqueo usando una regla transitiva.
- Dos filas se consideran parte del mismo episodio si:
  1. sus coordenadas están a una distancia geodésica menor o igual a
     `UMBRAL_DISTANCIA_METROS`, y
  2. la brecha entre sus intervalos de vigencia es menor o igual a
     `UMBRAL_BRECHA_TIEMPO`.
- La brecha temporal entre dos filas es cero si sus intervalos se superponen; si
  no se superponen, es el tiempo entre el fin del primer intervalo y el inicio
  del segundo.
- La relación es transitiva: si A se une con B y B con C, las tres filas forman
  un mismo episodio aunque A y C no cumplan directamente el umbral entre sí.
- Las filas sin `fecha_fin` se tratan como bloqueos que siguen activos hasta el
  momento de ejecución del script.

Producción de indicadores:
- `bloqueos_diarios.csv` cuenta episodios activos cada día al mediodía, más un
  punto adicional en el instante actual si aún no coincide con uno de esos cortes.
- `bloqueos_mensuales.csv` cuenta episodios que estuvieron activos en algún
  momento de cada mes; un episodio que cruza de un mes a otro cuenta en ambos.
- `bloqueos_puntos.csv` publica coordenadas únicas de episodios observados en tres
  ventanas de recencia: activos ahora, activos en los últimos 3 días y activos en
  los últimos 7 días, priorizando siempre la ventana más reciente.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

"""

"""


URL_FUENTE = (
    "https://raw.githubusercontent.com/mauforonda/transitabilidad-bolivia/"
    "refs/heads/master/data.csv"
)
TZ_BOLIVIA = ZoneInfo("America/La_Paz")
VENTANA_DIAS = 90
UMBRAL_DISTANCIA_METROS = 500.0
UMBRAL_BRECHA_TIEMPO = pd.Timedelta(hours=24)
# Celda angular conservadora para prefiltrar candidatos cercanos.
TAM_CELDA_GRADOS = UMBRAL_DISTANCIA_METROS / 100_000.0
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


def distancia_haversine_metres(
    latitud_1: float,
    longitud_1: float,
    latitud_2: float,
    longitud_2: float,
) -> float:
    """Calcula distancia geodésica aproximada entre dos coordenadas."""

    radio_tierra = 6_371_000.0
    latitud_1_rad = math.radians(latitud_1)
    latitud_2_rad = math.radians(latitud_2)
    delta_latitud = math.radians(latitud_2 - latitud_1)
    delta_longitud = math.radians(longitud_2 - longitud_1)

    a = (
        math.sin(delta_latitud / 2.0) ** 2
        + math.cos(latitud_1_rad)
        * math.cos(latitud_2_rad)
        * math.sin(delta_longitud / 2.0) ** 2
    )
    return 2.0 * radio_tierra * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def celda_espacial(latitud: float, longitud: float) -> tuple[int, int]:
    """Asigna una coordenada a una celda para prefiltrar vecinos."""

    return (
        math.floor(latitud / TAM_CELDA_GRADOS),
        math.floor(longitud / TAM_CELDA_GRADOS),
    )


def consolidar_conflictos(
    tabla: pd.DataFrame,
    ahora: pd.Timestamp,
) -> pd.DataFrame:
    """Consolida observaciones en episodios usando cercanía espacial y temporal."""

    if tabla.empty:
        return tabla.copy()

    conflictos = tabla.copy()
    conflictos["fecha_fin_efectiva"] = conflictos["fecha_fin"].fillna(ahora)
    conflictos = conflictos.sort_values(
        ["fecha_reporte", "fecha_fin_efectiva", "latitud", "longitud"],
        kind="stable",
    ).reset_index(drop=True)

    total = len(conflictos)
    padres = list(range(total))
    rangos = [0] * total

    def encontrar(indice: int) -> int:
        while padres[indice] != indice:
            padres[indice] = padres[padres[indice]]
            indice = padres[indice]
        return indice

    def unir(indice_a: int, indice_b: int) -> None:
        raiz_a = encontrar(indice_a)
        raiz_b = encontrar(indice_b)
        if raiz_a == raiz_b:
            return
        if rangos[raiz_a] < rangos[raiz_b]:
            padres[raiz_a] = raiz_b
            return
        if rangos[raiz_a] > rangos[raiz_b]:
            padres[raiz_b] = raiz_a
            return
        padres[raiz_b] = raiz_a
        rangos[raiz_a] += 1

    celdas_activas: dict[tuple[int, int], set[int]] = {}
    heap_eventos_activos: list[tuple[pd.Timestamp, int]] = []

    for indice, fila in conflictos.iterrows():
        fecha_reporte = fila["fecha_reporte"]
        latitud = fila["latitud"]
        longitud = fila["longitud"]
        umbral_inicio = fecha_reporte - UMBRAL_BRECHA_TIEMPO

        while heap_eventos_activos and heap_eventos_activos[0][0] < umbral_inicio:
            _, indice_expirado = heapq.heappop(heap_eventos_activos)
            latitud_expirada = conflictos.at[indice_expirado, "latitud"]
            longitud_expirada = conflictos.at[indice_expirado, "longitud"]
            if pd.isna(latitud_expirada) or pd.isna(longitud_expirada):
                continue
            celda_expirada = celda_espacial(latitud_expirada, longitud_expirada)
            activos = celdas_activas.get(celda_expirada)
            if activos is None:
                continue
            activos.discard(indice_expirado)
            if not activos:
                del celdas_activas[celda_expirada]

        if pd.notna(latitud) and pd.notna(longitud):
            celda = celda_espacial(latitud, longitud)
            vecinos: set[int] = set()
            for delta_latitud in (-1, 0, 1):
                for delta_longitud in (-1, 0, 1):
                    vecinos.update(
                        celdas_activas.get(
                            (celda[0] + delta_latitud, celda[1] + delta_longitud),
                            set(),
                        )
                    )

            for indice_vecino in vecinos:
                if (
                    distancia_haversine_metres(
                        latitud,
                        longitud,
                        conflictos.at[indice_vecino, "latitud"],
                        conflictos.at[indice_vecino, "longitud"],
                    )
                    <= UMBRAL_DISTANCIA_METROS
                ):
                    unir(indice, indice_vecino)

            celdas_activas.setdefault(celda, set()).add(indice)

        heapq.heappush(heap_eventos_activos, (fila["fecha_fin_efectiva"], indice))

    conflictos["episodio_id"] = [encontrar(indice) for indice in range(total)]

    episodios = []
    for _, grupo in conflictos.groupby("episodio_id", sort=False):
        grupo_ordenado = grupo.sort_values(
            ["fecha_reporte", "fecha_fin_efectiva"],
            kind="stable",
        )
        representativo = grupo_ordenado.iloc[-1]
        episodio = {
            "fecha_reporte": grupo_ordenado["fecha_reporte"].min(),
            "fecha_fin": (
                pd.NaT
                if grupo_ordenado["fecha_fin"].isna().any()
                else grupo_ordenado["fecha_fin"].max()
            ),
            "latitud": representativo["latitud"],
            "longitud": representativo["longitud"],
        }
        episodios.append(episodio)

    return (
        pd.DataFrame(episodios)
        .sort_values(["fecha_reporte", "fecha_fin", "latitud", "longitud"], kind="stable")
        .reset_index(drop=True)
    )


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
    """Genera indicadores consolidados de bloqueos."""

    ahora = pd.Timestamp.now(tz=TZ_BOLIVIA).floor("s")
    conflictos = filtrar_conflictos(descargar_fuente(URL_FUENTE))
    episodios = consolidar_conflictos(conflictos, ahora)
    bloqueos_diarios = construir_bloqueos_diarios(episodios, ahora)
    bloqueos_mensuales = construir_bloqueos_mensuales(episodios, ahora)
    bloqueos_puntos = construir_bloqueos_puntos(episodios, ahora)

    DIRECTORIO_DATOS.mkdir(parents=True, exist_ok=True)
    bloqueos_diarios.to_csv(RUTA_BLOQUEOS_DIARIOS, index=False)
    bloqueos_mensuales.to_csv(RUTA_BLOQUEOS_MENSUALES, index=False)
    bloqueos_puntos.to_csv(RUTA_BLOQUEOS_PUNTOS, index=False, float_format="%.5f")

    print(RUTA_BLOQUEOS_DIARIOS)
    print(RUTA_BLOQUEOS_MENSUALES)
    print(RUTA_BLOQUEOS_PUNTOS)


if __name__ == "__main__":
    main()
