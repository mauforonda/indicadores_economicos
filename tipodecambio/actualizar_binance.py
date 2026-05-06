#!/usr/bin/env python3

"""Actualiza indicadores compactos de tipo de cambio desde Binance/Kaggle.

El flujo es:
1. Descarga la ultima version de ``advice.parquet`` desde Kaggle.
2. Filtra USDT/BOB y corrige la inversion BUY/SELL del dataset original.
3. Guarda tablas base en parquet para inspeccion local.
4. Exporta seis CSV compactos ``fecha,valor`` para dashboard.
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd


DATASET = "andreschirinos/p2p-bob-exchange"
ARCHIVO_FUENTE = "advice.parquet"
PROPORCION_TRAMO_COMPETITIVO = 0.10
DIAS_SPARKLINE = 7
RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = RUTA_BASE / "datos"
COLUMNAS = [
    "timestamp",
    "asset",
    "tradetype",
    "fiatunit",
    "price",
    "tradablequantity",
    "advertiser_userno",
    "minsingletransamount",
    "maxsingletransamount",
]
MAPEO_LADOS = {
    # En Kaggle/Binance, BUY y SELL estan definidos desde la perspectiva
    # del anunciante. Para nuestros indicadores, invertimos ese mapeo.
    "buy": "SELL",
    "sell": "BUY",
}
METRICAS = [
    "precio_competitivo",
    "precio_profundidad",
    "precio_transado_estimado",
]


def descargar_parquet_fuente() -> Path:
    """Descarga el parquet fuente y devuelve su ruta local en cache."""
    ruta = kagglehub.dataset_download(DATASET, path=ARCHIVO_FUENTE)
    return Path(ruta)


def cargar_tabla_lado(ruta_parquet: Path, lado_salida: str) -> pd.DataFrame:
    """Carga un lado del mercado con solo las columnas necesarias."""
    tradetype_kaggle = MAPEO_LADOS[lado_salida]
    tabla = pd.read_parquet(
        ruta_parquet,
        columns=COLUMNAS,
        filters=[
            ("asset", "==", "USDT"),
            ("tradetype", "==", tradetype_kaggle),
        ],
    )
    tabla["timestamp"] = pd.to_datetime(tabla["timestamp"])
    return tabla.sort_values("timestamp").reset_index(drop=True)


def guardar_tablas_base(tablas: dict[str, pd.DataFrame]) -> None:
    """Guarda los cortes base buy/sell para auditoria y reproceso local."""
    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)
    tablas["buy"].to_parquet(DIRECTORIO_SALIDA / "buy.parquet", index=False)
    tablas["sell"].to_parquet(DIRECTORIO_SALIDA / "sell.parquet", index=False)


def calcular_vwap(tabla: pd.DataFrame) -> float | None:
    """Calcula el precio promedio ponderado por monto ofertado."""
    pesos = tabla["tradablequantity"]
    total = pesos.sum()
    if total == 0:
        return None
    return float((tabla["price"] * pesos).sum() / total)


def obtener_tramo_competitivo(tabla: pd.DataFrame, lado: str) -> pd.DataFrame:
    """Selecciona el tramo competitivo del libro.

    Para ``buy`` usamos el 10% inferior de precios.
    Para ``sell`` usamos el 10% superior.
    """
    cuantila = tabla["price"].quantile(
        PROPORCION_TRAMO_COMPETITIVO
        if lado == "buy"
        else 1 - PROPORCION_TRAMO_COMPETITIVO
    )
    if lado == "buy":
        return tabla[tabla["price"] <= cuantila]
    return tabla[tabla["price"] >= cuantila]


def calcular_vwap_transado(tabla: pd.DataFrame) -> float | None:
    """Estima precio transado a partir de caidas en montos entre snapshots."""
    montos = tabla.pivot_table(
        index="timestamp",
        columns="advertiser_userno",
        values="tradablequantity",
        aggfunc="last",
    ).sort_index()
    precios = tabla.pivot_table(
        index="timestamp",
        columns="advertiser_userno",
        values="price",
        aggfunc="last",
    ).sort_index()
    pesos = (-montos.diff()).clip(lower=0)
    total = pesos.sum().sum()
    if total == 0:
        return None
    return float((pesos * precios).sum().sum() / total)


def recortar_a_ventana_reciente(tabla: pd.DataFrame) -> pd.DataFrame:
    """Reduce el calculo a los ultimos siete dias con datos."""
    dias = (
        tabla["timestamp"]
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tail(DIAS_SPARKLINE)
    )
    return tabla[tabla["timestamp"].dt.normalize().isin(dias)].copy()


def construir_metricas_diarias(tabla: pd.DataFrame, lado: str) -> pd.DataFrame:
    """Construye las tres metricas diarias para el sparkline.

    Los dias cerrados usan agregacion diaria completa.
    El ultimo punto usa el timestamp mas reciente del dia en curso.
    """
    tabla = recortar_a_ventana_reciente(tabla)
    dia_actual = tabla["timestamp"].max().date()
    filas = []

    for dia, grupo in tabla.groupby(tabla["timestamp"].dt.date):
        tramo_competitivo = obtener_tramo_competitivo(grupo, lado)
        filas.append(
            {
                "fecha": pd.Timestamp(dia),
                "precio_competitivo": calcular_vwap(tramo_competitivo),
                "precio_profundidad": calcular_vwap(grupo),
                "precio_transado_estimado": calcular_vwap_transado(grupo),
                "es_dia_actual": dia == dia_actual,
                "timestamp_reciente": grupo["timestamp"].max(),
            }
        )

    resultado = pd.DataFrame(filas)
    dias_cerrados = resultado.loc[~resultado["es_dia_actual"]].tail(
        DIAS_SPARKLINE - 1
    )
    dia_en_curso = resultado.loc[resultado["es_dia_actual"]].copy()
    if not dia_en_curso.empty:
        dia_en_curso.loc[:, "fecha"] = dia_en_curso["timestamp_reciente"]
    serie_compacta = pd.concat([dias_cerrados, dia_en_curso], ignore_index=True)
    return serie_compacta[["fecha", *METRICAS]]


def exportar_metrica(tabla: pd.DataFrame, lado: str, metrica: str) -> Path:
    """Exporta una metrica a CSV con formato estricto ``fecha,valor``."""
    salida = tabla[["fecha", metrica]].rename(columns={metrica: "valor"}).copy()
    salida["fecha"] = pd.to_datetime(salida["fecha"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    ruta = DIRECTORIO_SALIDA / f"binance_{lado}_{metrica}.csv"
    salida.to_csv(ruta, index=False)
    return ruta


def main() -> None:
    """Ejecuta la actualizacion completa de tablas base y CSV compactos."""
    ruta_parquet = descargar_parquet_fuente()
    tablas = {
        lado: cargar_tabla_lado(ruta_parquet, lado) for lado in ("buy", "sell")
    }
    guardar_tablas_base(tablas)

    rutas_exportadas: list[Path] = []
    for lado, tabla in tablas.items():
        serie_compacta = construir_metricas_diarias(tabla, lado)
        for metrica in METRICAS:
            rutas_exportadas.append(exportar_metrica(serie_compacta, lado, metrica))

    for ruta in rutas_exportadas:
        print(ruta)


if __name__ == "__main__":
    main()
