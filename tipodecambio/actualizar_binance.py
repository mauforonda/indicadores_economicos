#!/usr/bin/env python3

"""Actualiza indicadores compactos de tipo de cambio desde Binance/Kaggle.

El flujo es:
1. Descarga la ultima version de ``advice.parquet`` desde Kaggle.
2. Filtra USDT/BOB y corrige la inversion BUY/SELL del dataset original.
3. Guarda tablas base en parquet para inspeccion local.
4. Agrega a resolucion diaria usando hora Bolivia.
5. Exporta series de precios y estructura para dashboard.
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd


DATASET = "andreschirinos/p2p-bob-exchange"
ARCHIVO_FUENTE = "advice.parquet"
PROPORCION_TRAMO_COMPETITIVO = 0.10
DIAS_SPARKLINE = 90
ZONA_HORARIA_BOLIVIA = "America/La_Paz"
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
METRICAS_PRECIO = [
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


def agregar_fecha_bolivia(tabla: pd.DataFrame) -> pd.DataFrame:
    """Anota el timestamp en hora Bolivia y la fecha local correspondiente."""
    tabla = tabla.copy()
    tabla["timestamp_bolivia"] = (
        tabla["timestamp"].dt.tz_localize("UTC").dt.tz_convert(ZONA_HORARIA_BOLIVIA)
    )
    tabla["fecha_bolivia"] = tabla["timestamp_bolivia"].dt.normalize().dt.tz_localize(
        None
    )
    return tabla


def recortar_a_ventana_reciente(tabla: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    """Reduce la tabla a los ultimos 90 dias con datos."""
    dias = (
        tabla[columna_fecha]
        .drop_duplicates()
        .sort_values()
        .tail(DIAS_SPARKLINE)
    )
    return tabla[tabla[columna_fecha].isin(dias)].copy()


def calcular_vwap(tabla: pd.DataFrame, columna_monto: str) -> float | None:
    """Calcula el precio promedio ponderado por monto."""
    pesos = tabla[columna_monto]
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


def construir_eventos_transados(tabla: pd.DataFrame) -> pd.DataFrame:
    """Construye eventos de monto probablemente tranzado por snapshot."""
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
    eventos = (
        pesos.stack()
        .rename("monto_probablemente_tranzado")
        .reset_index()
        .query("monto_probablemente_tranzado > 0")
    )
    precios_largos = precios.stack().rename("price").reset_index()
    eventos = eventos.merge(precios_largos, on=["timestamp", "advertiser_userno"])
    return agregar_fecha_bolivia(eventos)


def construir_metricas_diarias(tabla: pd.DataFrame, lado: str) -> pd.DataFrame:
    """Construye las tres metricas diarias de precios en hora Bolivia."""
    tabla = agregar_fecha_bolivia(tabla)
    eventos_transados = construir_eventos_transados(tabla)

    tabla = recortar_a_ventana_reciente(tabla, "fecha_bolivia")
    eventos_transados = recortar_a_ventana_reciente(
        eventos_transados, "fecha_bolivia"
    )

    filas = []
    for fecha, grupo in tabla.groupby("fecha_bolivia"):
        tramo_competitivo = obtener_tramo_competitivo(grupo, lado)
        grupo_transado = eventos_transados.loc[eventos_transados["fecha_bolivia"] == fecha]
        precio_transado = None
        if not grupo_transado.empty:
            precio_transado = calcular_vwap(
                grupo_transado, "monto_probablemente_tranzado"
            )
        filas.append(
            {
                "fecha": fecha,
                "precio_competitivo": calcular_vwap(
                    tramo_competitivo, "tradablequantity"
                ),
                "precio_profundidad": calcular_vwap(grupo, "tradablequantity"),
                "precio_transado_estimado": precio_transado,
            }
        )

    return pd.DataFrame(filas)[["fecha", *METRICAS_PRECIO]]


def resumir_estructura_oferta(tabla: pd.DataFrame) -> pd.DataFrame:
    """Resume liquidez y concentracion diaria del total de ofertas."""
    snapshots = (
        tabla.groupby(["timestamp", "advertiser_userno"], as_index=False)["tradablequantity"]
        .last()
    )
    snapshots = agregar_fecha_bolivia(snapshots)
    snapshots = recortar_a_ventana_reciente(snapshots, "fecha_bolivia")

    filas = []
    for _, grupo in snapshots.groupby("timestamp"):
        grupo = grupo.sort_values("tradablequantity", ascending=False)
        total = grupo["tradablequantity"].sum()
        top_5 = float(grupo["tradablequantity"].head(5).sum())
        filas.append(
            {
                "fecha": grupo["fecha_bolivia"].iloc[0],
                "total": total,
                "top_5": top_5,
            }
        )

    estructura = pd.DataFrame(filas)
    return (
        estructura.groupby("fecha", as_index=False)
        .agg(total=("total", "mean"), top_5=("top_5", "mean"))
        .sort_values("fecha")
        .reset_index(drop=True)
    )


def resumir_estructura_tranzado(tabla: pd.DataFrame) -> pd.DataFrame:
    """Resume liquidez y concentracion diaria del tranzado estimado."""
    eventos = recortar_a_ventana_reciente(
        construir_eventos_transados(tabla), "fecha_bolivia"
    )

    filas = []
    for _, grupo in eventos.groupby("timestamp"):
        grupo = grupo.sort_values("monto_probablemente_tranzado", ascending=False)
        total = grupo["monto_probablemente_tranzado"].sum()
        top_5 = float(grupo["monto_probablemente_tranzado"].head(5).sum())
        filas.append(
            {
                "fecha": grupo["fecha_bolivia"].iloc[0],
                "total": total,
                "top_5": top_5,
            }
        )

    estructura = pd.DataFrame(filas)
    return (
        estructura.groupby("fecha", as_index=False)
        .agg(total=("total", "mean"), top_5=("top_5", "mean"))
        .sort_values("fecha")
        .reset_index(drop=True)
    )


def exportar_metrica_precio(tabla: pd.DataFrame, lado: str, metrica: str) -> Path:
    """Exporta una metrica de precio a CSV con formato ``fecha,valor``."""
    salida = tabla[["fecha", metrica]].rename(columns={metrica: "valor"}).copy()
    salida["fecha"] = pd.to_datetime(salida["fecha"]).dt.strftime("%Y-%m-%d")
    ruta = DIRECTORIO_SALIDA / f"binance_{lado}_{metrica}.csv"
    salida.to_csv(ruta, float_format="%.3f", index=False)
    return ruta


def exportar_estructura(tabla: pd.DataFrame, tipo: str, lado: str) -> Path:
    """Exporta una serie diaria de estructura con columnas fecha,total,top_5."""
    salida = tabla.copy()
    salida["fecha"] = pd.to_datetime(salida["fecha"]).dt.strftime("%Y-%m-%d")
    ruta = DIRECTORIO_SALIDA / f"binance_estructura_{tipo}_{lado}.csv"
    salida.to_csv(ruta, float_format="%.3f", index=False)
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
        for metrica in METRICAS_PRECIO:
            rutas_exportadas.append(exportar_metrica_precio(serie_compacta, lado, metrica))
        rutas_exportadas.append(
            exportar_estructura(resumir_estructura_oferta(tabla), "oferta", lado)
        )
        rutas_exportadas.append(
            exportar_estructura(resumir_estructura_tranzado(tabla), "tranzado", lado)
        )

    for ruta in rutas_exportadas:
        print(ruta)


if __name__ == "__main__":
    main()
