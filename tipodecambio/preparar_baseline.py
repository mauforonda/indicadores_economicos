#!/usr/bin/env python3

"""Construye tablas baseline de precios Binance desde parquets historicos.

El flujo es:
1. Lee uno o mas parquets historicos generados por `descarga_binance.py`.
2. Replica la logica de precios de `actualizar_binance.py` para `buy` y `sell`.
3. Resume dos referencias fijas:
   - `8_nov`: todo el 2025-11-08 en hora Bolivia
   - `8_nov_3m`: 2025-08-08 a 2025-11-07 inclusive en hora Bolivia
4. Exporta una tabla `referencia,tipo,valor` para cada lado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROPORCION_TRAMO_COMPETITIVO = 0.10
ZONA_HORARIA_BOLIVIA = "America/La_Paz"
RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = RUTA_BASE / "datos"
COLUMNAS = [
    "timestamp",
    "asset",
    "tradetype",
    "advno",
    "fiatunit",
    "price",
    "tradablequantity",
    "advertiser_userno",
    "minsingletransamount",
    "maxsingletransamount",
]
MAPEO_LADOS = {
    "buy": "SELL",
    "sell": "BUY",
}
REFERENCIAS = {
    "8_nov": ("2025-11-08", "2025-11-08"),
    "8_nov_3m": ("2025-08-08", "2025-11-07"),
}
TIPOS_SALIDA = {
    "precio_competitivo": "competitivo",
    "precio_profundidad": "profundidad",
    "precio_transado_estimado": "tranzado_estimado",
}


def cargar_fuentes(rutas: list[Path]) -> pd.DataFrame:
    """Lee y concatena los parquets requeridos con solo columnas utiles."""
    tablas = [pd.read_parquet(ruta, columns=COLUMNAS) for ruta in rutas]
    if not tablas:
        raise SystemExit("Debes proveer al menos un parquet historico.")
    tabla = pd.concat(tablas, ignore_index=True)
    if pd.api.types.is_numeric_dtype(tabla["timestamp"]):
        tabla["timestamp"] = (
            pd.to_datetime(tabla["timestamp"], unit="s", utc=True)
            .dt.tz_localize(None)
        )
    else:
        tabla["timestamp"] = pd.to_datetime(tabla["timestamp"])
    return (
        tabla.drop_duplicates()
        .sort_values(["timestamp", "advno"])
        .reset_index(drop=True)
    )


def cargar_tabla_lado(tabla: pd.DataFrame, lado_salida: str) -> pd.DataFrame:
    """Filtra USDT y aplica el mismo mapeo de lados del script principal."""
    tradetype_kaggle = MAPEO_LADOS[lado_salida]
    filtrada = tabla.loc[
        (tabla["asset"] == "USDT") & (tabla["tradetype"] == tradetype_kaggle)
    ].copy()
    if filtrada.empty:
        raise SystemExit(f"No hay filas USDT/{tradetype_kaggle} para construir `{lado_salida}`.")
    return filtrada.sort_values("timestamp").reset_index(drop=True)


def agregar_fecha_bolivia(tabla: pd.DataFrame) -> pd.DataFrame:
    """Anota timestamp y fecha local en hora Bolivia."""
    tabla = tabla.copy()
    tabla["timestamp_bolivia"] = (
        tabla["timestamp"].dt.tz_localize("UTC").dt.tz_convert(ZONA_HORARIA_BOLIVIA)
    )
    tabla["fecha_bolivia"] = tabla["timestamp_bolivia"].dt.normalize().dt.tz_localize(
        None
    )
    return tabla


def calcular_vwap(tabla: pd.DataFrame, columna_monto: str) -> float | None:
    """Calcula un promedio ponderado por precio."""
    pesos = tabla[columna_monto]
    total = pesos.sum()
    if total == 0:
        return None
    return float((tabla["price"] * pesos).sum() / total)


def obtener_tramo_competitivo(tabla: pd.DataFrame, lado: str) -> pd.DataFrame:
    """Selecciona el mismo 10% extremo usado por `actualizar_binance.py`."""
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
        columns="advno",
        values="tradablequantity",
        aggfunc="last",
    ).sort_index()
    precios = tabla.pivot_table(
        index="timestamp",
        columns="advno",
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
    eventos = eventos.merge(precios_largos, on=["timestamp", "advno"])
    return agregar_fecha_bolivia(eventos)


def filtrar_periodo(tabla: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Filtra por fecha Bolivia inclusiva."""
    fecha_inicio = pd.Timestamp(inicio)
    fecha_fin = pd.Timestamp(fin)
    return tabla.loc[
        (tabla["fecha_bolivia"] >= fecha_inicio)
        & (tabla["fecha_bolivia"] <= fecha_fin)
    ].copy()


def resumir_referencias(tabla_lado: pd.DataFrame, lado: str) -> pd.DataFrame:
    """Construye las tres metricas para cada referencia fija."""
    tabla_lado = agregar_fecha_bolivia(tabla_lado)
    eventos_transados = construir_eventos_transados(tabla_lado)

    filas: list[dict[str, object]] = []
    for referencia, (inicio, fin) in REFERENCIAS.items():
        grupo = filtrar_periodo(tabla_lado, inicio, fin)
        if grupo.empty:
            raise SystemExit(
                f"No hay datos para `{referencia}` en el lado `{lado}`."
            )
        grupo_transado = filtrar_periodo(eventos_transados, inicio, fin)
        tramo_competitivo = obtener_tramo_competitivo(grupo, lado)

        metricas = {
            "precio_competitivo": calcular_vwap(
                tramo_competitivo, "tradablequantity"
            ),
            "precio_profundidad": calcular_vwap(grupo, "tradablequantity"),
            "precio_transado_estimado": (
                calcular_vwap(grupo_transado, "monto_probablemente_tranzado")
                if not grupo_transado.empty
                else None
            ),
        }
        for nombre_interno, tipo_salida in TIPOS_SALIDA.items():
            filas.append(
                {
                    "referencia": referencia,
                    "tipo": tipo_salida,
                    "valor": metricas[nombre_interno],
                }
            )

    return pd.DataFrame(filas)[["referencia", "tipo", "valor"]]


def exportar_tabla(tabla: pd.DataFrame, lado: str, output_dir: Path) -> Path:
    """Guarda la tabla baseline para un lado."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ruta = output_dir / f"binance_baseline_{lado}.csv"
    tabla.to_csv(ruta, float_format="%.3f", index=False)
    return ruta


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de linea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Lee parquets historicos y genera baselines buy/sell para "
            "8_nov y 8_nov_3m."
        )
    )
    parser.add_argument(
        "parquets",
        nargs="+",
        type=Path,
        help="Uno o mas parquets historicos producidos por descarga_binance.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIRECTORIO_SALIDA,
        help="Directorio donde se escriben binance_baseline_buy.csv y sell.csv",
    )
    return parser


def main() -> None:
    """Ejecuta la construccion de tablas baseline."""
    args = construir_parser().parse_args()
    rutas = [ruta.resolve() for ruta in args.parquets]
    faltantes = [str(ruta) for ruta in rutas if not ruta.exists()]
    if faltantes:
        raise SystemExit(f"No existen estos parquets: {', '.join(faltantes)}")

    tabla_fuente = cargar_fuentes(rutas)
    rutas_exportadas: list[Path] = []
    for lado in ("buy", "sell"):
        tabla_lado = cargar_tabla_lado(tabla_fuente, lado)
        baseline = resumir_referencias(tabla_lado, lado)
        rutas_exportadas.append(exportar_tabla(baseline, lado, args.output_dir))

    for ruta in rutas_exportadas:
        print(ruta)


if __name__ == "__main__":
    main()
