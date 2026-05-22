import os
from pathlib import Path

import pandas as pd


# Parametros principales
ffill_horas = 12
departamentos = [2, 7]
horas_resolucion_horaria = 6


# Parametros fijos del indicador
fecha_inicio_insumo = "2025-04-01"
fecha_inicio_historica = "2025-05-01"
meses_cobertura_horaria = 3
hora_historica = 12
percentil_critico = 0.10
pisos_absolutos = {
    "Diesel": 2000.0,
    "Gasolina": 2000.0,
}
horas_persistencia = 3
combustibles = {
    2: "Diesel",
    10: "Diesel",
    3: "Gasolina",
    7: "Gasolina",
}
combustible_codigo = {
    "Diesel": 0,
    "Gasolina": 1,
}
orden_productos = ["Diesel", "Gasolina"]


base_dir = Path(__file__).resolve().parent
source_dir = Path(os.environ.get("BO_COMBUSTIBLE_DIR", base_dir / "bo-combustible")).resolve()
data_dir = source_dir / "data"
stations_path = source_dir / "stations.csv"
output_dir = base_dir / "datos"
output_dir.mkdir(exist_ok=True)


def load_raw_events():
    frames = []
    usecols = ["fecha_actualizacion", "id_eess", "id_producto_bsa", "saldo_bsa"]

    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path, usecols=usecols, parse_dates=["fecha_actualizacion"])
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["id_producto_bsa"].isin(combustibles)].copy()
    raw = raw[raw["fecha_actualizacion"] >= fecha_inicio_insumo].copy()
    raw = raw[raw["saldo_bsa"] > 0].copy()
    raw["producto"] = raw["id_producto_bsa"].map(combustibles)

    stations = pd.read_csv(stations_path, usecols=["id_eess_saldo", "id_departamento"])
    raw = raw.merge(stations, left_on="id_eess", right_on="id_eess_saldo", how="left")
    raw = raw.drop(columns=["id_eess_saldo"])
    return raw.sort_values("fecha_actualizacion").reset_index(drop=True)


def build_hourly_reports(raw):
    hourly = raw.copy()
    hourly["ts_hour"] = hourly["fecha_actualizacion"].dt.floor("h")
    hourly = (
        hourly.sort_values("fecha_actualizacion")
        .groupby(["ts_hour", "id_eess", "producto", "id_departamento"], as_index=False)["saldo_bsa"]
        .last()
    )
    return hourly


def build_thresholds(hourly_reports):
    thresholds = (
        hourly_reports.groupby(["id_eess", "producto"])["saldo_bsa"]
        .quantile(percentil_critico)
        .rename("threshold_rel")
        .reset_index()
    )
    thresholds["threshold_abs"] = thresholds["producto"].map(pisos_absolutos)
    thresholds["threshold"] = thresholds[["threshold_rel", "threshold_abs"]].max(axis=1)
    return thresholds


def build_station_hourly(hourly_reports, thresholds):
    stations = pd.read_csv(stations_path, usecols=["id_eess_saldo", "id_departamento"])
    full_hours = pd.date_range(
        hourly_reports["ts_hour"].min(),
        hourly_reports["ts_hour"].max(),
        freq="h",
    )

    parts = []
    for producto in orden_productos:
        product_thresholds = thresholds[thresholds["producto"] == producto].copy()
        station_ids = product_thresholds["id_eess"].sort_values().unique()

        dfp = hourly_reports[hourly_reports["producto"] == producto][["ts_hour", "id_eess", "saldo_bsa"]].copy()
        wide = (
            dfp.pivot(index="ts_hour", columns="id_eess", values="saldo_bsa")
            .reindex(index=full_hours, columns=station_ids)
            .sort_index()
        )
        wide = wide.ffill(limit=ffill_horas).fillna(1.0)

        long = wide.stack(future_stack=True).rename("saldo_bsa").reset_index()
        long.columns = ["ts_hour", "id_eess", "saldo_bsa"]
        long["producto"] = producto
        parts.append(long)

    station_hourly = pd.concat(parts, ignore_index=True)
    station_hourly = station_hourly.merge(
        thresholds[["id_eess", "producto", "threshold"]],
        on=["id_eess", "producto"],
        how="left",
    )
    station_hourly = station_hourly.merge(
        stations,
        left_on="id_eess",
        right_on="id_eess_saldo",
        how="left",
    ).drop(columns=["id_eess_saldo"])

    station_hourly = station_hourly.sort_values(["producto", "id_eess", "ts_hour"])
    station_hourly["low_flag"] = station_hourly["saldo_bsa"] < station_hourly["threshold"]
    station_hourly["critical_flag"] = station_hourly.groupby(["producto", "id_eess"])["low_flag"].transform(
        lambda s: s.rolling(horas_persistencia, min_periods=horas_persistencia).sum().eq(horas_persistencia)
    )
    return station_hourly


def build_hourly_window(station_hourly):
    max_ts = station_hourly["ts_hour"].max()
    min_ts = max_ts - pd.DateOffset(months=meses_cobertura_horaria)
    hourly = station_hourly[station_hourly["ts_hour"] >= min_ts].copy()
    hourly = hourly[hourly["ts_hour"].dt.hour % horas_resolucion_horaria == 0].copy()
    return hourly


def build_historical_noon(station_hourly):
    historical = station_hourly[station_hourly["ts_hour"] >= pd.Timestamp(fecha_inicio_historica)].copy()
    historical = historical[historical["ts_hour"].dt.hour == hora_historica].copy()
    historical["fecha"] = historical["ts_hour"].dt.floor("D")
    return historical


def format_hourly_timestamp(series):
    local = pd.to_datetime(series)
    return local.dt.strftime("%Y-%m-%dT%H:%M:%S-04:00")


def build_national_output(station_hourly, date_col, hourly):
    national = (
        station_hourly.groupby([date_col, "producto"])["critical_flag"]
        .mean()
        .reset_index(name="porcentaje_critico")
    )
    national["combustible"] = national["producto"].map(combustible_codigo)
    if hourly:
        national["fecha"] = format_hourly_timestamp(national[date_col])
    else:
        national["fecha"] = pd.to_datetime(national[date_col]).dt.strftime("%Y-%m-%d")
    national["porcentaje_critico"] = national["porcentaje_critico"].round(3)
    national = national[["fecha", "combustible", "porcentaje_critico"]]
    return national.sort_values(["fecha", "combustible"]).reset_index(drop=True)


def build_department_output(station_hourly, date_col, hourly):
    by_dept = station_hourly[station_hourly["id_departamento"].isin(departamentos)].copy()
    by_dept = (
        by_dept.groupby([date_col, "id_departamento", "producto"])["critical_flag"]
        .mean()
        .reset_index(name="porcentaje_critico")
        .rename(columns={"id_departamento": "departamento"})
    )
    by_dept["combustible"] = by_dept["producto"].map(combustible_codigo)
    if hourly:
        by_dept["fecha"] = format_hourly_timestamp(by_dept[date_col])
    else:
        by_dept["fecha"] = pd.to_datetime(by_dept[date_col]).dt.strftime("%Y-%m-%d")
    by_dept["porcentaje_critico"] = by_dept["porcentaje_critico"].round(3)
    by_dept = by_dept[["fecha", "departamento", "combustible", "porcentaje_critico"]]
    return by_dept.sort_values(["fecha", "departamento", "combustible"]).reset_index(drop=True)


def main():
    raw = load_raw_events()
    hourly_reports = build_hourly_reports(raw)
    thresholds = build_thresholds(hourly_reports)
    station_hourly = build_station_hourly(hourly_reports, thresholds)

    hourly_window = build_hourly_window(station_hourly)
    historical_noon = build_historical_noon(station_hourly)

    national_hourly = build_national_output(hourly_window, "ts_hour", hourly=True)
    department_hourly = build_department_output(hourly_window, "ts_hour", hourly=True)
    national_historical = build_national_output(historical_noon, "fecha", hourly=False)
    department_historical = build_department_output(historical_noon, "fecha", hourly=False)

    national_hourly.to_csv(output_dir / "estaciones_criticas_nacional.csv", index=False, float_format="%.3f")
    department_hourly.to_csv(output_dir / "estaciones_criticas_departamentos.csv", index=False, float_format="%.3f")
    national_historical.to_csv(output_dir / "estaciones_criticas_historicas_nacional.csv", index=False, float_format="%.3f")
    department_historical.to_csv(output_dir / "estaciones_criticas_historicas_departamentos.csv", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
