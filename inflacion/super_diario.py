#!/usr/bin/env python3

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import unidecode

from numpy.linalg import svd


###############################################################################
# setup
###############################################################################

SKIP_SUBCATS = {
    "mascotas",
    "churrasqueria",
    "limpieza de zapatos",
    "cotillon",
    "mascotas general",
    "libreria",
    "living y dormitorio",
    "menaje plastico",
    "bijouteria/cuidado personal",
}

DROP_CATEGORIES = {
    "Juguetería",
    "Juguetería Importación",
    "Bazar Importación",
}

DATA_DIR = Path("/tmp/precios/data/hipermaxi")
PRODUCTS_CSV = DATA_DIR / "productos.csv"
MASK_PARQUET = Path("./inflacion/update_data/hm_df_mask_f.parquet")
OUTPUT_CSV = Path("./inflacion/datos/super_diario.csv")
CONFIDENCE_CSV = Path("./inflacion/datos/super_diario_confianza.csv")
STATE_FILE = Path("./inflacion/update_data/super_diario_state.pkl")
REFIT_FILTERS = False
CONFIDENCE_KEY_COLUMNS = ["departamento", "fecha", "componente"]
CONFIDENCE_SORT_COLUMNS = ["fecha", "departamento", "componente"]
BAND_QUANTILES = (.05, .50, .95)
BAND_COLUMNS = [f"band_q{int(100 * q):02d}" for q in BAND_QUANTILES]
CONFIDENCE_VALUE_COLUMNS = ["missing"] + BAND_COLUMNS
CONFIDENCE_COLUMNS = CONFIDENCE_KEY_COLUMNS + CONFIDENCE_VALUE_COLUMNS


def load_products(products_csv):
    df = pd.read_csv(products_csv)
    df["subcategoria_"] = df["subcategoria"].fillna("").astype(str).apply(
        lambda s: unidecode.unidecode(s).lower()
    )
    return df[~df["subcategoria_"].isin(SKIP_SUBCATS)].copy()


def load_department_price_frames(base_dir):
    out = {}
    for dept_path in sorted(glob.glob(os.path.join(base_dir, "*/"))):
        files = sorted(glob.glob(os.path.join(dept_path, "*.csv")))
        if not files:
            continue
        df = pd.concat((pd.read_csv(p) for p in files), ignore_index=True)

        dept_name = os.path.basename(os.path.normpath(dept_path))
        out[dept_name] = df

    return out


def load_filter_state():
    if REFIT_FILTERS or not STATE_FILE.exists():
        return {"svd": {}, "pca": {}}

    with open(STATE_FILE, "rb") as f:
        state = pickle.load(f)

    state.setdefault("svd", {})
    state.setdefault("pca", {})
    return state


def save_filter_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)


def empty_index_result():
    return {
        "index": pd.Series(dtype=float),
        "confidence": empty_confidence_frame(indexed=True),
    }


def empty_confidence_frame(indexed=False):
    columns = CONFIDENCE_VALUE_COLUMNS
    if not indexed:
        columns = ["componente", "fecha"] + CONFIDENCE_VALUE_COLUMNS

    return pd.DataFrame(columns=columns)


def gap_returns(pivot):
    px = pivot.asfreq("D").sort_index()

    miss = px.isna()
    gap_days = miss.apply(
        lambda s: s.groupby(s.ne(s.shift()).cumsum()).transform("sum")
    ).where(miss)
    gap_start = miss & ~miss.shift(1, fill_value=False)

    p0 = px.ffill()
    p1 = px.bfill()

    out = (
        pd.DataFrame({
            "gap_days": gap_days.where(gap_start).stack(),
            "price_before": p0.where(gap_start).stack(),
            "price_after": p1.where(gap_start).stack(),
        })
        .reset_index()
        .rename(columns={"level_0": "fecha", "level_1": "id_producto"})
    )

    out = out.dropna()
    out = out.loc[(out["price_before"] > 0) & (out["price_after"] > 0)]

    out["total_log_return"] = np.log(out["price_after"] / out["price_before"])
    out["daily_log_return"] = out["total_log_return"] / (out["gap_days"] + 1)

    return out


def missingness_log_band(pivot, weights, n=1000, q=BAND_QUANTILES, seed=0):
    columns = [f"q{int(100 * x):02d}" for x in q]
    if pivot.empty:
        return pd.DataFrame(columns=columns)

    pivot = pivot.asfreq("D").sort_index()
    weights = pd.Series(weights, index=pivot.columns).reindex(pivot.columns).fillna(0.0)
    gh = gap_returns(pivot)

    miss = pivot.isna()
    seen = pivot.notna().cummax().shift(fill_value=False)
    run = miss.ne(miss.shift()).cumsum()

    cur = miss & seen & miss.iloc[-1] & run.eq(run.iloc[-1])
    if cur.empty:
        return pd.DataFrame(columns=columns)

    start = (cur.index.max() - pd.DateOffset(months=1)).to_period("M").start_time
    cur = cur.loc[start:]

    active_dates = cur.sum(axis=1).replace(0, np.nan).dropna().index
    if active_dates.empty:
        return pd.DataFrame(columns=columns)

    cur = cur.loc[active_dates[0]:]
    W = cur.mul(weights, axis=1).fillna(0.0)
    W_aggregated = W.sum(axis=1).to_numpy()[:, None]

    pool = gh.loc[gh["gap_days"] > 1, ["gap_days", "daily_log_return"]]
    if pool.empty:
        return pd.DataFrame(np.nan, index=W.index, columns=columns)

    pool = pool["daily_log_return"].reindex(
        pool.index.repeat(pool["gap_days"].astype(int))
    ).dropna().to_numpy()
    if len(pool) == 0:
        return pd.DataFrame(np.nan, index=W.index, columns=columns)

    rng = np.random.default_rng(seed)
    shocks = rng.choice(pool, size=(1, n), replace=True)
    z = (W_aggregated @ shocks).cumsum(axis=0)

    return pd.DataFrame(
        np.quantile(z, q, axis=1).T,
        index=W.index,
        columns=columns,
    )


def svd_weights(Vt, X_sg, columns):
    X_sg = np.asarray(X_sg)
    weights = Vt.T @ Vt
    weights = (weights @ X_sg) / (len(X_sg) * X_sg)
    return pd.Series(weights, index=columns)


def uniform_weights(columns):
    return pd.Series(1.0, index=columns)


def confidence_from_pivot(pivot, weights):
    pivot = pivot.asfreq("D").sort_index()
    if pivot.empty or len(pivot.columns) == 0:
        return empty_confidence_frame(indexed=True)

    weights = pd.Series(weights, index=pivot.columns).reindex(pivot.columns).fillna(0.0)
    weights_abs = weights.abs()
    if weights_abs.sum() == 0:
        weights_abs = uniform_weights(pivot.columns)

    missing = 100 * pivot.isna().mul(weights_abs, axis=1).sum(axis=1) / weights_abs.sum()

    log_band = missingness_log_band(pivot, weights)
    log_band = log_band.rename(columns=lambda c: f"band_{c}")

    confidence = pd.DataFrame({"missing": missing.round(4)}, index=pivot.index)
    confidence = confidence.join(log_band, how="left")
    for column in BAND_COLUMNS:
        if column not in confidence:
            confidence[column] = np.nan
        confidence[column] = np.exp(
            pd.to_numeric(
                confidence[column], errors="coerce"
            )
        ).round(4)

    return confidence[CONFIDENCE_VALUE_COLUMNS]


def extract_inflation_index(df, filter_state, n_factors=5):
    df = df.loc[df["price"] > 0].copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    pivot = df.pivot_table(index='fecha', columns='id_producto', values='price')
    pivot = pivot.sort_index().sort_index(axis=1)

    if pivot.empty:
        return empty_index_result()

    if "columns" in filter_state:
        pivot = pivot.reindex(columns=filter_state["columns"])
    else:
        filter_state["columns"] = pivot.columns.to_list()

    pivot_daily = pivot.asfreq('D')
    logr = np.log(pivot_daily.ffill(limit_area='inside')).diff().fillna(0)

    if filter_state.get("kind") == "mean":
        index_s = np.exp(logr.mean(axis=1).cumsum()) * 100
        return {
            "index": index_s,
            "confidence": confidence_from_pivot(pivot_daily, uniform_weights(pivot.columns)),
        }

    X = logr.values

    if filter_state.get("kind") == "svd":
        X_mu = np.asarray(filter_state["X_mu"])
        X_sg = np.asarray(filter_state["X_sg"])
        Vt = np.asarray(filter_state["Vt"])

        Xc = (X - X_mu) / X_sg
        X_rec = Xc @ Vt.T @ Vt
        X_rec = (X_rec * X_sg) + X_mu

        logr_rec = X_rec.mean(axis=1)
        index_s = pd.Series(np.exp(np.cumsum(logr_rec)) * 100.0, index=logr.index)
        return {
            "index": index_s,
            "confidence": confidence_from_pivot(
                pivot_daily, svd_weights(Vt, X_sg, pivot.columns)
            ),
        }

    if (logr.shape[1] < (n_factors * 2)) or (logr.mean(axis=1).std() == 0):
        filter_state["kind"] = "mean"
        index_s = np.exp(logr.mean(axis=1).cumsum()) * 100
        return {
            "index": index_s,
            "confidence": confidence_from_pivot(pivot_daily, uniform_weights(pivot.columns)),
        }

    X_mu = X.mean(axis=0)
    X_sg = X.std(axis=0) + 1e-12
    Xc = (X - X_mu) / X_sg

    U, s, Vt = svd(Xc, full_matrices=False)

    # reconstruct
    s_factors = min(n_factors, len(s))
    X_rec = U[:, :s_factors] @ np.diag(s[:s_factors]) @ Vt[:s_factors, :]
    X_rec = (X_rec * X_sg) + X_mu

    filter_state["kind"] = "svd"
    filter_state["X_mu"] = X_mu
    filter_state["X_sg"] = X_sg
    filter_state["Vt"] = Vt[:s_factors, :]

    logr_rec = X_rec.mean(axis=1)

    index_s = pd.Series(
        np.exp(np.cumsum(logr_rec)) * 100.0,
        index=logr.index
    )

    return {
        "index": index_s,
        "confidence": confidence_from_pivot(
            pivot_daily, svd_weights(Vt[:s_factors, :], X_sg, pivot.columns)
        ),
    }


def extract_component_indexes(df_prices, product_categories, filter_state):
    if df_prices.empty:
        return pd.DataFrame(), empty_confidence_frame()

    categories = pd.Series(product_categories, index=df_prices.index, copy=False)
    valid = categories.notna()
    if not valid.any():
        return pd.DataFrame(), empty_confidence_frame()

    # Group on an in-frame column to avoid pandas reindexing a duplicated axis.
    grouped_prices = df_prices.loc[valid].copy()
    grouped_prices.insert(0, "_categoria", categories.loc[valid].to_numpy())

    component_map = {}
    confidence_frames = []
    for category, group in grouped_prices.groupby("_categoria", sort=True):
        if category in DROP_CATEGORIES:
            continue

        prices = (
            group.drop(columns="_categoria")
            .stack()
            .rename("price")
            .reset_index()
        )
        category_state = filter_state.setdefault(category, {})
        result = extract_inflation_index(prices, category_state, n_factors=5)
        index_s = result["index"]
        if not index_s.empty:
            component_map[category] = index_s

        confidence = result["confidence"]
        if not confidence.empty:
            confidence = confidence.rename_axis("fecha").reset_index()
            confidence.insert(0, "componente", category)
            confidence_frames.append(confidence)

    df_components = pd.DataFrame(component_map).T
    df_components.index.name = "categoria"
    df_components.columns.name = "fecha"

    if confidence_frames:
        df_confidence = pd.concat(confidence_frames, ignore_index=True)
    else:
        df_confidence = empty_confidence_frame()

    return df_components, df_confidence


def aggregate_components(df_components, filter_state, n_components=3):
    if isinstance(df_components, pd.Series):
        df_components = df_components.unstack()

    df_components = df_components.sort_index().sort_index(axis=1)
    X = np.log(df_components.T / 100).diff().iloc[1:]
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    if X.empty:
        return pd.Series(dtype=float, name="Compuesto")

    if "columns" in filter_state:
        X = X.reindex(columns=filter_state["columns"])
    else:
        filter_state["columns"] = X.columns.to_list()

    if filter_state.get("kind") == "mean":
        X_mu = pd.Series(filter_state["X_mu"], index=filter_state["columns"])
        X_st = pd.Series(filter_state["X_st"], index=filter_state["columns"])

        X = (X - X_mu) / X_st
        X = X.fillna(0)

        return X.mean(axis=1).rename("Compuesto")

    if filter_state.get("kind") == "pca":
        columns = filter_state["columns"]

        X_mu = pd.Series(filter_state["X_mu"], index=columns)
        X_st = pd.Series(filter_state["X_st"], index=columns)

        X = (X - X_mu) / X_st
        X = X.fillna(0)

        pca_mean = np.asarray(filter_state["pca_mean"])
        components = np.asarray(filter_state["components"])

        X_pca = (X.values - pca_mean) @ components.T
        X_rec = (X_pca @ components) + pca_mean
        X_rec = (pd.DataFrame(X_rec, index=X.index, columns=X.columns) * X_st) + X_mu

        return X_rec.mean(axis=1).rename("Compuesto")

    X_mu = X.mean()
    X_st = X.std(ddof=0).replace(0, 1)

    X = (X - X_mu) / X_st
    X = X.fillna(0)

    n_components = min(n_components, X.shape[0], X.shape[1])
    if n_components < 1:
        filter_state["kind"] = "mean"
        filter_state["X_mu"] = X_mu.to_numpy()
        filter_state["X_st"] = X_st.to_numpy()
        return X.mean(axis=1).rename("Compuesto")

    pca = PCA(n_components=n_components, svd_solver="full")
    X_pca = pca.fit_transform(X)

    # reconstruct
    X_rec = pca.inverse_transform(X_pca)
    X_rec = (pd.DataFrame(X_rec, index=X.index, columns=X.columns) * X_st) + X_mu

    filter_state["kind"] = "pca"
    filter_state["X_mu"] = X_mu.to_numpy()
    filter_state["X_st"] = X_st.to_numpy()
    filter_state["pca_mean"] = pca.mean_
    filter_state["components"] = pca.components_

    return X_rec.mean(axis=1).rename("Compuesto")


def build_department_index(df, df_mask, products_df, filter_state, dept):
    df = df.groupby(['fecha', 'id_producto'])['precio'].mean()
    df = df.unstack(level=0)

    df.columns = pd.to_datetime(df.columns)
    df = df.sort_index(axis=1)

    df = df.reindex(df_mask.index).loc[df_mask]
    product_categories = (
        products_df.loc[:, ["id_producto", "categoria"]]
        .dropna(subset=["id_producto", "categoria"])
        .drop_duplicates(subset="id_producto", keep="first")
        .set_index("id_producto")
        .reindex(df.index)["categoria"]
        .dropna()
    )
    df = df.loc[product_categories.index]

    svd_state = filter_state["svd"].setdefault(dept, {})
    pca_state = filter_state["pca"].setdefault(dept, {})

    df_components, df_confidence = extract_component_indexes(
        df, product_categories, svd_state
    )
    df_index = aggregate_components(df_components, pca_state)

    dept_index = pd.concat([
        df_index.rolling(window=7 * 4).sum(),
        np.log(df_components.T).diff().rolling(window=7 * 4).sum()
    ], axis=1).loc['2024/09/01':]

    if not df_confidence.empty:
        df_confidence = df_confidence.loc[
            pd.to_datetime(df_confidence["fecha"]) >= pd.Timestamp("2024-09-01")
        ]

    return (100 * np.exp(dept_index) - 100), df_confidence


def flatten_inflation_map(infl_map):
    frames = []
    for department, df in infl_map.items():
        frame = (
            df.rename_axis("fecha")
            .reset_index()
            .melt(id_vars="fecha", var_name="componente", value_name="inflacion_28d")
        )
        frame.insert(0, "departamento", department)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["departamento", "fecha", "componente", "inflacion_28d"]
        )

    return pd.concat(frames, ignore_index=True)


def flatten_confidence_map(confidence_map):
    frames = []
    for department, df in confidence_map.items():
        if df.empty:
            continue

        frame = df.copy()
        frame.insert(0, "departamento", department)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=CONFIDENCE_COLUMNS)

    return pd.concat(frames, ignore_index=True).loc[:, CONFIDENCE_COLUMNS]


def write_output(output_df):
    output_df = output_df.copy()
    output_df["fecha"] = pd.to_datetime(output_df["fecha"])
    output_df["inflacion_28d"] = output_df["inflacion_28d"].round(3)
    output_df = output_df.sort_values(["fecha", "departamento", "componente"])

    if OUTPUT_CSV.exists() and not REFIT_FILTERS:
        old_dates = pd.read_csv(OUTPUT_CSV, usecols=["fecha"])
        if not old_dates.empty:
            last_date = pd.to_datetime(old_dates["fecha"]).max()
            output_df = output_df.loc[output_df["fecha"] > last_date]

        if output_df.empty:
            return

        output_df.to_csv(
            OUTPUT_CSV, mode="a", header=False, index=False, float_format="%.3f"
        )
        return

    output_df.to_csv(OUTPUT_CSV, index=False, float_format="%.3f")


def write_confidence_output(output_df):
    output_df = output_df.copy()
    output_df = output_df.reindex(columns=CONFIDENCE_COLUMNS)

    if not output_df.empty:
        output_df["fecha"] = pd.to_datetime(output_df["fecha"])
        output_df["missing"] = output_df["missing"].round(4)
        for column in BAND_COLUMNS:
            output_df[column] = output_df[column].round(4)

    if not CONFIDENCE_CSV.exists() or REFIT_FILTERS:
        output_df = output_df.sort_values(CONFIDENCE_SORT_COLUMNS)
        output_df.to_csv(CONFIDENCE_CSV, index=False)
        return

    old_df = pd.read_csv(CONFIDENCE_CSV)
    old_df = old_df.reindex(columns=CONFIDENCE_COLUMNS)
    if old_df.empty:
        output_df = output_df.sort_values(CONFIDENCE_SORT_COLUMNS)
        output_df.to_csv(CONFIDENCE_CSV, index=False)
        return

    old_df["fecha"] = pd.to_datetime(old_df["fecha"])
    for column in CONFIDENCE_VALUE_COLUMNS:
        old_df[column] = pd.to_numeric(old_df[column], errors="coerce")

    if output_df.empty:
        old_df = old_df.sort_values(CONFIDENCE_SORT_COLUMNS)
        old_df.to_csv(CONFIDENCE_CSV, index=False)
        return

    old_keys = pd.MultiIndex.from_frame(old_df[CONFIDENCE_KEY_COLUMNS])
    new_keys = pd.MultiIndex.from_frame(output_df[CONFIDENCE_KEY_COLUMNS])
    band_keys = pd.MultiIndex.from_frame(
        output_df.loc[output_df[BAND_COLUMNS].notna().any(axis=1), CONFIDENCE_KEY_COLUMNS]
    )

    append_rows = output_df.loc[~new_keys.isin(old_keys)]
    update_rows = output_df.loc[new_keys.isin(old_keys) & new_keys.isin(band_keys)]

    if update_rows.empty and append_rows.empty:
        old_df = old_df.sort_values(CONFIDENCE_SORT_COLUMNS)
        old_df.to_csv(CONFIDENCE_CSV, index=False)
        return

    update_keys = pd.MultiIndex.from_frame(update_rows[CONFIDENCE_KEY_COLUMNS])
    keep_old = old_df.loc[~old_keys.isin(update_keys)]

    output_df = pd.concat([keep_old, update_rows, append_rows], ignore_index=True)
    output_df["fecha"] = pd.to_datetime(output_df["fecha"])
    output_df["missing"] = output_df["missing"].round(4)
    for column in BAND_COLUMNS:
        output_df[column] = output_df[column].round(4)

    output_df = output_df.sort_values(CONFIDENCE_SORT_COLUMNS)
    output_df.to_csv(CONFIDENCE_CSV, index=False)


###############################################################################
# run
###############################################################################

def main():
    filter_state = load_filter_state()
    products_df = load_products(PRODUCTS_CSV)
    dept_price_frames = load_department_price_frames(DATA_DIR)

    df_mask_f = pd.read_parquet(MASK_PARQUET)

    infl_map = {}
    confidence_map = {}
    for dept, df in dept_price_frames.items():
        df_mask = df_mask_f[dept]
        infl_map[dept], confidence_map[dept] = build_department_index(
            df, df_mask, products_df, filter_state, dept
        )

    output_df = flatten_inflation_map(infl_map)
    write_output(output_df)
    confidence_df = flatten_confidence_map(confidence_map)
    write_confidence_output(confidence_df)
    save_filter_state(filter_state)


if __name__ == "__main__":
    main()
