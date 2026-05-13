#!/usr/bin/env python3

import glob
import os
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


def extract_inflation_index(df, n_factors=5):
    df = df.loc[df["price"] > 0].copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    pivot = df.pivot_table(index='fecha', columns='id_producto', values='price')
    pivot = pivot.sort_index()

    if pivot.empty:
        return pd.Series(dtype=float)

    logr = np.log(
        pivot.asfreq('D').ffill(limit_area='inside')
    ).diff().fillna(0)

    if (logr.shape[1] < (n_factors * 2)) or (logr.mean(axis=1).std() == 0):
        return np.exp(logr.mean(axis=1).cumsum()) * 100

    X = logr.values
    X_mu = X.mean(axis=0)
    X_sg = X.std(axis=0) + 1e-12
    Xc = (X - X_mu) / X_sg

    U, s, Vt = svd(Xc, full_matrices=False)

    # reconstruct
    s_factors = min(n_factors, len(s))
    X_rec = U[:, :s_factors] @ np.diag(s[:s_factors]) @ Vt[:s_factors, :]
    X_rec = (X_rec * X_sg) + X_mu

    logr_rec = X_rec.mean(axis=1)

    index_s = pd.Series(
        np.exp(np.cumsum(logr_rec)) * 100.0,
        index=logr.index
    )

    return index_s


def extract_component_indexes(df_prices, product_categories):
    if df_prices.empty:
        return pd.DataFrame()

    categories = pd.Series(product_categories, index=df_prices.index, copy=False)
    valid = categories.notna()
    if not valid.any():
        return pd.DataFrame()

    # Group on an in-frame column to avoid pandas reindexing a duplicated axis.
    grouped_prices = df_prices.loc[valid].copy()
    grouped_prices.insert(0, "_categoria", categories.loc[valid].to_numpy())

    component_map = {}
    for category, group in grouped_prices.groupby("_categoria", sort=True):
        if category in DROP_CATEGORIES:
            continue

        prices = (
            group.drop(columns="_categoria")
            .stack()
            .rename("price")
            .reset_index()
        )
        index_s = extract_inflation_index(prices, n_factors=5)
        if not index_s.empty:
            component_map[category] = index_s

    df_components = pd.DataFrame(component_map).T
    df_components.index.name = "categoria"
    df_components.columns.name = "fecha"
    return df_components


def aggregate_components(df_components, n_components=3):
    if isinstance(df_components, pd.Series):
        df_components = df_components.unstack()

    df_components = df_components.sort_index().sort_index(axis=1)
    X = np.log(df_components.T / 100).diff().iloc[1:]
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    if X.empty:
        return pd.Series(dtype=float, name="Compuesto")

    X_mu = X.mean()
    X_st = X.std(ddof=0).replace(0, 1)

    X = (X - X_mu) / X_st
    X = X.fillna(0)

    n_components = min(n_components, X.shape[0], X.shape[1])
    if n_components < 1:
        return X.mean(axis=1).rename("Compuesto")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # reconstruct
    X_rec = pca.inverse_transform(X_pca)
    X_rec = (pd.DataFrame(X_rec, index=X.index, columns=X.columns) * X_st) + X_mu

    return X_rec.mean(axis=1).rename("Compuesto")


def build_department_index(df, df_mask, products_df):
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

    df_components = extract_component_indexes(df, product_categories)
    df_index = aggregate_components(df_components)

    dept_index = pd.concat([
        df_index.rolling(window=7 * 4).sum(),
        np.log(df_components.T).diff().rolling(window=7 * 4).sum()
    ], axis=1).loc['2024/09/01':]
    return (100 * np.exp(dept_index) - 100)


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


###############################################################################
# run
###############################################################################

def main():
    products_df = load_products(PRODUCTS_CSV)
    dept_price_frames = load_department_price_frames(DATA_DIR)

    df_mask_f = pd.read_parquet(MASK_PARQUET)

    infl_map = {}
    for dept, df in dept_price_frames.items():
        df_mask = df_mask_f[dept]
        infl_map[dept] = build_department_index(df, df_mask, products_df)

    output_df = flatten_inflation_map(infl_map)
    output_df.to_csv(OUTPUT_CSV, index=False) # shit


if __name__ == "__main__":
    main()
