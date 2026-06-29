import io
import requests
import unidecode
import pandas as pd
from pathlib import Path


NOW = pd.to_datetime('now').strftime('%Y-%m-%d')
URL = 'https://www.bcb.gob.bo/tco_tcreferencial_descargar_csv.php'
PARAMS = {
    'desde': NOW, 'hasta': NOW,
}
MAPEO_BANCOS = {
    "banco_bisa": "Banco BISA",
    "banco_de_credito": "Banco de Crédito",
    "banco_de_la_nacion_argentina": "Banco de la Nación Argentina",
    "banco_economico": "Banco Económico",
    "banco_fie": "Banco FIE",
    "banco_fortaleza": "Banco Fortaleza",
    "banco_ganadero": "Banco Ganadero",
    "banco_mercantil_santa_cruz": "Banco Mercantil Santa Cruz",
    "banco_nacional_de_bolivia": "Banco Nacional de Bolivia",
    "banco_prodem": "Banco Prodem",
    "banco_pyme_de_la_comunidad": "Banco PYME de la Comunidad",
    "banco_pyme_ecofuturo": "Banco PYME Ecofuturo",
    "banco_solidario": "Banco Solidario",
    "banco_union": "Banco Unión",
}
RUTA_BASE = Path(__file__).resolve().parent
DIRECTORIO_SALIDA = RUTA_BASE / "datos"
OUT_S = DIRECTORIO_SALIDA / 'oficial_bcb.csv'
OUT_D = DIRECTORIO_SALIDA / 'oficial_bcb_desagregado.csv'

def do_process(req):
    df = pd.read_csv(io.StringIO(req.text), skiprows=5, sep=';', header=None)

    df.columns = pd.MultiIndex.from_frame(
        df.iloc[:2].T.apply(
            lambda _: _.str.lower().str.replace(
                r'\([^\)]+\)', '', regex=True
            ).str.strip().str.replace(' ', '_')
        ).ffill()
    )
    df.iloc[-1] = df.iloc[-1].ffill()
    df = df.drop(columns='n°', level=1).droplevel(1, axis=1).iloc[2:]

    df = df.iloc[-2:]
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.set_index(['fecha', 'tc'])

    df = df.map(
        lambda _: _.replace('.', '').replace(',', '.')
    ).replace('-', 'NaN').astype(float)

    df.columns = (df.columns).map(unidecode.unidecode)
    df.columns.name = 'banco'

    return df


def do_merge(new, storage_path, mkeys):
    storage_path = Path(storage_path)
    _extra_opts = {
        'index': False,
        'float_format': '%.2f',
    }

    if not storage_path.is_file():
        return new.sort_values(mkeys).to_csv(storage_path, **_extra_opts)

    old = pd.read_csv(storage_path)
    old['fecha'] = pd.to_datetime(old['fecha'])

    if old['fecha'].max() >= new['fecha'].max():
        return

    pd.concat([
        old, new
    ]).sort_values(mkeys).to_csv(storage_path, **_extra_opts)

if __name__ == '__main__':
    req = requests.get(URL, params=PARAMS)
    df = do_process(req)

    df = df.stack().unstack(level='tc')
    df = df.reset_index()

    df.columns = ['fecha', 'banco', 'valor', 'monto']
    df['monto'] = df['monto'].astype(int)

    df_mask = df['banco'] == 'total_bancos'
    do_merge(df[df_mask].drop(columns='banco'), OUT_S, ['fecha'])
    do_merge(df[~df_mask].assign(banco=df.loc[~df_mask, 'banco'].map(MAPEO_BANCOS)), OUT_D, ['fecha', 'banco'])
