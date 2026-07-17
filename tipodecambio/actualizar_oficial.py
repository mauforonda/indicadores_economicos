import io
import requests
import unidecode
import pandas as pd

from bs4 import BeautifulSoup
from pathlib import Path


NOW = pd.to_datetime('now').normalize()
BASE_URL = 'https://www.bcb.gob.bo/tco_reporte_ultima_cotizacion.php'
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


def do_clean(_):
    _ = _.str.lower().str.replace(
        r'\([^\)]+\)', '', regex=True
    ).str.strip().str.replace(' ', '_')
    return _.map(unidecode.unidecode)


def do_process(req):
    table_html = str(
        BeautifulSoup(req.text, 'html.parser').select_one('.tco-public-table')
    )

    df = pd.read_html(
        io.StringIO(table_html),
        decimal=',', thousands='.',
        flavor='bs4',
    )
    assert len(df) > 0
    df = df[0]

    df.columns = do_clean(df.columns)
    df['banco'] = do_clean(df['entidad'])

    df = df.rename(columns={'compra': 'valor'})
    df['fecha'] = pd.to_datetime('now').normalize()

    df = df[['fecha', 'banco', 'valor', 'monto']]

    return df


def do_merge(new, storage_path, mkeys):
    storage_path = Path(storage_path)
    _extra_opts = {
        'index': False,
        'float_format': '%.2f',
        'date_format': '%Y-%m-%d',
    }

    if not storage_path.is_file():
        return new.sort_values(mkeys).to_csv(storage_path, **_extra_opts)

    old = pd.read_csv(storage_path)
    old['fecha'] = pd.to_datetime(old['fecha'])
    old_fm = old['fecha'].max()

    on = pd.concat([old, new])

    if (
        (old_fm >= new['fecha'].max()) |
        (on[on['fecha'] >= old_fm].drop(columns='fecha').duplicated().sum() == new.shape[0])
    ):
        return

    on.sort_values(mkeys).to_csv(storage_path, **_extra_opts)

if __name__ == '__main__':
    req = requests.get(BASE_URL)
    df = do_process(req)

    df_mask = (
        (df['banco'] == 'total_bancos') |
        (df['banco'] == 'bancos') |
        (df['banco'] == 'totales')
    )
    do_merge(df[df_mask].ffill().iloc[[-1]].drop(columns='banco'), OUT_S, ['fecha'])
    do_merge(df[~df_mask].assign(banco=df.loc[~df_mask, 'banco'].map(MAPEO_BANCOS)), OUT_D, ['fecha', 'banco'])
