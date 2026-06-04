import glob

import numpy as np
import pandas as pd


# load

df = pd.concat([
    pd.read_csv(_) for _ in sorted(glob.glob('/tmp/despacho_de_carga/data_demanda/*.csv'))
], ignore_index=True)
df['hora'] = pd.to_datetime(df['hora'])
df = df.set_index(['agente', 'hora'])['demanda'].astype(float)

agents = df.reset_index()['agente'].drop_duplicates()
agents_cre = agents[agents.str.startswith('CRE')]


# build basic ts

s = (
    df[
        ~df.index.get_level_values(0).isin(agents_cre.values)
    ]
    .groupby(level=1).sum()
    .loc['2016':]
    .resample('D').mean()
)
s = s[s.index.dayofweek < 5]

t = (
    df
    .groupby(level=1).sum()
    .loc['2016':]
    .resample('D').mean()
)
t = t[t.index.dayofweek < 5]


# fix ts

carnavales = [
    ('2017-02-24', '2017-03-01'),
    ('2018-02-09', '2018-02-14'),
    ('2019-03-01', '2019-03-06'),
    ('2020-02-21', '2020-02-26'),
    ('2021-02-12', '2021-02-17'),
    ('2022-02-25', '2022-03-02'),
    ('2023-02-17', '2023-02-22'),
    ('2024-02-09', '2024-02-14'),
    ('2025-02-28', '2025-03-05'),
    ('2026-02-13', '2026-02-18'),
]

for _, __ in carnavales:
    _m = (s.index >= _) & (s.index <= __)
    s.loc[_m] = np.nan
    t.loc[_m] = np.nan


# build sts

st = pd.concat([s.rename('nacional_sin_scz'), t.rename('nacional')], axis=1)
st['scz'] = st['nacional'] - st['nacional_sin_scz']

sts = np.log(st).asfreq('D').resample('D').mean().rolling(window=7 * 4, min_periods=0).mean()
sts = np.exp(sts)

sts = sts / sts.shift(365)
sts = 100 * sts.loc['2018':] - 100


# fix sts

sts['fixed'] = sts['nacional'].copy()
sts.loc['2023-06-25':'2024-07-14', 'fixed'] = np.nan
sts['fixed'] = sts['fixed'].fillna(sts['nacional_sin_scz'])

sts.index.name = 'fecha'
sts.round(4).to_csv('./actividad_economica/datos/demada_energia_electrica.csv')
