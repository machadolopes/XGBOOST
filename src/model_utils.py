"""Preparação, métricas e faixas de celeridade (XGBoost Survival AFT)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.survival_utils import HORIZONTES_DIAS, estimar_km, tempo_em_dias_inteiros

COVARIATEIS = ("classe_processual", "orgao_julgador", "ano_ajuizamento")
N_AMOSTRA_AVAL = 80_000
ANO_REF_PERFIS = 2022
N_SUB_CURVA_CLASSE = 200
MIN_N_SEGMENTO_FAIXA = 40
MIN_N_ORGAO_CLASSE = 80
LIMIAR_ANOMALIA = 2.0

ORDEM_FAIXAS = (
    "Muito rápido",
    "Rápido",
    "Típico",
    "Lento",
    "Muito lento",
)


def y_sobrevivencia(evento: np.ndarray | pd.Series, tempo: np.ndarray | pd.Series) -> np.ndarray:
    """Array estruturado (event, time) exigido pelo scikit-survival."""
    from sksurv.util import Surv

    event = np.asarray(evento).astype(bool)
    time = tempo_em_dias_inteiros(tempo)
    return Surv.from_arrays(event, time)


def amostrar_estratificado(
    df: pd.DataFrame,
    n: int = N_AMOSTRA_AVAL,
    colunas_strata: tuple[str, ...] = ("evento", "ano_ajuizamento"),
    random_state: int = 42,
) -> pd.DataFrame:
    """Amostra sem reposição, estratificada (preserva censura e coortes anuais)."""
    from sklearn.model_selection import train_test_split

    if n >= len(df):
        return df.copy().reset_index(drop=True)
    strata = df[list(colunas_strata)].astype(str).agg("_".join, axis=1)
    cont = strata.value_counts()
    raros = set(cont[cont < 2].index)
    if raros:
        strata = strata.where(~strata.isin(raros), other="_raro")
    _, amostra = train_test_split(df, test_size=n, stratify=strata, random_state=random_state)
    return amostra.reset_index(drop=True)


def ajustar_encoder(X_cat: pd.DataFrame) -> OneHotEncoder:
    """One-hot denso (fallback); o treino AFT usa encoder_esparso()."""
    enc = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    enc.fit(X_cat[list(COVARIATEIS)])
    return enc


def transformar_X(enc: OneHotEncoder, X_cat: pd.DataFrame):
    return enc.transform(X_cat[list(COVARIATEIS)])


def nomes_features(enc: OneHotEncoder) -> np.ndarray:
    return enc.get_feature_names_out(COVARIATEIS)


def cindex(y, risco: np.ndarray) -> float:
    """Índice de concordância de Harrell."""
    from sksurv.metrics import concordance_index_censored

    return float(concordance_index_censored(y["event"], y["time"], risco)[0])


def grelha_brier(t_max: float = 1825.0, passo: int = 30) -> np.ndarray:
    """Grelha temporal para a curva do Brier (inclui os horizontes académicos)."""
    grelha = np.arange(passo, t_max + 1, passo, dtype=float)
    extra = np.array(list(HORIZONTES_DIAS.values()), dtype=float)
    grelha = np.unique(np.concatenate([grelha, extra]))
    return grelha[grelha <= t_max]


def brier_modelo_e_km(
    y_train,
    y_test,
    S_mod: np.ndarray,
    tempos: np.ndarray,
    km_treino=None,
    evento_treino=None,
    tempo_treino=None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Brier pontual (modelo vs. KM de referência) e IBS no mesmo suporte."""
    from sksurv.metrics import brier_score, integrated_brier_score

    if km_treino is None:
        if evento_treino is None:
            evento_treino = y_train["event"]
            tempo_treino = y_train["time"]
        km_treino = estimar_km(evento_treino, tempo_treino)
    S_km_t = np.array([km_treino.em(t)[0] for t in tempos])
    est_km = np.tile(S_km_t, (len(y_test), 1))
    _, bs_mod = brier_score(y_train, y_test, S_mod, tempos)
    _, bs_km = brier_score(y_train, y_test, est_km, tempos)
    ibs_mod = float(integrated_brier_score(y_train, y_test, S_mod, tempos))
    ibs_km = float(integrated_brier_score(y_train, y_test, est_km, tempos))
    return np.asarray(bs_mod), np.asarray(bs_km), ibs_mod, ibs_km


# Alias usado pelos geradores portados.
brier_rsf_e_km = brier_modelo_e_km


def calibracao_decis(y_test, s_pred: np.ndarray, t: float, n_decis: int = 10) -> pd.DataFrame:
    """Média de Ŝ(t) prevista vs. KM observado, por decil da previsão."""
    s_pred = np.asarray(s_pred, dtype=float)
    ranks = pd.Series(s_pred).rank(method="first")
    try:
        decil = pd.qcut(ranks, q=n_decis, labels=False) + 1
    except ValueError:
        decil = pd.Series(np.full(len(s_pred), 1, dtype=int))
    linhas = []
    for d in range(1, int(decil.max()) + 1):
        mask = decil.to_numpy() == d
        if mask.sum() == 0:
            continue
        km = estimar_km(y_test["event"][mask], y_test["time"][mask])
        s_obs, lo, hi = km.em(t)
        linhas.append(
            {
                "decil": d,
                "n": int(mask.sum()),
                "eventos": int(y_test["event"][mask].sum()),
                "S_previsto_medio": round(float(s_pred[mask].mean()), 4),
                "S_observado_KM": round(s_obs, 4),
                "IC_95_inf": round(lo, 4),
                "IC_95_sup": round(hi, 4),
            }
        )
    return pd.DataFrame(linhas)


def grelha_curvas(t_max: float = 3650.0, passo: int = 30) -> np.ndarray:
    """Grelha para desenhar Ŝ(t | x), incluindo horizontes académicos e de dashboard."""
    from src.survival_utils import HORIZONTES_DASHBOARD

    grelha = np.arange(passo, t_max + 1, passo, dtype=float)
    extra = np.array(
        list(HORIZONTES_DIAS.values()) + list(HORIZONTES_DASHBOARD.values()),
        dtype=float,
    )
    return np.unique(np.concatenate([grelha, extra]))


def tempo_ate_limiar(tempos: np.ndarray, sobrevivencia: np.ndarray, limiar: float = 0.5) -> float:
    """Primeiro t com Ŝ(t) ≤ limiar (mediana se 0,5). NaN se não for atingido."""
    t = np.asarray(tempos, dtype=float)
    s = np.asarray(sobrevivencia, dtype=float)
    hit = np.where(s <= limiar)[0]
    if hit.size == 0:
        return float("nan")
    return float(t[hit[0]])


def faixa_por_percentis(valor: float, p10: float, p25: float, p75: float, p90: float) -> str:
    """Faixa de celeridade sobre exp(μ): valores baixos = término mais cedo."""
    if not np.isfinite(valor):
        return "Indeterminada"
    if valor <= p10:
        return "Muito rápido"
    if valor <= p25:
        return "Rápido"
    if valor <= p75:
        return "Típico"
    if valor <= p90:
        return "Lento"
    return "Muito lento"
