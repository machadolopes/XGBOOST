"""XGBoost Survival AFT: DMatrix, treino e importância."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.model_utils import COVARIATEIS, amostrar_estratificado, cindex
from src.survival_utils import (
    S_aft,
    ajustar_sigma_aft as ajustar_sigma_aft_dist,
    localizacao_from_pred,
    tempo_em_dias_inteiros,
)

RNG = 42
FRACAO_TESTE = 0.20
N_CINDEX_UNIVERSO = 80_000

AFT_NUM_BOOST = 500
AFT_EARLY_STOP = 30
PARAMS_AFT_DISSERTACAO: dict[str, Any] = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution_scale": 1.0,
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "min_child_weight": 10,
    "seed": RNG,
    "verbosity": 0,
}


def covariaveis_como_treino(df: pd.DataFrame) -> pd.DataFrame:
    """Garante os tipos com que o encoder foi ajustado (ano em string)."""
    out = df[list(COVARIATEIS)].copy()
    out["ano_ajuizamento"] = out["ano_ajuizamento"].astype(str)
    return out


def prever_mu_e_S(modelo, enc, df_cat, grelha, sigma: float, dist: str):
    """μ e matriz Ŝ(t | x) para um quadro de covariáveis de ajuizamento."""
    X = enc.transform(covariaveis_como_treino(df_cat))
    mu = localizacao_from_pred(np.asarray(modelo.predict(xgb.DMatrix(X)), dtype=float))
    S = S_aft(mu, np.asarray(grelha, dtype=float), float(sigma), dist)
    return mu, S


def encoder_esparso():
    """One-hot CSR (float32); categorias não vistas no treino → zeros."""
    from sklearn.preprocessing import OneHotEncoder

    return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)


def preparar_split(df: pd.DataFrame, n: int | None, random_state: int = RNG):
    """Amostra opcional, 80/20 estratificado (evento × ano), tempos ao dia."""
    bloco = df if n is None or n >= len(df) else amostrar_estratificado(df, n=n, random_state=random_state)
    bloco = bloco.reset_index(drop=True).copy()
    bloco["ano_ajuizamento"] = bloco["ano_ajuizamento"].astype(str)
    bloco["tempo"] = tempo_em_dias_inteiros(bloco["tempo"])
    strata = bloco["evento"].astype(str) + "_" + bloco["ano_ajuizamento"]
    cont = strata.value_counts()
    raros = set(cont[cont < 2].index)
    if raros:
        strata = strata.where(~strata.isin(raros), other="_raro")
    df_tr, df_te = train_test_split(
        bloco, test_size=FRACAO_TESTE, stratify=strata, random_state=random_state
    )
    return df_tr.reset_index(drop=True), df_te.reset_index(drop=True)


def _dmatrix_aft(X, evento, tempo, feature_names: list[str] | None = None) -> xgb.DMatrix:
    evento = np.asarray(evento).astype(bool)
    tempo = np.asarray(tempo, dtype=np.float32)
    y_lo = tempo.copy()
    y_hi = np.where(evento, tempo, np.finfo(np.float32).max)
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    dmat.set_float_info("label_lower_bound", y_lo)
    dmat.set_float_info("label_upper_bound", y_hi)
    return dmat


def dmatrix_aft(X, evento, tempo, feature_names: list[str] | None = None) -> xgb.DMatrix:
    """DMatrix AFT com limites [t, t] (evento) e [t, +∞) (censura à direita)."""
    return _dmatrix_aft(X, evento, tempo, feature_names=feature_names)


def treinar_aft_distribuicao(
    dtrain: xgb.DMatrix,
    dist: str,
    dval: xgb.DMatrix | None = None,
    num_boost: int = AFT_NUM_BOOST,
    early_stopping_rounds: int = AFT_EARLY_STOP,
    nthread: int | None = 8,
) -> tuple[xgb.Booster, dict]:
    """Treina um AFT com a distribuição pedida e early stopping na validação interna."""
    params = {**PARAMS_AFT_DISSERTACAO, "aft_loss_distribution": dist}
    if nthread is not None:
        params["nthread"] = int(nthread)
    historico: dict = {}
    kwargs: dict[str, Any] = {
        "params": params,
        "dtrain": dtrain,
        "num_boost_round": int(num_boost),
        "verbose_eval": False,
        "evals_result": historico,
    }
    if dval is not None:
        kwargs["evals"] = [(dtrain, "train"), (dval, "val")]
        try:
            from xgboost.callback import EarlyStopping

            kwargs["callbacks"] = [
                EarlyStopping(rounds=int(early_stopping_rounds), save_best=True)
            ]
        except Exception:
            kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    else:
        kwargs["evals"] = [(dtrain, "train")]
    modelo = xgb.train(**kwargs)
    return modelo, historico


def localizacao_aft(modelo, X) -> np.ndarray:
    pred = np.asarray(modelo.predict(xgb.DMatrix(X)), dtype=float)
    return localizacao_from_pred(pred)


def ajustar_sigma_aft(mu_tr: np.ndarray, evento, tempo, dist: str = "normal") -> float:
    return ajustar_sigma_aft_dist(mu_tr, evento, tempo, distribuicao=dist)


def importancia_gain_grupos(booster, feature_names: np.ndarray | list[str]) -> pd.DataFrame:
    """Gain do XGBoost agregado às três covariáveis originais."""
    nomes = np.asarray(feature_names).astype(str)
    indice = {str(n): i for i, n in enumerate(nomes)}
    scores = booster.get_score(importance_type="gain")
    grupos = {"classe_processual": 0.0, "orgao_julgador": 0.0, "ano_ajuizamento": 0.0}
    for chave, val in scores.items():
        chave_s = str(chave)
        nome = None
        if chave_s in indice:
            nome = chave_s
        elif chave_s.startswith("f"):
            try:
                i = int(chave_s[1:])
            except ValueError:
                continue
            if 0 <= i < len(nomes):
                nome = str(nomes[i])
        if nome is None:
            continue
        for g in grupos:
            if nome.startswith(g):
                grupos[g] += float(val)
                break
    total = sum(grupos.values()) or 1.0
    return pd.DataFrame(
        [
            {"covariavel": g, "gain": round(v, 2), "gain_relativo": round(v / total, 4)}
            for g, v in grupos.items()
        ]
    )


def cindex_sub(y, risco: np.ndarray, n_max: int | None = None, random_state: int = RNG) -> float:
    """C-index; no universo usa subamostra (Harrell em ~700 mil é inviável)."""
    risco = np.asarray(risco, dtype=float)
    risco = np.nan_to_num(risco, nan=0.0, posinf=1e6, neginf=-1e6)
    n = len(y)
    if n_max is None or n <= n_max:
        return cindex(y, risco)
    from sksurv.metrics import concordance_index_censored

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=n_max, replace=False)
    return float(concordance_index_censored(y["event"][idx], y["time"][idx], risco[idx])[0])
