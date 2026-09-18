"""Reconstrução de S(t) AFT, horizontes e Kaplan–Meier (bloco interno de calibração)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Horizontes de avaliação académica (Notebook 3) — não alterar.
HORIZONTES_DIAS = {
    "6 meses": 182,
    "1 ano": 365,
    "2 anos": 730,
    "3 anos": 1095,
    "5 anos": 1825,
}

# Horizontes de exibição do dashboard (Notebook 6) — conjunto próprio.
HORIZONTES_DASHBOARD = {
    "6 meses": 182,
    "1 ano": 365,
    "2 anos": 730,
    "5 anos": 1825,
    "mais de 7 anos": 2555,
}

HORIZONTES_FIGURA = ("6 meses", "1 ano", "2 anos", "5 anos")
T_MAX_PLOT_DIAS = 3650
MAX_PONTOS_PLOT = 2000
AFT_DISTRIBUICOES = ("normal", "logistic", "extreme")


def tempo_em_dias_inteiros(tempo: np.ndarray | pd.Series) -> np.ndarray:
    """Agrupa o tempo contínuo ao dia civil (mínimo de 1 dia)."""
    t = np.asarray(tempo, dtype=np.float64)
    return np.maximum(np.floor(t), 1.0)


@dataclass(frozen=True)
class CurvaKM:
    """Estimativa de Kaplan–Meier com intervalo de confiança log-log a 95%."""

    tempo: np.ndarray
    sobrevivencia: np.ndarray
    ic_inf: np.ndarray
    ic_sup: np.ndarray

    def em(self, t: float) -> tuple[float, float, float]:
        idx = int(np.searchsorted(self.tempo, t, side="right") - 1)
        if idx < 0:
            return 1.0, 1.0, 1.0
        return (
            float(self.sobrevivencia[idx]),
            float(self.ic_inf[idx]),
            float(self.ic_sup[idx]),
        )


def estimar_km(
    evento: np.ndarray | pd.Series,
    tempo: np.ndarray | pd.Series,
    conf_level: float = 0.95,
) -> CurvaKM:
    """Kaplan–Meier via sksurv (usado na curva de calibração por decil)."""
    from sksurv.nonparametric import kaplan_meier_estimator

    event = np.asarray(evento).astype(bool)
    time_exit = tempo_em_dias_inteiros(tempo)
    time, surv, ci = kaplan_meier_estimator(
        event, time_exit, conf_type="log-log", conf_level=conf_level
    )
    return CurvaKM(
        tempo=np.asarray(time, dtype=np.float64),
        sobrevivencia=np.asarray(surv, dtype=np.float64),
        ic_inf=np.asarray(ci[0], dtype=np.float64),
        ic_sup=np.asarray(ci[1], dtype=np.float64),
    )


def localizacao_from_pred(pred: np.ndarray | list[float]) -> np.ndarray:
    """Converte a saída do booster AFT em μ (log-dias), finito."""
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, np.nan)
    if pred.size == 0 or not np.isfinite(pred).any():
        return np.zeros_like(pred, dtype=float)
    med = float(np.nanmedian(pred))
    if np.isfinite(med) and med > 20.0:
        mu = np.log(np.maximum(pred, 1e-6))
    else:
        mu = pred
    mu = np.clip(mu, -2.0, 12.0)
    return np.nan_to_num(mu, nan=float(np.nanmedian(mu)), posinf=12.0, neginf=-2.0)


def z_aft(mu: np.ndarray, t: np.ndarray | float, sigma: float) -> np.ndarray:
    """z = (ln t − μ) / σ."""
    mu = np.asarray(mu, dtype=float)
    t_arr = np.maximum(np.asarray(t, dtype=float), 1.0)
    s = max(float(sigma), 1e-3)
    return (np.log(t_arr) - mu) / s


def sobrevivencia_aft_z(z: np.ndarray, distribuicao: str) -> np.ndarray:
    """Ŝ a partir de z padronizado, segundo a distribuição AFT."""
    from scipy.stats import norm

    z = np.asarray(z, dtype=float)
    dist = str(distribuicao).lower()
    if dist == "normal":
        s = 1.0 - norm.cdf(z)
    elif dist == "logistic":
        s = 1.0 / (1.0 + np.exp(np.clip(z, -40.0, 40.0)))
    elif dist in ("extreme", "extreme_value", "gumbel"):
        s = np.exp(-np.exp(np.clip(z, -40.0, 40.0)))
    else:
        raise ValueError(f"Distribuição AFT desconhecida: {distribuicao!r}")
    return np.clip(s, 0.0, 1.0)


def S_aft(
    mu: np.ndarray,
    tempos: np.ndarray | list[float],
    sigma: float,
    distribuicao: str,
) -> np.ndarray:
    """Matriz Ŝ(t | x): forma (n observações × n tempos)."""
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    t = np.maximum(np.asarray(tempos, dtype=float).reshape(1, -1), 1.0)
    z = z_aft(mu, t, sigma)
    return sobrevivencia_aft_z(z, distribuicao)


def nll_aft_sigma(
    sigma: float,
    mu: np.ndarray,
    logt: np.ndarray,
    evento: np.ndarray,
    distribuicao: str,
) -> float:
    """NLL média de σ com μ fixo (eventos: log-densidade; censurados: log-S)."""
    from scipy.special import log_expit
    from scipy.stats import norm

    s = max(float(sigma), 1e-3)
    z = (np.asarray(logt, dtype=float) - np.asarray(mu, dtype=float)) / s
    ev = np.asarray(evento).astype(bool)
    dist = str(distribuicao).lower()
    ll = np.empty(z.shape[0], dtype=float)

    if dist == "normal":
        ll[ev] = norm.logpdf(z[ev]) - np.log(s)
        ll[~ev] = norm.logsf(z[~ev])
    elif dist == "logistic":
        zc = np.clip(z, -40.0, 40.0)
        ll[ev] = zc[ev] - 2.0 * np.logaddexp(0.0, zc[ev]) - np.log(s)
        ll[~ev] = log_expit(-zc[~ev])
    elif dist in ("extreme", "extreme_value", "gumbel"):
        ez = np.exp(np.clip(z, -40.0, 40.0))
        ll[ev] = z[ev] - ez[ev] - np.log(s)
        ll[~ev] = -ez[~ev]
    else:
        raise ValueError(f"Distribuição AFT desconhecida: {distribuicao!r}")

    fin = np.isfinite(ll)
    return float(-np.mean(ll[fin])) if fin.any() else 1e6


def ajustar_sigma_aft(
    mu: np.ndarray,
    evento: np.ndarray | pd.Series,
    tempo: np.ndarray | pd.Series,
    distribuicao: str = "normal",
) -> float:
    """σ por máxima verosimilhança censurada, com μ já fixo."""
    from scipy.optimize import minimize_scalar

    mu = np.asarray(mu, dtype=float)
    logt = np.log(np.maximum(np.asarray(tempo, dtype=float), 1.0))
    ev = np.asarray(evento).astype(bool)

    def objetivo(sigma: float) -> float:
        return nll_aft_sigma(sigma, mu, logt, ev, distribuicao)

    res = minimize_scalar(objetivo, bounds=(0.15, 4.0), method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def predict_survival_function_aft(modelo, dmatrix, sigma, distribuicao, horizontes):
    """Reconstrói Ŝ(t) nos horizontes a partir do modelo AFT.

    μ = log(predict) se o booster devolver tempo em dias; caso contrário, predict é já μ.
    z = (ln t − μ) / σ
      normal   → 1 − Φ(z)
      logistic → 1 / (1 + exp(z))
      extreme  → exp(−exp(z))
    """
    pred = np.asarray(modelo.predict(dmatrix), dtype=float)
    mu = localizacao_from_pred(pred)
    if isinstance(horizontes, dict):
        chaves = list(horizontes.values())
    else:
        chaves = list(horizontes)
    curvas = {}
    for t in chaves:
        t_f = float(t)
        z = z_aft(mu, t_f, sigma)
        curvas[t] = sobrevivencia_aft_z(z, distribuicao).reshape(-1)
    return curvas
