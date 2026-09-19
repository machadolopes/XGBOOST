"""Funções auxiliares da EDA (associações, intervalos, faixas temporais)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FAIXAS_TEMPO_DIAS = [
    ("0–6 meses", 0, 182),
    ("6 meses–1 ano", 182, 365),
    ("1–2 anos", 365, 730),
    ("2–3 anos", 730, 1095),
    ("3–5 anos", 1095, 1825),
    ("5 anos+", 1825, np.inf),
]


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float, int, float]:
    """Cramér's V, estatística qui-quadrado, graus de liberdade e p-valor."""
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(x, y)
    chi2, p, gl, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    r, c = ct.shape
    k = min(r, c) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else np.nan
    return v, float(chi2), int(gl), float(p)


def ic_mediana(valores: np.ndarray | pd.Series, z: float = 1.96) -> tuple[float, float, float]:
    """Intervalo de confiança não paramétrico da mediana (ordem binomial)."""
    x = np.sort(np.asarray(valores, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    if n < 5:
        return med, med, med
    i = int(np.floor((n - z * np.sqrt(n)) / 2.0))
    j = int(np.ceil(1.0 + (n + z * np.sqrt(n)) / 2.0))
    i = max(0, min(i, n - 1))
    j = max(0, min(j, n - 1))
    return med, float(x[i]), float(x[j])


def classificar_faixa_tempo(dias: pd.Series) -> pd.Series:
    """Assinala a faixa temporal (desde o ajuizamento) de cada processo."""
    out = pd.Series(index=dias.index, dtype="object")
    for nome, lo, hi in FAIXAS_TEMPO_DIAS:
        mask = (dias >= lo) & (dias < hi)
        out.loc[mask] = nome
    return pd.Categorical(out, categories=[n for n, _, _ in FAIXAS_TEMPO_DIAS], ordered=True)


def classes_por_volume(df: pd.DataFrame, coluna: str = "classe_processual") -> list[str]:
    """Classes da amostra analítica (top 95% + «Outras classes»), por volume decrescente."""
    return df[coluna].value_counts().index.tolist()


def encoding_frequencia(serie: pd.Series) -> pd.Series:
    """Substitui cada categoria pela respectiva frequência relativa."""
    freq = serie.value_counts(normalize=True)
    return serie.map(freq).astype(float)


def percentagens_que_somam_100(contagens, casas: int = 2) -> np.ndarray:
    """Percentagens com `casas` decimais cuja soma é exactamente 100.

    Usa o método dos maiores restos (Hamilton): evita que o arredondamento
    independente de cada linha produza 99,99 % ou 100,01 %.
    """
    n = np.asarray(contagens, dtype=np.int64)
    total = int(n.sum())
    if total == 0 or n.size == 0:
        return np.zeros(n.shape, dtype=float)
    factor = 10**casas
    quota = 100 * factor
    numerador = n.astype(object) * quota
    quociente = np.array([int(x // total) for x in numerador], dtype=np.int64)
    restos = np.array([int(x % total) for x in numerador], dtype=np.int64)
    falta = quota - int(quociente.sum())
    if falta > 0:
        ordem = np.argsort(-restos, kind="mergesort")
        quociente[ordem[:falta]] += 1
    elif falta < 0:
        ordem = np.argsort(restos, kind="mergesort")
        removidos = 0
        for idx in ordem:
            if removidos >= -falta:
                break
            if quociente[idx] > 0:
                quociente[idx] -= 1
                removidos += 1
    return quociente / float(factor)
