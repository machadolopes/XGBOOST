"""Dashboard gerencial: modelo AFT congelado, sem re-treino."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_utils import NOME_OUTRAS_CLASSES, project_root
from src.model_utils import (
    LIMIAR_ANOMALIA,
    MIN_N_ORGAO_CLASSE,
    ORDEM_FAIXAS,
    faixa_por_percentis,
    grelha_curvas,
    tempo_ate_limiar,
)
from src.survival_utils import HORIZONTES_DASHBOARD, HORIZONTES_DIAS
from src.xgb_survival_utils import prever_mu_e_S

ANO_REF = "2022"
ANO_TREINO_MIN = 2015
ANO_TREINO_MAX = 2025
MAX_ORGAOS_OVERLAY = 6
N_EXEMPLOS_FAIXA = 4
ANOS_LEITURA_CAUTELA = {"2024", "2025"}
ANOS_FOLLOWUP_OK = {str(a) for a in range(2015, 2024)}

FAIXA_COR = {
    "Muito rápido": "green",
    "Rápido": "blue",
    "Típico": "gray",
    "Lento": "orange",
    "Muito lento": "red",
    "Indeterminada": "gray",
}

HORIZONTES_DESTAQUE = HORIZONTES_DASHBOARD


def _raiz() -> Path:
    return project_root()


def caminho_booster() -> Path:
    return _raiz() / "modelo" / "xgboost_aft_modelo_final.json"


def caminho_encoder() -> Path:
    return _raiz() / "modelo" / "xgboost_aft_encoder.joblib"


def caminho_meta() -> Path:
    return _raiz() / "modelo" / "xgboost_aft_meta.json"


def dir_tab() -> Path:
    return _raiz() / "tabelas"


def carregar_modelo_aft() -> dict:
    """Booster + encoder + metadados. Não treina."""
    meta = json.loads(caminho_meta().read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(str(caminho_booster()))
    encoder = joblib.load(caminho_encoder())
    cats = {}
    if hasattr(encoder, "categories_") and hasattr(encoder, "feature_names_in_"):
        cats = {
            str(nome): [str(c) for c in cat]
            for nome, cat in zip(encoder.feature_names_in_, encoder.categories_)
        }
    return {
        "modelo": booster,
        "encoder": encoder,
        "meta": meta,
        "dist": meta["distribuicao"],
        "sigma": float(meta["sigma"]),
        "classes_retidas": cats.get("classe_processual", []),
        "anos_retidos": cats.get("ano_ajuizamento", []),
    }


def categorias_encoder(enc) -> dict[str, list[str]]:
    nomes = list(enc.feature_names_in_)
    out: dict[str, list[str]] = {}
    for col, cats in zip(nomes, enc.categories_):
        out[col] = [str(c) for c in cats]
    return out


def ano_para_modelo(ano: int | str) -> tuple[str, bool]:
    """Devolve (ano usado no one-hot, extrapolação?). Nunca deixa o vector de ano a zeros."""
    try:
        a = int(ano)
    except (TypeError, ValueError):
        return str(ANO_TREINO_MAX), True
    if ANO_TREINO_MIN <= a <= ANO_TREINO_MAX:
        return str(a), False
    proxy = min(max(a, ANO_TREINO_MIN), ANO_TREINO_MAX)
    return str(proxy), True


def mapear_classe(nome: str | None, classes_retidas: list[str]) -> tuple[str, bool]:
    """Mapeia classes fora do top 95% para «Outras classes»."""
    if nome and str(nome) in classes_retidas:
        return str(nome), False
    return NOME_OUTRAS_CLASSES, True


def montar_perfis(classes, orgaos, anos) -> pd.DataFrame:
    if isinstance(classes, str):
        classes = [classes]
    if isinstance(orgaos, str):
        orgaos = [orgaos]
    if isinstance(anos, str):
        anos = [anos]
    n = max(len(classes), len(orgaos), len(anos))
    if len(classes) == 1:
        classes = classes * n
    if len(orgaos) == 1:
        orgaos = orgaos * n
    if len(anos) == 1:
        anos = anos * n
    return pd.DataFrame(
        {
            "classe_processual": classes,
            "orgao_julgador": orgaos,
            "ano_ajuizamento": [str(a) for a in anos],
        }
    )


def prever_perfis(pacote: dict, x_cat: pd.DataFrame, grelha: np.ndarray | None = None):
    grelha = grelha if grelha is not None else grelha_curvas()
    mu, S = prever_mu_e_S(
        pacote["modelo"], pacote["encoder"], x_cat, grelha, pacote["sigma"], pacote["dist"]
    )
    return mu, S, grelha


def s_nos_horizontes(tempos: np.ndarray, s: np.ndarray, horizontes: dict | None = None) -> dict[str, float]:
    horizontes = horizontes or HORIZONTES_DASHBOARD
    out = {}
    for nome, t in horizontes.items():
        idx = int(np.searchsorted(tempos, t, side="right") - 1)
        out[nome] = float(s[idx]) if idx >= 0 else 1.0
    return out


def curva_para_grafico(tempos, sobrevivencias, rotulos) -> pd.DataFrame:
    linhas = []
    for i, rotulo in enumerate(rotulos):
        for t, s in zip(tempos, sobrevivencias[i]):
            linhas.append(
                {
                    "perfil": rotulo,
                    "dias": float(t),
                    "anos": float(t) / 365.25,
                    "ainda_pendente": float(s),
                    "ja_terminado": float(1.0 - s),
                }
            )
    return pd.DataFrame(linhas)


def classificar_t(valor: float, p10, p25, p75, p90) -> str:
    return faixa_por_percentis(float(valor), float(p10), float(p25), float(p75), float(p90))


def formatar_cnj(processo_id: str) -> str:
    digits = "".join(ch for ch in str(processo_id) if ch.isdigit())
    if len(digits) < 20:
        return str(processo_id)
    d = digits[-20:]
    return f"{d[:7]}-{d[7:9]}.{d[9:13]}.{d[13]}.{d[14:16]}.{d[16:20]}"


def dias_para_texto(dias: float) -> str:
    if not np.isfinite(dias):
        return "não definido até 10 anos"
    meses = dias / 30.437
    if dias < 400:
        return f"{dias:.0f} dias ({meses:.0f} meses)"
    anos = dias / 365.25
    return f"{dias:.0f} dias ({anos:.1f} anos)"


def pct_em_100(s: float) -> str:
    return f"{100.0 * float(s):.1f} %".replace(".", ",")


def racio_vs_restantes(valores: np.ndarray) -> np.ndarray:
    t = np.asarray(valores, dtype=float)
    out = np.full(t.shape, np.nan, dtype=float)
    for i in range(len(t)):
        rest = np.delete(t, i)
        if rest.size == 0:
            continue
        med = float(np.median(rest))
        if med > 0 and np.isfinite(t[i]):
            out[i] = t[i] / med
    return out


def construir_artefactos_dash(*, forcar: bool = False) -> dict[str, Path]:
    """Catálogo, prazos previstos por célula, percentis, exemplos e anomalias."""
    tab = dir_tab()
    tab.mkdir(parents=True, exist_ok=True)
    paths = {
        "volume": tab / "dash_volume_classe_orgao_ano.csv",
        "prazos": tab / "dash_prazos_celulas.csv",
        "percentis": tab / "dash_percentis_classe_ano.csv",
        "catalogo": tab / "dash_catalogo_classe_orgao.csv",
        "exemplos": tab / "dash_exemplos_processos.csv",
        "anomalias": tab / "dash_anomalias.csv",
    }
    if (not forcar) and all(p.is_file() for p in paths.values()):
        return paths

    parquet = _raiz() / "dados" / "dataset_limpo.parquet"
    cols = [
        "processo_id",
        "classe_processual",
        "orgao_julgador",
        "ano_ajuizamento",
        "evento",
        "tempo",
    ]
    print("A ler o universo (só colunas de capa)…", flush=True)
    df = pd.read_parquet(parquet, columns=cols)
    df["ano_ajuizamento"] = df["ano_ajuizamento"].astype(str)
    df["classe_processual"] = df["classe_processual"].astype(str)
    df["orgao_julgador"] = df["orgao_julgador"].astype(str)
    df["processo_id"] = df["processo_id"].astype(str)

    catalogo = (
        df.groupby(["classe_processual", "orgao_julgador"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["classe_processual", "n"], ascending=[True, False])
    )
    catalogo.to_csv(paths["catalogo"], index=False, encoding="utf-8-sig")

    volume = (
        df.groupby(["classe_processual", "orgao_julgador", "ano_ajuizamento"], observed=True)
        .size()
        .reset_index(name="n")
    )
    volume.to_csv(paths["volume"], index=False, encoding="utf-8-sig")

    print(f"Células classe × órgão × ano: {len(volume):,}".replace(",", " "), flush=True)
    pacote = carregar_modelo_aft()
    celulas = volume[["classe_processual", "orgao_julgador", "ano_ajuizamento"]].drop_duplicates()
    mu, S, grelha = prever_perfis(pacote, celulas)
    prazos = celulas.copy()
    prazos["t_previsto_d"] = np.exp(mu)
    prazos["mu"] = mu
    s_h = [s_nos_horizontes(grelha, S[i]) for i in range(len(prazos))]
    prazos["S_6meses"] = [h["6 meses"] for h in s_h]
    prazos["S_1ano"] = [h["1 ano"] for h in s_h]
    prazos["S_2anos"] = [h["2 anos"] for h in s_h]
    prazos["S_5anos"] = [h["5 anos"] for h in s_h]
    prazos["S_7anos"] = [h["mais de 7 anos"] for h in s_h]
    prazos["mediana_curva_d"] = [tempo_ate_limiar(grelha, S[i], 0.5) for i in range(len(prazos))]
    prazos = prazos.merge(volume, on=["classe_processual", "orgao_julgador", "ano_ajuizamento"], how="left")
    prazos.to_csv(paths["prazos"], index=False, encoding="utf-8-sig")

    linhas_p = []
    for (classe, ano), g in prazos.groupby(["classe_processual", "ano_ajuizamento"]):
        if len(g) < 3:
            continue
        s = g["t_previsto_d"]
        linhas_p.append(
            {
                "classe_processual": classe,
                "ano_ajuizamento": str(ano),
                "n_orgaos": int(len(g)),
                "p10": float(s.quantile(0.10)),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "p90": float(s.quantile(0.90)),
                "mediana": float(s.median()),
            }
        )
    percentis = pd.DataFrame(linhas_p)
    percentis.to_csv(paths["percentis"], index=False, encoding="utf-8-sig")

    an = []
    sub_ano = prazos.loc[prazos["ano_ajuizamento"] == ANO_REF]
    for classe, g in sub_ano.groupby("classe_processual"):
        g = g.loc[g["n"] >= MIN_N_ORGAO_CLASSE]
        if len(g) < 3:
            continue
        t = g["t_previsto_d"].to_numpy()
        orgs = g["orgao_julgador"].to_numpy()
        ns = g["n"].to_numpy()
        s2 = g["S_2anos"].to_numpy()
        racios = racio_vs_restantes(t)
        for i, org in enumerate(orgs):
            racio = float(racios[i])
            an.append(
                {
                    "classe_processual": classe,
                    "orgao_julgador": org,
                    "ano_ajuizamento": ANO_REF,
                    "n": int(ns[i]),
                    "t_orgao_d": float(t[i]),
                    "t_restantes_d": float(np.median(np.delete(t, i))),
                    "racio": racio,
                    "S_2anos": float(s2[i]),
                    "alerta": bool(np.isfinite(racio) and racio >= LIMIAR_ANOMALIA),
                }
            )
    anomalias = pd.DataFrame(an).sort_values("racio", ascending=False)
    anomalias.to_csv(paths["anomalias"], index=False, encoding="utf-8-sig")

    from src.xgb_survival_utils import preparar_split

    print("A amostrar processos da base de teste (seed=42)…", flush=True)
    _, df_te = preparar_split(df, n=None, random_state=42)
    df_te["processo_id"] = df_te["processo_id"].astype(str)
    chave = ["classe_processual", "orgao_julgador", "ano_ajuizamento"]
    te = df_te.merge(prazos[chave + ["t_previsto_d", "S_2anos"]], on=chave, how="left")
    te = te.merge(percentis, on=["classe_processual", "ano_ajuizamento"], how="left")
    te = te.dropna(subset=["t_previsto_d", "p10"])
    te["faixa"] = [
        classificar_t(r.t_previsto_d, r.p10, r.p25, r.p75, r.p90) for r in te.itertuples(index=False)
    ]
    exemplos = []
    for f in ORDEM_FAIXAS:
        bloco = te.loc[te["faixa"] == f]
        if bloco.empty:
            continue
        usavel = bloco.loc[~bloco["ano_ajuizamento"].isin(ANOS_LEITURA_CAUTELA)]
        fonte = usavel if not usavel.empty else bloco
        exemplos.append(fonte.sample(n=min(N_EXEMPLOS_FAIXA, len(fonte)), random_state=42))
    ex = pd.concat(exemplos, ignore_index=True)
    ex["numero_cnj"] = ex["processo_id"].map(formatar_cnj)
    ex[
        [
            "processo_id",
            "numero_cnj",
            "classe_processual",
            "orgao_julgador",
            "ano_ajuizamento",
            "t_previsto_d",
            "S_2anos",
            "faixa",
        ]
    ].to_csv(paths["exemplos"], index=False, encoding="utf-8-sig")
    print("Artefactos do painel gravados em tabelas/dash_*.csv", flush=True)
    return paths


def carregar_csv(nome: str) -> pd.DataFrame:
    path = dir_tab() / nome
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta {path.name}. Execute o Notebook 6 para gerar os catálogos do painel."
        )
    return pd.read_csv(path)


def prazos_df() -> pd.DataFrame:
    df = carregar_csv("dash_prazos_celulas.csv")
    df["ano_ajuizamento"] = df["ano_ajuizamento"].astype(str)
    return df


def percentis_df() -> pd.DataFrame:
    df = carregar_csv("dash_percentis_classe_ano.csv")
    df["ano_ajuizamento"] = df["ano_ajuizamento"].astype(str)
    return df


def catalogo_df() -> pd.DataFrame:
    return carregar_csv("dash_catalogo_classe_orgao.csv")


def anomalias_df() -> pd.DataFrame:
    df = carregar_csv("dash_anomalias.csv")
    df["ano_ajuizamento"] = df["ano_ajuizamento"].astype(str)
    return df


def orgaos_da_classe(catalogo: pd.DataFrame, classe: str) -> list[str]:
    g = catalogo.loc[catalogo["classe_processual"] == classe].sort_values("n", ascending=False)
    return g["orgao_julgador"].astype(str).tolist()


def n_par(catalogo: pd.DataFrame, classe: str, orgao: str) -> int:
    hit = catalogo.loc[
        (catalogo["classe_processual"] == classe) & (catalogo["orgao_julgador"] == orgao)
    ]
    if hit.empty:
        return 0
    return int(hit["n"].iloc[0])


def faixa_celula(prazos: pd.DataFrame, percentis: pd.DataFrame, classe: str, orgao: str, ano: str):
    row = prazos.loc[
        (prazos["classe_processual"] == classe)
        & (prazos["orgao_julgador"] == orgao)
        & (prazos["ano_ajuizamento"] == str(ano))
    ]
    pc = percentis.loc[
        (percentis["classe_processual"] == classe) & (percentis["ano_ajuizamento"] == str(ano))
    ]
    if row.empty:
        return None
    r = row.iloc[0]
    if pc.empty:
        faixa = "Indeterminada"
    else:
        p = pc.iloc[0]
        faixa = classificar_t(r["t_previsto_d"], p["p10"], p["p25"], p["p75"], p["p90"])
    return r, faixa


def ranking_classe_ano(prazos: pd.DataFrame, classe: str, ano: str, n_min: int) -> pd.DataFrame:
    g = prazos.loc[
        (prazos["classe_processual"] == classe)
        & (prazos["ano_ajuizamento"] == str(ano))
        & (prazos["n"] >= n_min)
    ].copy()
    if g.empty:
        return g
    g["racio"] = racio_vs_restantes(g["t_previsto_d"].to_numpy())
    g = g.sort_values("t_previsto_d", ascending=False)
    g["alerta"] = g["racio"] >= LIMIAR_ANOMALIA
    return g.reset_index(drop=True)


def tendencia_par(prazos: pd.DataFrame, classe: str, orgao: str) -> pd.DataFrame:
    par = prazos.loc[
        (prazos["classe_processual"] == classe) & (prazos["orgao_julgador"] == orgao)
    ].copy()
    par["ano_ajuizamento"] = par["ano_ajuizamento"].astype(str)
    par["cautela"] = par["ano_ajuizamento"].isin(ANOS_LEITURA_CAUTELA)
    par["followup_ok"] = par["ano_ajuizamento"].isin(ANOS_FOLLOWUP_OK)
    return par.sort_values("ano_ajuizamento")
