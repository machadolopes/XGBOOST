"""Gera notebooks/04_curvas_individualizadas.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_curvas_individualizadas.ipynb"


def md(src: str):
    return new_markdown_cell(src.strip() + "\n")


def code(src: str):
    return new_code_cell(src.strip() + "\n")


CELLS = [
    md(
        """# Notebook 4 — Curvas individualizadas, faixas e anomalias

**Dissertação:** previsão, no momento do ajuizamento, da probabilidade de término com **XGBoost Survival (AFT)**.

**Objectivo:** $\\hat S(t\\mid x)$ para perfis-tipo, heterogeneidade de classe e de órgão, evolução por painel anual (Figura 18 — comparação válida entre coortes), faixas de celeridade e anomalias estruturais. Tabelas 14–17; Figuras 15–20.

**Input:** modelo do Notebook 3. **Sem re-treino.**

**Secção da dissertação:** Resultados (curvas individualizadas, heterogeneidade, anomalias)."""
    ),
    md("## 0. Configuração e carregamento do modelo"),
    code(
        """
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from IPython.display import Markdown, display

warnings.filterwarnings("ignore")

HERE = Path.cwd().resolve()
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apa_style import (
    CMAP_HEATMAP,
    apa_figure_title, apa_multipanel_title, exportar_tabela_apa, save_apa_figure, setup_apa_style,
)
from src.data_utils import project_root
from src.eda_utils import classes_por_volume
from src.model_utils import (
    ANO_REF_PERFIS, LIMIAR_ANOMALIA, MIN_N_ORGAO_CLASSE, MIN_N_SEGMENTO_FAIXA,
    ORDEM_FAIXAS, grelha_curvas, tempo_ate_limiar,
)
from src.survival_utils import HORIZONTES_DIAS, HORIZONTES_FIGURA, T_MAX_PLOT_DIAS
from src.xgb_survival_utils import preparar_split, prever_mu_e_S

ROOT = project_root()
DIR_DADOS = ROOT / "dados"
DIR_FIG = ROOT / "figuras"
DIR_TAB = ROOT / "tabelas"
DIR_MOD = ROOT / "modelo"
setup_apa_style()
CORES = sns.color_palette("colorblind")
ANO_REF = ANO_REF_PERFIS
GRELHA = grelha_curvas(t_max=T_MAX_PLOT_DIAS, passo=30)


def exportar_figura(fig, ax, number, title, note, stem):
    apa_figure_title(fig, ax, number, title, note)
    caminho = save_apa_figure(fig, DIR_FIG / f"{stem}.png")
    print(f"PNG 300 dpi: {caminho}")
    plt.show()
    plt.close(fig)
    return caminho


def marcar_horizontes(ax, ymax=1.02):
    for nome in HORIZONTES_FIGURA:
        t = HORIZONTES_DIAS[nome]
        ax.axvline(t, color="#888888", ls="--", lw=0.8, zorder=0)
        ax.text(t, ymax, nome, rotation=90, va="bottom", ha="right", fontsize=8, color="#555555")


def eixo_anos(ax):
    ax.set_xlim(0, T_MAX_PLOT_DIAS)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650])
    ax.set_xticklabels(["0", "1 a", "2 a", "3 a", "4 a", "5 a", "6 a", "7 a", "8 a", "9 a", "10 a"])
    ax.set_xlabel("Tempo desde o ajuizamento (anos civis)")
    ax.set_ylabel("Probabilidade de ainda tramitar, S(t | x)")


def s_em(s, t):
    return float(s[int(np.searchsorted(GRELHA, t))])


meta = json.loads((DIR_MOD / "xgboost_aft_meta.json").read_text(encoding="utf-8"))
DIST = meta["distribuicao"]
SIGMA = float(meta["sigma"])
modelo = xgb.Booster()
modelo.load_model(str(DIR_MOD / "xgboost_aft_modelo_final.json"))
enc = joblib.load(DIR_MOD / "xgboost_aft_encoder.joblib")
print(f"AFT {DIST} | σ={SIGMA:.3f} | árvores={meta['n_arvores']} | C-index teste={meta['c_index_teste']:.3f}")
print("Ano de referência dos perfis isolados:", ANO_REF)
"""
    ),
    md(
        """Reconstitui-se a divisão 80/20 estratificada (`seed=42`) para as curvas medianas por classe e as faixas usarem os mesmos processos da avaliação. O encoder **não** é reajustado."""
    ),
    code(
        """
df_raw = pd.read_parquet(DIR_DADOS / "dataset_limpo.parquet")
print(f"Universo {len(df_raw):,}")
_, df_te = preparar_split(df_raw, n=None, random_state=42)
df = df_raw.copy()
df["ano_ajuizamento"] = df["ano_ajuizamento"].astype(int)
CLASSES_95 = classes_por_volume(df)
print(f"Teste {len(df_te):,} | censura teste {(1 - df_te['evento'].mean()) * 100:.2f} %")
print(f"Classes analíticas (corte de Pareto a 95% + residual): {len(CLASSES_95)}")
print(df_te["classe_processual"].value_counts().reindex(CLASSES_95).to_string())
# Cinco perfis-tipo contrastantes (Figura 15 / Tabela 14), não o recorte das figuras de classe.
PERFIS_TIPO = CLASSES_95[:5]
"""
    ),
    md(
        """## 1. Perfis-tipo contrastantes

Três a cinco combinações reais (classe + órgão modal + ano 2022). Cada curva é **uma** previsão $\\hat S(t\\mid x)$."""
    ),
    code(
        """
def orgao_modal(classe: str) -> str:
    return df.loc[df["classe_processual"] == classe, "orgao_julgador"].value_counts().index[0]

perfis_alvo = []
for classe in PERFIS_TIPO:
    org = orgao_modal(classe)
    n = int(((df["classe_processual"] == classe) & (df["orgao_julgador"] == org)).sum())
    rotulo = f"{classe.split(',')[0][:32]} · {org[:28]}"
    perfis_alvo.append({
        "classe_processual": classe, "orgao_julgador": org,
        "ano_ajuizamento": ANO_REF, "rotulo": rotulo, "n_par_universo": n,
    })
df_perfis = pd.DataFrame(perfis_alvo)
mu_p, S_p = prever_mu_e_S(modelo, enc, df_perfis, GRELHA, SIGMA, DIST)

linhas14 = []
for i, row in df_perfis.iterrows():
    s = S_p[i]
    rec = {
        "perfil": row["rotulo"], "classe": row["classe_processual"],
        "orgao": row["orgao_julgador"], "ano": ANO_REF,
        "mu": round(float(mu_p[i]), 3),
        "t_previsto_d": round(float(np.exp(mu_p[i])), 0),
        "t_mediana_curva_d": tempo_ate_limiar(GRELHA, s, 0.5),
    }
    for nome, t in HORIZONTES_DIAS.items():
        rec[f"S({nome})"] = round(s_em(s, t), 4)
    linhas14.append(rec)
tab14 = pd.DataFrame(linhas14)
tab14["t_mediana_curva_d"] = tab14["t_mediana_curva_d"].round(0)
exportar_tabela_apa(
    tab14.drop(columns=["classe", "orgao"]), 14,
    "S(t | x) nos horizontes académicos para cinco perfis-tipo (ano de ajuizamento = 2022)",
    "Cada linha é um único perfil (classe + órgão modal dessa classe + 2022). "
    f"Distribuição AFT {DIST}: a mediana de T é exp(μ). Não é Kaplan–Meier.",
    DIR_TAB, "tab14_perfis",
)
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(9.2, 5.8))
for i, row in df_perfis.iterrows():
    ax.plot(GRELHA, S_p[i], lw=1.8, color=CORES[i], label=str(row["rotulo"])[:48])
marcar_horizontes(ax)
eixo_anos(ax)
ax.legend(frameon=False, loc="upper right", fontsize=8)
sns.despine()
exportar_figura(
    fig, ax, 15,
    "Curvas de sobrevivência individualizadas para cinco perfis-tipo contrastantes",
    "Ano fixo em 2022. Órgão = o de maior volume da respectiva classe no TRF2. "
    f"XGBoost AFT ({DIST}); as curvas partilham a forma e diferem pelo deslocamento μ(x).",
    "fig15_perfis",
)
"""
    ),
    md(
        """**Interpretação da Figura 15 e da Tabela 14.** O AFT produz curvas deslocadas no tempo para combinações distintas de classe e órgão, com a mesma forma. O recurso inominado cai depressa; a execução fiscal permanece alta em horizontes longos. Isto é o output central da dissertação: uma curva por $x$, não um único valor pontual de duração."""
    ),
    md(
        """## 2. Heterogeneidade entre classes

Curva mediana prevista por classe no conjunto de teste — **todas** as classes do corte de Pareto a 95 %, incluindo «Outras classes»."""
    ),
    code(
        """
S_cls, n_cls, mu_cls = {}, {}, {}
for classe in CLASSES_95:
    sub = df_te.loc[df_te["classe_processual"] == classe]
    mu_c, Sc = prever_mu_e_S(modelo, enc, sub, GRELHA, SIGMA, DIST)
    S_cls[classe] = np.median(Sc, axis=0)
    mu_cls[classe] = float(np.median(mu_c))
    n_cls[classe] = len(sub)
    print(f"{classe[:42]:42s}  teste={len(sub):6d}  μ med={mu_cls[classe]:.3f}")

linhas15 = []
for classe in CLASSES_95:
    s = S_cls[classe]
    rec = {"classe": classe, "n_teste": n_cls[classe],
           "mu_mediano": round(mu_cls[classe], 3),
           "t_previsto_med_d": round(float(np.exp(mu_cls[classe])), 0),
           "t_mediana_curva_d": tempo_ate_limiar(GRELHA, s, 0.5)}
    for nome, t in HORIZONTES_DIAS.items():
        rec[f"S({nome})"] = round(s_em(s, t), 4)
    linhas15.append(rec)
tab15 = pd.DataFrame(linhas15)
tab15["t_mediana_curva_d"] = tab15["t_mediana_curva_d"].round(0)
exportar_tabela_apa(
    tab15, 15,
    "S(t) mediano do XGBoost AFT por classe processual (conjunto de teste)",
    "Mediana pontual das curvas individuais de todos os processos de teste da classe. "
    "Inclui as 16 classes do corte de Pareto a 95 % e o residual «Outras classes». "
    "Órgão e ano não estão fixos nesta figura.",
    DIR_TAB, "tab15_classe",
)
"""
    ),
    code(
        """
n_cls_n = len(CLASSES_95)
n_cols = 4
n_rows = int(np.ceil(n_cls_n / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.4, 10.2), sharex=True, sharey=True)
axes_flat = np.asarray(axes).ravel()
for i, classe in enumerate(CLASSES_95):
    ax = axes_flat[i]
    ax.plot(GRELHA, S_cls[classe], lw=1.7, color=CORES[0])
    marcar_horizontes(ax)
    eixo_anos(ax)
    rotulo = classe if len(classe) <= 34 else classe[:32] + "…"
    ax.set_title(rotulo, fontsize=9, fontstyle="italic", loc="left")
    if i % n_cols != 0:
        ax.set_ylabel("")
    if i < n_cls_n - n_cols:
        ax.set_xlabel("")
for j in range(n_cls_n, len(axes_flat)):
    axes_flat[j].axis("off")
sns.despine(fig=fig)
fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.08, hspace=0.45, wspace=0.18)
apa_multipanel_title(
    fig, 16,
    "Curvas de sobrevivência medianas do XGBoost AFT em todas as classes do corte a 95%",
    "Cada painel é a mediana pontual das previsões de teste da classe (órgãos e anos misturados). "
    "Inclui «Outras classes». A forma é a mesma (AFT normal); o deslocamento é μ(x).",
)
save_apa_figure(fig, DIR_FIG / "fig16_classe.png")
plt.show()
plt.close(fig)
"""
    ),
    md(
        """**Interpretação da Figura 16 e da Tabela 15.** A ordenação das classes no AFT deve separar execução fiscal (lenta) de recursos (rápidos). O AFT suaviza um único molde paramétrico; a diferença face a um KM estratificado é o facto de combinar classe com órgão e ano na mesma previsão (Figuras 17–18)."""
    ),
    md(
        """## 3. Heterogeneidade entre órgãos (classe fixa)

Isola-se o órgão: execução fiscal, ano 2022, os oito órgãos de maior volume nessa classe."""
    ),
    code(
        """
CLASSE_ORGAO = "Execução Fiscal"
if CLASSE_ORGAO not in set(df["classe_processual"]):
    CLASSE_ORGAO = CLASSES_95[0]
orgaos8 = (
    df.loc[df["classe_processual"] == CLASSE_ORGAO, "orgao_julgador"].value_counts().head(8).index.tolist()
)
df_org = pd.DataFrame({
    "classe_processual": CLASSE_ORGAO, "orgao_julgador": orgaos8, "ano_ajuizamento": ANO_REF,
})
mu_org, S_org = prever_mu_e_S(modelo, enc, df_org, GRELHA, SIGMA, DIST)
fig, ax = plt.subplots(figsize=(9.2, 6.0))
for i, nome in enumerate(orgaos8):
    rotulo = nome if len(nome) <= 40 else nome[:38] + "…"
    ax.plot(GRELHA, S_org[i], lw=1.6, color=CORES[i], label=rotulo)
marcar_horizontes(ax)
eixo_anos(ax)
ax.legend(frameon=False, loc="upper right", fontsize=8)
sns.despine()
exportar_figura(
    fig, ax, 17,
    f"Curvas AFT por órgão julgador na classe {CLASSE_ORGAO} (ano {ANO_REF})",
    "Classe e ano fixos; varia só o órgão (os oito de maior volume nessa classe). "
    "Afastamento horizontal = heterogeneidade de localização μ.",
    "fig17_orgao",
)
"""
    ),
    md(
        """**Interpretação da Figura 17.** Com a classe e o ano fixos, as curvas não coincidem. Órgãos do mesmo tribunal e da mesma classe não são intercambiáveis para a previsão no ajuizamento."""
    ),
    md(
        """## 4. Evolução por painel anual (Figura 18)

Classe e órgão fixos; 2015–2025. **Esta é a comparação metodologicamente válida entre coortes** nesta dissertação. Não se lê tendência causal de celeridade: o ano mistura efeito de coorte e truncamento à direita das coortes mais recentes."""
    ),
    code(
        """
pares = df.groupby(["classe_processual", "orgao_julgador"], observed=True).size().sort_values(ascending=False)
classe_ano, orgao_ano = pares.index[0]
print(f"Par fixo: {classe_ano} | {orgao_ano} | n={pares.iloc[0]:,}")
anos = list(range(2015, 2026))
df_anos = pd.DataFrame({
    "classe_processual": classe_ano, "orgao_julgador": orgao_ano, "ano_ajuizamento": anos,
})
mu_anos, S_anos = prever_mu_e_S(modelo, enc, df_anos, GRELHA, SIGMA, DIST)
cores_ano = [CMAP_HEATMAP(x) for x in np.linspace(0.22, 0.95, len(anos))]
fig, ax = plt.subplots(figsize=(9.2, 6.0))
for i, a in enumerate(anos):
    ax.plot(GRELHA, S_anos[i], lw=1.5, color=cores_ano[i], label=str(a))
marcar_horizontes(ax)
eixo_anos(ax)
ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8, title="Ano")
sns.despine()
exportar_figura(
    fig, ax, 18,
    f"Curvas AFT por ano de ajuizamento ({classe_ano[:40]}; órgão fixo)",
    "Classe e órgão fixos. Escala preto–cinza: 2015 (claro) → 2025 (escuro). "
    "Deslocamentos anuais absorvem coorte e truncamento, não uma reforma da justiça. "
    "Confrontar com a Figura EDA-16: uma mediana dos terminados a cair não basta para concluir celeridade.",
    "fig18_ano",
)
"""
    ),
    md(
        """**Interpretação da Figura 18.** O AFT condiciona no ano, por isso as curvas 2015–2025 não coincidem mesmo com classe e órgão fixos. Esse deslocamento mistura mudanças reais e truncatura das coortes recentes. **Não há inferência causal de celeridade.** Esta figura substitui a leitura inválida da mediana dos já terminados (Tabela 2, Figuras 3–4, EDA-16)."""
    ),
    md(
        """## 5. Faixas de celeridade

Cinco faixas sobre $\\exp(\\mu)$, dentro de cada segmento **classe × ano** (p10 / p25 / p75 / p90). μ é constante na célula classe × órgão × ano, por isso os percentis intra-célula seriam degenerados."""
    ),
    code(
        """
print("A prever μ no conjunto de teste...")
mu_te, _ = prever_mu_e_S(modelo, enc, df_te, np.array([730.0]), SIGMA, DIST)
df_te = df_te.copy()
df_te["mu"] = mu_te
df_te["t_previsto"] = np.exp(mu_te)
df_te["grupo_faixa"] = df_te["classe_processual"].astype(str) + " | " + df_te["ano_ajuizamento"].astype(str)

def percentis_grupo(s: pd.Series) -> pd.Series:
    return pd.Series({"p10": s.quantile(0.10), "p25": s.quantile(0.25),
                      "p75": s.quantile(0.75), "p90": s.quantile(0.90), "n": s.size})

p_grp = df_te.groupby("grupo_faixa")["t_previsto"].apply(percentis_grupo).unstack()
p_cls = df_te.groupby("classe_processual")["t_previsto"].apply(percentis_grupo).unstack()
use_grp = df_te["grupo_faixa"].map(p_grp["n"] >= MIN_N_SEGMENTO_FAIXA).astype(bool)
p10 = np.where(use_grp, df_te["grupo_faixa"].map(p_grp["p10"]), df_te["classe_processual"].map(p_cls["p10"]))
p25 = np.where(use_grp, df_te["grupo_faixa"].map(p_grp["p25"]), df_te["classe_processual"].map(p_cls["p25"]))
p75 = np.where(use_grp, df_te["grupo_faixa"].map(p_grp["p75"]), df_te["classe_processual"].map(p_cls["p75"]))
p90 = np.where(use_grp, df_te["grupo_faixa"].map(p_grp["p90"]), df_te["classe_processual"].map(p_cls["p90"]))
tv = df_te["t_previsto"].to_numpy()
faixas = np.full(len(df_te), "Típico", dtype=object)
faixas[tv <= p10] = "Muito rápido"
faixas[(tv > p10) & (tv <= p25)] = "Rápido"
faixas[(tv > p25) & (tv <= p75)] = "Típico"
faixas[(tv > p75) & (tv <= p90)] = "Lento"
faixas[tv > p90] = "Muito lento"
df_te["faixa"] = pd.Categorical(faixas, categories=ORDEM_FAIXAS, ordered=True)
df_te["faixa_ref"] = np.where(use_grp, "classe×ano", "classe")
print(df_te["faixa"].value_counts().reindex(ORDEM_FAIXAS).to_string())

exemplos = []
for f in ORDEM_FAIXAS:
    bloco = df_te.loc[df_te["faixa"] == f]
    if len(bloco) == 0:
        continue
    exemplos.append(bloco.sample(n=min(4, len(bloco)), random_state=42))
tab16 = pd.concat(exemplos, ignore_index=True)
tab16_out = pd.DataFrame({
    "processo_id": tab16["processo_id"].astype(str).str[-20:],
    "classe": tab16["classe_processual"].str[:36],
    "orgao": tab16["orgao_julgador"].str[:32],
    "ano": tab16["ano_ajuizamento"],
    "t_previsto_d": tab16["t_previsto"].round(0),
    "faixa": tab16["faixa"].astype(str),
    "percentis": tab16["faixa_ref"],
})
exportar_tabela_apa(
    tab16_out, 16,
    "Exemplos de classificação em faixas de celeridade (conjunto de teste)",
    "Faixas sobre exp(μ) dentro da classe × ano "
    f"(n ≥ {MIN_N_SEGMENTO_FAIXA}); caso contrário, percentis da classe. "
    "Quatro processos por faixa (random_state=42).",
    DIR_TAB, "tab16_faixas_exemplos",
)
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.2), gridspec_kw={"width_ratios": [1.0, 1.55]})
cnt = df_te["faixa"].value_counts().reindex(ORDEM_FAIXAS)
axes[0].bar(range(len(cnt)), cnt.values, color=CORES[0])
axes[0].set_xticks(range(len(cnt)))
axes[0].set_xticklabels(list(cnt.index), rotation=30, ha="right", fontsize=9)
axes[0].set_ylabel("N.º de processos (teste)")
axes[0].set_title("Global", loc="left", fontsize=11)
ct = pd.crosstab(
    df_te["classe_processual"],
    df_te["faixa"],
    normalize="index",
).reindex(index=list(reversed(CLASSES_95)), columns=ORDEM_FAIXAS) * 100
ct.plot(kind="barh", stacked=True, ax=axes[1], color=sns.color_palette("colorblind", 5), legend=False)
axes[1].set_xlabel("% dentro da classe")
axes[1].set_ylabel("Classe processual")
axes[1].set_yticklabels(
    [c[:28] + ("…" if len(c) > 28 else "") for c in reversed(CLASSES_95)],
    fontsize=8,
)
axes[1].set_title("Por classe (corte a 95%)", loc="left", fontsize=11)
axes[1].legend(ORDEM_FAIXAS, frameon=False, fontsize=7, loc="lower right")
sns.despine(fig=fig)
apa_multipanel_title(
    fig, 19,
    "Distribuição pelas faixas de celeridade no conjunto de teste",
    "A faixa é relativa à classe×ano: «típico» na execução fiscal não é o mesmo prazo que «típico» no recurso inominado. "
    "O painel direito inclui as 16 classes do corte de Pareto a 95 % e o residual «Outras classes».",
)
save_apa_figure(fig, DIR_FIG / "fig19_faixas.png")
plt.show()
plt.close(fig)
"""
    ),
    md(
        """**Interpretação da Tabela 16 e da Figura 19.** As faixas são relativas, não absolutas em dias. Um processo «muito lento» no juizado pode ter $\\exp(\\mu)$ inferior ao de um «típico» em execução fiscal."""
    ),
    md(
        """## 6. Anomalias estruturais (rácio ≥ 2)

Pares classe × órgão com rácio ≥ 2 face à mediana dos restantes órgãos da mesma classe, ano de referência 2022, n mínimo 80 no par ao longo da série. Interpretação: diagnóstico de fluxo, **não** juízo disciplinar. Nomes completos dos órgãos na Tabela 17."""
    ),
    code(
        """
linhas_an = []
for classe in CLASSES_95:
    vc = df.loc[df["classe_processual"] == classe, "orgao_julgador"].value_counts()
    orgs = vc[vc >= MIN_N_ORGAO_CLASSE].index.tolist()
    if len(orgs) < 3:
        continue
    dfo = pd.DataFrame({"classe_processual": classe, "orgao_julgador": orgs, "ano_ajuizamento": ANO_REF})
    mu_o, So = prever_mu_e_S(modelo, enc, dfo, GRELHA, SIGMA, DIST)
    tmed = np.exp(mu_o)
    s2 = So[:, int(np.searchsorted(GRELHA, 730))]
    for i, org in enumerate(orgs):
        outros = tmed[np.arange(len(orgs)) != i]
        outros = outros[np.isfinite(outros)]
        t_rest = float(np.median(outros)) if outros.size else np.nan
        t_i = float(tmed[i])
        racio = t_i / t_rest if (np.isfinite(t_i) and np.isfinite(t_rest) and t_rest > 0) else np.nan
        linhas_an.append({
            "classe": classe, "orgao": org, "n_classe_orgao": int(vc[org]),
            "t_mediano_orgao_d": t_i, "t_mediano_restantes_d": t_rest,
            "racio": racio, "S(2 anos)": round(float(s2[i]), 4),
            "anomalia": bool(np.isfinite(racio) and racio >= LIMIAR_ANOMALIA),
        })
tab_an = pd.DataFrame(linhas_an)
n_an = int(tab_an["anomalia"].sum()) if len(tab_an) else 0
print(f"Anomalias (rácio ≥ {LIMIAR_ANOMALIA}): {n_an}")
tab17 = tab_an.loc[tab_an["anomalia"]].sort_values("racio", ascending=False).copy()
if tab17.empty:
    tab17 = tab_an.sort_values("racio", ascending=False).head(15).copy()
    nota17 = "Nenhum órgão atingiu rácio ≥ 2. Mostram-se os 15 maiores rácios. O limiar permanece 2."
else:
    nota17 = (
        "Tempo mediano = exp(μ) do perfil (classe, órgão, 2022). "
        "Restantes = mediana dos outros órgãos da mesma classe. "
        f"Limiar: rácio ≥ {LIMIAR_ANOMALIA}. n mínimo: {MIN_N_ORGAO_CLASSE}. "
        "Diagnóstico de fluxo, não juízo disciplinar. Nomes completos dos órgãos."
    )
tab17_out = tab17[["orgao", "classe", "n_classe_orgao", "t_mediano_orgao_d",
                   "t_mediano_restantes_d", "racio", "S(2 anos)"]].copy()
for c in ["t_mediano_orgao_d", "t_mediano_restantes_d"]:
    tab17_out[c] = tab17_out[c].round(0)
tab17_out["racio"] = tab17_out["racio"].round(2)
exportar_tabela_apa(tab17_out, 17, "Órgãos com tempo mediano AFT desproporcionado na mesma classe (TRF2)",
                    nota17, DIR_TAB, "tab17_anomalias")
tab_an.to_csv(DIR_TAB / "tab17_anomalias_completo.csv", index=False, encoding="utf-8-sig")
"""
    ),
    code(
        """
plot_df = tab_an.dropna(subset=["t_mediano_orgao_d", "t_mediano_restantes_d"]).copy()
fig, ax = plt.subplots(figsize=(7.8, 7.2))
marcadores = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "p", "h", "8", "d", "H", "+"]
for i, classe in enumerate(CLASSES_95):
    sub = plot_df.loc[plot_df["classe"] == classe]
    if sub.empty:
        continue
    ax.scatter(
        sub["t_mediano_restantes_d"],
        sub["t_mediano_orgao_d"],
        s=26,
        color=CORES[i % len(CORES)],
        marker=marcadores[i % len(marcadores)],
        alpha=0.85,
        label=classe[:30],
    )
mx = max(plot_df["t_mediano_restantes_d"].max(), plot_df["t_mediano_orgao_d"].max())
xx = np.linspace(0, mx * 1.05, 50)
ax.plot(xx, xx, ls="--", color="#888888", lw=1.0, label="Igualdade")
ax.plot(xx, 2 * xx, ls=":", color=CORES[3], lw=1.2, label="Rácio = 2")
ax.set_xlabel("Tempo mediano dos restantes órgãos da classe (dias)")
ax.set_ylabel("Tempo mediano do órgão (dias)")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=1)
sns.despine()
exportar_figura(
    fig, ax, 20,
    "Tempo mediano AFT por órgão versus mediana dos restantes órgãos da mesma classe",
    "Cada ponto é um par classe × órgão (n ≥ 80). Ano do perfil = 2022. "
    "Pontos acima da linha pontilhada têm rácio ≥ 2 (Tabela 17).",
    "fig20_anomalias",
)
"""
    ),
    md(
        """**Interpretação da Tabela 17 e da Figura 20.** Um rácio ≥ 2, com classe e ano fixos, é o critério pré-especificado de anomalia estrutural. Não é um juízo disciplinar: o modelo não vê recursos humanos, acervo legado nem qualidade da capa. É um alerta relativo de fluxo.

## Síntese

A classe ordena a celeridade (Figura 16); o órgão desloca $S(t)$ dentro da classe (Figura 17); o ano desloca coortes (Figura 18) sem autorização causal. As faixas (Figura 19) traduzem $\\exp(\\mu)$ num rótulo comunicável, relativo à classe×ano. As anomalias (Figura 20) são o extremo dessa lógica. A forma partilhada da AFT é a contrapartida de ter treinado o universo."""
    ),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbformat.write(nb, OUT)
    print(f"Escrito {OUT} ({len(CELLS)} células)")


if __name__ == "__main__":
    main()
