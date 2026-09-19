"""Gera notebooks/02_eda_exaustiva.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_eda_exaustiva.ipynb"


def md(src: str):
    return new_markdown_cell(src.strip() + "\n")


def code(src: str):
    return new_code_cell(src.strip() + "\n")


CELLS = [
    md(
        """# Notebook 2 — Análise exploratória exaustiva

**Dissertação:** previsão, no momento do ajuizamento, da probabilidade de término (XGBoost Survival AFT).

**Objectivo:** descrever univariadas, bivariadas, associações e qualidade dos dados sobre `dados/dataset_limpo.parquet` (Notebook 1). Figuras **EDA-1 a EDA-25** e Tabelas **EDA-1 a EDA-13**.

**Secção da dissertação:** Resultados (EDA) e Metodologia (aviso de censura por coorte).

**Aviso que se repete.** A mediana do tempo *só nos terminados* **não** prova melhoria de celeridade nas coortes recentes. A leitura válida entre anos é a Figura 18 do Notebook 4."""
    ),
    md("## 0. Configuração e carregamento"),
    code(
        """
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

HERE = Path.cwd().resolve()
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apa_style import (
    apa_figure_title,
    apa_heatmap,
    apa_multipanel_title,
    exportar_tabela_apa,
    fmt_milhares,
    save_apa_figure,
    setup_apa_style,
)
from src.data_utils import project_root
import importlib
import src.eda_utils as _eda_utils
importlib.reload(_eda_utils)
from src.eda_utils import (
    classificar_faixa_tempo,
    classes_por_volume,
    cramers_v,
    encoding_frequencia,
    ic_mediana,
    percentagens_que_somam_100,
)

setup_apa_style()
CORES = sns.color_palette("colorblind")
ROOT = project_root()
DIR_FIG = ROOT / "figuras"
DIR_TAB = ROOT / "tabelas"
df = pd.read_parquet(ROOT / "dados" / "dataset_limpo.parquet")
df["ano_ajuizamento"] = df["ano_ajuizamento"].astype(int)
CLASSES_95 = classes_por_volume(df)
print(df.shape)
print(df.dtypes)
print("censura %", round(100 * (1 - df["evento"].mean()), 2))
print(f"Classes analíticas (top 95% + residual): {len(CLASSES_95)}")
print(df.head(3))
"""
    ),
    md(
        """## 1. Univariadas

### Tempo — Figuras EDA-1 a EDA-4 e Tabela EDA-1"""
    ),
    code(
        """
term = df.loc[df["evento"] == 1, "tempo"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(term.clip(upper=term.quantile(0.99)), bins=70, color=CORES[0], edgecolor="white")
ax.axvline(term.median(), color=CORES[3], ls="--", lw=1.6, label=f"Mediana = {term.median():.0f} d")
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Número de processos")
fmt_milhares(ax, "y")
ax.legend(frameon=False)
apa_figure_title(fig, ax, "EDA-1", "Histograma do tempo até ao término (escala linear)",
    "Apenas processos terminados. Truncado visualmente no p99.")
save_apa_figure(fig, DIR_FIG / "fig_eda01_hist_tempo_linear.png", close=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(np.log10(term.clip(lower=1)), bins=70, color=CORES[0], edgecolor="white")
ax.set_xlabel("log10 do tempo (dias)")
ax.set_ylabel("Número de processos")
fmt_milhares(ax, "y")
apa_figure_title(fig, ax, "EDA-2", "Histograma do tempo até ao término (escala log10)",
    "A escala logarítmica evidencia a massa entre 2 e 3 (100–1000 dias).")
save_apa_figure(fig, DIR_FIG / "fig_eda02_hist_tempo_log10.png", close=True)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.sort(term.values)
y = np.arange(1, len(x) + 1) / len(x)
ax.plot(x, y, color=CORES[0], lw=1.6)
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("F(t) empírica")
ax.set_xlim(0, float(term.quantile(0.99)))
apa_figure_title(fig, ax, "EDA-3", "Função de distribuição empírica do tempo (terminados)",
    "ECDF dos eventos. Não trata censura; para S(t) usa-se o AFT (Notebook 3).")
save_apa_figure(fig, DIR_FIG / "fig_eda03_ecdf_tempo.png", close=True)

fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(y=term.clip(upper=term.quantile(0.99)), color=CORES[0], ax=ax, cut=0)
ax.set_ylabel("Tempo (dias)")
ax.set_xlabel("")
apa_figure_title(fig, ax, "EDA-4", "Violino do tempo até ao término (terminados)",
    "A largura é densidade. Truncado no p99.")
save_apa_figure(fig, DIR_FIG / "fig_eda04_violin_tempo.png", close=True)

pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
tab_eda1 = pd.DataFrame({"percentil": pcts, "tempo_dias": [round(float(term.quantile(p / 100)), 1) for p in pcts]})
exportar_tabela_apa(tab_eda1, "EDA-1", "Percentis do tempo até ao término (processos com evento)",
    "Não inclui censurados.", DIR_TAB, "tab_eda01_percentis_tempo")
"""
    ),
    md("**Interpretação (tempo).** A distribuição é assimétrica: a mediana dos terminados cai na casa das centenas de dias, o p90 e o p99 documentam a cauda longa. Um único valor esperado de duração ocultaria esta heterogeneidade — daí $S(t)$."),
    md("### Classe — Figuras EDA-5, EDA-6 e Tabela EDA-2"),
    code(
        """
freq = df["classe_processual"].value_counts()
tab_eda2 = pd.DataFrame({
    "classe_processual": freq.index, "n": freq.values,
    "percentagem": percentagens_que_somam_100(freq.values),
})
tab_eda2["percentagem_acumulada"] = tab_eda2["percentagem"].cumsum().round(2)
exportar_tabela_apa(tab_eda2, "EDA-2", "Frequência das classes processuais (após agregação do residual)",
    "«Outras classes» é o residual do corte de Pareto a 95% (Notebook 1). "
    "Percentagens a duas casas pelo método dos maiores restos, para o total ser 100,00 %.",
    DIR_TAB, "tab_eda02_classes_completa")

fig, ax = plt.subplots(figsize=(8, 7))
ord_ = list(reversed(CLASSES_95))
ax.barh(ord_, freq.loc[ord_].values, color=CORES[0])
ax.set_xlabel("Número de processos")
ax.set_ylabel("Classe processual")
fmt_milhares(ax, "x")
apa_figure_title(fig, ax, "EDA-5", "Volume por classe processual",
    "Todas as classes do corte de Pareto a 95%, incluindo o residual «Outras classes».")
save_apa_figure(fig, DIR_FIG / "fig_eda05_barras_classes.png", close=True)

acum = tab_eda2["percentagem_acumulada"].values
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(tab_eda2)), tab_eda2["n"], color=CORES[0])
ax2 = ax.twinx()
ax2.plot(range(len(tab_eda2)), acum, color=CORES[3], marker="o", lw=1.5)
ax2.axhline(95, color=CORES[2], ls="--", lw=1.2, label="95%")
ax2.set_ylabel("% acumulada")
ax.set_ylabel("Número de processos")
fmt_milhares(ax, "y")
ax.set_xlabel("Classes ordenadas por volume")
ax.set_xticks([])
ax2.legend(frameon=False)
apa_figure_title(fig, ax, "EDA-6", "Curva de Pareto das classes processuais (corte 95%)",
    "A linha tracejada marca o limiar usado para agregar «Outras classes».")
save_apa_figure(fig, DIR_FIG / "fig_eda06_pareto_classes.png", close=True)
"""
    ),
    md("### Órgão — Figuras EDA-7, EDA-8 e Tabela EDA-3"),
    code(
        """
vol = df["orgao_julgador"].value_counts()
tab_eda3 = pd.DataFrame({
    "estatistica": ["n órgãos", "volume mínimo", "p25", "mediana", "p75", "máximo", "média"],
    "valor": [len(vol), int(vol.min()), int(vol.quantile(0.25)), int(vol.median()),
              int(vol.quantile(0.75)), int(vol.max()), round(float(vol.mean()), 1)],
})
exportar_tabela_apa(tab_eda3, "EDA-3", "Distribuição do volume de processos por órgão julgador",
    "Cada órgão é uma unidade de julgamento do TRF2 (todos os graus).", DIR_TAB, "tab_eda03_volume_orgaos")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(vol.values, bins=40, color=CORES[0], edgecolor="white")
ax.set_xlabel("Processos por órgão")
ax.set_ylabel("Número de órgãos")
apa_figure_title(fig, ax, "EDA-7", "Histograma do volume por órgão julgador",
    "Muitos órgãos de volume moderado; poucos de volume muito alto.")
save_apa_figure(fig, DIR_FIG / "fig_eda07_hist_volume_orgaos.png", close=True)

top30 = vol.head(30).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top30.index, top30.values, color=CORES[0])
ax.set_xlabel("Número de processos")
fmt_milhares(ax, "x")
ax.tick_params(axis="y", labelsize=8)
apa_figure_title(fig, ax, "EDA-8", "Os 30 órgãos julgadores com maior volume",
    "Nenhum órgão concentra a litigância; a heterogeneidade institucional é estimável.")
save_apa_figure(fig, DIR_FIG / "fig_eda08_top30_orgaos.png", close=True)
"""
    ),
    md(
        """### Ano — Figura EDA-9 e Tabela EDA-4

> **Aviso metodológico.** A percentagem de censura e a mediana dos *terminados* por ano **não** autorizam a conclusão de aceleração da justiça. Nas coortes de 2024–2025 o follow-up é curto: só os processos rápidos já terminaram. A comparação válida é a Figura 18 (Notebook 4)."""
    ),
    code(
        """
tab_eda4 = (
    df.groupby("ano_ajuizamento")
    .agg(n=("processo_id", "size"), n_terminados=("evento", "sum"))
    .reset_index()
)
tab_eda4["n_censurados"] = tab_eda4["n"] - tab_eda4["n_terminados"]
tab_eda4["pct_censura"] = (100 * tab_eda4["n_censurados"] / tab_eda4["n"]).round(2)
med = df.loc[df["evento"] == 1].groupby("ano_ajuizamento")["tempo"].median()
tab_eda4 = tab_eda4.merge(med.rename("mediana_terminados"), on="ano_ajuizamento")
tab_eda4["mediana_terminados"] = tab_eda4["mediana_terminados"].round(1)
exportar_tabela_apa(tab_eda4, "EDA-4", "Volume, censura e mediana dos terminados por ano de ajuizamento",
    "A mediana dos terminados sofre viés de censura nas coortes recentes.", DIR_TAB, "tab_eda04_por_ano")

stack = df.groupby(["ano_ajuizamento", "evento"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(stack.index, stack.get(1, 0), color=CORES[0], label="Terminados")
ax.bar(stack.index, stack.get(0, 0), bottom=stack.get(1, 0), color=CORES[1], label="Censurados")
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Número de processos")
fmt_milhares(ax, "y")
ax.legend(frameon=False)
apa_figure_title(fig, ax, "EDA-9", "Terminados e censurados por coorte anual",
    "A fatia de censura em 2024–2025 é truncamento, não prova de mudança de celeridade.")
save_apa_figure(fig, DIR_FIG / "fig_eda09_ano_empilhado.png", close=True)
"""
    ),
    md("## 2. Bivariadas"),
    md("### Classe × evento — Tabela EDA-5 e Figura EDA-10"),
    code(
        """
ct = pd.crosstab(df["classe_processual"], df["evento"], margins=True)
ct.columns = ["censurados", "terminados", "total"] if list(ct.columns)[:2] == [0, 1] else ct.columns
pct_ev = pd.crosstab(df["classe_processual"], df["evento"], normalize="index") * 100
tab_eda5 = pct_ev.rename(columns={0: "pct_censura", 1: "pct_evento"}).round(2)
tab_eda5["n"] = df.groupby("classe_processual").size()
tab_eda5 = tab_eda5.reset_index()
exportar_tabela_apa(tab_eda5, "EDA-5", "Classe processual × evento (percentagens em linha e n)",
    "Percentagem de censura por classe. Execuções tendem a censura mais alta.", DIR_TAB, "tab_eda05_classe_evento")

fig, ax = plt.subplots(figsize=(8, 7))
pct_plot = pct_ev.reindex(CLASSES_95)[[0, 1]]
pct_plot.plot(kind="barh", stacked=True, color=[CORES[1], CORES[0]], ax=ax, width=0.8)
ax.set_xlabel("Percentagem")
ax.set_ylabel("Classe processual")
ax.legend(["Censurados", "Terminados"], frameon=False)
apa_figure_title(fig, ax, "EDA-10", "Composição de evento e censura por classe (100% empilhado)",
    "Todas as classes do corte a 95%. Barras comparáveis independentemente do volume.")
save_apa_figure(fig, DIR_FIG / "fig_eda10_classe_evento_stacked.png", close=True)
"""
    ),
    md("### Classe × tempo — Figuras EDA-11, EDA-12 e Tabela EDA-6"),
    code(
        """
term_df = df.loc[df["evento"] == 1]
desc = term_df.groupby("classe_processual")["tempo"].agg(["count", "mean", "median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
desc.columns = ["n_terminados", "media", "mediana", "Q1", "Q3"]
tab_eda6 = desc.round(1).reset_index()
exportar_tabela_apa(tab_eda6, "EDA-6", "Tempo até ao término por classe (apenas eventos)",
    "Médias e quartis dos terminados; não corrige censura.", DIR_TAB, "tab_eda06_tempo_por_classe")

ordem = CLASSES_95
fig, ax = plt.subplots(figsize=(8, 7))
sns.boxplot(data=term_df, y="classe_processual", x="tempo", order=ordem, showfliers=False, color=CORES[0], ax=ax)
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Classe processual")
apa_figure_title(fig, ax, "EDA-11", "Boxplot do tempo até ao término por classe",
    "Todas as classes do corte a 95%, incluindo «Outras classes»; sem outliers. Só terminados.")
save_apa_figure(fig, DIR_FIG / "fig_eda11_box_tempo_classe.png", close=True)

fig, ax = plt.subplots(figsize=(8, 7))
sns.violinplot(data=term_df, y="classe_processual", x="tempo", order=CLASSES_95, cut=0, color=CORES[0], ax=ax)
ax.set_xlim(0, float(term_df["tempo"].quantile(0.98)))
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Classe processual")
apa_figure_title(fig, ax, "EDA-12", "Violino do tempo nas classes do corte de Pareto a 95%",
    "Densidade por classe (terminados), incluindo o residual. Escala truncada no p98.")
save_apa_figure(fig, DIR_FIG / "fig_eda12_violin_classes.png", close=True)
"""
    ),
    md("### Órgão × tempo — Figuras EDA-13, EDA-14 e Tabela EDA-7"),
    code(
        """
top20 = df["orgao_julgador"].value_counts().head(20).index.tolist()
sub = term_df.loc[term_df["orgao_julgador"].isin(top20)]
fig, ax = plt.subplots(figsize=(8, 7))
sns.boxplot(data=sub, y="orgao_julgador", x="tempo", order=top20, showfliers=False, color=CORES[0], ax=ax)
ax.set_xlabel("Tempo (dias)")
ax.tick_params(axis="y", labelsize=8)
apa_figure_title(fig, ax, "EDA-13", "Tempo até ao término nos 20 órgãos de maior volume",
    "Só terminados, sem outliers. Diferenças de localização entre órgãos grandes.")
save_apa_figure(fig, DIR_FIG / "fig_eda13_box_orgao_top20.png", close=True)

med_o = term_df.groupby("orgao_julgador")["tempo"].median()
n_o = df.groupby("orgao_julgador").size()
tab_eda7 = pd.DataFrame({"orgao_julgador": n_o.index, "n": n_o.values}).merge(
    med_o.rename("mediana_terminados"), on="orgao_julgador", how="left"
)
tab_eda7 = tab_eda7.sort_values("n", ascending=False).head(30)
tab_eda7["mediana_terminados"] = tab_eda7["mediana_terminados"].round(1)
exportar_tabela_apa(tab_eda7, "EDA-7", "Volume e mediana do tempo (terminados) nos 30 órgãos de maior volume",
    "Mediana condicionada a evento=1; órgãos pequenos omitidos nesta tabela.", DIR_TAB, "tab_eda07_top30_orgaos")

plot_o = pd.DataFrame({"n": n_o, "mediana": med_o}).dropna()
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(plot_o["n"], plot_o["mediana"], s=18, alpha=0.6, color=CORES[0])
ax.axhline(term.median(), color=CORES[3], ls="--", lw=1.2, label="Mediana global (terminados)")
ax.set_xlabel("Volume do órgão (n processos)")
ax.set_ylabel("Mediana do tempo (dias)")
fmt_milhares(ax, "x")
ax.legend(frameon=False)
apa_figure_title(fig, ax, "EDA-14", "Mediana do tempo (terminados) versus volume do órgão",
    "Cada ponto é um órgão. A dispersão vertical entre órgãos grandes fundamenta a covariável órgão.")
save_apa_figure(fig, DIR_FIG / "fig_eda14_scatter_mediana_volume.png", close=True)
"""
    ),
    md(
        """### Ano × tempo — Figuras EDA-15 e EDA-16

> **Não concluir «melhoria de celeridade»** a partir da queda da mediana nas coortes recentes. A leitura válida é a do Notebook 4 (Figura 18)."""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=term_df, x="ano_ajuizamento", y="tempo", showfliers=False, color=CORES[0], ax=ax)
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Tempo (dias)")
apa_figure_title(fig, ax, "EDA-15", "Tempo até ao término por ano (só eventos observados)",
    "A queda recente da caixa é compatível com truncamento: os lentos de 2024–2025 ainda não terminaram.")
save_apa_figure(fig, DIR_FIG / "fig_eda15_box_tempo_ano.png", close=True)

linhas = []
for ano, g in term_df.groupby("ano_ajuizamento"):
    med, lo, hi = ic_mediana(g["tempo"])
    linhas.append({"ano": int(ano), "mediana": med, "ic_inf": lo, "ic_sup": hi, "n": len(g)})
ic_ano = pd.DataFrame(linhas)
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(ic_ano["ano"], ic_ano["mediana"],
            yerr=[ic_ano["mediana"] - ic_ano["ic_inf"], ic_ano["ic_sup"] - ic_ano["mediana"]],
            fmt="o-", color=CORES[0], capsize=3)
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Mediana do tempo (dias)")
apa_figure_title(fig, ax, "EDA-16", "Mediana do tempo dos terminados e intervalo de confiança, por ano",
    "IC não paramétrico da mediana. A descida recente NÃO é evidência de aceleração (viés de censura). "
    "Comparação válida: Figura 18, Notebook 4.")
save_apa_figure(fig, DIR_FIG / "fig_eda16_mediana_ic_ano.png", close=True)
"""
    ),
    md("**Interpretação (ano × tempo).** A Figura EDA-16 mostra medianas decrescentes nas coortes recentes *entre quem já terminou*. Isso é o viés clássico de amostrar os rápidos. Não se interpreta como reforma da justiça."),
    md("### Classe × ano — Tabela EDA-8 e Figura EDA-17"),
    code(
        """
tab_eda8 = pd.crosstab(df["classe_processual"], df["ano_ajuizamento"]).reindex(CLASSES_95)
tab_eda8.to_csv(DIR_TAB / "tab_eda08_classe_ano.csv", encoding="utf-8-sig")
exportar_tabela_apa(tab_eda8.reset_index(), "EDA-8", "Volume de processos por classe e ano de ajuizamento",
    "Contagens absolutas. O mix de classes pode mudar ao longo do painel.", DIR_TAB, "tab_eda08_classe_ano")
heat_vol = np.log1p(tab_eda8)
heat_vol.columns = [str(int(c)) for c in heat_vol.columns]
fig, ax = plt.subplots(figsize=(9, 7))
apa_heatmap(heat_vol, ax, cbar_label="log(1 + n)", xtick_rotation=45, ytick_size=9)
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Classe processual")
apa_figure_title(fig, ax, "EDA-17", "Mapa de calor do volume classe × ano (escala log)",
    "Escala preto–cinza. A intensidade é log(1+n) para não esmagar classes de menor volume.")
save_apa_figure(fig, DIR_FIG / "fig_eda17_heatmap_classe_ano.png", close=True)
"""
    ),
    md("### Órgão × classe e órgão × ano — Tabela EDA-9, Figuras EDA-18 e EDA-19"),
    code(
        """
top15o = df["orgao_julgador"].value_counts().head(15).index
ct_oc = pd.crosstab(df.loc[df["orgao_julgador"].isin(top15o), "orgao_julgador"],
                    df.loc[df["orgao_julgador"].isin(top15o), "classe_processual"])
ct_oc = ct_oc.reindex(index=list(top15o), columns=CLASSES_95, fill_value=0)
tab_eda9 = ct_oc.reset_index()
exportar_tabela_apa(tab_eda9, "EDA-9", "Volume órgão × classe (15 órgãos de maior volume)",
    "Recorte dos 15 órgãos mais volumosos; colunas = classes do corte a 95%.", DIR_TAB, "tab_eda09_orgao_classe")
fig, ax = plt.subplots(figsize=(10, 7))
apa_heatmap(np.log1p(ct_oc), ax, cbar_label="log(1 + n)", xtick_rotation=90, ytick_size=8, xtick_size=8)
ax.set_xlabel("Classe processual")
ax.set_ylabel("Órgão julgador")
apa_figure_title(fig, ax, "EDA-18", "Mapa de calor órgão × classe (top 15 órgãos; classes do corte a 95%)",
    "Escala preto–cinza. Alguns órgãos especializam-se em classes; outros são generalistas.")
save_apa_figure(fig, DIR_FIG / "fig_eda18_heatmap_orgao_classe.png", close=True)

ct_oa = pd.crosstab(df.loc[df["orgao_julgador"].isin(top15o), "orgao_julgador"],
                    df.loc[df["orgao_julgador"].isin(top15o), "ano_ajuizamento"])
ct_oa = ct_oa.reindex(list(top15o))
ct_oa.columns = [str(int(c)) for c in ct_oa.columns]
fig, ax = plt.subplots(figsize=(9, 7))
apa_heatmap(np.log1p(ct_oa), ax, cbar_label="log(1 + n)", xtick_rotation=45, ytick_size=8)
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Órgão julgador")
apa_figure_title(fig, ax, "EDA-19", "Mapa de calor do volume órgão × ano (top 15 órgãos)",
    "Escala preto–cinza. Permite ver se o crescimento de 2021 se espalhou ou se concentrou em unidades.")
save_apa_figure(fig, DIR_FIG / "fig_eda19_heatmap_orgao_ano.png", close=True)
"""
    ),
    md("## 3. Correlações e associações"),
    code(
        """
ano_n = df["ano_ajuizamento"].astype(float)
tempo = df["tempo"].astype(float)
evento = df["evento"].astype(float)
cls_f = encoding_frequencia(df["classe_processual"])
org_f = encoding_frequencia(df["orgao_julgador"])
mat = pd.DataFrame({"tempo": tempo, "evento": evento, "ano": ano_n, "classe_freq": cls_f, "orgao_freq": org_f})
pear = mat.corr(method="pearson")
spear = mat.corr(method="spearman")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
apa_heatmap(pear, axes[0], cbar_label="r", vmin=-1, vmax=1, annot=True, fmt=".2f",
            diverging=True, xtick_rotation=45, square=True)
axes[0].set_title("Pearson", fontstyle="italic")
apa_heatmap(spear, axes[1], cbar_label="ρ", vmin=-1, vmax=1, annot=True, fmt=".2f",
            diverging=True, xtick_rotation=45, square=True)
axes[1].set_title("Spearman", fontstyle="italic")
apa_multipanel_title(fig, "EDA-20",
    "Matrizes de correlação (Pearson e Spearman) entre tempo, evento, ano e frequências das categorias",
    "classe_freq e orgao_freq são a frequência relativa da categoria (não o one-hot). "
    "A correlação negativa ano–tempo nos terminados mistura truncamento e eventual mudança real.")
save_apa_figure(fig, DIR_FIG / "fig_eda20_correlacoes.png", close=True)

pares = [
    ("classe_processual", "orgao_julgador"),
    ("classe_processual", "ano_ajuizamento"),
    ("orgao_julgador", "ano_ajuizamento"),
]
linhas_v = []
for a, b in pares:
    v, chi2, gl, p = cramers_v(df[a].astype(str), df[b].astype(str))
    linhas_v.append({"par": f"{a} × {b}", "Cramers_V": round(v, 3), "chi2": round(chi2, 1), "gl": gl, "p": "< .001" if p < 0.001 else f"{p:.3f}"})
tab_eda10 = pd.DataFrame(linhas_v)
exportar_tabela_apa(tab_eda10, "EDA-10", "Cramér's V entre as três covariáveis categóricas",
    "V próximo de 0 = associação fraca. Classe × órgão costuma ser a associação mais forte (especialização).",
    DIR_TAB, "tab_eda10_cramer_v")

linhas_q = []
for col in ("classe_processual", "ano_ajuizamento"):
    ct = pd.crosstab(df[col], df["evento"])
    chi2, p, gl, _ = chi2_contingency(ct)
    linhas_q.append({"teste": f"evento × {col}", "chi2": round(chi2, 1), "gl": int(gl),
                     "p": "< .001" if p < 0.001 else f"{p:.3f}"})
tab_eda11 = pd.DataFrame(linhas_q)
exportar_tabela_apa(tab_eda11, "EDA-11", "Testes de independência qui-quadrado (evento × classe e evento × ano)",
    "Rejeitar independência não quantifica a magnitude; ver Cramér's V e as taxas de censura por classe.",
    DIR_TAB, "tab_eda11_qui_quadrado_evento")
"""
    ),
    md("## 4. Síntese visual — Figuras EDA-21 a EDA-24"),
    code(
        """
med_ca = term_df.pivot_table(index="classe_processual", columns="ano_ajuizamento", values="tempo", aggfunc="median")
med_ca = med_ca.reindex(CLASSES_95)
med_ca.columns = [str(int(c)) for c in med_ca.columns]
fig, ax = plt.subplots(figsize=(9, 7))
apa_heatmap(med_ca, ax, cbar_label="Mediana do tempo (dias)", xtick_rotation=45, ytick_size=9)
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Classe processual")
apa_figure_title(fig, ax, "EDA-21", "Mediana do tempo (terminados) por classe e ano",
    "Escala preto–cinza. Células recentes enviesadas para processos rápidos. Não ler como aceleração.")
save_apa_figure(fig, DIR_FIG / "fig_eda21_heatmap_mediana_classe_ano.png", close=True)

cens_ca = df.pivot_table(index="classe_processual", columns="ano_ajuizamento", values="evento",
                         aggfunc=lambda s: 100 * (1 - s.mean()))
cens_ca = cens_ca.reindex(CLASSES_95)
cens_ca.columns = [str(int(c)) for c in cens_ca.columns]
fig, ax = plt.subplots(figsize=(9, 7))
apa_heatmap(cens_ca, ax, cbar_label="% de censura", xtick_rotation=45, ytick_size=9)
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Classe processual")
apa_figure_title(fig, ax, "EDA-22", "Percentagem de censura por classe e ano",
    "Escala preto–cinza. A censura sobe em 2024–2025 em quase todas as classes (follow-up curto).")
save_apa_figure(fig, DIR_FIG / "fig_eda22_heatmap_censura_classe_ano.png", close=True)

df["faixa_t"] = classificar_faixa_tempo(df["tempo"])
area = df.groupby(["faixa_t", "evento"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(area))
ax.stackplot(x, area.get(1, 0), area.get(0, 0), labels=["Terminados", "Censurados"],
             colors=[CORES[0], CORES[1]])
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in area.index], rotation=20, ha="right")
ax.set_ylabel("Número de processos")
ax.set_xlabel("Faixa de tempo desde o ajuizamento")
ax.legend(frameon=False)
apa_figure_title(fig, ax, "EDA-23", "Terminados e censurados segundo a faixa de tempo observado",
    "Nas faixas longas aumenta a quota de censura: quem ainda tramita só é observado se o tempo já for longo.")
save_apa_figure(fig, DIR_FIG / "fig_eda23_area_faixas_tempo.png", close=True)

n_cls = len(CLASSES_95)
n_rows = int(np.ceil(n_cls / 2))
fig, axes = plt.subplots(n_rows, 6, figsize=(14, max(12.0, 1.7 * n_rows)))
for i, cls in enumerate(CLASSES_95):
    bloco = 0 if i < n_rows else 3
    linha = i if i < n_rows else i - n_rows
    ax_t, ax_c, ax_v = axes[linha, bloco], axes[linha, bloco + 1], axes[linha, bloco + 2]
    g = df.loc[df["classe_processual"] == cls]
    gt = g.loc[g["evento"] == 1, "tempo"]
    ax_t.hist(gt.clip(upper=gt.quantile(0.99)), bins=30, color=CORES[0], edgecolor="white")
    fmt_milhares(ax_t, "y")
    ax_t.set_ylabel(cls if len(cls) <= 28 else cls[:26] + "…", fontsize=8)
    cens = 100 * (1 - g["evento"].mean())
    ax_c.bar(["Cens.", "Term."], [cens, 100 - cens], color=[CORES[1], CORES[0]])
    vol = g.groupby("ano_ajuizamento").size()
    ax_v.bar(vol.index.astype(int), vol.values, color="#4d4d4d")
    fmt_milhares(ax_v, "y")
    ax_t.tick_params(labelsize=8)
    ax_c.tick_params(labelsize=8)
    ax_v.tick_params(labelsize=8)
    if linha == 0:
        ax_t.set_title("Tempo (terminados)", fontsize=10)
        ax_c.set_title("% censura", fontsize=10)
        ax_v.set_title("Volume por ano", fontsize=10)
n_slots = n_rows * 2
for j in range(n_cls, n_slots):
    bloco = 0 if j < n_rows else 3
    linha = j if j < n_rows else j - n_rows
    for k in range(3):
        axes[linha, bloco + k].axis("off")
fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.04, hspace=0.55, wspace=0.35)
apa_multipanel_title(fig, "EDA-24",
    "Small multiples das classes do corte de Pareto a 95% (tempo, censura, volume anual)",
    "Cada linha é uma classe retida (incluindo «Outras classes»), em duas colunas por volume decrescente. "
    "A comparação visual antecipa a heterogeneidade que o AFT deve deslocar via μ(x).")
save_apa_figure(fig, DIR_FIG / "fig_eda24_small_multiples_classes.png", close=True)
print("EDA-21 a EDA-24 gravadas")
"""
    ),
    md("## 5. Qualidade dos dados — Figura EDA-25 e Tabela EDA-12"),
    code(
        """
miss = df.isna().mean().rename("pct_missing").reset_index().rename(columns={"index": "coluna"})
miss["pct_missing"] = (100 * miss["pct_missing"]).round(3)
miss["n_missing"] = df.isna().sum().values
miss["n_distintos"] = [df[c].nunique(dropna=False) for c in df.columns]
miss["dtype"] = [str(df[c].dtype) for c in df.columns]
exportar_tabela_apa(miss, "EDA-12", "Resumo de qualidade: missing, cardinalidade e tipo por coluna",
    "data_termino em falta é censura, não um defeito de extração.", DIR_TAB, "tab_eda12_qualidade_colunas")

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(miss["coluna"], miss["pct_missing"], color=CORES[0])
ax.set_xlabel("Valores em falta (%)")
apa_figure_title(fig, ax, "EDA-25", "Percentagem de valores em falta por coluna",
    "A única coluna com missing estrutural esperado é data_termino (processos censurados).")
save_apa_figure(fig, DIR_FIG / "fig_eda25_missing.png", close=True)
"""
    ),
    md("## 6. Tabela EDA-13 — resumo geral e interpretação"),
    code(
        """
tab_eda13 = pd.DataFrame({
    "item": ["n processos", "n eventos", "pct censura", "n classes", "n órgãos",
             "ano min", "ano max", "mediana tempo (terminados)", "média tempo (terminados)"],
    "valor": [
        f"{len(df):,}".replace(",", " "),
        f"{int(df['evento'].sum()):,}".replace(",", " "),
        f"{100 * (1 - df['evento'].mean()):.2f} %",
        str(df["classe_processual"].nunique()),
        str(df["orgao_julgador"].nunique()),
        str(int(df["ano_ajuizamento"].min())),
        str(int(df["ano_ajuizamento"].max())),
        f"{term.median():.1f}",
        f"{term.mean():.1f}",
    ],
})
exportar_tabela_apa(tab_eda13, "EDA-13", "Síntese do dataset limpo após a análise exploratória",
    "Amostra do Notebook 1, sem exclusões adicionais neste caderno.", DIR_TAB, "tab_eda13_resumo_dataset")
print(tab_eda13.to_string(index=False))
"""
    ),
    md(
        """## Interpretação final

A EDA confirma, antes de qualquer modelo, que a probabilidade de término não se resume a uma duração média. O tempo dos processos já baixados é assimétrico (EDA-1 a EDA-4): a mediana situa-se bem abaixo da média, e os percentis superiores documentam uma cauda longa típica de execuções. As classes (EDA-5, EDA-6, EDA-10 a EDA-12) separam regimes — recursos tendem a terminar cedo e com pouca censura; a execução fiscal combina tempos longos e censura elevada. Os órgãos (EDA-7, EDA-8, EDA-13, EDA-14) são numerosos e o volume está disperso; mesmo entre unidades grandes a mediana dos terminados afasta-se da linha global, o que justifica `orgao_julgador` como covariável e não como mero identificador administrativo.

O painel anual (EDA-9, EDA-15, EDA-16, EDA-21, EDA-22) é o ponto em que a leitura ingénua falha. A censura dispara em 2024–2025 e a mediana dos *já terminados* cai: ambos os padrões são, em primeira instância, truncamento à direita. A dissertação **não** interpreta essa queda como melhoria de celeridade. A comparação entre coortes só será válida quando o XGBoost AFT produzir $\\hat S(t\\mid x)$ com classe e órgão fixos (Notebook 4, Figura 18).

As associações (EDA-20, Tabelas EDA-10 e EDA-11) mostram que as três covariáveis não são independentes — em especial classe e órgão, por especialização funcional — mas nenhuma colapsa na outra: há informação residual em cada uma. A qualidade dos dados (EDA-25, EDA-12) é compatível com o desenho: o missing de `data_termino` *é* a censura. O dataset que segue para o Notebook 3 é o mesmo `dataset_limpo.parquet`, sem exclusões adicionais."""
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
