"""Gera notebooks/01_exploracao_dados.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_exploracao_dados.ipynb"


def md(src: str):
    return new_markdown_cell(src.strip() + "\n")


def code(src: str):
    return new_code_cell(src.strip() + "\n")


CELLS = [
    md(
        """# Notebook 1 — Carregamento e exploração dos dados

**Dissertação:** previsão, no momento do ajuizamento, da probabilidade de término de processos judiciais do TRF2 (DataJud / CNJ), com **XGBoost Survival em modo AFT** (*Accelerated Failure Time*).

**Objectivo:** localizar os Parquets extraídos da API do DataJud, construir a variável de sobrevivência (`tempo`, `evento`), documentar a limpeza, reter as classes do top 95% (residual em «Outras classes») e produzir as Tabelas 1–5 e as Figuras 1–5.

**Secção da dissertação:** Metodologia (unidade de análise, evento, recorte) e Resultados (estatística descritiva)."""
    ),
    md(
        """## 0. Configuração

Estilo APA 7.ª edição, `seed=42`. Os Parquets brutos já extraídos da API do DataJud estão em `data/raw/` (`trf2_{ano}_processos.parquet` e `trf2_{ano}_movimentos.parquet`). Este notebook **não** consulta a API: lê os Parquet com PyArrow (`ParquetFile` para o inventário; `read_table` para o carregamento)."""
    ),
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

warnings.filterwarnings("ignore")

HERE = Path.cwd().resolve()
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apa_style import (
    apa_figure_title,
    exportar_tabela_apa,
    fmt_milhares,
    save_apa_figure,
    setup_apa_style,
)
from src.data_utils import (
    ANOS_ANALISE,
    agregar_classes_top_percentil,
    agregar_por_processo_id,
    aplicar_limpeza,
    carregar_datas_termino,
    carregar_processos,
    colunas_dataset_limpo,
    construir_variaveis_sobrevivencia,
    listar_ficheiros_brutos,
    project_root,
    raw_data_dir,
)
import importlib
import src.eda_utils as _eda_utils
importlib.reload(_eda_utils)
from src.eda_utils import percentagens_que_somam_100
import pyarrow.parquet as pq

setup_apa_style()
CORES = sns.color_palette("colorblind")
ROOT = project_root()
DIR_FIG = ROOT / "figuras"
DIR_TAB = ROOT / "tabelas"
DIR_DADOS = ROOT / "dados"
DIR_FIG.mkdir(exist_ok=True)
DIR_TAB.mkdir(exist_ok=True)
DIR_DADOS.mkdir(exist_ok=True)

print("Raiz:", ROOT)
print("Parquets:", raw_data_dir())
print("Janela de análise:", ANOS_ANALISE)
"""
    ),
    md(
        """## 1. Carregamento e inventário

Cada ano da extração DataJud gera dois Parquets: a **capa** do processo e a lista de **movimentos**. Documentam-se formato, dimensões e esquema antes de qualquer transformação. A janela analítica é **2015–2025** (inclusive): 2026, quando existir no dump, é coorte incompleta e fica de fora do treino."""
    ),
    code(
        """
raw = raw_data_dir()
print("Pasta:", raw)
print("Ficheiros:", sorted(p.name for p in raw.glob("trf2_*.parquet")))

# Inventário via ParquetFile (metadados, sem carregar o conteúdo).
inv = listar_ficheiros_brutos()
inv_anos = inv.loc[inv["ano"].between(*ANOS_ANALISE)].copy()
proc_ex = raw / "trf2_2024_processos.parquet"
pf = pq.ParquetFile(proc_ex)
print()
print("Exemplo", proc_ex.name)
print("  num_row_groups=", pf.num_row_groups, " num_rows=", pf.metadata.num_rows)
print("  esquema:")
print(pf.schema_arrow)
print()
print(inv_anos.to_string(index=False))
print()
print("Totais na janela 2015–2025:")
print(inv_anos.groupby("tipo")[["n_registos", "tamanho_mb"]].sum().to_string())
print()
print("Anos presentes no dump e fora da janela analítica:")
print(inv.loc[~inv["ano"].between(*ANOS_ANALISE)].to_string(index=False))

exportar_tabela_apa(
    inv_anos,
    "0a",
    "Inventário dos ficheiros Parquet DataJud (TRF2) na janela 2015–2025",
    "Cada ano tem um ficheiro de processos (capa) e um de movimentos. "
    "n_registos é o número de linhas do Parquet, não ainda a unidade processo_id.",
    DIR_TAB,
    "tab00_inventario_ficheiros",
)
"""
    ),
    md(
        """**Interpretação do inventário.** Os ficheiros são Parquet colunar, adequados a ~3,5 milhões de capas e dezenas de milhões de movimentos. O esquema da capa inclui `dataAjuizamento` (texto AAAAMMDDHHMMSS), classe, órgão julgador, grau e `numeroProcesso`. Os movimentos trazem `codigo` TPU e `dataHora`. A unidade analítica **não** é a linha do dump: é `processo_id` = tribunal + grau + número (passo 2)."""
    ),
    md(
        """## 2. Variável de sobrevivência e unidade de análise

- **Evento** (`evento = 1`): primeira ocorrência do movimento TPU **22** (Baixa Definitiva). Códigos 246, 861 e 488 não ocorrem neste recorte do TRF2.
- **Tempo** (`tempo`): dias civis desde o ajuizamento até à baixa (evento) ou até à data de referência da extração (censura à direita).
- **`processo_id`**: um recurso (G2/TR) **não** se funde com a acção originária (G1/JE).
- **Covariáveis do ajuizamento:** `orgao_julgador`, `classe_processual`, `ano_ajuizamento`. Não se usam movimentos posteriores."""
    ),
    code(
        """
print("A carregar capas 2015–2025…", flush=True)
processos = carregar_processos(anos=ANOS_ANALISE)
print(f"Documentos DataJud (capas): {len(processos):,}")
print("A localizar a primeira baixa TPU 22…", flush=True)
baixas = carregar_datas_termino(anos=ANOS_ANALISE)
print(f"Documentos com pelo menos uma baixa 22: {len(baixas):,}")

df = construir_variaveis_sobrevivencia(processos, baixas)
print("data_referencia:", df.attrs.get("data_referencia"))
print(df[["processo_id", "tempo", "evento", "ano_ajuizamento", "classe_processual", "orgao_julgador"]].head())
print("n documentos", len(df), "| n processo_id", df["processo_id"].nunique())
print("graus:", sorted(df["grau"].dropna().unique().tolist()))
"""
    ),
    md(
        """**Agregação por `processo_id`.** O mesmo número CNJ pode existir em mais do que um grau. Redistribuições geram documentos DataJud duplicados na mesma chave. Retém-se a capa no ajuizamento mais antigo (empate: mais movimentos) e a **primeira** baixa TPU 22."""
    ),
    code(
        """
df, log_agg = agregar_por_processo_id(df)
print(log_agg.to_string(index=False))
print(f"Após agregação: {len(df):,} processos")
"""
    ),
    md(
        """## 3. Limpeza

Remove-se: data de ajuizamento inválida; classe ou órgão em falta; `tempo` ≤ 0; ano em falta. Cada decisão fica no log (Tabela 0b). **Não se imputam** covariáveis — um processo sem órgão no ajuizamento não pode entrar no one-hot do AFT."""
    ),
    code(
        """
df, log_limp = aplicar_limpeza(df)
log_all = pd.concat([log_agg, log_limp], ignore_index=True)
print(log_all.to_string(index=False))
exportar_tabela_apa(
    log_all,
    "0b",
    "Log de limpeza e agregação da amostra analítica",
    "n_removidos em agregar_processo_id são documentos DataJud colapsados, não processos excluídos da análise.",
    DIR_TAB,
    "tab00_log_limpeza",
)
print(f"Amostra após limpeza: {len(df):,} | censura {(1 - df['evento'].mean()) * 100:.2f} %")
"""
    ),
    md(
        """**Interpretação da limpeza.** As perdas por data inválida ou tempo não positivo devem ser residuais (erros de registo). A agregação por `processo_id` reduz linhas mas **não** elimina unidades de análise: funde capas duplicadas. A amostra resultante é a população de processos ajuizados no TRF2 em 2015–2025 com capa observável."""
    ),
    md(
        """## 4. Classes processuais (top 95%)

A TPU tem dezenas de classes residuais. Incluir todas no *one-hot* geraria colunas esparsas sem ganho interpretativo. **Retêm-se as classes que, em conjunto, somam 95% do volume**; o resto agrega-se em **«Outras classes»**. Nenhum processo é excluído."""
    ),
    code(
        """
df, resumo_cls = agregar_classes_top_percentil(df, limiar=0.95)
n_retidas = int(resumo_cls["retida"].sum())
n_outras = int(resumo_cls.loc[~resumo_cls["retida"], "n"].sum())
print(resumo_cls.to_string(index=False))
print(f"Classes retidas: {n_retidas} | Outras classes: {n_outras:,} processos")
print("Categorias finais:", df["classe_processual"].nunique())
"""
    ),
    md(
        """## 5. Estatística descritiva

### Tabela 1 — Classes (frequência, %, % acumulada)"""
    ),
    code(
        """
freq = df["classe_processual"].value_counts()
tab1 = pd.DataFrame(
    {
        "classe_processual": freq.index,
        "frequencia": freq.values,
        "percentagem": percentagens_que_somam_100(freq.values),
    }
)
tab1["percentagem_acumulada"] = tab1["percentagem"].cumsum().round(2)
exportar_tabela_apa(
    tab1,
    1,
    "Distribuição das classes processuais na amostra analítica (top 95% e residual agregado)",
    "Percentagens a duas casas pelo método dos maiores restos, para o total ser 100,00 %. "
    "«Outras classes» é a agregação do residual (~5%), não uma classe TPU.",
    DIR_TAB,
    "tab01_descritiva_classes",
)
"""
    ),
    code(
        """
from IPython.display import Markdown, display
display(Markdown(f'''
**Interpretação da Tabela 1.** {tab1.iloc[0]["classe_processual"]} concentra {tab1.iloc[0]["percentagem"]:.2f} % da amostra. As três primeiras classes, se forem de famílias distintas (juízo, execução, recurso), mostram que o modelo terá de acomodar regimes de celeridade muito diferentes. «Outras classes» evita perder o residual sem inflacionar o *one-hot*.
'''))
"""
    ),
    md(
        """### Tabela 2 — Painel anual (frequência e situação censura/evento)

> **Aviso metodológico (obrigatório).** A mediana do tempo calculada **só sobre os processos já terminados**, comparada entre anos, sofre **viés de censura**: nas coortes recentes (2024–2025) apenas os processos *rápidos* já tiveram baixa. Isso faz a mediana «dos terminados» parecer decrescente mesmo **sem melhoria real de celeridade**. A comparação válida entre coortes anuais nesta dissertação faz-se pelas previsões do XGBoost AFT com classe e órgão fixos (Notebook 4, Figura 18) — não por esta tabela nem pelas Figuras 3 e 4."""
    ),
    code(
        """
por_ano = (
    df.groupby("ano_ajuizamento")
    .agg(n=("processo_id", "size"), n_terminados=("evento", "sum"))
    .reset_index()
)
por_ano["n_censurados"] = por_ano["n"] - por_ano["n_terminados"]
por_ano["pct_terminados"] = (100 * por_ano["n_terminados"] / por_ano["n"]).round(2)
por_ano["pct_censurados"] = (100 * por_ano["n_censurados"] / por_ano["n"]).round(2)
med_term = (
    df.loc[df["evento"] == 1]
    .groupby("ano_ajuizamento")["tempo"]
    .median()
    .rename("mediana_tempo_terminados")
)
por_ano = por_ano.merge(med_term, on="ano_ajuizamento", how="left")
por_ano["mediana_tempo_terminados"] = por_ano["mediana_tempo_terminados"].round(1)
exportar_tabela_apa(
    por_ano,
    2,
    "Ajuizamentos por ano: volume, percentagem de terminados/censurados e mediana do tempo (só terminados)",
    "A coluna da mediana dos terminados está sujeita a viés de censura nas coortes recentes "
    "e não deve ser lida como evidência de aceleração. Comparação válida: Figura 18 (Notebook 4).",
    DIR_TAB,
    "tab02_distribuicao_ano",
)
"""
    ),
    md(
        """### Tabela 3 — Órgão julgador (20 de maior volume + resumo dos restantes)"""
    ),
    code(
        """
freq_o = df["orgao_julgador"].value_counts()
top20 = freq_o.head(20)
resto_n = int(freq_o.iloc[20:].sum()) if len(freq_o) > 20 else 0
tab3 = pd.DataFrame(
    {
        "orgao_julgador": list(top20.index) + [f"Restantes ({len(freq_o) - 20} órgãos)"],
        "frequencia": list(top20.values) + [resto_n],
        "percentagem": list((100 * top20.values / freq_o.sum()).round(2))
        + [round(100 * resto_n / freq_o.sum(), 2)],
    }
)
exportar_tabela_apa(
    tab3,
    3,
    "Órgãos julgadores com maior volume (top 20) e resumo dos restantes",
    f"Total de órgãos na amostra: {df['orgao_julgador'].nunique()}. "
    "A litigância está dispersa: nenhum órgão monopoliza o volume.",
    DIR_TAB,
    "tab03_orgaos_julgadores",
)
"""
    ),
    md(
        """### Tabela 4 — Resumo de `tempo` (apenas processos terminados)"""
    ),
    code(
        """
t = df.loc[df["evento"] == 1, "tempo"]
tab4 = pd.DataFrame(
    {
        "estatistica": ["n", "média", "mediana", "Q1", "Q3", "mínimo", "máximo"],
        "tempo_dias": [
            int(t.size),
            round(float(t.mean()), 1),
            round(float(t.median()), 1),
            round(float(t.quantile(0.25)), 1),
            round(float(t.quantile(0.75)), 1),
            round(float(t.min()), 1),
            round(float(t.max()), 1),
        ],
    }
)
exportar_tabela_apa(
    tab4,
    4,
    "Resumo do tempo até ao término (dias), processos com evento observado",
    "Exclui censurados. A mediana e os quartis descrevem a duração observada, não a função de sobrevivência. "
    "A cauda (máximo) reflecte processos muito longos, típicos de execução.",
    DIR_TAB,
    "tab04_resumo_tempo",
)
"""
    ),
    md(
        """### Tabela 5 — Proporção de censura global e por classe"""
    ),
    code(
        """
linhas = [
    {
        "estrato": "Global",
        "n": len(df),
        "n_eventos": int(df["evento"].sum()),
        "n_censurados": int((df["evento"] == 0).sum()),
        "pct_censura": round(100 * (1 - df["evento"].mean()), 2),
    }
]
for cls, g in df.groupby("classe_processual"):
    linhas.append(
        {
            "estrato": cls,
            "n": len(g),
            "n_eventos": int(g["evento"].sum()),
            "n_censurados": int((g["evento"] == 0).sum()),
            "pct_censura": round(100 * (1 - g["evento"].mean()), 2),
        }
    )
tab5 = pd.DataFrame(linhas).sort_values(["estrato"]).reset_index(drop=True)
# Global no topo
tab5 = pd.concat(
    [tab5.loc[tab5["estrato"] == "Global"], tab5.loc[tab5["estrato"] != "Global"]],
    ignore_index=True,
)
exportar_tabela_apa(
    tab5,
    5,
    "Proporção de censura à direita, global e por classe processual",
    "Censura = processos sem baixa TPU 22 até à data de referência da extração. "
    "Classes com censura elevada tendem a ser mais lentas e/ou mais recentes no mix de coortes.",
    DIR_TAB,
    "tab05_censura_por_classe",
)
"""
    ),
    md(
        """## 6. Figuras

### Figura 1 — Histograma do tempo (terminados)"""
    ),
    code(
        """
t = df.loc[df["evento"] == 1, "tempo"]
med = float(t.median())
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(t.clip(upper=t.quantile(0.99)), bins=60, color=CORES[0], edgecolor="white")
ax.axvline(med, color=CORES[3], ls="--", lw=1.8, label=f"Mediana = {med:.0f} dias")
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Número de processos")
fmt_milhares(ax, "y")
ax.legend(frameon=False)
apa_figure_title(
    fig,
    ax,
    1,
    "Distribuição do tempo até ao término (processos com baixa definitiva)",
    "Apenas eventos observados. A linha tracejada marca a mediana. "
    "O histograma está truncado visualmente no percentil 99 para não comprimir a massa.",
)
save_apa_figure(fig, DIR_FIG / "fig01_histograma_tempo.png", close=False)
plt.show()
plt.close(fig)
"""
    ),
    md(
        """**Interpretação da Figura 1.** A duração observada é assimétrica à direita: a massa concentra-se no primeiro ou segundo ano e a cauda estende-se por vários anos. Uma média única de duração seria enganadora; a dissertação trabalha com $S(t)$, não com um ponto."""
    ),
    md("### Figura 2 — Classes processuais (top 95% + residual)"),
    code(
        """
vol_cls = df["classe_processual"].value_counts().iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(vol_cls.index, vol_cls.values, color=CORES[0])
ax.set_xlabel("Número de processos")
ax.set_ylabel("Classe processual")
fmt_milhares(ax, "x")
apa_figure_title(
    fig,
    ax,
    2,
    "Classes processuais retidas na amostra analítica (corte de Pareto a 95%)",
    "Inclui todas as classes que, em conjunto, cobrem 95% dos processos, mais o residual «Outras classes».",
)
save_apa_figure(fig, DIR_FIG / "fig02_classes_volume.png", close=False)
plt.show()
plt.close(fig)
"""
    ),
    md("### Figura 3 — Processos ajuizados por ano"),
    code(
        """
n_ano = df.groupby("ano_ajuizamento").size()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(n_ano.index.astype(int), n_ano.values, color=CORES[0])
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Número de processos")
fmt_milhares(ax, "y")
apa_figure_title(
    fig,
    ax,
    3,
    "Volume de processos ajuizados no TRF2, por ano (2015–2025)",
    "Aviso: este gráfico descreve procura, não celeridade. A comparação de durações entre "
    "anos exige o modelo AFT (Figura 18), por causa da censura diferencial das coortes recentes.",
)
save_apa_figure(fig, DIR_FIG / "fig03_ajuizamentos_por_ano.png", close=False)
plt.show()
plt.close(fig)
"""
    ),
    md("### Figura 4 — Censurados versus terminados por ano"),
    code(
        """
stack = (
    df.groupby(["ano_ajuizamento", "evento"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={0: "Censurados", 1: "Terminados"})
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(stack.index.astype(int), stack["Terminados"], color=CORES[0], label="Terminados")
ax.bar(
    stack.index.astype(int),
    stack["Censurados"],
    bottom=stack["Terminados"],
    color=CORES[1],
    label="Censurados",
)
ax.set_xlabel("Ano de ajuizamento")
ax.set_ylabel("Número de processos")
fmt_milhares(ax, "y")
ax.legend(frameon=False)
apa_figure_title(
    fig,
    ax,
    4,
    "Composição de terminados e censurados por coorte de ajuizamento",
    "A fatia de censura sobe em 2024–2025 porque o follow-up é curto. Excluir os censurados "
    "enviesaria as coortes recentes para os processos mais rápidos.",
)
save_apa_figure(fig, DIR_FIG / "fig04_censura_vs_terminados_por_ano.png", close=False)
plt.show()
plt.close(fig)
"""
    ),
    md("### Figura 5 — Tempo por classe (top 95% + residual, só terminados)"),
    code(
        """
ordem_cls = df["classe_processual"].value_counts().index.tolist()
sub = df.loc[df["evento"] == 1].copy()
fig, ax = plt.subplots(figsize=(8, 7))
sns.boxplot(
    data=sub,
    y="classe_processual",
    x="tempo",
    order=ordem_cls,
    showfliers=False,
    color=CORES[0],
    ax=ax,
)
ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Classe processual")
apa_figure_title(
    fig,
    ax,
    5,
    "Distribuição do tempo até ao término nas classes do corte de Pareto a 95%",
    "Apenas processos terminados; outliers omitidos para leitura dos quartis. "
    "Inclui «Outras classes». Diferenças de localização entre classes são o primeiro "
    "indício de heterogeneidade preditiva.",
)
save_apa_figure(fig, DIR_FIG / "fig05_boxplot_tempo_classes.png", close=False)
plt.show()
plt.close(fig)
"""
    ),
    md(
        """## 7. Exportação do dataset limpo

O ficheiro `dados/dataset_limpo.parquet` é o input dos Notebooks 2–6. Colunas: `tempo`, `evento`, `orgao_julgador`, `classe_processual`, `ano_ajuizamento`, mais identificadores e datas para rastreio."""
    ),
    code(
        """
cols = colunas_dataset_limpo()
out = df[cols].copy()
path = DIR_DADOS / "dataset_limpo.parquet"
out.to_parquet(path, index=False)
resumo = pd.DataFrame(
    {
        "item": [
            "n_processos",
            "n_eventos",
            "n_censurados",
            "pct_censura",
            "n_classes",
            "n_orgaos",
            "ano_min",
            "ano_max",
            "data_referencia",
        ],
        "valor": [
            len(out),
            int(out["evento"].sum()),
            int((out["evento"] == 0).sum()),
            round(100 * (1 - out["evento"].mean()), 2),
            int(out["classe_processual"].nunique()),
            int(out["orgao_julgador"].nunique()),
            int(out["ano_ajuizamento"].min()),
            int(out["ano_ajuizamento"].max()),
            str(df.attrs.get("data_referencia")),
        ],
    }
)
resumo.to_csv(DIR_TAB / "tab00_resumo_dataset_limpo.csv", index=False, encoding="utf-8-sig")
print("Exportado:", path, "|", path.stat().st_size / 1e6, "MB")
print(resumo.to_string(index=False))
print(out.dtypes)
"""
    ),
    md(
        """## Interpretação final

Este notebook constrói a amostra analítica da dissertação a partir dos Parquets DataJud do TRF2 (2015–2025). A unidade é o `processo_id` (tribunal + grau + número CNJ): um recurso não se confunde com a acção originária. O evento é a primeira baixa definitiva (TPU 22); os processos ainda em curso entram como censura à direita, com tempo medido até à data de referência da extração. As três covariáveis — órgão julgador, classe processual (top 95% mais «Outras classes») e ano de ajuizamento — são as únicas informações usadas pelo XGBoost AFT, todas observáveis no ajuizamento.

A estatística descritiva já antecipa heterogeneidade: classes de volume elevado misturam juizado, execução e recursos; a censura é baixa nas coortes com follow-up longo e sobe em 2024–2025; os órgãos são muitos e nenhum monopoliza o acervo. O histograma do tempo dos terminados é assimétrico, o que torna inadequada qualquer síntese por uma duração média única — exactamente a motivação para estimar $S(t\\mid x)$ e não um ponto.

O aviso que acompanha a Tabela 2 e as Figuras 3–4 é substantivo, não de estilo: a mediana dos já terminados **não** autoriza a conclusão de que o tribunal ficou mais célere nas coortes recentes. Essa comparação só será válida no Notebook 4 (Figura 18), com classe e órgão fixos e curvas previstas pelo AFT. O Notebook 2 aprofunda a EDA sem alterar esta amostra."""
    ),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbformat.write(nb, OUT)
    print(f"Escrito {OUT} ({len(CELLS)} células)")


if __name__ == "__main__":
    main()
