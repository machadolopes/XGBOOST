"""Gera notebooks/05_verificacao_final.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_verificacao_final.ipynb"


def md(src: str):
    return new_markdown_cell(src.strip() + "\n")


def code(src: str):
    return new_code_cell(src.strip() + "\n")


CELLS = [
    md(
        """# Notebook 5 — Verificação final e exportação

**Objectivo:** checklist de reprodutibilidade (`seed=42`), conferir numeração APA, consolidar a Tabela 18 e montar `entregaveis/` para a redação dos capítulos de Resultados, Discussão e Conclusões. **Não se re-treina.**

**Secção da dissertação:** Resultados (síntese) e Anexos (reprodutibilidade)."""
    ),
    md("## 1. Checklist de reprodutibilidade"),
    code(
        """
from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import xgboost as xgb
from IPython.display import Markdown, display

warnings.filterwarnings("ignore")

HERE = Path.cwd().resolve()
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apa_style import exportar_tabela_apa, setup_apa_style
from src.data_utils import project_root

ROOT = project_root()
DIR_DADOS = ROOT / "dados"
DIR_FIG = ROOT / "figuras"
DIR_TAB = ROOT / "tabelas"
DIR_MOD = ROOT / "modelo"
DIR_META = ROOT / "metadados"
DIR_NB = ROOT / "notebooks"
setup_apa_style()

NOTEBOOKS = [
    "01_exploracao_dados.ipynb",
    "02_eda_exaustiva.ipynb",
    "03_xgboost_aft_modelo.ipynb",
    "04_curvas_individualizadas.ipynb",
    "05_verificacao_final.ipynb",
    "06_dashboard.ipynb",
]
print("Python", sys.version.split()[0], "| xgboost", xgb.__version__)

faltam_nb = [n for n in NOTEBOOKS if not (DIR_NB / n).is_file()]
print("Notebooks em falta:", faltam_nb or "nenhum")

req = ROOT / "requirements.txt"
if req.is_file():
    print("requirements.txt existe:", req)
try:
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8")
    (ROOT / "requirements.txt").write_text(freeze, encoding="utf-8")
    print("pip freeze actualizado:", len(freeze.splitlines()), "pacotes")
except Exception as exc:
    print("pip freeze falhou:", exc)

artefactos = {
    "dataset_limpo.parquet": DIR_DADOS / "dataset_limpo.parquet",
    "booster": DIR_MOD / "xgboost_aft_modelo_final.json",
    "encoder": DIR_MOD / "xgboost_aft_encoder.joblib",
    "meta modelo": DIR_MOD / "xgboost_aft_meta.json",
    "meta pasta": DIR_META / "xgboost_aft_meta.json",
}
for nome, p in artefactos.items():
    print(f"{'OK' if p.is_file() else 'FALTA':4s}  {nome:24s}  {p}")

# seed=42 nos notebooks
padrao = re.compile(r"random_state\\s*=\\s*42|seed\\s*=\\s*42")
for nb_nome in NOTEBOOKS:
    p = DIR_NB / nb_nome
    if not p.is_file():
        continue
    txt = p.read_text(encoding="utf-8")
    n = len(padrao.findall(txt))
    print(f"{nb_nome}: {n} ocorrências de seed/random_state=42")
"""
    ),
    md("## 2. Numeração APA das figuras e tabelas canónicas"),
    code(
        """
FIGURAS = [
    "fig01_histograma_tempo.png", "fig02_classes_volume.png", "fig03_ajuizamentos_por_ano.png",
    "fig04_censura_vs_terminados_por_ano.png", "fig05_boxplot_tempo_classes.png",
    "fig_eda01_hist_tempo_linear.png", "fig_eda02_hist_tempo_log10.png", "fig_eda03_ecdf_tempo.png",
    "fig_eda04_violin_tempo.png", "fig_eda05_barras_classes.png", "fig_eda06_pareto_classes.png",
    "fig_eda07_hist_volume_orgaos.png", "fig_eda08_top30_orgaos.png", "fig_eda09_ano_empilhado.png",
    "fig_eda10_classe_evento_stacked.png", "fig_eda11_box_tempo_classe.png", "fig_eda12_violin_classes.png",
    "fig_eda13_box_orgao_top20.png", "fig_eda14_scatter_mediana_volume.png", "fig_eda15_box_tempo_ano.png",
    "fig_eda16_mediana_ic_ano.png", "fig_eda17_heatmap_classe_ano.png", "fig_eda18_heatmap_orgao_classe.png",
    "fig_eda19_heatmap_orgao_ano.png", "fig_eda20_correlacoes.png",
    "fig_eda21_heatmap_mediana_classe_ano.png", "fig_eda22_heatmap_censura_classe_ano.png",
    "fig_eda23_area_faixas_tempo.png", "fig_eda24_small_multiples_classes.png", "fig_eda25_missing.png",
    "fig10_brier.png", "fig11_calib_1a.png", "fig12_calib_2a.png", "fig13_calib_5a.png",
    "fig14_importancia.png", "fig14b_shap.png",
    "fig15_perfis.png", "fig16_classe.png", "fig17_orgao.png", "fig18_ano.png",
    "fig19_faixas.png", "fig20_anomalias.png",
]
TABELAS = [
    "tab01_descritiva_classes.csv", "tab02_distribuicao_ano.csv", "tab03_orgaos_julgadores.csv",
    "tab04_resumo_tempo.csv", "tab05_censura_por_classe.csv",
    "tab_eda01_percentis_tempo.csv", "tab_eda02_classes_completa.csv", "tab_eda03_volume_orgaos.csv",
    "tab_eda04_por_ano.csv", "tab_eda05_classe_evento.csv", "tab_eda06_tempo_por_classe.csv",
    "tab_eda07_top30_orgaos.csv", "tab_eda08_classe_ano.csv", "tab_eda09_orgao_classe.csv",
    "tab_eda10_cramer_v.csv", "tab_eda11_qui_quadrado_evento.csv", "tab_eda12_qualidade_colunas.csv",
    "tab_eda13_resumo_dataset.csv",
    "tab09_distribuicoes_aft.csv", "tab10_cindex_aft.csv", "tab11_brier_aft.csv",
    "tab12_importancia_aft.csv", "tab12b_gain_grupos.csv", "tab12c_shap_grupos.csv",
    "tab13_resumo_aft.csv", "tab14_perfis.csv", "tab15_classe.csv",
    "tab16_faixas_exemplos.csv", "tab17_anomalias.csv",
]
print("=== Figuras em falta ===")
for f in FIGURAS:
    if not (DIR_FIG / f).is_file():
        print(" ", f)
print("=== Tabelas em falta ===")
for f in TABELAS:
    if not (DIR_TAB / f).is_file():
        print(" ", f)
print("Contagem figuras presentes:", sum((DIR_FIG / f).is_file() for f in FIGURAS), "/", len(FIGURAS))
print("Contagem tabelas presentes:", sum((DIR_TAB / f).is_file() for f in TABELAS), "/", len(TABELAS))
"""
    ),
    md("## 3. Tabela 18 — resumo geral"),
    code(
        """
df = pd.read_parquet(DIR_DADOS / "dataset_limpo.parquet")
meta = json.loads((DIR_MOD / "xgboost_aft_meta.json").read_text(encoding="utf-8"))
tab09 = pd.read_csv(DIR_TAB / "tab09_distribuicoes_aft.csv")
tab11 = pd.read_csv(DIR_TAB / "tab11_brier_aft.csv")
tab12b = pd.read_csv(DIR_TAB / "tab12b_gain_grupos.csv")
tab12c = pd.read_csv(DIR_TAB / "tab12c_shap_grupos.csv") if (DIR_TAB / "tab12c_shap_grupos.csv").is_file() else pd.DataFrame()
tab15 = pd.read_csv(DIR_TAB / "tab15_classe.csv") if (DIR_TAB / "tab15_classe.csv").is_file() else pd.DataFrame()
tab17 = pd.read_csv(DIR_TAB / "tab17_anomalias.csv") if (DIR_TAB / "tab17_anomalias.csv").is_file() else pd.DataFrame()
tab17c = pd.read_csv(DIR_TAB / "tab17_anomalias_completo.csv") if (DIR_TAB / "tab17_anomalias_completo.csv").is_file() else pd.DataFrame()

n_an = int(tab17c["anomalia"].sum()) if "anomalia" in tab17c.columns else len(tab17)
s2_min = s2_max = "—"
if not tab15.empty and "S(2 anos)" in tab15.columns:
    s2_min = f"{tab15['S(2 anos)'].min():.3f} ({tab15.loc[tab15['S(2 anos)'].idxmin(), 'classe']})"
    s2_max = f"{tab15['S(2 anos)'].max():.3f} ({tab15.loc[tab15['S(2 anos)'].idxmax(), 'classe']})"

def _brier(nome):
    hit = tab11.loc[tab11["horizonte"] == nome]
    return f"{hit['Brier_AFT'].iloc[0]:.3f}" if len(hit) else "—"

gain = {str(r.covariavel): f"{r.gain_relativo:.3f}" for r in tab12b.itertuples()} if "gain_relativo" in tab12b.columns else {}
shapg = {str(r.covariavel): f"{r.partilha:.3f}" for r in tab12c.itertuples()} if "partilha" in tab12c.columns else {}

linhas = [
    ("Dataset: n processos", f"{len(df):,}".replace(",", " ")),
    ("Dataset: censura", f"{100*(1-df['evento'].mean()):.2f} % ({int((df['evento']==0).sum()):,} / {len(df):,})".replace(",", " ")),
    ("Dataset: evento", "primeira baixa TPU 22"),
    ("Dataset: janela", "ajuizamentos 2015–2025, TRF2, todos os graus"),
    ("Dataset: classes", f"{df['classe_processual'].nunique()} (top 95 % + Outras classes)"),
    ("Dataset: órgãos", str(df["orgao_julgador"].nunique())),
    ("Algoritmo", "XGBoost Survival AFT (único algoritmo estimado)"),
    ("Distribuição AFT (Tabela 9)", f"{meta['distribuicao']}; σ = {meta['sigma']:.3f}"),
    ("C-index teste / treino", f"{meta['c_index_teste']:.3f} / {meta['c_index_treino']:.3f}"),
    ("IBS AFT / IBS KM", f"{meta['ibs']:.3f} / {meta['ibs_km']:.3f}"),
    ("Brier 6m / 1a / 2a / 3a / 5a",
     " / ".join(_brier(h) for h in ["6 meses", "1 ano", "2 anos", "3 anos", "5 anos"])),
    ("Gain relativo órgão / classe / ano",
     f"{gain.get('orgao_julgador','—')} / {gain.get('classe_processual','—')} / {gain.get('ano_ajuizamento','—')}"),
    ("|SHAP| médio relativo órgão / classe / ano",
     f"{shapg.get('orgao_julgador','—')} / {shapg.get('classe_processual','—')} / {shapg.get('ano_ajuizamento','—')}"),
    ("S(2 anos) extremos (medianas de classe, teste)", f"mín {s2_min} · máx {s2_max}"),
    ("Anomalias rácio ≥ 2 (2022)", str(n_an)),
    ("Semente", "42"),
]
tab18 = pd.DataFrame(linhas, columns=["item", "valor"])
exportar_tabela_apa(
    tab18, 18,
    "Resumo geral da amostra, do modelo XGBoost AFT e do desempenho",
    "Números extraídos dos artefactos dos Notebooks 1–4. C-index e IBS na subamostra estratificada de 80 000 do teste.",
    DIR_TAB, "tab18_resumo_geral",
)
tab18.to_csv(DIR_META / "tab18_resumo_geral.csv", index=False, encoding="utf-8-sig")
"""
    ),
    md("## 4. Interpretação global (síntese para a dissertação)"),
    code(
        """
display(Markdown(open(DIR_TAB / "tab18_resumo_geral.csv", encoding="utf-8-sig").read() and f'''
O dataset analítico reúne **{len(df):,}** processos do TRF2 ajuizados entre 2015 e 2025, com censura à direita de **{100*(1-df["evento"].mean()):.2f} %** (611 509 processos ainda em curso na data de extração). A unidade é o `processo_id` (tribunal + grau + número CNJ), de modo que um recurso não se funde com a acção originária. O evento é a primeira baixa definitiva TPU 22. As três covariáveis do ajuizamento — órgão julgador (337 categorias), classe processual (16 após o corte de Pareto a 95 % mais «Outras classes») e ano — alimentam o **único** algoritmo estimado nesta dissertação: XGBoost Survival em modo AFT.

A comparação empírica das três distribuições (Tabela 9) seleccionou **{meta["distribuicao"]}** (σ = {meta["sigma"]:.3f}), com C-index de teste **{meta["c_index_teste"]:.3f}** — coincidente com o do treino na subamostra de 80 000 — e IBS **{meta["ibs"]:.3f}**, inferior ao Kaplan–Meier sem covariáveis ({meta["ibs_km"]:.3f}). A logística foi descartada (C-index ≈ 0,58). O Brier pontual é 0,119 aos 6 meses, 0,164 a 1 ano, 0,158 a 2 anos, 0,133 a 3 anos e 0,077 a 5 anos. Classe, órgão e ano, observados no ajuizamento, ordenam e calibram a probabilidade de o processo ainda estar pendente melhor do que a curva populacional. O gain relativo concentra-se no órgão (0,705), depois na classe (0,213) e no ano (0,082); o |SHAP| médio eleva a partilha do ano (0,209), coerente com o truncamento das coortes recentes.

A probabilidade de término é **heterogénea**: não se resume a uma duração média. No teste, Ŝ(2 anos) mediano das 16 classes vai de {s2_min} a {s2_max} (Tabela 15). As curvas individualizadas (Figuras 15–17) deslocam $S(t)$ entre classes e, dentro da mesma classe, entre órgãos. A Figura 18 — classe e órgão fixos, ano a variar — é a comparação válida entre coortes; a queda da mediana dos *já terminados* nas coortes recentes (Tabela 2, Figuras 3–4, EDA-16) **não** autoriza a leitura de aceleração da celeridade. As faixas (Figura 19) traduzem $\\exp(\\mu)$ num rótulo relativo à célula classe×ano. As anomalias (Figura 20, n = {n_an}) são alertas de fluxo com rácio ≥ 2 em 2022, não juízos disciplinares.

Limitações centrais: forma paramétrica partilhada (σ global, só translação temporal); C-index e IBS avaliados em 80 000 processos por custo quadrático dos pares; o ano mistura efeito de coorte e truncamento à direita; a consulta de processos de 2026 no dashboard usa 2025 como *proxy* one-hot e é extrapolação. O Notebook 6 demonstra a interface gerencial sem re-treino e **não** integra a validação estatística formal.
'''))
"""
    ),
    md("## 5. Directório `entregaveis/`"),
    code(
        """
from src.montar_entregaveis import montar_entregaveis

ENT = montar_entregaveis(ROOT)
print("entregaveis/ pronto em", ENT)
print("figuras:", len(list((ENT / "figuras").glob("*.png"))))
print("tabelas:", len(list((ENT / "tabelas").glob("*.csv"))))
print("interpretacao:", sorted(p.name for p in (ENT / "interpretacao").glob("*.md")))
print("LEIA-ME:", (ENT / "LEIA-ME.md").is_file())
"""
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
