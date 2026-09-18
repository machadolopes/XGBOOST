"""Gera notebooks/03_xgboost_aft_modelo.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "03_xgboost_aft_modelo.ipynb"


def md(src: str):
    return new_markdown_cell(src.strip() + "\n")


def code(src: str):
    return new_code_cell(src.strip() + "\n")


CELLS = [
    md(
        """# Notebook 3 — XGBoost Survival (AFT): treino e avaliação

**Dissertação:** previsão, no momento do ajuizamento, da probabilidade de término de processos judiciais (DataJud / TRF2).

**Algoritmo único:** XGBoost Survival em modo AFT. O modelo de Cox é apenas enquadramento teórico da revisão de literatura — **não corre** neste caderno.

**Objectivo:** treinar o AFT no universo, comparar `normal`, `logistic` e `extreme`, seleccionar a vencedora e avaliar C-index, Brier/IBS e calibração. Tabelas 9–13; Figuras 10–14b.

**Input:** `dados/dataset_limpo.parquet`. **Output:** `modelo/xgboost_aft_modelo_final.json` e `metadados/xgboost_aft_meta.json`.

**Secção da dissertação:** Resultados (modelo) e Metodologia (AFT, métricas, subamostra de avaliação)."""
    ),
    md(
        """## Nota técnica: o que o AFT assume

O XGBoost AFT modela $\\log T$ com localização $\\mu(x)$ (árvores) e escala $\\sigma$ **global**. As covariáveis deslocam a curva no tempo, mas não alteram a forma relativa.

| `aft_loss_distribution` | $S(t)$ | $z=(\\ln t-\\mu)/\\sigma$ |
|---|---|---|
| `normal` | $1-\\Phi(z)$ | log-normal |
| `logistic` | $1/(1+e^{z})$ | log-logística |
| `extreme` | $\\exp(-e^{z})$ | valor extremo (Gumbel) |

Não existe `predict_survival_function()` no XGBoost. As curvas são reconstruídas em `src/survival_utils.py` (`predict_survival_function_aft`). A comparação das três distribuições **é um resultado** (Tabela 9)."""
    ),
    md(
        """## 0. Configuração

Estilo APA 7.ª ed., `seed=42`. O *early stopping* usa 10% do **treino** (paciência 30); o teste nunca entra na paragem antecipada."""
    ),
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
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

HERE = Path.cwd().resolve()
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apa_style import apa_figure_title, exportar_tabela_apa, save_apa_figure, setup_apa_style
from src.model_utils import (
    COVARIATEIS,
    amostrar_estratificado,
    brier_modelo_e_km,
    calibracao_decis,
    grelha_brier,
    y_sobrevivencia,
)
from src.survival_utils import (
    AFT_DISTRIBUICOES,
    HORIZONTES_DIAS,
    S_aft,
    ajustar_sigma_aft,
    localizacao_from_pred,
    predict_survival_function_aft,
)
from src.xgb_survival_utils import (
    AFT_EARLY_STOP,
    AFT_NUM_BOOST,
    PARAMS_AFT_DISSERTACAO,
    RNG,
    cindex_sub,
    dmatrix_aft,
    encoder_esparso,
    importancia_gain_grupos,
    preparar_split,
    treinar_aft_distribuicao,
)

setup_apa_style()
CORES = sns.color_palette("colorblind")
DIR_FIG = ROOT / "figuras"
DIR_TAB = ROOT / "tabelas"
DIR_MOD = ROOT / "modelo"
DIR_META = ROOT / "metadados"
for d in (DIR_FIG, DIR_TAB, DIR_MOD, DIR_META):
    d.mkdir(exist_ok=True)

N_AVAL = 80_000
N_SHAP = 4_000
FRACAO_VAL = 0.10

print("Python", sys.version.split()[0], "| xgboost", xgb.__version__, "| RNG", RNG)
print("Hiperparâmetros AFT (fixos na comparação):")
for k, v in PARAMS_AFT_DISSERTACAO.items():
    print(f"  {k}: {v}")
print(f"num_boost_round={AFT_NUM_BOOST} | early_stopping={AFT_EARLY_STOP} | N_AVAL={N_AVAL}")
"""
    ),
    md(
        """## 1. Preparação

*One-hot* das três covariáveis (`handle_unknown='ignore'`), matriz **CSR float32**. Limites AFT: evento $[t,t]$; censura $[t,+\\infty)$.

Divisão **80/20** estratificada por **evento × ano**; 10% do treino para validação interna."""
    ),
    code(
        """
df = pd.read_parquet(ROOT / "dados" / "dataset_limpo.parquet")
print(f"Universo: {len(df):,} processos | censura {(1 - df['evento'].mean()) * 100:.2f} %")
print(df[list(COVARIATEIS) + ["tempo", "evento"]].dtypes)

df_tr_full, df_te = preparar_split(df, n=None, random_state=RNG)
print(f"Treino+val: {len(df_tr_full):,} | Teste: {len(df_te):,}")

strata_tr = df_tr_full["evento"].astype(str) + "_" + df_tr_full["ano_ajuizamento"].astype(str)
cont = strata_tr.value_counts()
raros = set(cont[cont < 2].index)
if raros:
    strata_tr = strata_tr.where(~strata_tr.isin(raros), other="_raro")
df_tr, df_val = train_test_split(
    df_tr_full, test_size=FRACAO_VAL, stratify=strata_tr, random_state=RNG
)
df_tr = df_tr.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
print(f"Ajuste: {len(df_tr):,} | Validação (early stop): {len(df_val):,}")

enc = encoder_esparso()
enc.fit(df_tr[list(COVARIATEIS)])
nomes_raw = enc.get_feature_names_out(COVARIATEIS)
nomes = [str(n).replace("[", "(").replace("]", ")").replace("<", "lt") for n in nomes_raw]
print(f"Dummies one-hot: {len(nomes)} (esparso CSR)")

X_tr = enc.transform(df_tr[list(COVARIATEIS)])
X_val = enc.transform(df_val[list(COVARIATEIS)])
X_te = enc.transform(df_te[list(COVARIATEIS)])
X_tr_full = enc.transform(df_tr_full[list(COVARIATEIS)])

dtrain = dmatrix_aft(X_tr, df_tr["evento"], df_tr["tempo"])
dval = dmatrix_aft(X_val, df_val["evento"], df_val["tempo"])
dtest = dmatrix_aft(X_te, df_te["evento"], df_te["tempo"])
dtrain_full = dmatrix_aft(X_tr_full, df_tr_full["evento"], df_tr_full["tempo"])

y_tr_full = y_sobrevivencia(df_tr_full["evento"], df_tr_full["tempo"])
y_te = y_sobrevivencia(df_te["evento"], df_te["tempo"])

df_te_i = df_te.copy()
df_te_i["_i"] = np.arange(len(df_te))
idx_te = amostrar_estratificado(df_te_i, n=min(N_AVAL, len(df_te)), random_state=RNG)["_i"].to_numpy()
df_tr_i = df_tr_full.copy()
df_tr_i["_i"] = np.arange(len(df_tr_full))
idx_tr = amostrar_estratificado(df_tr_i, n=min(N_AVAL, len(df_tr_full)), random_state=RNG)["_i"].to_numpy()
print(f"Avaliação: {len(idx_te):,} no teste | {len(idx_tr):,} no treino")

y_te_sub = y_te[idx_te]
y_tr_sub = y_tr_full[idx_tr]
dtest_sub = dmatrix_aft(X_te[idx_te], df_te.iloc[idx_te]["evento"], df_te.iloc[idx_te]["tempo"])
dtrain_sub = dmatrix_aft(
    X_tr_full[idx_tr], df_tr_full.iloc[idx_tr]["evento"], df_tr_full.iloc[idx_tr]["tempo"]
)
X_te_sub = X_te[idx_te]
"""
    ),
    md(
        """**Interpretação da preparação.** A estratificação evento × ano evita que o teste fique enviesado para coortes antigas (quase todas terminadas) ou recentes (muito censuradas). Os limites AFT traduzem a mesma informação `(evento, tempo)` na linguagem de intervalos do `survival:aft`."""
    ),
    md(
        """## 2. Treino comparativo das três distribuições

Hiperparâmetros **fixos** entre `normal`, `logistic` e `extreme`. Documenta-se o tempo de parede e o número de árvores retidas pelo *early stopping*."""
    ),
    code(
        """
resultados = {}
for dist in AFT_DISTRIBUICOES:
    print(f"=== AFT {dist} ===", flush=True)
    t0 = time.perf_counter()
    modelo, hist = treinar_aft_distribuicao(
        dtrain, dist, dval=dval, num_boost=AFT_NUM_BOOST, early_stopping_rounds=AFT_EARLY_STOP
    )
    dt = time.perf_counter() - t0
    best_iter = int(getattr(modelo, "best_iteration", AFT_NUM_BOOST - 1) or (AFT_NUM_BOOST - 1))
    n_arvores = best_iter + 1
    val_ll = None
    if hist and "val" in hist:
        chave = next(iter(hist["val"]))
        serie = hist["val"][chave]
        if serie:
            val_ll = float(serie[best_iter] if best_iter < len(serie) else serie[-1])
    resultados[dist] = {
        "modelo": modelo, "historico": hist, "tempo_s": dt,
        "n_arvores": n_arvores, "val_nloglik": val_ll,
    }
    print(f"  {dt / 60:.2f} min | {n_arvores} árvores | val aft-nloglik={val_ll}", flush=True)

tab_tempos = pd.DataFrame([
    {"distribuicao": d, "tempo_treino_min": round(resultados[d]["tempo_s"] / 60.0, 2),
     "n_arvores": resultados[d]["n_arvores"], "val_aft_nloglik": resultados[d]["val_nloglik"]}
    for d in AFT_DISTRIBUICOES
])
print(tab_tempos.to_string(index=False))
"""
    ),
    md(
        """**Interpretação dos tempos.** As três especificações partilham o mesmo orçamento de árvores e `tree_method='hist'`. Diferenças de tempo reflectem sobretudo o número de rondas até ao *early stopping*."""
    ),
    md(
        """## 3. Reconstrução de $S(t)$ e selecção da distribuição

Para cada distribuição: prevê-se $\\mu$; calibra-se $\\sigma$ por MV censurada no treino; reconstrói-se $\\hat S(t)$; calculam-se C-index e IBS na **subamostra estratificada de 80 000** do teste.

Critério: maximizar **C-index de teste − IBS**. Se a diferença de saldos for $< 0{,}002$, privilegia-se o maior C-index."""
    ),
    md(
        """### Nota de custo computacional (C-index e IBS)

O C-index de Harrell compara processos **par a par**; o número de pares comparáveis cresce quadraticamente ($n^2/2$). O treino do XGBoost com `tree_method='hist'` é aproximadamente linear/log-linear no número de observações. Sobre os ~700 000 processos de teste, o C-index passaria de **200 mil milhões de pares** — inviável nas implementações padrão (`sksurv.concordance_index_censored`).

Por isso, o C-index e o IBS são avaliados numa **subamostra estratificada de 80 000** do teste (evento × ano). O mesmo cálculo repete-se em 80 000 do treino, para verificar estabilidade (coincidência treino–teste). Esta restrição é computacional, não amostral no sentido de reduzir o *treino*: o *booster* vê o universo."""
    ),
    code(
        """
from src.survival_utils import estimar_km

km_tr = estimar_km(y_tr_sub["event"], y_tr_sub["time"])
t_ibs_max = min(1825.0, float(y_te_sub["time"].max()) - 1.0, float(y_tr_sub["time"].max()) - 1.0)
tempos_g = grelha_brier(t_max=t_ibs_max, passo=30)
print(f"Grelha IBS: {len(tempos_g)} pontos até {t_ibs_max:.0f} d")

linhas_tab9 = []
for dist in AFT_DISTRIBUICOES:
    modelo = resultados[dist]["modelo"]
    pred_tr = np.asarray(modelo.predict(dtrain_full), dtype=float)
    n_fin = float(np.mean(np.isfinite(pred_tr)))
    print(f"{dist}: predict finitos={n_fin:.3f}  mediana={np.nanmedian(pred_tr):.3g}", flush=True)
    mu_tr = localizacao_from_pred(pred_tr)
    sigma = ajustar_sigma_aft(mu_tr, df_tr_full["evento"], df_tr_full["tempo"], distribuicao=dist)
    resultados[dist]["sigma"] = float(sigma)
    resultados[dist]["mu_mediana_treino"] = float(np.median(mu_tr))
    print(f"{dist}: μ mediana={resultados[dist]['mu_mediana_treino']:.3f}  σ={sigma:.3f}", flush=True)

    mu_tr_sub = localizacao_from_pred(modelo.predict(dtrain_sub))
    mu_te_sub = localizacao_from_pred(modelo.predict(dtest_sub))
    c_tr = cindex_sub(y_tr_sub, -mu_tr_sub, n_max=None, random_state=RNG)
    c_te = cindex_sub(y_te_sub, -mu_te_sub, n_max=None, random_state=RNG)

    S_te = S_aft(mu_te_sub, tempos_g, sigma, dist)
    _, _, ibs, ibs_km = brier_modelo_e_km(y_tr_sub, y_te_sub, S_te, tempos_g, km_treino=km_tr)

    resultados[dist]["c_treino"] = float(c_tr)
    resultados[dist]["c_teste"] = float(c_te)
    resultados[dist]["ibs"] = float(ibs)
    resultados[dist]["ibs_km"] = float(ibs_km)
    resultados[dist]["score"] = float(c_te - ibs)
    resultados[dist]["S_te"] = S_te
    resultados[dist]["mu_te_sub"] = mu_te_sub

    linhas_tab9.append({
        "distribuicao": dist,
        "C_index_treino": round(c_tr, 3),
        "C_index_teste": round(c_te, 3),
        "IBS": round(ibs, 3),
        "IBS_KM": round(ibs_km, 3),
        "sigma_calibrado": round(sigma, 3),
        "n_arvores": resultados[dist]["n_arvores"],
        "tempo_treino_min": round(resultados[dist]["tempo_s"] / 60.0, 2),
        "score_C_menos_IBS": round(c_te - ibs, 3),
    })
    print(f"{dist:10s}  C-tr={c_tr:.3f}  C-te={c_te:.3f}  IBS={ibs:.3f}  σ={sigma:.3f}", flush=True)

tab9 = pd.DataFrame(linhas_tab9)
vencedor = max(AFT_DISTRIBUICOES, key=lambda d: (resultados[d]["score"], resultados[d]["c_teste"]))
melhores = [d for d in AFT_DISTRIBUICOES if abs(resultados[d]["score"] - resultados[vencedor]["score"]) < 0.002]
if len(melhores) > 1:
    vencedor = max(melhores, key=lambda d: resultados[d]["c_teste"])
tab9["seleccionada"] = np.where(tab9["distribuicao"] == vencedor, "sim", "não")
resultados["vencedor"] = vencedor
print("Distribuição seleccionada:", vencedor)

exportar_tabela_apa(
    tab9, 9,
    "Comparação das três distribuições AFT no conjunto de teste (C-index, IBS e tempo de treino)",
    "N = 80 000 processos estratificados (evento × ano) extraídos do teste (20 % do universo). "
    "C-index de Harrell com risco = −μ. IBS até 5 anos (ou t máximo do teste − 1 dia). "
    "σ calibrado por MV censurada no treino. Hiperparâmetros restantes fixos; seed = 42.",
    DIR_TAB, "tab09_distribuicoes_aft",
)
"""
    ),
    code(
        """
v = resultados["vencedor"]
display(Markdown(f'''
**Interpretação da Tabela 9.** A distribuição **{v}** foi seleccionada com C-index de teste {resultados[v]["c_teste"]:.3f} e IBS {resultados[v]["ibs"]:.3f} (KM: {resultados[v]["ibs_km"]:.3f}). O saldo C-index − IBS é o critério combinado da metodologia.

As três formas deslocam $\\mu(x)$ da mesma maneira; o que muda é a cauda de $T$. Este quadro **entra no corpo da dissertação**: a escolha da distribuição é um resultado empírico. Um IBS inferior ao do Kaplan–Meier indica que as três covariáveis do ajuizamento melhoram a probabilidade prevista.
'''))
"""
    ),
    md("## 4. Métricas de desempenho (distribuição escolhida)"),
    md(
        """### 4.a Índice de concordância (Tabela 10)

O C-index estima a probabilidade de, em dois processos, aquele com menor tempo de término observado (e evento) ter maior risco previsto ($-\\mu$). 0,5 = acaso; 1,0 = ordenação perfeita."""
    ),
    code(
        """
v = resultados["vencedor"]
tab10 = pd.DataFrame({
    "conjunto": ["Treino (subamostra)", "Teste (subamostra)"],
    "n": [len(idx_tr), len(idx_te)],
    "C_index": [round(resultados[v]["c_treino"], 3), round(resultados[v]["c_teste"], 3)],
    "distribuicao": [v, v],
})
exportar_tabela_apa(
    tab10, 10,
    "Índice de concordância do XGBoost AFT (distribuição seleccionada)",
    f"Distribuição {v}; risco = −μ; seed = 42. Subamostras estratificadas de {len(idx_tr):,} (treino) e "
    f"{len(idx_te):,} (teste). Ver nota de custo computacional na secção 3.",
    DIR_TAB, "tab10_cindex_aft",
)
display(Markdown(f'''
**Interpretação da Tabela 10.** O C-index de teste ({resultados[v]["c_teste"]:.3f}) está acima de 0,50: classe, órgão e ano ordenam a celeridade melhor do que o acaso. A diferença treino–teste ({resultados[v]["c_treino"]:.3f} vs. {resultados[v]["c_teste"]:.3f}) mede o sobreajustamento. Valores na casa de 0,70 são típicos com apenas três covariáveis de ajuizamento. O C-index não avalia calibração — daí as secções 4.b e 4.c.
'''))
"""
    ),
    md(
        """### 4.b Brier score dependente do tempo e IBS

Horizontes académicos **fixos**: 6 meses, 1 ano, 2 anos, 3 anos, 5 anos. A referência é o Kaplan–Meier do treino, sem covariáveis."""
    ),
    code(
        """
v = resultados["vencedor"]
modelo = resultados[v]["modelo"]
sigma = resultados[v]["sigma"]
mu_te_sub = resultados[v]["mu_te_sub"]
S_te = resultados[v]["S_te"]

bs_mod, bs_km, ibs_mod, ibs_km = brier_modelo_e_km(y_tr_sub, y_te_sub, S_te, tempos_g)

linhas_brier = []
S_horiz = predict_survival_function_aft(modelo, dtest_sub, sigma, v, HORIZONTES_DIAS)
for nome, t in HORIZONTES_DIAS.items():
    if t > t_ibs_max:
        continue
    j = int(np.argmin(np.abs(tempos_g - t)))
    linhas_brier.append({
        "horizonte": nome, "t_dias": int(t),
        "Brier_AFT": round(float(bs_mod[j]), 3),
        "Brier_KM": round(float(bs_km[j]), 3),
        "S_previsto_medio": round(float(np.mean(S_horiz[t])), 3),
    })
tab11 = pd.DataFrame(linhas_brier)
tab11.loc[len(tab11)] = {
    "horizonte": "IBS (integrado)", "t_dias": int(round(t_ibs_max)),
    "Brier_AFT": round(ibs_mod, 3), "Brier_KM": round(ibs_km, 3), "S_previsto_medio": np.nan,
}
exportar_tabela_apa(
    tab11, 11,
    "Brier score nos horizontes académicos e Integrated Brier Score (XGBoost AFT vs. Kaplan–Meier)",
    f"Distribuição {v}; σ = {sigma:.3f}. Mesma subamostra de teste da Tabela 9. "
    "Horizontes: 6 meses, 1, 2, 3 e 5 anos (fixos na dissertação). Valores mais baixos são melhores.",
    DIR_TAB, "tab11_brier_aft",
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tempos_g / 365.25, bs_km, color=CORES[1], lw=1.8, label="Kaplan–Meier (sem covariáveis)")
ax.plot(tempos_g / 365.25, bs_mod, color=CORES[0], lw=1.8, label=f"XGBoost AFT ({v})")
ax.set_xlabel("Tempo desde o ajuizamento (anos)")
ax.set_ylabel("Brier score")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.legend(frameon=False, loc="upper left")
apa_figure_title(
    fig, ax, 10,
    "Brier score dependente do tempo: XGBoost AFT e Kaplan–Meier de referência",
    "Curva mais baixa indica melhor probabilidade prevista. A referência KM não usa covariáveis. "
    f"Distribuição {v}; subamostra de teste estratificada (n = {len(idx_te):,}).",
)
save_apa_figure(fig, DIR_FIG / "fig10_brier.png", close=False)
plt.show()
plt.close(fig)
"""
    ),
    code(
        """
v = resultados["vencedor"]
display(Markdown(f'''
**Interpretação da Figura 10 e da Tabela 11.** O IBS do AFT ({resultados[v]["ibs"]:.3f}) compara-se com o do KM ({resultados[v]["ibs_km"]:.3f}). Uma redução sistemática nos horizontes de 6 meses a 5 anos significa que classe, órgão e ano melhoram a probabilidade de o processo ainda estar pendente. É esperado que o Brier suba com $t$; o que importa é o hiato face ao KM.
'''))
"""
    ),
    md(
        """### 4.c Curvas de calibração (1, 2 e 5 anos)

Agrupam-se as previsões em **decis** de $\\hat S(t)$. Dentro de cada decil estima-se a sobrevivência observada por Kaplan–Meier (tratamento da censura) e compara-se com a média prevista."""
    ),
    code(
        """
v = resultados["vencedor"]
sigma = resultados[v]["sigma"]
mu_te_sub = resultados[v]["mu_te_sub"]
horiz_calib = [
    ("1 ano", 365, 11, "fig11_calib_1a.png"),
    ("2 anos", 730, 12, "fig12_calib_2a.png"),
    ("5 anos", 1825, 13, "fig13_calib_5a.png"),
]
tabs_cal = []
t_max_obs = float(y_te_sub["time"].max())
for nome, t, num, stem in horiz_calib:
    if t >= t_max_obs:
        print(f"Calibração a {nome}: horizonte omitido (t máx. teste = {t_max_obs:.0f} d).")
        continue
    s_hat = S_aft(mu_te_sub, np.array([t], dtype=float), sigma, v).ravel()
    cal = calibracao_decis(y_te_sub, s_hat, t)
    cal.insert(0, "horizonte", nome)
    tabs_cal.append(cal)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#888888", label="Calibração perfeita")
    ax.errorbar(
        cal["S_previsto_medio"], cal["S_observado_KM"],
        yerr=np.vstack([cal["S_observado_KM"] - cal["IC_95_inf"], cal["IC_95_sup"] - cal["S_observado_KM"]]),
        fmt="o", color=CORES[0], ms=7, capsize=3, label="Decis do teste",
    )
    ax.set_xlabel("S(t) média prevista (XGBoost AFT)")
    ax.set_ylabel("S(t) observada (Kaplan–Meier no decil)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    apa_figure_title(
        fig, ax, num,
        f"Curva de calibração a {nome}: probabilidade de pendência prevista vs. observada",
        f"Cada ponto é um decil de S({nome}) prevista no teste (n = {len(idx_te):,}). "
        f"Barras: IC 95 % log-log do KM no decil. Distribuição {v}.",
    )
    save_apa_figure(fig, DIR_FIG / stem, close=False)
    plt.show()
    plt.close(fig)

if tabs_cal:
    tab_cal = pd.concat(tabs_cal, ignore_index=True)
    tab_cal.to_csv(DIR_TAB / "tab_calib_decis_aft.csv", index=False, encoding="utf-8-sig")
"""
    ),
    code(
        """
display(Markdown('''
**Interpretação das Figuras 11–13.** A calibração pergunta se as **probabilidades** estão certas, não só a ordem. Pontos sobre a diagonal = calibração perfeita; acima = o modelo é pessimista (prevê demasiada pendência); abaixo = optimista. Se a calibração se degradar aos 5 anos, a cauda paramétrica e a censura das coortes recentes são a explicação mais plausível — e devem constar das limitações.
'''))
"""
    ),
    md(
        """## 5. Importância de variáveis

Gain nativo (Figura 14, Tabelas 12 e 12b) e SHAP (Figura 14b, Tabela 12c). Gain sem sinal; SHAP com sinal sobre $\\mu$."""
    ),
    code(
        """
v = resultados["vencedor"]
modelo = resultados[v]["modelo"]
gain_map = modelo.get_score(importance_type="gain")
linhas_gain = []
for i, nome in enumerate(nomes):
    g = float(gain_map.get(nome, gain_map.get(f"f{i}", 0.0)))
    linhas_gain.append({"variavel": nome, "gain": g})
tab12 = pd.DataFrame(linhas_gain).sort_values("gain", ascending=False)
tab12["gain_relativo"] = (tab12["gain"] / tab12["gain"].sum()).round(4) if tab12["gain"].sum() else 0.0
tab12["gain"] = tab12["gain"].round(2)
tab12_top = tab12.head(20).reset_index(drop=True)

exportar_tabela_apa(
    tab12_top, 12,
    "Importância gain do XGBoost AFT (20 variáveis dummy com maior contribuição)",
    f"Distribuição {v}. A tabela completa está em tab12_importancia_aft_completa.csv.",
    DIR_TAB, "tab12_importancia_aft",
)
tab12.to_csv(DIR_TAB / "tab12_importancia_aft_completa.csv", index=False, encoding="utf-8-sig")

tab_grupos = importancia_gain_grupos(modelo, nomes)
exportar_tabela_apa(
    tab_grupos, "12b",
    "Gain agregado por grupo de covariáveis (órgão, classe, ano)",
    "Soma do gain dos dummies de cada covariável original, reescalada para total 1.",
    DIR_TAB, "tab12b_gain_grupos",
)

fig, ax = plt.subplots(figsize=(8, 6))
plot_df = tab12_top.iloc[::-1]
ypos = np.arange(len(plot_df))
ax.barh(ypos, plot_df["gain"], color=CORES[0], height=0.65)
ax.set_yticks(ypos)
ax.set_yticklabels(plot_df["variavel"], fontsize=8)
ax.set_xlabel("Gain (redução da perda AFT)")
apa_figure_title(
    fig, ax, "14",
    "Importância nativa (gain) das 20 variáveis dummy com maior contribuição no XGBoost AFT",
    f"Distribuição {v}. Barras mais longas = splits mais úteis para μ(x).",
)
save_apa_figure(fig, DIR_FIG / "fig14_importancia.png", close=False)
plt.show()
plt.close(fig)
"""
    ),
    code(
        """
v = resultados["vencedor"]
modelo = resultados[v]["modelo"]
shap_ok = False
shap_err = None
try:
    import shap
    rng = np.random.default_rng(RNG)
    n_sh = min(N_SHAP, X_te_sub.shape[0])
    escolhidos = rng.choice(X_te_sub.shape[0], size=n_sh, replace=False)
    X_sh = X_te_sub[escolhidos]
    if hasattr(X_sh, "toarray"):
        X_sh = X_sh.toarray()
    explainer = shap.TreeExplainer(modelo)
    shap_vals = explainer.shap_values(X_sh)
    shap_ok = True
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, X_sh, feature_names=nomes, max_display=20, show=False, color_bar=True)
    fig = plt.gcf()
    fig.suptitle("Figura 14b", fontweight="bold", fontsize=13, x=0.01, ha="left", y=1.04)
    ax = fig.axes[0]
    ax.set_title("Valores SHAP (resumo) do XGBoost AFT — 20 variáveis com maior impacto em μ",
                 fontstyle="italic", fontsize=12, loc="left", pad=10)
    fig.text(0.01, -0.06,
             f"Nota. Subamostra aleatória de {n_sh:,} processos do teste (seed = 42). "
             "SHAP positivo desloca μ para cima (tramitação mais longa). "
             f"Distribuição {v}. Vermelho = presença da categoria (dummy = 1).",
             fontsize=10, fontstyle="italic", ha="left", va="top")
    save_apa_figure(fig, DIR_FIG / "fig14b_shap.png", close=False)
    plt.show()
    plt.close(fig)
    abs_mean = np.abs(shap_vals).mean(axis=0)
    grupos = {"classe_processual": 0.0, "orgao_julgador": 0.0, "ano_ajuizamento": 0.0}
    for i, nome in enumerate(nomes):
        for g in grupos:
            if nome.startswith(g):
                grupos[g] += float(abs_mean[i])
                break
    tot = sum(grupos.values()) or 1.0
    tab_shap_g = pd.DataFrame([
        {"covariavel": g, "mean_abs_SHAP": round(val, 4), "partilha": round(val / tot, 4)}
        for g, val in grupos.items()
    ])
    exportar_tabela_apa(
        tab_shap_g, "12c",
        "Valor absoluto médio de SHAP agregado por grupo de covariáveis",
        f"Média de |SHAP| em {n_sh:,} processos do teste. Partilha = proporção do total.",
        DIR_TAB, "tab12c_shap_grupos",
    )
except Exception as exc:
    shap_err = f"{type(exc).__name__}: {exc}"
    print("SHAP indisponível:", shap_err)
resultados["shap_ok"] = shap_ok
resultados["shap_err"] = shap_err
"""
    ),
    code(
        """
v = resultados["vencedor"]
txt_shap = (
    "O gráfico SHAP (Figura 14b) mostra o sentido do efeito: a presença de certas classes "
    "(p. ex. execução fiscal) empurra μ para valores altos, enquanto classes sumárias o empurram para baixo."
    if resultados.get("shap_ok")
    else (
        "O resumo SHAP não pôde ser gerado nesta execução"
        + (f" ({resultados.get('shap_err')})." if resultados.get("shap_err") else ".")
        + " A interpretação recai sobre o gain (Figura 14 e Tabela 12)."
    )
)
display(Markdown(f'''
**Interpretação da importância (Figuras 14 e 14b; Tabelas 12, 12b e 12c).** O gain concentra-se nos *dummies* que mais reduzem a perda AFT. É frequente o **órgão julgador** aparecer com grande gain agregado (dezenas de níveis). O **ano** pode ter gain baixo se o seu papel for sobretudo truncatura administrativa. Essa leitura alinha-se ao aviso dos Notebooks 1–2: o ano não deve ser interpretado, isoladamente, como «o tribunal ficou mais rápido».

{txt_shap}
'''))
"""
    ),
    md("## 6. Tabela 13 e exportação do modelo"),
    code(
        """
v = resultados["vencedor"]
tab13 = pd.DataFrame([
    {"metrica": "Distribuição AFT", "valor": v},
    {"metrica": "σ calibrado", "valor": f"{resultados[v]['sigma']:.3f}"},
    {"metrica": "Árvores (early stopping)", "valor": str(resultados[v]["n_arvores"])},
    {"metrica": "C-index treino", "valor": f"{resultados[v]['c_treino']:.3f}"},
    {"metrica": "C-index teste", "valor": f"{resultados[v]['c_teste']:.3f}"},
    {"metrica": "IBS (AFT)", "valor": f"{resultados[v]['ibs']:.3f}"},
    {"metrica": "IBS (KM, referência)", "valor": f"{resultados[v]['ibs_km']:.3f}"},
    {"metrica": "Tempo de treino (min)", "valor": f"{resultados[v]['tempo_s'] / 60:.2f}"},
    {"metrica": "n treino+val / n teste", "valor": f"{len(df_tr_full):,} / {len(df_te):,}"},
    {"metrica": "n avaliação (subamostra teste)", "valor": f"{len(idx_te):,}"},
])
exportar_tabela_apa(
    tab13, 13,
    "Resumo consolidado do XGBoost AFT (modelo principal da dissertação)",
    "Métricas na subamostra estratificada de teste, excepto o tempo de treino (universo de ajuste). "
    "Hiperparâmetros: max_depth = 6, eta = 0.05, min_child_weight = 10, tree_method = hist, seed = 42.",
    DIR_TAB, "tab13_resumo_aft",
)

path_modelo = DIR_MOD / "xgboost_aft_modelo_final.json"
resultados[v]["modelo"].save_model(path_modelo)
joblib.dump(enc, DIR_MOD / "xgboost_aft_encoder.joblib")
meta = {
    "algoritmo": "xgboost_survival_aft",
    "distribuicao": v,
    "sigma": resultados[v]["sigma"],
    "sigma_treino_inicial": 1.0,
    "n_arvores": resultados[v]["n_arvores"],
    "c_index_treino": resultados[v]["c_treino"],
    "c_index_teste": resultados[v]["c_teste"],
    "ibs": resultados[v]["ibs"],
    "ibs_km": resultados[v]["ibs_km"],
    "seed": RNG,
    "covariaveis": list(COVARIATEIS),
    "n_features": len(nomes),
    "n_treino": int(len(df_tr_full)),
    "n_teste": int(len(df_te)),
    "n_avaliacao_teste": int(len(idx_te)),
    "hiperparametros": {**PARAMS_AFT_DISSERTACAO, "aft_loss_distribution": v},
    "xgboost_versao": xgb.__version__,
    "anos_treino": [2015, 2025],
}
texto_meta = json.dumps(meta, indent=2, ensure_ascii=False)
(DIR_MOD / "xgboost_aft_meta.json").write_text(texto_meta, encoding="utf-8")
(DIR_META / "xgboost_aft_meta.json").write_text(texto_meta, encoding="utf-8")
print("Modelo:", path_modelo)
print("Encoder:", DIR_MOD / "xgboost_aft_encoder.joblib")
print("Meta:", DIR_META / "xgboost_aft_meta.json")
"""
    ),
    code(
        """
v = resultados["vencedor"]
display(Markdown(f'''
## Interpretação global do desempenho

O XGBoost AFT com distribuição **{v}** é o modelo da dissertação porque treina o universo (n treino = {len(df_tr_full):,}) em {resultados[v]["tempo_s"] / 60:.1f} minutos e produz, no teste, C-index {resultados[v]["c_teste"]:.3f} e IBS {resultados[v]["ibs"]:.3f} (KM: {resultados[v]["ibs_km"]:.3f}). A pergunta de investigação — *no ajuizamento, que probabilidade tem o processo de ainda estar pendente em 6 meses, 1, 2, 3 ou 5 anos?* — fica operacionalizada por $\\hat S(t\\mid x)$ reconstruída a partir de $\\mu(x)$ e $\\sigma$.

Três cautelas acompanham este resultado:

1. **Forma paramétrica.** Todas as curvas individualizadas partilham a família {v}; o que muda entre um processo «rápido» e um «lento» é o deslocamento temporal, não a forma.
2. **Ano de ajuizamento.** Entra como covariável de painel / truncatura. Qualquer leitura de «melhoria de celeridade» exige as curvas estratificadas do Notebook 4 (Figura 18).
3. **Subamostra de métricas.** C-index e IBS foram calculados em 80 000 processos estratificados do teste, não nos {len(df_te):,} do teste integral, por custo quadrático do C-index.

O passo seguinte (Notebook 4) produz as curvas individualizadas, as faixas de celeridade e as anomalias estruturais — sempre com este *booster* e esta distribuição.
'''))
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
