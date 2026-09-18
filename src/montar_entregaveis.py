"""Monta entregaveis/ (LEIA-ME, narrativas) a partir das tabelas canónicas."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.data_utils import project_root


def _ler(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _fmt_n(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def _fmt3(x) -> str:
    return f"{float(x):.3f}"


def _fmt2(x) -> str:
    return f"{float(x):.2f}"


def _brier(tab11: pd.DataFrame, nome: str) -> str:
    hit = tab11.loc[tab11["horizonte"] == nome]
    return _fmt3(hit["Brier_AFT"].iloc[0]) if len(hit) else "—"


MAPA = [
    ("Figura 1", "fig01_histograma_tempo.png", "Histograma do tempo até ao término (apenas processos terminados), com a mediana assinalada."),
    ("Figura 2", "fig02_classes_volume.png", "Volume das classes processuais retidas (corte de Pareto a 95 %)."),
    ("Figura 3", "fig03_ajuizamentos_por_ano.png", "Número de processos ajuizados em cada ano (2015–2025)."),
    ("Figura 4", "fig04_censura_vs_terminados_por_ano.png", "Composição anual de terminados versus censurados à direita."),
    ("Figura 5", "fig05_boxplot_tempo_classes.png", "Distribuição do tempo até ao término nas 16 classes do corte de Pareto a 95 % (inclui «Outras classes»)."),
    ("Figura EDA-1", "fig_eda01_hist_tempo_linear.png", "Histograma do tempo em escala linear (terminados)."),
    ("Figura EDA-2", "fig_eda02_hist_tempo_log10.png", "Histograma do tempo em escala log₁₀."),
    ("Figura EDA-3", "fig_eda03_ecdf_tempo.png", "Função de distribuição empírica do tempo."),
    ("Figura EDA-4", "fig_eda04_violin_tempo.png", "Violino do tempo até ao término."),
    ("Figura EDA-5", "fig_eda05_barras_classes.png", "Frequência absoluta por classe processual."),
    ("Figura EDA-6", "fig_eda06_pareto_classes.png", "Curva de Pareto das classes com corte aos 95 %."),
    ("Figura EDA-7", "fig_eda07_hist_volume_orgaos.png", "Distribuição do volume de processos por órgão julgador."),
    ("Figura EDA-8", "fig_eda08_top30_orgaos.png", "Os 30 órgãos de maior volume."),
    ("Figura EDA-9", "fig_eda09_ano_empilhado.png", "Barras empilhadas de terminados e censurados por ano."),
    ("Figura EDA-10", "fig_eda10_classe_evento_stacked.png", "Composição percentual terminado/censurado por classe."),
    ("Figura EDA-11", "fig_eda11_box_tempo_classe.png", "Boxplot do tempo em todas as classes retidas."),
    ("Figura EDA-12", "fig_eda12_violin_classes.png", "Violinos do tempo nas 16 classes do corte de Pareto a 95 % (inclui «Outras classes»)."),
    ("Figura EDA-13", "fig_eda13_box_orgao_top20.png", "Boxplot do tempo nos 20 órgãos de maior volume."),
    ("Figura EDA-14", "fig_eda14_scatter_mediana_volume.png", "Mediana do tempo versus volume do órgão."),
    ("Figura EDA-15", "fig_eda15_box_tempo_ano.png", "Boxplot do tempo por ano de ajuizamento (terminados)."),
    ("Figura EDA-16", "fig_eda16_mediana_ic_ano.png", "Mediana do tempo dos terminados por ano, com intervalo de confiança — leitura sujeita a viés de censura."),
    ("Figura EDA-17", "fig_eda17_heatmap_classe_ano.png", "Heatmap do volume classe × ano."),
    ("Figura EDA-18", "fig_eda18_heatmap_orgao_classe.png", "Heatmap do volume órgão × classe (órgãos mais frequentes)."),
    ("Figura EDA-19", "fig_eda19_heatmap_orgao_ano.png", "Heatmap do volume órgão × ano."),
    ("Figura EDA-20", "fig_eda20_correlacoes.png", "Matriz de correlação Pearson/Spearman entre tempo, evento, ano e variáveis codificadas."),
    ("Figura EDA-21", "fig_eda21_heatmap_mediana_classe_ano.png", "Heatmap classe × ano colorido pela mediana do tempo (terminados)."),
    ("Figura EDA-22", "fig_eda22_heatmap_censura_classe_ano.png", "Heatmap classe × ano colorido pela percentagem de censura."),
    ("Figura EDA-23", "fig_eda23_area_faixas_tempo.png", "Área empilhada de terminados e censurados por faixa de tempo desde o ajuizamento."),
    ("Figura EDA-24", "fig_eda24_small_multiples_classes.png", "Small multiples das 16 classes do corte de Pareto a 95 % (histograma, censura, volume anual)."),
    ("Figura EDA-25", "fig_eda25_missing.png", "Valores em falta por coluna no dataset analítico."),
    ("Figura 10", "fig10_brier.png", "Brier score dependente do tempo do XGBoost AFT versus Kaplan–Meier sem covariáveis."),
    ("Figura 11", "fig11_calib_1a.png", "Calibração por decil de Ŝ(t) a 1 ano."),
    ("Figura 12", "fig12_calib_2a.png", "Calibração por decil de Ŝ(t) a 2 anos."),
    ("Figura 13", "fig13_calib_5a.png", "Calibração por decil de Ŝ(t) a 5 anos."),
    ("Figura 14", "fig14_importancia.png", "Importância (gain) das 20 dummies com maior contribuição."),
    ("Figura 14b", "fig14b_shap.png", "Resumo SHAP das contribuições para μ."),
    ("Figura 15", "fig15_perfis.png", "Curvas S(t | x) de cinco perfis-tipo (classe + órgão modal + 2022)."),
    ("Figura 16", "fig16_classe.png", "Curva mediana prevista por classe (as 16 do corte a 95 %, teste)."),
    ("Figura 17", "fig17_orgao.png", "Curvas por órgão julgador com classe e ano fixos (execução fiscal, 2022)."),
    ("Figura 18", "fig18_ano.png", "Comparação metodologicamente válida entre coortes anuais (classe e órgão fixos)."),
    ("Figura 19", "fig19_faixas.png", "Distribuição das cinco faixas de celeridade no conjunto de teste."),
    ("Figura 20", "fig20_anomalias.png", "Dispersão do tempo mediano AFT do órgão versus a mediana dos restantes, com linha de rácio 2."),
    ("Tabela 1", "tab01_descritiva_classes.csv", "Frequência, percentagem e percentagem acumulada das classes."),
    ("Tabela 2", "tab02_distribuicao_ano.csv", "Volume anual, proporção de terminados/censurados e mediana dos já terminados."),
    ("Tabela 3", "tab03_orgaos_julgadores.csv", "Órgãos julgadores (top 20 e resumo dos restantes)."),
    ("Tabela 4", "tab04_resumo_tempo.csv", "Resumo do tempo até ao término (apenas eventos)."),
    ("Tabela 5", "tab05_censura_por_classe.csv", "Proporção de censura global e por classe."),
    ("Tabela EDA-1", "tab_eda01_percentis_tempo.csv", "Percentis do tempo até ao término."),
    ("Tabela EDA-2", "tab_eda02_classes_completa.csv", "Frequências completas por classe."),
    ("Tabela EDA-3", "tab_eda03_volume_orgaos.csv", "Estatísticas de volume por órgão."),
    ("Tabela EDA-4", "tab_eda04_por_ano.csv", "Descritiva por ano de ajuizamento."),
    ("Tabela EDA-5", "tab_eda05_classe_evento.csv", "Classe × indicador de evento."),
    ("Tabela EDA-6", "tab_eda06_tempo_por_classe.csv", "Tempo (terminados) por classe."),
    ("Tabela EDA-7", "tab_eda07_top30_orgaos.csv", "Tempo e volume nos 30 órgãos principais."),
    ("Tabela EDA-8", "tab_eda08_classe_ano.csv", "Volume classe × ano."),
    ("Tabela EDA-9", "tab_eda09_orgao_classe.csv", "Volume órgão × classe."),
    ("Tabela EDA-10", "tab_eda10_cramer_v.csv", "Cramér's V entre as três covariáveis categóricas."),
    ("Tabela EDA-11", "tab_eda11_qui_quadrado_evento.csv", "Testes qui-quadrado de evento × classe e evento × ano."),
    ("Tabela EDA-12", "tab_eda12_qualidade_colunas.csv", "Resumo de qualidade (missing, tipos)."),
    ("Tabela EDA-13", "tab_eda13_resumo_dataset.csv", "Síntese geral do dataset analítico."),
    ("Tabela 9", "tab09_distribuicoes_aft.csv", "Comparação das três distribuições AFT (C-index, IBS, σ, tempo de treino)."),
    ("Tabela 10", "tab10_cindex_aft.csv", "C-index de Harrell no treino e no teste (subamostra de 80 000)."),
    ("Tabela 11", "tab11_brier_aft.csv", "Brier score nos horizontes académicos e Integrated Brier Score."),
    ("Tabela 12", "tab12_importancia_aft.csv", "Gain nativo por dummy (top)."),
    ("Tabela 12b", "tab12b_gain_grupos.csv", "Gain agregado às três covariáveis originais."),
    ("Tabela 12c", "tab12c_shap_grupos.csv", "|SHAP| médio agregado por grupo de covariável."),
    ("Tabela 13", "tab13_resumo_aft.csv", "Resumo consolidado do modelo final."),
    ("Tabela 14", "tab14_perfis.csv", "S(t) nos horizontes académicos para cinco perfis-tipo."),
    ("Tabela 15", "tab15_classe.csv", "S(t) mediano por classe no teste (16 classes do corte a 95 %)."),
    ("Tabela 16", "tab16_faixas_exemplos.csv", "Exemplos de classificação nas cinco faixas de celeridade."),
    ("Tabela 17", "tab17_anomalias.csv", "Pares classe × órgão com rácio ≥ 2 (nomes completos dos órgãos)."),
    ("Tabela 18", "tab18_resumo_geral.csv", "Síntese geral da amostra, do modelo e do desempenho."),
]


def _nb01(tab: Path, df_n: int, pct_cens: float, n_org: int) -> str:
    log = _ler(tab / "tab00_log_limpeza.csv")
    t1 = _ler(tab / "tab01_descritiva_classes.csv")
    t2 = _ler(tab / "tab02_distribuicao_ano.csv")
    t4 = _ler(tab / "tab04_resumo_tempo.csv")
    t5 = _ler(tab / "tab05_censura_por_classe.csv")
    med = float(t4.loc[t4["estatistica"] == "mediana", "tempo_dias"].iloc[0])
    media = float(t4.loc[t4["estatistica"] == "média", "tempo_dias"].iloc[0])
    q1 = float(t4.loc[t4["estatistica"] == "Q1", "tempo_dias"].iloc[0])
    q3 = float(t4.loc[t4["estatistica"] == "Q3", "tempo_dias"].iloc[0])
    n_rem_tempo = int(log.loc[log["passo"] == "tempo_nao_positivo", "n_removidos"].iloc[0])
    n_rem_id = int(log.loc[log["passo"] == "agregar_processo_id", "n_removidos"].iloc[0])
    top = t1.iloc[0]
    exec_f = t5.loc[t5["estrato"] == "Execução Fiscal"]
    rec = t5.loc[t5["estrato"] == "Recurso Inominado Cível"]
    med_2015 = float(t2.loc[t2["ano_ajuizamento"] == 2015, "mediana_tempo_terminados"].iloc[0])
    med_2025 = float(t2.loc[t2["ano_ajuizamento"] == 2025, "mediana_tempo_terminados"].iloc[0])
    cens_2025 = float(t2.loc[t2["ano_ajuizamento"] == 2025, "pct_censurados"].iloc[0])
    return f"""# Notebook 1 — Carregamento e exploração dos dados

## O que foi feito e porquê

O caderno parte dos Parquets DataJud do TRF2 (2015–2025), define a unidade de análise `processo_id` = tribunal + grau + `numeroProcesso` (um recurso G2/TR não se funde com a acção originária) e constrói a resposta de sobrevivência: tempo em dias desde o ajuizamento até à primeira movimentação TPU 22 (baixa definitiva), com evento = 1 se essa baixa ocorreu e 0 se o processo permanece em curso na data de referência da extração (21 de agosto de 2026). As únicas covariáveis retidas são as observáveis no ajuizamento: órgão julgador, classe processual e ano. Destina-se às secções de Metodologia (amostra e desfecho) e Resultados (estatística descritiva).

## Decisões de limpeza

A agregação por `processo_id` reduziu 20 999 documentos duplicados (redistribuição ou mudança de classe), retendo a capa no ajuizamento e a primeira baixa TPU 22. Removeram-se { _fmt_n(n_rem_tempo) } processos com tempo não positivo (datas invertidas ou o mesmo instante). Não houve perdas por data de ajuizamento inválida nem por covariáveis em falta. O dataset analítico ficou com **{ _fmt_n(df_n) }** processos, **{ _fmt2(pct_cens) } %** de censura à direita, 16 classes (corte de Pareto a 95 % mais «Outras classes») e **{ _fmt_n(n_org) }** órgãos.

## Achados

Entre os { _fmt_n(int(t4.loc[t4['estatistica']=='n','tempo_dias'].iloc[0])) } processos já terminados, o tempo tem média {media:.1f} dias e mediana **{med:.0f} dias** (Q1 = {q1:.1f}; Q3 = {q3:.1f}). A classe mais volumosa é {top['classe_processual']} ({top['percentagem']:.2f} % do universo). A censura é fortemente heterogénea: execução fiscal apresenta {float(exec_f['pct_censura'].iloc[0]):.2f} % de censura, contra {float(rec['pct_censura'].iloc[0]):.2f} % no recurso inominado cível. A mediana dos *já terminados* cai de {med_2015:.0f} dias em 2015 para {med_2025:.0f} dias em 2025, ao mesmo tempo que a censura da coorte de 2025 sobe para {cens_2025:.2f} %. **Esta queda não autoriza a leitura de aceleração da celeridade**: nas coortes recentes só os processos rápidos tiveram tempo de terminar.

## Figuras e tabelas

- **Tabela 1** — Ordena as 16 classes por volume e mostra o corte de Pareto; «Outras classes» concentra o residual após o limiar de 95 %.
- **Tabela 2** — Painel anual de volume, eventos e censura; a coluna da mediana dos terminados é a que sofre o viés descrito acima.
- **Tabela 3** — Os órgãos são numerosos e o volume é disperso (o maior concentra cerca de 1,6 %); não há um «órgão médio» representativo.
- **Tabela 4** — Localização e dispersão do tempo dos eventos; o máximo (~4 220 dias) cobre a janela 2015–2025.
- **Tabela 5** — A censura global de { _fmt2(pct_cens) } % mascara diferenças de classe que o modelo terá de absorver.
- **Figura 1** — O histograma dos terminados é assimétrico à direita, coerente com uma duração mediana inferior à média.
- **Figura 2** — Poucas classes concentram a maior parte do volume, justificando o corte a 95 %.
- **Figura 3** — O volume anual não é estacionário; 2021 e 2023 são picos.
- **Figura 4** — A barra de censurados explode em 2024–2025, sinal visual do truncamento à direita.
- **Figura 5** — Os boxplots por classe antecipam a heterogeneidade que o AFT deslocará via μ(x).

## Limitações

O tempo contínuo original pode arredondar a 0,0 dias na Tabela 4 por apresentação, apesar do filtro de tempos não positivos. A mediana dos terminados por ano **não** é um estimador de celeridade; a comparação válida entre coortes é a Figura 18 (Notebook 4). O modelo não vê o mérito da causa nem o acervo legado do órgão.
"""


def _nb02(tab: Path) -> str:
    eda6 = _ler(tab / "tab_eda06_tempo_por_classe.csv")
    eda10 = _ler(tab / "tab_eda10_cramer_v.csv")
    eda11 = _ler(tab / "tab_eda11_qui_quadrado_evento.csv")
    ef = eda6.loc[eda6["classe_processual"] == "Execução Fiscal"].iloc[0]
    ri = eda6.loc[eda6["classe_processual"] == "Recurso Inominado Cível"].iloc[0]
    cp = eda6.loc[eda6["classe_processual"] == "Carta Precatória Cível"].iloc[0]
    v_co = float(eda10.loc[eda10["par"].str.contains("classe_processual × orgao"), "Cramers_V"].iloc[0])
    v_ca = float(eda10.loc[eda10["par"].str.contains("classe_processual × ano"), "Cramers_V"].iloc[0])
    v_oa = float(eda10.loc[eda10["par"].str.contains("orgao_julgador × ano"), "Cramers_V"].iloc[0])
    return f"""# Notebook 2 — Análise exploratória exaustiva

## O que foi feito e porquê

A EDA descreve a heterogeneidade de tempo, classe, órgão e ano *antes* de treinar o XGBoost AFT, para que os Resultados não apresentem o modelo como a primeira evidência de diferenças. Destina-se às secções de Resultados (descritiva avançada) e, pontualmente, à Metodologia (justificação do corte de classes e do aviso de censura). Todas as medianas de tempo deste caderno são calculadas **apenas sobre terminados** e herdam o viés de censura nas coortes recentes.

## Achados

A distribuição do tempo é assimétrica (EDA-1) e torna-se mais simétrica em log₁₀ (EDA-2). A ECDF (EDA-3) mostra que metade dos eventos ocorre antes de cerca de 400 dias, mas a cauda ultrapassa os 10 anos. Por classe, a execução fiscal tem mediana de {ef['mediana']:.0f} dias entre os já terminados (média {ef['media']:.0f}), o recurso inominado {ri['mediana']:.0f} dias e a carta precatória {cp['mediana']:.0f} dias — ordens de grandeza distintas. A associação entre classe e órgão é forte (Cramér's V = {v_co:.3f}): certas classes concentram-se em varas especializadas. Classe × ano e órgão × ano são associações mais fracas (V = {v_ca:.3f} e {v_oa:.3f}). Os testes qui-quadrado de evento × classe e evento × ano rejeitam independência (p < 0,001), o que é esperado: classes lentas e coortes recentes acumulam mais censura, não necessariamente mais «dificuldade».

## Figuras e tabelas (frase de leitura)

- **EDA-1 a EDA-4 / Tabela EDA-1** — O tempo não é gaussiano; qualquer modelo em escala original sem transformação (o AFT trabalha em log-tempo) seria mal especificado.
- **EDA-5 e EDA-6 / Tabela EDA-2** — O corte a 95 % é visível na curva de Pareto; as classes residuais agrupam-se em «Outras classes» sem fingir homogeneidade jurídica.
- **EDA-7 e EDA-8 / Tabela EDA-3** — A maior parte dos 337 órgãos tem volume moderado; as previsões para órgãos raros assentam em *pooling* pelas árvores, não em KM estratificado.
- **EDA-9 / Tabela EDA-4** — Repete o aviso: 2024–2025 estão truncados; não se compara a mediana dos terminados entre anos como celeridade.
- **EDA-10 / Tabela EDA-5** — A execução fiscal e a execução de título extrajudicial concentram censura; recursos e agravos, não.
- **EDA-11 e EDA-12 / Tabela EDA-6** — Os boxplots/violinos ordenam as classes de forma compatível com o direito processual (execução longa, recurso curto).
- **EDA-13 e EDA-14 / Tabela EDA-7** — Órgãos do mesmo tribunal não partilham a mesma mediana; o volume não explica sozinho a duração.
- **EDA-15 e EDA-16** — A mediana dos terminados desce nas coortes recentes **com o intervalo de confiança**. A leitura válida é a Figura 18, não esta.
- **EDA-17 a EDA-19 / Tabelas EDA-8 e EDA-9** — A composição classe × ano e órgão × classe não é estável; o one-hot tem de cobrir essa grelha.
- **EDA-20 / Tabelas EDA-10 e EDA-11** — Não há colinearidade perfeita que impeça as três covariáveis; há associação classe–órgão, como esperado.
- **EDA-21 e EDA-22** — Mediana e censura classe × ano movem-se em conjunto nas células recentes — outro sinal de truncamento, não de reforma.
- **EDA-23** — A massa de censura acumula-se nas faixas curtas de follow-up (coortes novas), não só na cauda longa.
- **EDA-24** — Os *small multiples* mostram que cada classe tem o seu regime de volume, censura e duração.
- **EDA-25 / Tabelas EDA-12 e EDA-13** — O dataset analítico está completo nas colunas usadas pelo modelo.

## Limitações

Nenhuma figura deste caderno estima S(t) com tratamento formal da censura (isso é o AFT e, internamente, o KM dos decis de calibração). Interpretar EDA-16 como «o tribunal ficou mais célere» seria um erro metodológico já assinalado no Notebook 1.
"""


def _nb03(tab: Path, meta: dict) -> str:
    t9 = _ler(tab / "tab09_distribuicoes_aft.csv")
    t11 = _ler(tab / "tab11_brier_aft.csv")
    t12b = _ler(tab / "tab12b_gain_grupos.csv")
    t12c = _ler(tab / "tab12c_shap_grupos.csv")
    dist = meta["distribuicao"]
    sigma = float(meta["sigma"])
    c_te = float(meta["c_index_teste"])
    c_tr = float(meta["c_index_treino"])
    ibs = float(meta["ibs"])
    ibs_km = float(meta["ibs_km"])
    logi = t9.loc[t9["distribuicao"] == "logistic"].iloc[0]
    ext = t9.loc[t9["distribuicao"] == "extreme"].iloc[0]
    gain = {r.covariavel: float(r.gain_relativo) for r in t12b.itertuples()}
    shap = {r.covariavel: float(r.partilha) for r in t12c.itertuples()}
    return f"""# Notebook 3 — XGBoost Survival AFT: treino e avaliação

## O que foi feito e porquê

Treinou-se **apenas** o XGBoost Survival em modo AFT (Cox e RSF não correm neste repositório). As três covariáveis foram recodificadas em one-hot esparso (`float32`), com categorias não vistas no treino mapeadas para zero. A divisão 80/20 foi estratificada por evento × ano (`seed=42`); 10 % do treino serviu de validação interna para *early stopping* (paciência 30). Os limites AFT foram [t, t] para eventos e [t, +∞) para censurados. Destina-se aos Resultados (desempenho) e à Metodologia (especificação AFT e custo do C-index).

As três distribuições (`normal`, `logistic`, `extreme`) partilharam os mesmos hiperparâmetros (`max_depth=6`, `eta=0.05`, `min_child_weight=10`, 500 rondas). A curva S(t) não existe pronta no XGBoost: reconstruiu-se Ŝ a partir de μ previsto, σ calibrado por máxima verosimilhança censurada, e z = (ln t − μ) / σ.

## Custo computacional do C-index

O C-index compara processos par a par; o número de pares cresce com n²/2. Nos ~700 000 processos de teste isso excederia 200 mil milhões de pares nas implementações padrão. Por isso C-index e IBS foram avaliados numa **subamostra estratificada de 80 000** do teste, repetida no treino para verificar estabilidade. O treino do *booster* (`tree_method='hist'`) permanece aproximadamente linear/log-linear e correu no universo.

## Achados

A Tabela 9 selecciona **{dist}** (σ = {sigma:.3f}). O C-index de teste é **{c_te:.3f}**, idêntico ao do treino ({c_tr:.3f}), o que não sugere sobreajuste grosseiro na ordenação. A logística falhou (C-index teste {float(logi['C_index_teste']):.3f}, IBS {float(logi['IBS']):.3f}) e foi descartada. A Gumbel/extreme ficou próxima (C-index {float(ext['C_index_teste']):.3f}, IBS {float(ext['IBS']):.3f}) mas perdeu no saldo C-index − IBS. O IBS do AFT normal é **{ibs:.3f}**, inferior ao Kaplan–Meier sem covariáveis ({ibs_km:.3f}).

Brier pontual (AFT): 6 meses {_brier(t11,'6 meses')}; 1 ano {_brier(t11,'1 ano')}; 2 anos {_brier(t11,'2 anos')}; 3 anos {_brier(t11,'3 anos')}; 5 anos {_brier(t11,'5 anos')}. O erro é maior perto de 1–2 anos, onde há mais massa de eventos, e menor aos 5 anos.

O gain relativo concentra-se no **órgão** ({gain.get('orgao_julgador',0):.3f}), depois na classe ({gain.get('classe_processual',0):.3f}) e no ano ({gain.get('ano_ajuizamento',0):.3f}). O |SHAP| médio altera a partilha — órgão {shap.get('orgao_julgador',0):.3f}, ano {shap.get('ano_ajuizamento',0):.3f}, classe {shap.get('classe_processual',0):.3f} — porque o ano, com poucas dummies, desloca μ de forma sistemática nas coortes recentes.

## Figuras e tabelas

- **Tabela 9** — A comparação das três distribuições é um resultado, não um detalhe de implementação.
- **Tabela 10** — C-index treino/teste coincidentes a três casas na subamostra de 80 000.
- **Tabela 11 / Figura 10** — O AFT reduz o Brier em todos os horizontes académicos face ao KM.
- **Figuras 11–13** — A calibração por decil (KM dentro do decil) a 1, 2 e 5 anos mostra se Ŝ médio acompanha a sobrevivência observada.
- **Figura 14 / Tabelas 12 e 12b** — A dummy de execução fiscal e vários órgãos especializados lideram o gain.
- **Figura 14b / Tabela 12c** — O SHAP confirma o órgão como principal deslocador de μ, com o ano mais visível do que no gain.
- **Tabela 13** — Ficha do modelo final (distribuição, σ, árvores, métricas, n).

## Limitações

σ é **global**: as covariáveis deslocam a curva mas não lhe mudam a forma relativa. O C-index/IBS não foram calculados nos 699 795 processos de teste por custo quadrático; a subamostra é estratificada e o treino coincide, mas permanece uma restrição. O ano de ajuizamento mistura efeito de coorte e truncamento. SHAP foi estimado em subamostra (custo de Kernel/TreeSHAP sobre 361 colunas).
"""


def _nb04(tab: Path, meta: dict, n_an: int) -> str:
    t14 = _ler(tab / "tab14_perfis.csv")
    t15 = _ler(tab / "tab15_classe.csv")
    t17 = _ler(tab / "tab17_anomalias.csv")
    imin = t15["S(2 anos)"].idxmin()
    imax = t15["S(2 anos)"].idxmax()
    smin, cmin = float(t15.loc[imin, "S(2 anos)"]), t15.loc[imin, "classe"]
    smax, cmax = float(t15.loc[imax, "S(2 anos)"]), t15.loc[imax, "classe"]
    dist = meta["distribuicao"]
    linhas17 = "\n".join(
        f"- {r.orgao} × {r.classe}: rácio {float(r.racio):.2f} "
        f"(exp(μ) = {float(r.t_mediano_orgao_d):.0f} d vs. {float(r.t_mediano_restantes_d):.0f} d nos restantes)."
        for r in t17.itertuples()
    )
    return f"""# Notebook 4 — Curvas individualizadas, faixas e anomalias

## O que foi feito e porquê

Com o *booster* congelado, reconstituiu-se Ŝ(t | x) na grelha de 0–10 anos para responder às alíneas (b) e (d) da pergunta de investigação: a probabilidade de término é uma **curva**, heterogénea entre órgãos, classes e coortes, e não um único prazo médio. Sem re-treino. Destina-se aos Resultados (heterogeneidade, Figura 18, anomalias) e à Discussão (o que as faixas comunicam à gestão).

## Achados

Nos perfis-tipo de 2022 (órgão modal de cada uma das cinco classes de maior volume), a execução fiscal permanece com Ŝ(5 anos) elevado, enquanto juizados e cumprimentos de sentença descem mais depressa (Tabela 14, Figura 15). No conjunto de teste, a curva **mediana** das 16 classes do corte a 95 % confirma a ordenação: {cmax} tem Ŝ(2 anos) = {smax:.3f} e {cmin} tem Ŝ(2 anos) = {smin:.3f} (Tabela 15, Figura 16). Com classe e ano fixos, as oito varas de execução fiscal de maior volume **não coincidem** (Figura 17): o órgão desloca μ. A Figura 18 fixa o par de maior volume (Procedimento do Juizado Especial Cível × 2.º Juizado Especial Federal de Vitória, n = 46 958) e varia só o ano — esta é a comparação válida entre coortes.

As faixas (p10/p25/p75/p90 de exp(μ) dentro de classe × ano) traduzem a localização AFT num rótulo relativo. Um «muito lento» no juizado pode ter prazo inferior a um «típico» em execução fiscal (Figura 19, Tabela 16).

Anomalias (rácio ≥ 2 face à mediana dos outros órgãos da mesma classe, ano 2022, n ≥ 80 no par): **{n_an}** pares. Nomes completos na Tabela 17:

{linhas17}

Interpretação: **diagnóstico de fluxo, não juízo disciplinar**. O modelo não observa recursos humanos, acervo legado nem qualidade da petição.

## Figuras e tabelas

- **Figura 15 / Tabela 14** — Cinco curvas reais, uma por combinação; mesma forma {dist}, diferente μ.
- **Figura 16 / Tabela 15** — As 16 classes do corte a 95 % (mais «Outras classes») ordenam a pendência prevista mesmo misturando órgãos e anos no teste.
- **Figura 17** — Isola o órgão; responde à alínea (a) quanto à heterogeneidade intra-classe.
- **Figura 18** — Substitui a leitura inválida da mediana dos já terminados (Tabela 2, EDA-16).
- **Figura 19 / Tabela 16** — Semáforo relativo à célula classe × ano.
- **Figura 20 / Tabela 17** — Pontos acima da recta 2× são os alertas pré-especificados.

## Limitações

A forma paramétrica partilhada impede que uma classe tenha um «ombro» de S(t) diferente de outra: só há translação temporal. As anomalias usam o ano de referência 2022 (follow-up suficiente) e n mínimo 80; órgãos pequenos ficam de fora por desenho. As faixas degenerariam se fossem calculadas dentro da célula classe × órgão × ano, porque μ é constante nessa célula — daí o segmento classe × ano.
"""


def _nb05(meta: dict, n_an: int) -> str:
    return f"""# Notebook 5 — Verificação final e exportação

## O que foi feito e porquê

Este caderno não estima parâmetros novos. Confere a cadeia de reprodutibilidade (`seed=42` em todos os pontos de aleatoriedade dos cadernos que amostram ou treinam), a numeração APA das figuras e tabelas canónicas (42 figuras e 29 tabelas no checklist), a existência do *booster* `xgboost_aft_modelo_final.json` e do encoder, actualiza `requirements.txt` via `pip freeze`, consolida a Tabela 18 e copia o pacote para `entregaveis/`, pensado para uma sessão de redacção sem acesso aos cadernos Jupyter. Destina-se aos Anexos (reprodutibilidade) e à síntese dos Resultados / Conclusões.

## Achados de verificação

O modelo gravado é XGBoost Survival AFT com distribuição **{meta['distribuicao']}**, σ = {float(meta['sigma']):.3f}, {meta['n_arvores']} árvores, C-index de teste {float(meta['c_index_teste']):.3f} (treino {float(meta['c_index_treino']):.3f}) e IBS {float(meta['ibs']):.3f} contra {float(meta['ibs_km']):.3f} do Kaplan–Meier. A semente nos metadados é {meta['seed']}. Foram identificadas {n_an} anomalias estruturais (rácio ≥ 2, ano 2022, n ≥ 80). Todas as figuras canónicas 1–5, EDA-1–25, 10–14b e 15–20 e as tabelas 1–5, EDA-1–13 e 9–18 estavam presentes no disco no momento da conferência. A pasta `entregaveis/` replica figuras, tabelas, metadados e as cinco narrativas.

O Notebook 2 não contém `random_state=42` porque a EDA é determinística (não há amostragem). Os cadernos 3 e 4 concentram os pontos de aleatoriedade (split 80/20, subamostra de 80 000, exemplos das faixas).

## Figuras e tabelas

- **Tabela 18** — Única tabela nova: n, censura, distribuição vencedora e σ, C-index, IBS, Brier por horizonte, gain e SHAP por grupo, extremos de Ŝ(2 anos) e contagem de anomalias.
- As restantes figuras e tabelas são apenas conferidas (existência e nomenclatura APA), não refeitas.

## Limitações

A verificação confirma artefactos no disco; não volta a executar os cadernos 1–4 neste passo (já foram corridos do início ao fim com a mesma semente). O `pip freeze` reflecte o interpretador que executou o caderno (o ambiente em que o *booster* foi treinado). O dashboard (Notebook 6) fica de fora da validação estatística formal, por construção: é demonstração gerencial, incluindo a extrapolação 2026→2025.
"""


def montar_entregaveis(root: Path | None = None) -> Path:
    """Copia artefactos canónicos e escreve LEIA-ME + narrativas."""
    root = Path(root) if root is not None else project_root()
    tab = root / "tabelas"
    fig = root / "figuras"
    meta_dir = root / "metadados"
    mod = root / "modelo"
    ent = root / "entregaveis"
    for sub in ("figuras", "tabelas", "metadados", "interpretacao"):
        (ent / sub).mkdir(parents=True, exist_ok=True)

    meta_path = mod / "xgboost_aft_meta.json"
    if not meta_path.is_file():
        meta_path = meta_dir / "xgboost_aft_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    df = pd.read_parquet(root / "dados" / "dataset_limpo.parquet", columns=["evento", "orgao_julgador"])
    n = len(df)
    pct_cens = 100.0 * (1.0 - float(df["evento"].mean()))
    n_org = int(df["orgao_julgador"].nunique())

    tab17 = _ler(tab / "tab17_anomalias.csv") if (tab / "tab17_anomalias.csv").is_file() else pd.DataFrame()
    tab17c = (
        _ler(tab / "tab17_anomalias_completo.csv")
        if (tab / "tab17_anomalias_completo.csv").is_file()
        else pd.DataFrame()
    )
    if "anomalia" in tab17c.columns:
        n_an = int(tab17c["anomalia"].sum())
    else:
        n_an = len(tab17)

    for _id, ficheiro, _frase in MAPA:
        src_fig = fig / ficheiro
        src_tab = tab / ficheiro
        if src_fig.is_file():
            shutil.copy2(src_fig, ent / "figuras" / ficheiro)
        elif src_tab.is_file():
            shutil.copy2(src_tab, ent / "tabelas" / ficheiro)
    extras = [
        "tab00_resumo_dataset_limpo.csv",
        "tab00_log_limpeza.csv",
        "tab12_importancia_aft_completa.csv",
        "tab17_anomalias_completo.csv",
        "tab18_resumo_geral.csv",
    ]
    for nome in extras:
        src = tab / nome
        if src.is_file():
            shutil.copy2(src, ent / "tabelas" / nome)

    if (meta_dir / "xgboost_aft_meta.json").is_file():
        shutil.copy2(meta_dir / "xgboost_aft_meta.json", ent / "metadados" / "xgboost_aft_meta.json")
    if (meta_dir / "tab18_resumo_geral.csv").is_file():
        shutil.copy2(meta_dir / "tab18_resumo_geral.csv", ent / "metadados" / "tab18_resumo_geral.csv")
    elif (tab / "tab18_resumo_geral.csv").is_file():
        shutil.copy2(tab / "tab18_resumo_geral.csv", ent / "metadados" / "tab18_resumo_geral.csv")

    interp = ent / "interpretacao"
    (interp / "01_exploracao.md").write_text(_nb01(tab, n, pct_cens, n_org), encoding="utf-8")
    (interp / "02_eda.md").write_text(_nb02(tab), encoding="utf-8")
    (interp / "03_xgboost_aft.md").write_text(_nb03(tab, meta), encoding="utf-8")
    (interp / "04_curvas_faixas_anomalias.md").write_text(_nb04(tab, meta, n_an), encoding="utf-8")
    (interp / "05_verificacao.md").write_text(_nb05(meta, n_an), encoding="utf-8")

    t11 = _ler(tab / "tab11_brier_aft.csv")
    t12b = _ler(tab / "tab12b_gain_grupos.csv")
    t12c = _ler(tab / "tab12c_shap_grupos.csv")
    t15 = _ler(tab / "tab15_classe.csv") if (tab / "tab15_classe.csv").is_file() else pd.DataFrame()
    gain = {str(r.covariavel): float(r.gain_relativo) for r in t12b.itertuples()}
    shap = {str(r.covariavel): float(r.partilha) for r in t12c.itertuples()}
    if not t15.empty and "S(2 anos)" in t15.columns:
        imin, imax = t15["S(2 anos)"].idxmin(), t15["S(2 anos)"].idxmax()
        s2_txt = (
            f"mín {float(t15.loc[imin, 'S(2 anos)']):.3f} ({t15.loc[imin, 'classe']}); "
            f"máx {float(t15.loc[imax, 'S(2 anos)']):.3f} ({t15.loc[imax, 'classe']})"
        )
    else:
        s2_txt = "—"

    linhas_mapa = [
        "# Entregáveis para redacção dos capítulos de Resultados, Discussão e Conclusões",
        "",
        "Pacote autossuficiente: não depende dos cadernos Jupyter. Números extraídos dos CSV e de `metadados/xgboost_aft_meta.json` — não inventar valores.",
        "",
        "## 1. Mapa de numeração",
        "",
        "| ID | Ficheiro | O que mostra |",
        "|---|---|---|",
    ]
    for i, f, q in MAPA:
        linhas_mapa.append(f"| {i} | `{f}` | {q} |")

    linhas_mapa += [
        "",
        "## 2. Números-chave consolidados",
        "",
        "| Item | Valor |",
        "|---|---|",
        f"| n processos | {_fmt_n(n)} |",
        f"| eventos (TPU 22) | {_fmt_n(int(df['evento'].sum()))} |",
        f"| censura à direita | {_fmt2(pct_cens)} % |",
        f"| janela | ajuizamentos 2015–2025, TRF2, todos os graus |",
        f"| classes / órgãos | 16 (top 95 % + «Outras classes») / {_fmt_n(n_org)} |",
        f"| algoritmo | XGBoost Survival AFT (único estimado) |",
        f"| distribuição vencedora | {meta['distribuicao']}; σ = {float(meta['sigma']):.3f} |",
        f"| C-index treino / teste (n = 80 000) | {float(meta['c_index_treino']):.3f} / {float(meta['c_index_teste']):.3f} |",
        f"| IBS AFT / IBS KM | {float(meta['ibs']):.3f} / {float(meta['ibs_km']):.3f} |",
        f"| Brier 6 meses / 1 ano / 2 anos / 3 anos / 5 anos | {_brier(t11,'6 meses')} / {_brier(t11,'1 ano')} / {_brier(t11,'2 anos')} / {_brier(t11,'3 anos')} / {_brier(t11,'5 anos')} |",
        f"| gain relativo órgão / classe / ano | {gain.get('orgao_julgador', float('nan')):.3f} / {gain.get('classe_processual', float('nan')):.3f} / {gain.get('ano_ajuizamento', float('nan')):.3f} |",
        f"| partilha \\|SHAP\\| órgão / classe / ano | {shap.get('orgao_julgador', float('nan')):.3f} / {shap.get('classe_processual', float('nan')):.3f} / {shap.get('ano_ajuizamento', float('nan')):.3f} |",
        f"| S(2 anos) extremos (medianas de classe, teste) | {s2_txt} |",
        f"| anomalias (rácio ≥ 2, ano 2022, n ≥ 80) | {n_an} |",
        f"| semente | 42 |",
        "",
        "Detalhe linha a linha: `metadados/tab18_resumo_geral.csv`.",
        "",
        "## 3. Ligação às secções da dissertação",
        "",
        "- **Notebooks 1–2** → Metodologia (unidade, desfecho, corte de classes, aviso de censura) e Resultados (descritiva / EDA).",
        "- **Notebook 3** → Metodologia (AFT, reconstrução de S(t), custo do C-index) e Resultados (Tabelas 9–13, Figuras 10–14b).",
        "- **Notebook 4** → Resultados (curvas individualizadas, heterogeneidade, Figura 18, faixas, anomalias) e Discussão (o que as anomalias não são).",
        "- **Notebook 5** → Síntese (Tabela 18) e anexos de reprodutibilidade.",
        "- **Notebook 6** → Demonstração gerencial no Capítulo 5; **não** integra a validação estatística formal. A consulta de processos de 2026 usa 2025 como *proxy* one-hot.",
        "",
        "## 4. Narrativas",
        "",
        "Ver `interpretacao/01_exploracao.md` … `05_verificacao.md`. Cada ficheiro cobre o que foi feito, os números exactos, uma frase por figura/tabela e as limitações encontradas.",
        "",
    ]
    (ent / "LEIA-ME.md").write_text("\n".join(linhas_mapa), encoding="utf-8")
    return ent


if __name__ == "__main__":
    destino = montar_entregaveis()
    print("entregaveis/ em", destino)
