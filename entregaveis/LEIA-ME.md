# Entregáveis para redacção dos capítulos de Resultados, Discussão e Conclusões

Pacote autossuficiente: não depende dos cadernos Jupyter. Números extraídos dos CSV e de `metadados/xgboost_aft_meta.json` — não inventar valores.

## 1. Mapa de numeração

| ID | Ficheiro | O que mostra |
|---|---|---|
| Figura 1 | `fig01_histograma_tempo.png` | Histograma do tempo até ao término (apenas processos terminados), com a mediana assinalada. |
| Figura 2 | `fig02_classes_volume.png` | Volume das classes processuais retidas (corte de Pareto a 95 %). |
| Figura 3 | `fig03_ajuizamentos_por_ano.png` | Número de processos ajuizados em cada ano (2015–2025). |
| Figura 4 | `fig04_censura_vs_terminados_por_ano.png` | Composição anual de terminados versus censurados à direita. |
| Figura 5 | `fig05_boxplot_tempo_classes.png` | Distribuição do tempo até ao término nas 16 classes do corte de Pareto a 95 % (inclui «Outras classes»). |
| Figura EDA-1 | `fig_eda01_hist_tempo_linear.png` | Histograma do tempo em escala linear (terminados). |
| Figura EDA-2 | `fig_eda02_hist_tempo_log10.png` | Histograma do tempo em escala log₁₀. |
| Figura EDA-3 | `fig_eda03_ecdf_tempo.png` | Função de distribuição empírica do tempo. |
| Figura EDA-4 | `fig_eda04_violin_tempo.png` | Violino do tempo até ao término. |
| Figura EDA-5 | `fig_eda05_barras_classes.png` | Frequência absoluta por classe processual. |
| Figura EDA-6 | `fig_eda06_pareto_classes.png` | Curva de Pareto das classes com corte aos 95 %. |
| Figura EDA-7 | `fig_eda07_hist_volume_orgaos.png` | Distribuição do volume de processos por órgão julgador. |
| Figura EDA-8 | `fig_eda08_top30_orgaos.png` | Os 30 órgãos de maior volume. |
| Figura EDA-9 | `fig_eda09_ano_empilhado.png` | Barras empilhadas de terminados e censurados por ano. |
| Figura EDA-10 | `fig_eda10_classe_evento_stacked.png` | Composição percentual terminado/censurado por classe. |
| Figura EDA-11 | `fig_eda11_box_tempo_classe.png` | Boxplot do tempo em todas as classes retidas. |
| Figura EDA-12 | `fig_eda12_violin_classes.png` | Violinos do tempo nas 16 classes do corte de Pareto a 95 % (inclui «Outras classes»). |
| Figura EDA-13 | `fig_eda13_box_orgao_top20.png` | Boxplot do tempo nos 20 órgãos de maior volume. |
| Figura EDA-14 | `fig_eda14_scatter_mediana_volume.png` | Mediana do tempo versus volume do órgão. |
| Figura EDA-15 | `fig_eda15_box_tempo_ano.png` | Boxplot do tempo por ano de ajuizamento (terminados). |
| Figura EDA-16 | `fig_eda16_mediana_ic_ano.png` | Mediana do tempo dos terminados por ano, com intervalo de confiança — leitura sujeita a viés de censura. |
| Figura EDA-17 | `fig_eda17_heatmap_classe_ano.png` | Heatmap do volume classe × ano. |
| Figura EDA-18 | `fig_eda18_heatmap_orgao_classe.png` | Heatmap do volume órgão × classe (órgãos mais frequentes). |
| Figura EDA-19 | `fig_eda19_heatmap_orgao_ano.png` | Heatmap do volume órgão × ano. |
| Figura EDA-20 | `fig_eda20_correlacoes.png` | Matriz de correlação Pearson/Spearman entre tempo, evento, ano e variáveis codificadas. |
| Figura EDA-21 | `fig_eda21_heatmap_mediana_classe_ano.png` | Heatmap classe × ano colorido pela mediana do tempo (terminados). |
| Figura EDA-22 | `fig_eda22_heatmap_censura_classe_ano.png` | Heatmap classe × ano colorido pela percentagem de censura. |
| Figura EDA-23 | `fig_eda23_area_faixas_tempo.png` | Área empilhada de terminados e censurados por faixa de tempo desde o ajuizamento. |
| Figura EDA-24 | `fig_eda24_small_multiples_classes.png` | Small multiples das 16 classes do corte de Pareto a 95 % (histograma, censura, volume anual). |
| Figura EDA-25 | `fig_eda25_missing.png` | Valores em falta por coluna no dataset analítico. |
| Figura 10 | `fig10_brier.png` | Brier score dependente do tempo do XGBoost AFT versus Kaplan–Meier sem covariáveis. |
| Figura 11 | `fig11_calib_1a.png` | Calibração por decil de Ŝ(t) a 1 ano. |
| Figura 12 | `fig12_calib_2a.png` | Calibração por decil de Ŝ(t) a 2 anos. |
| Figura 13 | `fig13_calib_5a.png` | Calibração por decil de Ŝ(t) a 5 anos. |
| Figura 14 | `fig14_importancia.png` | Importância (gain) das 20 dummies com maior contribuição. |
| Figura 14b | `fig14b_shap.png` | Resumo SHAP das contribuições para μ. |
| Figura 15 | `fig15_perfis.png` | Curvas S(t | x) de cinco perfis-tipo (classe + órgão modal + 2022). |
| Figura 16 | `fig16_classe.png` | Curva mediana prevista por classe (as 16 do corte a 95 %, teste). |
| Figura 17 | `fig17_orgao.png` | Curvas por órgão julgador com classe e ano fixos (execução fiscal, 2022). |
| Figura 18 | `fig18_ano.png` | Comparação metodologicamente válida entre coortes anuais (classe e órgão fixos). |
| Figura 19 | `fig19_faixas.png` | Distribuição das cinco faixas de celeridade no conjunto de teste. |
| Figura 20 | `fig20_anomalias.png` | Dispersão do tempo mediano AFT do órgão versus a mediana dos restantes, com linha de rácio 2. |
| Tabela 1 | `tab01_descritiva_classes.csv` | Frequência, percentagem e percentagem acumulada das classes. |
| Tabela 2 | `tab02_distribuicao_ano.csv` | Volume anual, proporção de terminados/censurados e mediana dos já terminados. |
| Tabela 3 | `tab03_orgaos_julgadores.csv` | Órgãos julgadores (top 20 e resumo dos restantes). |
| Tabela 4 | `tab04_resumo_tempo.csv` | Resumo do tempo até ao término (apenas eventos). |
| Tabela 5 | `tab05_censura_por_classe.csv` | Proporção de censura global e por classe. |
| Tabela EDA-1 | `tab_eda01_percentis_tempo.csv` | Percentis do tempo até ao término. |
| Tabela EDA-2 | `tab_eda02_classes_completa.csv` | Frequências completas por classe. |
| Tabela EDA-3 | `tab_eda03_volume_orgaos.csv` | Estatísticas de volume por órgão. |
| Tabela EDA-4 | `tab_eda04_por_ano.csv` | Descritiva por ano de ajuizamento. |
| Tabela EDA-5 | `tab_eda05_classe_evento.csv` | Classe × indicador de evento. |
| Tabela EDA-6 | `tab_eda06_tempo_por_classe.csv` | Tempo (terminados) por classe. |
| Tabela EDA-7 | `tab_eda07_top30_orgaos.csv` | Tempo e volume nos 30 órgãos principais. |
| Tabela EDA-8 | `tab_eda08_classe_ano.csv` | Volume classe × ano. |
| Tabela EDA-9 | `tab_eda09_orgao_classe.csv` | Volume órgão × classe. |
| Tabela EDA-10 | `tab_eda10_cramer_v.csv` | Cramér's V entre as três covariáveis categóricas. |
| Tabela EDA-11 | `tab_eda11_qui_quadrado_evento.csv` | Testes qui-quadrado de evento × classe e evento × ano. |
| Tabela EDA-12 | `tab_eda12_qualidade_colunas.csv` | Resumo de qualidade (missing, tipos). |
| Tabela EDA-13 | `tab_eda13_resumo_dataset.csv` | Síntese geral do dataset analítico. |
| Tabela 9 | `tab09_distribuicoes_aft.csv` | Comparação das três distribuições AFT (C-index, IBS, σ, tempo de treino). |
| Tabela 10 | `tab10_cindex_aft.csv` | C-index de Harrell no treino e no teste (subamostra de 80 000). |
| Tabela 11 | `tab11_brier_aft.csv` | Brier score nos horizontes académicos e Integrated Brier Score. |
| Tabela 12 | `tab12_importancia_aft.csv` | Gain nativo por dummy (top). |
| Tabela 12b | `tab12b_gain_grupos.csv` | Gain agregado às três covariáveis originais. |
| Tabela 12c | `tab12c_shap_grupos.csv` | |SHAP| médio agregado por grupo de covariável. |
| Tabela 13 | `tab13_resumo_aft.csv` | Resumo consolidado do modelo final. |
| Tabela 14 | `tab14_perfis.csv` | S(t) nos horizontes académicos para cinco perfis-tipo. |
| Tabela 15 | `tab15_classe.csv` | S(t) mediano por classe no teste (16 classes do corte a 95 %). |
| Tabela 16 | `tab16_faixas_exemplos.csv` | Exemplos de classificação nas cinco faixas de celeridade. |
| Tabela 17 | `tab17_anomalias.csv` | Pares classe × órgão com rácio ≥ 2 (nomes completos dos órgãos). |
| Tabela 18 | `tab18_resumo_geral.csv` | Síntese geral da amostra, do modelo e do desempenho. |

## 2. Números-chave consolidados

| Item | Valor |
|---|---|
| n processos | 3 498 973 |
| eventos (TPU 22) | 2 887 464 |
| censura à direita | 17.48 % |
| janela | ajuizamentos 2015–2025, TRF2, todos os graus |
| classes / órgãos | 16 (top 95 % + «Outras classes») / 337 |
| algoritmo | XGBoost Survival AFT (único estimado) |
| distribuição vencedora | normal; σ = 0.963 |
| C-index treino / teste (n = 80 000) | 0.760 / 0.760 |
| IBS AFT / IBS KM | 0.126 / 0.194 |
| Brier 6 meses / 1 ano / 2 anos / 3 anos / 5 anos | 0.119 / 0.164 / 0.158 / 0.133 / 0.077 |
| gain relativo órgão / classe / ano | 0.705 / 0.213 / 0.082 |
| partilha \|SHAP\| órgão / classe / ano | 0.662 / 0.129 / 0.209 |
| S(2 anos) extremos (medianas de classe, teste) | mín 0.001 (Carta Precatória Cível); máx 0.802 (Execução Fiscal) |
| anomalias (rácio ≥ 2, ano 2022, n ≥ 80) | 19 |
| semente | 42 |

Detalhe linha a linha: `metadados/tab18_resumo_geral.csv`.

## 3. Ligação às secções da dissertação

- **Notebooks 1–2** → Metodologia (unidade, desfecho, corte de classes, aviso de censura) e Resultados (descritiva / EDA).
- **Notebook 3** → Metodologia (AFT, reconstrução de S(t), custo do C-index) e Resultados (Tabelas 9–13, Figuras 10–14b).
- **Notebook 4** → Resultados (curvas individualizadas, heterogeneidade, Figura 18, faixas, anomalias) e Discussão (o que as anomalias não são).
- **Notebook 5** → Síntese (Tabela 18) e anexos de reprodutibilidade.
- **Notebook 6** → Demonstração gerencial no Capítulo 5; **não** integra a validação estatística formal. A consulta de processos de 2026 usa 2025 como *proxy* one-hot.

## 4. Narrativas

Ver `interpretacao/01_exploracao.md` … `05_verificacao.md`. Cada ficheiro cobre o que foi feito, os números exactos, uma frase por figura/tabela e as limitações encontradas.
