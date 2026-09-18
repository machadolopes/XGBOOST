# Notebook 5 — Verificação final e exportação

## O que foi feito e porquê

Este caderno não estima parâmetros novos. Confere a cadeia de reprodutibilidade (`seed=42` em todos os pontos de aleatoriedade dos cadernos que amostram ou treinam), a numeração APA das figuras e tabelas canónicas (42 figuras e 29 tabelas no checklist), a existência do *booster* `xgboost_aft_modelo_final.json` e do encoder, actualiza `requirements.txt` via `pip freeze`, consolida a Tabela 18 e copia o pacote para `entregaveis/`, pensado para uma sessão de redacção sem acesso aos cadernos Jupyter. Destina-se aos Anexos (reprodutibilidade) e à síntese dos Resultados / Conclusões.

## Achados de verificação

O modelo gravado é XGBoost Survival AFT com distribuição **normal**, σ = 0.963, 500 árvores, C-index de teste 0.760 (treino 0.760) e IBS 0.126 contra 0.194 do Kaplan–Meier. A semente nos metadados é 42. Foram identificadas 19 anomalias estruturais (rácio ≥ 2, ano 2022, n ≥ 80). Todas as figuras canónicas 1–5, EDA-1–25, 10–14b e 15–20 e as tabelas 1–5, EDA-1–13 e 9–18 estavam presentes no disco no momento da conferência. A pasta `entregaveis/` replica figuras, tabelas, metadados e as cinco narrativas.

O Notebook 2 não contém `random_state=42` porque a EDA é determinística (não há amostragem). Os cadernos 3 e 4 concentram os pontos de aleatoriedade (split 80/20, subamostra de 80 000, exemplos das faixas).

## Figuras e tabelas

- **Tabela 18** — Única tabela nova: n, censura, distribuição vencedora e σ, C-index, IBS, Brier por horizonte, gain e SHAP por grupo, extremos de Ŝ(2 anos) e contagem de anomalias.
- As restantes figuras e tabelas são apenas conferidas (existência e nomenclatura APA), não refeitas.

## Limitações

A verificação confirma artefactos no disco; não volta a executar os cadernos 1–4 neste passo (já foram corridos do início ao fim com a mesma semente). O `pip freeze` reflecte o interpretador que executou o caderno (o ambiente em que o *booster* foi treinado). O dashboard (Notebook 6) fica de fora da validação estatística formal, por construção: é demonstração gerencial, incluindo a extrapolação 2026→2025.
