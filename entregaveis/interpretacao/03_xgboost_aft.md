# Notebook 3 — XGBoost Survival AFT: treino e avaliação

## O que foi feito e porquê

Treinou-se **apenas** o XGBoost Survival em modo AFT (Cox e RSF não correm neste repositório). As três covariáveis foram recodificadas em one-hot esparso (`float32`), com categorias não vistas no treino mapeadas para zero. A divisão 80/20 foi estratificada por evento × ano (`seed=42`); 10 % do treino serviu de validação interna para *early stopping* (paciência 30). Os limites AFT foram [t, t] para eventos e [t, +∞) para censurados. Destina-se aos Resultados (desempenho) e à Metodologia (especificação AFT e custo do C-index).

As três distribuições (`normal`, `logistic`, `extreme`) partilharam os mesmos hiperparâmetros (`max_depth=6`, `eta=0.05`, `min_child_weight=10`, 500 rondas). A curva S(t) não existe pronta no XGBoost: reconstruiu-se Ŝ a partir de μ previsto, σ calibrado por máxima verosimilhança censurada, e z = (ln t − μ) / σ.

## Custo computacional do C-index

O C-index compara processos par a par; o número de pares cresce com n²/2. Nos ~700 000 processos de teste isso excederia 200 mil milhões de pares nas implementações padrão. Por isso C-index e IBS foram avaliados numa **subamostra estratificada de 80 000** do teste, repetida no treino para verificar estabilidade. O treino do *booster* (`tree_method='hist'`) permanece aproximadamente linear/log-linear e correu no universo.

## Achados

A Tabela 9 selecciona **normal** (σ = 0.963). O C-index de teste é **0.760**, idêntico ao do treino (0.760), o que não sugere sobreajuste grosseiro na ordenação. A logística falhou (C-index teste 0.583, IBS 0.313) e foi descartada. A Gumbel/extreme ficou próxima (C-index 0.754, IBS 0.128) mas perdeu no saldo C-index − IBS. O IBS do AFT normal é **0.126**, inferior ao Kaplan–Meier sem covariáveis (0.194).

Brier pontual (AFT): 6 meses 0.119; 1 ano 0.164; 2 anos 0.158; 3 anos 0.133; 5 anos 0.077. O erro é maior perto de 1–2 anos, onde há mais massa de eventos, e menor aos 5 anos.

O gain relativo concentra-se no **órgão** (0.705), depois na classe (0.213) e no ano (0.082). O |SHAP| médio altera a partilha — órgão 0.662, ano 0.209, classe 0.129 — porque o ano, com poucas dummies, desloca μ de forma sistemática nas coortes recentes.

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
