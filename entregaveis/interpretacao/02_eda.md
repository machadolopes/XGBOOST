# Notebook 2 — Análise exploratória exaustiva

## O que foi feito e porquê

A EDA descreve a heterogeneidade de tempo, classe, órgão e ano *antes* de treinar o XGBoost AFT, para que os Resultados não apresentem o modelo como a primeira evidência de diferenças. Destina-se às secções de Resultados (descritiva avançada) e, pontualmente, à Metodologia (justificação do corte de classes e do aviso de censura). Todas as medianas de tempo deste caderno são calculadas **apenas sobre terminados** e herdam o viés de censura nas coortes recentes.

## Achados

A distribuição do tempo é assimétrica (EDA-1) e torna-se mais simétrica em log₁₀ (EDA-2). A ECDF (EDA-3) mostra que metade dos eventos ocorre antes de cerca de 400 dias, mas a cauda ultrapassa os 10 anos. Por classe, a execução fiscal tem mediana de 1942 dias entre os já terminados (média 1722), o recurso inominado 139 dias e a carta precatória 29 dias — ordens de grandeza distintas. A associação entre classe e órgão é forte (Cramér's V = 0.538): certas classes concentram-se em varas especializadas. Classe × ano e órgão × ano são associações mais fracas (V = 0.127 e 0.159). Os testes qui-quadrado de evento × classe e evento × ano rejeitam independência (p < 0,001), o que é esperado: classes lentas e coortes recentes acumulam mais censura, não necessariamente mais «dificuldade».

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
