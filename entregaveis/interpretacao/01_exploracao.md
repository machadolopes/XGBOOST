# Notebook 1 — Carregamento e exploração dos dados

## O que foi feito e porquê

O caderno parte dos Parquets DataJud do TRF2 (2015–2025), define a unidade de análise `processo_id` = tribunal + grau + `numeroProcesso` (um recurso G2/TR não se funde com a acção originária) e constrói a resposta de sobrevivência: tempo em dias desde o ajuizamento até à primeira movimentação TPU 22 (baixa definitiva), com evento = 1 se essa baixa ocorreu e 0 se o processo permanece em curso na data de referência da extração (21 de agosto de 2026). As únicas covariáveis retidas são as observáveis no ajuizamento: órgão julgador, classe processual e ano. Destina-se às secções de Metodologia (amostra e desfecho) e Resultados (estatística descritiva).

## Decisões de limpeza

A agregação por `processo_id` reduziu 20 999 documentos duplicados (redistribuição ou mudança de classe), retendo a capa no ajuizamento e a primeira baixa TPU 22. Removeram-se 2 028 processos com tempo não positivo (datas invertidas ou o mesmo instante). Não houve perdas por data de ajuizamento inválida nem por covariáveis em falta. O dataset analítico ficou com **3 498 973** processos, **17.48 %** de censura à direita, 16 classes (corte de Pareto a 95 % mais «Outras classes») e **337** órgãos.

## Achados

Entre os 2 887 464 processos já terminados, o tempo tem média 645.6 dias e mediana **401 dias** (Q1 = 174.7; Q3 = 872.4). A classe mais volumosa é Procedimento do Juizado Especial Cível (31.11 % do universo). A censura é fortemente heterogénea: execução fiscal apresenta 47.04 % de censura, contra 5.78 % no recurso inominado cível. A mediana dos *já terminados* cai de 2410 dias em 2015 para 167 dias em 2025, ao mesmo tempo que a censura da coorte de 2025 sobe para 46.81 %. **Esta queda não autoriza a leitura de aceleração da celeridade**: nas coortes recentes só os processos rápidos tiveram tempo de terminar.

## Figuras e tabelas

- **Tabela 1** — Ordena as 16 classes por volume e mostra o corte de Pareto; «Outras classes» concentra o residual após o limiar de 95 %.
- **Tabela 2** — Painel anual de volume, eventos e censura; a coluna da mediana dos terminados é a que sofre o viés descrito acima.
- **Tabela 3** — Os órgãos são numerosos e o volume é disperso (o maior concentra cerca de 1,6 %); não há um «órgão médio» representativo.
- **Tabela 4** — Localização e dispersão do tempo dos eventos; o máximo (~4 220 dias) cobre a janela 2015–2025.
- **Tabela 5** — A censura global de 17.48 % mascara diferenças de classe que o modelo terá de absorver.
- **Figura 1** — O histograma dos terminados é assimétrico à direita, coerente com uma duração mediana inferior à média.
- **Figura 2** — Poucas classes concentram a maior parte do volume, justificando o corte a 95 %.
- **Figura 3** — O volume anual não é estacionário; 2021 e 2023 são picos.
- **Figura 4** — A barra de censurados explode em 2024–2025, sinal visual do truncamento à direita.
- **Figura 5** — Os boxplots por classe antecipam a heterogeneidade que o AFT deslocará via μ(x).

## Limitações

O tempo contínuo original pode arredondar a 0,0 dias na Tabela 4 por apresentação, apesar do filtro de tempos não positivos. A mediana dos terminados por ano **não** é um estimador de celeridade; a comparação válida entre coortes é a Figura 18 (Notebook 4). O modelo não vê o mérito da causa nem o acervo legado do órgão.
