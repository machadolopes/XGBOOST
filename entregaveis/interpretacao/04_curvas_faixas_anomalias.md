# Notebook 4 — Curvas individualizadas, faixas e anomalias

## O que foi feito e porquê

Com o *booster* congelado, reconstituiu-se Ŝ(t | x) na grelha de 0–10 anos para responder às alíneas (b) e (d) da pergunta de investigação: a probabilidade de término é uma **curva**, heterogénea entre órgãos, classes e coortes, e não um único prazo médio. Sem re-treino. Destina-se aos Resultados (heterogeneidade, Figura 18, anomalias) e à Discussão (o que as faixas comunicam à gestão).

## Achados

Nos perfis-tipo de 2022 (órgão modal de cada uma das cinco classes de maior volume), a execução fiscal permanece com Ŝ(5 anos) elevado, enquanto juizados e cumprimentos de sentença descem mais depressa (Tabela 14, Figura 15). No conjunto de teste, a curva **mediana** das 16 classes do corte a 95 % confirma a ordenação: Execução Fiscal tem Ŝ(2 anos) = 0.802 e Carta Precatória Cível tem Ŝ(2 anos) = 0.001 (Tabela 15, Figura 16). Com classe e ano fixos, as oito varas de execução fiscal de maior volume **não coincidem** (Figura 17): o órgão desloca μ. A Figura 18 fixa o par de maior volume (Procedimento do Juizado Especial Cível × 2.º Juizado Especial Federal de Vitória, n = 46 958) e varia só o ano — esta é a comparação válida entre coortes.

As faixas (p10/p25/p75/p90 de exp(μ) dentro de classe × ano) traduzem a localização AFT num rótulo relativo. Um «muito lento» no juizado pode ter prazo inferior a um «típico» em execução fiscal (Figura 19, Tabela 16).

Anomalias (rácio ≥ 2 face à mediana dos outros órgãos da mesma classe, ano 2022, n ≥ 80 no par): **19** pares. Nomes completos na Tabela 17:

- Subsecretaria de Distribuição de Atividades Judiciárias × Carta Precatória Cível: rácio 22.47 (exp(μ) = 1664 d vs. 74 d nos restantes).
- Gabinete da Vice-Presidência × Agravo de Instrumento: rácio 6.71 (exp(μ) = 2143 d vs. 319 d nos restantes).
- Gabinete da Vice-Presidência × Apelação / Remessa Necessária: rácio 4.97 (exp(μ) = 2009 d vs. 404 d nos restantes).
- Gabinete da Vice-Presidência × Remessa Necessária Cível: rácio 4.96 (exp(μ) = 1101 d vs. 222 d nos restantes).
- Gabinete da Vice-Presidência × Apelação Cível: rácio 4.93 (exp(μ) = 1980 d vs. 402 d nos restantes).
- Núcleo Permanente de Métodos Consensuais de Solução de Conflitos × Procedimento Comum Cível: rácio 4.31 (exp(μ) = 3073 d vs. 713 d nos restantes).
- Gabinete da Vice-Presidência × Outras classes: rácio 3.03 (exp(μ) = 768 d vs. 253 d nos restantes).
- Gabinete do Juízo Gestor das Turmas Recursais × Recurso Inominado Cível: rácio 2.92 (exp(μ) = 512 d vs. 175 d nos restantes).
- 4ª Turma Recursal - Juiz Relator 2 - RJ × Recurso Inominado Cível: rácio 2.87 (exp(μ) = 502 d vs. 175 d nos restantes).
- 1ª Vara federal de cachoeiro de itapemirim × Cumprimento de sentença: rácio 2.55 (exp(μ) = 1533 d vs. 600 d nos restantes).
- 1ª Vara federal de cachoeiro de itapemirim × Procedimento Comum Cível: rácio 2.44 (exp(μ) = 1741 d vs. 713 d nos restantes).
- 1ª Vara federal de cachoeiro de itapemirim × Embargos à Execução: rácio 2.35 (exp(μ) = 944 d vs. 402 d nos restantes).
- 1ª Vara federal de cachoeiro de itapemirim × Procedimento do Juizado Especial Cível: rácio 2.35 (exp(μ) = 910 d vs. 388 d nos restantes).
- 1ª Vara federal de cachoeiro de itapemirim × Cumprimento de Sentença contra a Fazenda Pública: rácio 2.33 (exp(μ) = 1384 d vs. 595 d nos restantes).
- 1ª Vara federal de cachoeiro de itapemirim × Execução de Título Extrajudicial: rácio 2.30 (exp(μ) = 1604 d vs. 699 d nos restantes).
- Vara Federal de Magé × Cumprimento de Sentença contra a Fazenda Pública: rácio 2.18 (exp(μ) = 1298 d vs. 595 d nos restantes).
- 8ª Vara Federal do Rio de Janeiro × Outras classes: rácio 2.12 (exp(μ) = 538 d vs. 253 d nos restantes).
- 4ª Vara Federal de Campos × Outras classes: rácio 2.05 (exp(μ) = 518 d vs. 253 d nos restantes).
- Gabinete do Juízo Vice-Gestor das Turmas Recursais × Recurso Inominado Cível: rácio 2.02 (exp(μ) = 355 d vs. 175 d nos restantes).

Interpretação: **diagnóstico de fluxo, não juízo disciplinar**. O modelo não observa recursos humanos, acervo legado nem qualidade da petição.

## Figuras e tabelas

- **Figura 15 / Tabela 14** — Cinco curvas reais, uma por combinação; mesma forma normal, diferente μ.
- **Figura 16 / Tabela 15** — As 16 classes do corte a 95 % (mais «Outras classes») ordenam a pendência prevista mesmo misturando órgãos e anos no teste.
- **Figura 17** — Isola o órgão; responde à alínea (a) quanto à heterogeneidade intra-classe.
- **Figura 18** — Substitui a leitura inválida da mediana dos já terminados (Tabela 2, EDA-16).
- **Figura 19 / Tabela 16** — Semáforo relativo à célula classe × ano.
- **Figura 20 / Tabela 17** — Pontos acima da recta 2× são os alertas pré-especificados.

## Limitações

A forma paramétrica partilhada impede que uma classe tenha um «ombro» de S(t) diferente de outra: só há translação temporal. As anomalias usam o ano de referência 2022 (follow-up suficiente) e n mínimo 80; órgãos pequenos ficam de fora por desenho. As faixas degenerariam se fossem calculadas dentro da célula classe × órgão × ano, porque μ é constante nessa célula — daí o segmento classe × ano.
