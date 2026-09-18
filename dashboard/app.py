"""Painel de celeridade processual — gestores (XGBoost Survival AFT).

Arranque (na raiz do repositório):

    streamlit run dashboard/app.py

Não re-treina o modelo. Requer os CSV gerados pelo Notebook 6.
Linguagem de gestão em todo o painel (pendência, prazo típico).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dash_aft import (  # noqa: E402
    ANOS_LEITURA_CAUTELA,
    ANO_REF,
    ANO_TREINO_MAX,
    ANO_TREINO_MIN,
    FAIXA_COR,
    HORIZONTES_DASHBOARD,
    MAX_ORGAOS_OVERLAY,
    ORDEM_FAIXAS,
    ano_para_modelo,
    catalogo_df,
    categorias_encoder,
    carregar_modelo_aft,
    curva_para_grafico,
    faixa_celula,
    grelha_curvas,
    mapear_classe,
    montar_perfis,
    n_par,
    orgaos_da_classe,
    pct_em_100,
    percentis_df,
    prazos_df,
    prever_perfis,
    s_nos_horizontes,
    tendencia_par,
)
from src.datajud_api import consultar_processo_por_numero  # noqa: E402

st.set_page_config(
    page_title="Celeridade processual — TRF2",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

HORIZONTE_ROTULO = {
    "6 meses": "6 meses",
    "1 ano": "1 ano",
    "2 anos": "2 anos",
    "5 anos": "5 anos",
    "mais de 7 anos": "mais de 7 anos",
}


@st.cache_resource(show_spinner="A carregar o modelo…")
def _pacote():
    return carregar_modelo_aft()


@st.cache_data(show_spinner=False)
def _catalogos():
    return catalogo_df(), prazos_df(), percentis_df()


def _curva_plotly(
    df_plot: pd.DataFrame,
    horizontes_s: dict[str, float] | None = None,
    *,
    linhas_referencia: bool = True,
) -> go.Figure:
    fig = go.Figure()
    for perfil, g in df_plot.groupby("perfil"):
        fig.add_trace(
            go.Scatter(
                x=g["anos"],
                y=g["ainda_pendente"] * 100,
                mode="lines",
                name=str(perfil)[:48],
                hovertemplate="%{x:.1f} anos<br>ainda pendente: %{y:.1f}%<extra></extra>",
            )
        )
    if linhas_referencia:
        for nome, t in HORIZONTES_DASHBOARD.items():
            anos = t / 365.25
            fig.add_vline(x=anos, line_dash="dash", line_color="#888888")
            s = None if horizontes_s is None else horizontes_s.get(nome)
            if s is None:
                continue
            pct = 100.0 * float(s)
            fig.add_annotation(
                x=anos,
                y=pct,
                text=f"{HORIZONTE_ROTULO[nome]}: {pct:.1f}%".replace(".", ","),
                showarrow=True,
                arrowhead=0,
                ax=40,
                ay=-20,
                font=dict(size=11),
            )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Tempo desde o ajuizamento (anos)",
        yaxis_title="Probabilidade de ainda estar pendente (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=40, b=40),
        height=480,
    )
    return fig


def _semaforo(faixa: str) -> None:
    cor = FAIXA_COR.get(faixa, "gray")
    st.markdown(
        f"**Faixa de celeridade:** :{cor}[{faixa}]  \n"
        "Comparação com os outros órgãos da *mesma classe* e do *mesmo ano*. "
        "«Típico» em execução fiscal não é o mesmo prazo que «típico» num recurso."
    )


try:
    pacote = _pacote()
    catalogo, prazos, percentis = _catalogos()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

cats = categorias_encoder(pacote["encoder"])
classes = cats.get("classe_processual") or sorted(catalogo["classe_processual"].unique().tolist())
anos = [str(a) for a in range(ANO_TREINO_MIN, ANO_TREINO_MAX + 1)]
grelha = grelha_curvas()

st.title("Celeridade processual no TRF2")
st.caption(
    "Estimativa no momento do ajuizamento — só com classe, órgão julgador e ano. "
    "Demonstração aplicada; **não** faz parte da validação estatística da dissertação."
)

pagina = st.sidebar.radio(
    "Página",
    [
        "Consulta por combinação",
        "Comparar órgãos",
        "Tendência no tempo",
        "Evolução no tempo (visão global)",
        "Consultar um processo (CNJ)",
    ],
)

# ---------------------------------------------------------------------------
if pagina == "Consulta por combinação":
    st.subheader("Consulta por combinação de covariáveis")
    c1, c2, c3 = st.columns(3)
    with c1:
        classe = st.selectbox("Classe processual", classes)
    orgaos_cls = orgaos_da_classe(catalogo, classe) or cats.get("orgao_julgador") or []
    with c2:
        orgao = st.selectbox("Órgão julgador", orgaos_cls)
    with c3:
        ano = st.selectbox("Ano de ajuizamento", anos, index=anos.index(ANO_REF) if ANO_REF in anos else 0)

    x = montar_perfis(classe, orgao, ano)
    mu, S, grelha = prever_perfis(pacote, x, grelha)
    horiz = s_nos_horizontes(grelha, S[0])
    plot = curva_para_grafico(grelha, S, [f"{classe[:28]} · {orgao[:24]} · {ano}"])
    st.plotly_chart(_curva_plotly(plot, horiz), width="stretch")

    cards = st.columns(5)
    chaves = list(HORIZONTES_DASHBOARD)
    for i, nome in enumerate(chaves):
        with cards[i]:
            st.metric(
                f"Pendente em {HORIZONTE_ROTULO[nome]}",
                pct_em_100(horiz[nome]),
                help="Em 100 processos iguais a este, quantos o modelo estima que ainda estariam pendentes.",
            )

    cel = faixa_celula(prazos, percentis, classe, orgao, ano)
    faixa = cel[1] if cel is not None else "Indeterminada"
    _semaforo(faixa)
    st.caption(f"{n_par(catalogo, classe, orgao):,} processos desta classe neste órgão (todos os anos).".replace(",", " "))

# ---------------------------------------------------------------------------
elif pagina == "Comparar órgãos":
    st.subheader("Comparação entre órgãos julgadores (mesma classe)")
    st.caption("A classe fica fixa para isolar o efeito do órgão — a mesma lógica da Figura 17 da dissertação.")
    classe = st.selectbox("Classe processual (fixa)", classes, key="cmp_cls")
    orgaos_cls = orgaos_da_classe(catalogo, classe)
    orgaos_sel = st.multiselect(
        "Órgãos julgadores (mesma classe)",
        orgaos_cls,
        default=orgaos_cls[: min(3, len(orgaos_cls))],
        max_selections=MAX_ORGAOS_OVERLAY,
    )
    ano = st.selectbox("Ano de ajuizamento", anos, index=anos.index(ANO_REF) if ANO_REF in anos else 0, key="cmp_ano")
    if len(orgaos_sel) < 2:
        st.info("Seleccione pelo menos dois órgãos da mesma classe.")
    else:
        x = montar_perfis(classe, orgaos_sel, ano)
        mu, S, grelha = prever_perfis(pacote, x, grelha)
        plot = curva_para_grafico(grelha, S, orgaos_sel)
        st.plotly_chart(_curva_plotly(plot, None, linhas_referencia=True), width="stretch")
        st.caption(
            "Linhas verticais tracejadas: 6 meses, 1 ano, 2 anos, 5 anos e mais de 7 anos. "
            "Os rótulos percentuais ficam nos cartões da consulta individual, para não sobrepor as curvas."
        )

# ---------------------------------------------------------------------------
elif pagina == "Tendência no tempo":
    st.subheader("Tendência no tempo para classe × órgão")
    classe = st.selectbox("Classe processual", classes, key="tr_cls")
    orgaos_cls = orgaos_da_classe(catalogo, classe)
    orgao = st.selectbox("Órgão julgador", orgaos_cls, key="tr_org")
    st.warning(
        "A variação entre coortes reflecte efeito de coorte e truncamento à direita das coortes mais recentes, "
        "não uma tendência causal de aceleração ou desaceleração da celeridade. "
        "É a mesma ressalva da Figura 18 da dissertação."
    )
    tend = tendencia_par(prazos, classe, orgao)
    if tend.empty:
        st.info("Não há previsões para este par classe × órgão.")
    else:
        anos_tr = tend["ano_ajuizamento"].astype(str).tolist()
        x = montar_perfis(classe, orgao, anos_tr)
        mu, S, grelha = prever_perfis(pacote, x, grelha)
        plot = curva_para_grafico(grelha, S, anos_tr)
        st.plotly_chart(_curva_plotly(plot, None), width="stretch")
        st.markdown("**Pendência prevista aos 2 anos, por coorte**")
        st.dataframe(
            tend[["ano_ajuizamento", "S_2anos", "t_previsto_d", "n", "cautela"]].rename(
                columns={
                    "ano_ajuizamento": "ano",
                    "S_2anos": "ainda pendente (2 anos)",
                    "t_previsto_d": "prazo típico (dias)",
                    "n": "n processos",
                    "cautela": "coorte muito censurada",
                }
            ),
            hide_index=True,
        )

# ---------------------------------------------------------------------------
elif pagina == "Evolução no tempo (visão global)":
    st.subheader("Evolução no tempo — visão global das coortes")
    st.markdown(
        """
- **Até 2023:** coortes com *follow-up* suficiente para ler a pendência prevista com menos truncamento.
- **2024–2025:** coortes **fortemente censuradas** — muitos processos ainda em curso. Não ler a queda do prazo típico como aceleração real.
        """
    )
    classe = st.selectbox("Classe processual", classes, key="gl_cls")
    sub = prazos.loc[prazos["classe_processual"] == classe]
    med = (
        sub.groupby("ano_ajuizamento", as_index=False)["S_2anos"]
        .median()
        .sort_values("ano_ajuizamento")
    )
    med["grupo"] = np.where(med["ano_ajuizamento"].astype(str).isin(ANOS_LEITURA_CAUTELA), "2024–2025 (censura alta)", "até 2023")
    fig = go.Figure()
    for grupo, g in med.groupby("grupo"):
        fig.add_trace(
            go.Scatter(
                x=g["ano_ajuizamento"],
                y=g["S_2anos"] * 100,
                mode="lines+markers",
                name=grupo,
            )
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Ano de ajuizamento",
        yaxis_title="Pendência mediana prevista aos 2 anos (%)",
        yaxis=dict(range=[0, 100]),
        height=420,
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
else:
    st.subheader("Consulta por número de processo (processo novo)")
    st.caption(
        "Para processos recentes (por exemplo ajuizados em 2026) que **não** estão no dataset de treino (2015–2025). "
        "A capa é obtida na API pública do DataJud em tempo real. "
        "Isto é uma demonstração aplicada — **não** faz parte da validação estatística formal."
    )
    numero = st.text_input("Número do processo (formato CNJ)", placeholder="0000000-00.2026.4.02.0000")
    if st.button("Consultar DataJud") and numero.strip():
        try:
            capa = consultar_processo_por_numero(numero.strip())
        except ValueError as exc:
            st.error(str(exc))
            capa = None
        except Exception as exc:
            st.error(f"Não foi possível contactar a API do DataJud: {exc}")
            capa = None
        if capa is None:
            st.error("Processo não encontrado na API do TRF2, ou faltam classe, órgão ou data de ajuizamento.")
        elif not capa.get("classe_nome") or not capa.get("orgao_nome") or capa.get("ano_ajuizamento") is None:
            st.error("Os dados obrigatórios (classe, órgão, data de ajuizamento) não puderam ser extraídos.")
        else:
            classe_m, cls_out = mapear_classe(capa["classe_nome"], classes)
            ano_m, extra = ano_para_modelo(capa["ano_ajuizamento"])
            orgaos_ok = cats.get("orgao_julgador") or []
            orgao = capa["orgao_nome"]
            if orgao not in orgaos_ok:
                st.error(
                    f"O órgão «{orgao}» não consta das categorias treinadas. "
                    "Não é possível produzir uma previsão fiável para esta unidade."
                )
            else:
                if extra:
                    st.warning(
                        f"Ano de ajuizamento ({capa['ano_ajuizamento']}) fora do intervalo de treino do modelo "
                        f"({ANO_TREINO_MIN}–{ANO_TREINO_MAX}); esta estimativa usa {ano_m} como referência mais próxima "
                        "e deve ser lida com cautela adicional — é uma extrapolação, não uma previsão validada "
                        "nas mesmas condições dos restantes resultados desta dissertação."
                    )
                if cls_out:
                    st.info(
                        f"A classe «{capa['classe_nome']}» não está entre as classes retidas (top 95%). "
                        f"Foi mapeada para «{classe_m}»."
                    )
                x = montar_perfis(classe_m, orgao, ano_m)
                mu, S, grelha = prever_perfis(pacote, x, grelha)
                horiz = s_nos_horizontes(grelha, S[0])
                plot = curva_para_grafico(
                    grelha, S, [f"{capa['numero_cnj']} · {classe_m[:24]}"]
                )
                st.plotly_chart(_curva_plotly(plot, horiz), width="stretch")
                cards = st.columns(5)
                for i, nome in enumerate(HORIZONTES_DASHBOARD):
                    with cards[i]:
                        st.metric(f"Pendente em {HORIZONTE_ROTULO[nome]}", pct_em_100(horiz[nome]))
                cel = faixa_celula(prazos, percentis, classe_m, orgao, ano_m)
                faixa = cel[1] if cel is not None else "Indeterminada"
                _semaforo(faixa)
                st.write(
                    {
                        "número CNJ": capa["numero_cnj"],
                        "classe DataJud": capa["classe_nome"],
                        "classe no modelo": classe_m,
                        "órgão": orgao,
                        "ano real": capa["ano_ajuizamento"],
                        "ano no modelo": ano_m,
                    }
                )
