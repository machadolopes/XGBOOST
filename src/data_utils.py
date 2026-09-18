"""Carregamento, mapeamento e limpeza dos Parquets DataJud (TRF2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Movimento TPU 22 — Baixa Definitiva (encerramento da distribuição).
# Se o processo for desarquivado e baixado de novo, interessa a PRIMEIRA ocorrência.
CODIGOS_EVENTO_TERMINO = (22,)

CODIGO_OUTRAS_CLASSES = 0
NOME_OUTRAS_CLASSES = "Outras classes"

ANOS_ANALISE = (2015, 2025)

COLUNAS_PROCESSOS = [
    "id",
    "tribunal",
    "grau",
    "numeroProcesso",
    "dataAjuizamento",
    "dataHoraUltimaAtualizacao",
    "timestamp",
    "classe_codigo",
    "classe_nome",
    "orgaoJulgador_codigo",
    "orgaoJulgador_nome",
    "n_movimentos",
]


def project_root() -> Path:
    """Raiz do repositório, mesmo se o kernel Jupyter abrir em notebooks/."""
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            if (candidate / "src" / "data_utils.py").is_file():
                return candidate
    return here


def raw_data_dir() -> Path:
    """Pasta dos Parquets brutos já extraídos: ``data/raw/`` neste repositório."""
    pasta = project_root() / "data" / "raw"
    if pasta.is_dir() and any(pasta.glob("trf2_*_processos.parquet")):
        return pasta
    raise FileNotFoundError(
        f"Não foram encontrados os Parquets DataJud em {pasta}. "
        "Coloque os ficheiros trf2_{ano}_processos.parquet e "
        "trf2_{ano}_movimentos.parquet em data/raw/."
    )


def listar_ficheiros_brutos(raw_dir: Path | None = None) -> pd.DataFrame:
    """Inventário dos Parquets (nome, tipo, tamanho, n.º de linhas)."""
    raw_dir = raw_dir or raw_data_dir()
    linhas = []
    for path in sorted(raw_dir.glob("trf2_*.parquet")):
        nome = path.name
        tipo = "movimentos" if "_movimentos" in nome else "processos"
        ano = int(nome.split("_")[1])
        n_linhas = pq.ParquetFile(path).metadata.num_rows
        linhas.append(
            {
                "ficheiro": nome,
                "ano": ano,
                "tipo": tipo,
                "tamanho_mb": round(path.stat().st_size / 1e6, 2),
                "n_registos": n_linhas,
            }
        )
    return pd.DataFrame(linhas)


def _concat_parquets(ficheiros: list[Path], colunas: list[str] | None = None) -> pd.DataFrame:
    """Lê e concatena Parquets, unificando esquemas (string vs large_string)."""
    tabelas = [pq.read_table(f, columns=colunas) for f in ficheiros]
    try:
        return pa.concat_tables(tabelas, promote_options="permissive").to_pandas()
    except TypeError:
        return pa.concat_tables(tabelas, promote=True).to_pandas()


def carregar_processos(
    raw_dir: Path | None = None,
    anos: tuple[int, int] | None = ANOS_ANALISE,
) -> pd.DataFrame:
    """Carrega a capa dos processos no intervalo de anos indicado (inclusive)."""
    raw_dir = raw_dir or raw_data_dir()
    ficheiros = sorted(raw_dir.glob("trf2_*_processos.parquet"))
    if anos is not None:
        ano_min, ano_max = anos
        ficheiros = [
            f for f in ficheiros if ano_min <= int(f.name.split("_")[1]) <= ano_max
        ]
    if not ficheiros:
        raise FileNotFoundError(f"Nenhum Parquet de processos em {raw_dir}")
    return _concat_parquets(ficheiros, COLUNAS_PROCESSOS)


def carregar_datas_termino(
    raw_dir: Path | None = None,
    anos: tuple[int, int] | None = ANOS_ANALISE,
    codigos: tuple[int, ...] = CODIGOS_EVENTO_TERMINO,
) -> pd.DataFrame:
    """Primeira data de baixa definitiva (TPU 22) por documento DataJud (`id`)."""
    raw_dir = raw_dir or raw_data_dir()
    ficheiros = sorted(raw_dir.glob("trf2_*_movimentos.parquet"))
    if anos is not None:
        ano_min, ano_max = anos
        ficheiros = [
            f for f in ficheiros if ano_min <= int(f.name.split("_")[1]) <= ano_max
        ]
    partes: list[pd.DataFrame] = []
    for path in ficheiros:
        tab = pq.read_table(path, columns=["id_processo", "codigo", "dataHora"])
        tab = tab.filter(pc.is_in(tab["codigo"], value_set=pa.array(list(codigos))))
        if tab.num_rows == 0:
            continue
        partes.append(tab.select(["id_processo", "dataHora"]).to_pandas())
    if not partes:
        return pd.DataFrame(columns=["id", "data_termino"])
    baixas = pd.concat(partes, ignore_index=True)
    baixas["data_termino"] = pd.to_datetime(
        baixas["dataHora"], utc=True, errors="coerce"
    )
    baixas = (
        baixas.dropna(subset=["data_termino"])
        .groupby("id_processo", as_index=False)["data_termino"]
        .min()
        .rename(columns={"id_processo": "id"})
    )
    return baixas


def criar_processo_id(df: pd.DataFrame) -> pd.Series:
    """Chave primária composta: tribunal + instância (grau) + número do processo.

    A apelação cível (G2/TR) partilha o `numeroProcesso` da acção originária (G1/JE)
    mas constitui unidade autónoma: tramita noutro grau, noutro órgão julgador,
    com classe processual própria.
    """
    return (
        df["tribunal"].astype(str).str.strip()
        + "_"
        + df["grau"].astype(str).str.strip()
        + "_"
        + df["numeroProcesso"].astype(str).str.strip()
    )


def agregar_por_processo_id(
    df: pd.DataFrame,
    data_referencia: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Colapsa documentos DataJud duplicados numa linha por `processo_id`."""
    n0 = len(df)
    out = df.copy()
    if "processo_id" not in out.columns:
        out["processo_id"] = criar_processo_id(out)

    out["_ts"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.sort_values(
        ["processo_id", "data_ajuizamento", "n_movimentos", "_ts"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    capa = out.drop_duplicates("processo_id", keep="first").copy()

    agg = out.groupby("processo_id", as_index=False).agg(
        data_ajuizamento=("data_ajuizamento", "min"),
        data_termino=("data_termino", "min"),
        n_documentos_datajud=("id", "size"),
    )
    capa = capa.drop(columns=["data_ajuizamento", "data_termino"], errors="ignore")
    capa = capa.merge(agg, on="processo_id", how="left")
    capa = capa.drop(columns=["_ts"], errors="ignore")

    if data_referencia is None:
        data_referencia = out.attrs.get("data_referencia")
    if isinstance(data_referencia, str):
        data_referencia = pd.Timestamp(data_referencia)
    if data_referencia is not None and data_referencia.tzinfo is None:
        data_referencia = data_referencia.tz_localize("UTC")

    capa["evento"] = capa["data_termino"].notna().astype("int8")
    fim = capa["data_termino"].copy()
    if data_referencia is not None:
        fim = fim.fillna(data_referencia)
    capa["tempo"] = (fim - capa["data_ajuizamento"]).dt.total_seconds() / 86400.0
    capa["ano_ajuizamento"] = capa["data_ajuizamento"].dt.year.astype("Int64")
    if "data_referencia" in out.attrs:
        capa.attrs["data_referencia"] = out.attrs["data_referencia"]

    n_dup_linhas = int((out.groupby("processo_id").size() > 1).sum())
    log = pd.DataFrame(
        [
            {
                "passo": "agregar_processo_id",
                "n_antes": n0,
                "n_depois": len(capa),
                "n_removidos": n0 - len(capa),
                "pct_removidos": round(100 * (n0 - len(capa)) / n0, 3) if n0 else 0.0,
                "justificacao": (
                    "Unidade de análise = tribunal + grau + numeroProcesso. "
                    f"{n_dup_linhas} chaves tinham mais do que um documento DataJud "
                    "(redistribuição ou mudança de classe); retém-se a capa no ajuizamento "
                    "e a primeira baixa TPU 22."
                ),
            }
        ]
    )
    return capa.reset_index(drop=True), log


def parse_data_ajuizamento(serie: pd.Series) -> pd.Series:
    """Converte `dataAjuizamento` AAAAMMDDHHMMSS em datetime com fuso UTC."""
    texto = serie.astype("string").str.slice(0, 14)
    dt = pd.to_datetime(texto, format="%Y%m%d%H%M%S", errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    else:
        dt = dt.dt.tz_convert("UTC")
    return dt


def _nome_modal_por_codigo(df: pd.DataFrame, codigo: str, nome: str) -> pd.Series:
    """Associa a cada código o nome mais frequente (evita variantes ortográficas)."""
    contagens = (
        df.groupby([codigo, nome], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates(codigo)
    )
    return df[codigo].map(contagens.set_index(codigo)[nome])


def construir_variaveis_sobrevivencia(
    processos: pd.DataFrame,
    baixas: pd.DataFrame,
    data_referencia: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cria `tempo` (dias), `evento`, `ano_ajuizamento` e rótulos das covariáveis."""
    df = processos.copy()
    df["data_ajuizamento"] = parse_data_ajuizamento(df["dataAjuizamento"])
    df = df.merge(baixas, on="id", how="left")

    if data_referencia is None:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        data_referencia = ts.max()
    df.attrs["data_referencia"] = (
        data_referencia.isoformat() if hasattr(data_referencia, "isoformat") else str(data_referencia)
    )

    df["processo_id"] = criar_processo_id(df)
    df["evento"] = df["data_termino"].notna().astype("int8")
    fim = df["data_termino"].copy()
    fim = fim.fillna(data_referencia)
    df["tempo"] = (fim - df["data_ajuizamento"]).dt.total_seconds() / 86400.0
    df["ano_ajuizamento"] = df["data_ajuizamento"].dt.year.astype("Int64")

    df["classe_processual"] = _nome_modal_por_codigo(df, "classe_codigo", "classe_nome")
    df["orgao_julgador"] = _nome_modal_por_codigo(
        df, "orgaoJulgador_codigo", "orgaoJulgador_nome"
    )
    df = df.rename(
        columns={
            "classe_codigo": "classe_processual_codigo",
            "orgaoJulgador_codigo": "orgao_julgador_codigo",
        }
    )
    return df


def aplicar_limpeza(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove datas inválidas e tempos não positivos. Devolve (limpo, log)."""
    n0 = len(df)
    passos: list[dict] = []

    def _registar(nome: str, antes: int, depois: int, justificacao: str) -> None:
        passos.append(
            {
                "passo": nome,
                "n_antes": antes,
                "n_depois": depois,
                "n_removidos": antes - depois,
                "pct_removidos": round(100 * (antes - depois) / n0, 3) if n0 else 0.0,
                "justificacao": justificacao,
            }
        )

    n = len(df)
    mask = df["data_ajuizamento"].notna()
    df = df.loc[mask].copy()
    _registar(
        "datas_ajuizamento_invalidas",
        n,
        len(df),
        "Impossível calcular o tempo de tramitação sem data de ajuizamento válida.",
    )

    n = len(df)
    mask = df["classe_processual"].notna() & df["orgao_julgador"].notna()
    df = df.loc[mask].copy()
    _registar(
        "covariaveis_em_falta",
        n,
        len(df),
        "O XGBoost AFT exige classe processual e órgão julgador observados no ajuizamento.",
    )

    n = len(df)
    mask = df["tempo"].notna() & (df["tempo"] > 0)
    df = df.loc[mask].copy()
    _registar(
        "tempo_nao_positivo",
        n,
        len(df),
        "Tempos ≤ 0 resultam de datas invertidas ou do mesmo instante (erro de registo).",
    )

    n = len(df)
    mask = df["ano_ajuizamento"].notna()
    df = df.loc[mask].copy()
    _registar(
        "ano_ajuizamento_em_falta",
        n,
        len(df),
        "O ano de ajuizamento é covariável da dissertação (painéis anuais).",
    )

    log = pd.DataFrame(passos)
    return df, log


def agregar_classes_top_percentil(
    df: pd.DataFrame,
    limiar: float = 0.95,
    coluna: str = "classe_processual_codigo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mantém as classes do top `limiar` e agrega o residual em «Outras classes»."""
    freq = df[coluna].value_counts(dropna=False)
    pct = freq / freq.sum()
    acum = pct.cumsum()
    n_keep = int((acum < limiar).sum() + 1)
    n_keep = min(n_keep, len(freq))
    codigos_retidos = freq.index[:n_keep]
    nomes = df[["classe_processual_codigo", "classe_processual"]].drop_duplicates(
        "classe_processual_codigo"
    )
    resumo = pd.DataFrame(
        {
            "classe_processual_codigo": freq.index,
            "n": freq.values,
            "pct": pct.values,
            "pct_acumulada": acum.values,
        }
    ).merge(nomes, on="classe_processual_codigo", how="left")
    resumo["retida"] = resumo["classe_processual_codigo"].isin(codigos_retidos)

    out = df.copy()
    mask_outras = ~out[coluna].isin(codigos_retidos)
    out.loc[mask_outras, "classe_processual_codigo"] = CODIGO_OUTRAS_CLASSES
    out.loc[mask_outras, "classe_processual"] = NOME_OUTRAS_CLASSES

    n_outras = int(mask_outras.sum())
    pct_outras = n_outras / len(out) if len(out) else 0.0
    linha_outras = pd.DataFrame(
        {
            "classe_processual_codigo": [CODIGO_OUTRAS_CLASSES],
            "classe_processual": [NOME_OUTRAS_CLASSES],
            "n": [n_outras],
            "pct": [pct_outras],
            "pct_acumulada": [1.0],
            "retida": [False],
            "agregada": [True],
        }
    )
    resumo["agregada"] = False
    resumo = pd.concat([resumo.loc[resumo["retida"]], linha_outras], ignore_index=True)
    return out, resumo


def filtrar_classes_top_percentil(*args, **kwargs):
    """Compatibilidade: agrega o residual em vez de o excluir."""
    return agregar_classes_top_percentil(*args, **kwargs)


def colunas_dataset_limpo() -> list[str]:
    """Colunas exportadas para os notebooks seguintes."""
    return [
        "processo_id",
        "id",
        "numeroProcesso",
        "tribunal",
        "grau",
        "data_ajuizamento",
        "data_termino",
        "tempo",
        "evento",
        "ano_ajuizamento",
        "classe_processual_codigo",
        "classe_processual",
        "orgao_julgador_codigo",
        "orgao_julgador",
        "n_movimentos",
        "n_documentos_datajud",
    ]
