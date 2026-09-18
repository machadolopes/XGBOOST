"""Consulta pontual à API pública do DataJud (CNJ) — TRF2.

A chave por omissão é a chave pública documentada pelo CNJ
(https://datajud-wiki.cnj.jus.br/api-publica/acesso/). Pode ser
substituída pela variável de ambiente DATAJUD_API_KEY.
"""

from __future__ import annotations

import os
import re
from typing import Any

import requests

TRIBUNAL = "trf2"
URL = f"https://api-publica.datajud.cnj.jus.br/api_publica_{TRIBUNAL}/_search"
API_KEY_PUBLICA = "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="

# Formato CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO (20 dígitos).
CNJ_DIGITS = re.compile(r"\D+")


def api_key() -> str:
    return os.environ.get("DATAJUD_API_KEY", API_KEY_PUBLICA)


def headers() -> dict[str, str]:
    return {
        "Authorization": f"APIKey {api_key()}",
        "Content-Type": "application/json",
    }


def normalizar_numero_cnj(texto: str) -> str | None:
    """Devolve os 20 dígitos do número CNJ, ou None se inválido."""
    digits = CNJ_DIGITS.sub("", str(texto or ""))
    if len(digits) < 20:
        return None
    return digits[-20:]


def formatar_cnj(digits20: str) -> str:
    d = "".join(ch for ch in digits20 if ch.isdigit())[-20:]
    return f"{d[:7]}-{d[7:9]}.{d[9:13]}.{d[13]}.{d[14:16]}.{d[16:20]}"


def consultar(corpo: dict, timeout: int = 60) -> dict[str, Any]:
    """POST na API DataJud. Levanta RuntimeError se HTTP ≠ 2xx."""
    resp = requests.post(URL, headers=headers(), json=corpo, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")
    return resp.json()


def consultar_processo_por_numero(numero: str) -> dict[str, Any] | None:
    """Obtém a capa (classe, órgão, data de ajuizamento) de um número CNJ.

    Devolve None se o processo não existir no índice do TRF2.
    """
    d20 = normalizar_numero_cnj(numero)
    if d20 is None:
        raise ValueError(
            "Número de processo inválido. Use o formato CNJ "
            "(NNNNNNN-DD.AAAA.J.TR.OOOO) com 20 dígitos."
        )
    corpo = {
        "size": 5,
        "query": {"term": {"numeroProcesso.keyword": d20}},
        "_source": [
            "numeroProcesso",
            "classe",
            "orgaoJulgador",
            "dataAjuizamento",
            "grau",
            "tribunal",
        ],
    }
    dados = consultar(corpo)
    hits = (dados.get("hits") or {}).get("hits") or []
    if not hits:
        # Alguns índices indexam numeroProcesso sem .keyword
        corpo["query"] = {"term": {"numeroProcesso": d20}}
        dados = consultar(corpo)
        hits = (dados.get("hits") or {}).get("hits") or []
    if not hits:
        return None

    src = hits[0].get("_source") or {}
    classe = src.get("classe") or {}
    orgao = src.get("orgaoJulgador") or {}
    data_aj = str(src.get("dataAjuizamento") or "")[:14]
    ano = None
    if len(data_aj) >= 4 and data_aj[:4].isdigit():
        ano = int(data_aj[:4])
    return {
        "numeroProcesso": str(src.get("numeroProcesso") or d20),
        "numero_cnj": formatar_cnj(d20),
        "classe_nome": (classe.get("nome") if isinstance(classe, dict) else None),
        "classe_codigo": (classe.get("codigo") if isinstance(classe, dict) else None),
        "orgao_nome": (orgao.get("nome") if isinstance(orgao, dict) else None),
        "orgao_codigo": (orgao.get("codigo") if isinstance(orgao, dict) else None),
        "data_ajuizamento": data_aj or None,
        "ano_ajuizamento": ano,
        "grau": src.get("grau"),
        "tribunal": src.get("tribunal") or "TRF2",
        "n_documentos": len(hits),
    }
