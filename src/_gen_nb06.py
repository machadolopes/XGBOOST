"""Gera notebooks/06_dashboard.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "06_dashboard.ipynb"


def md(src: str):
    return new_markdown_cell(src.strip() + "\n")


def code(src: str):
    return new_code_cell(src.strip() + "\n")


CELLS = [
    md(
        """# Notebook 6 — Dashboard de celeridade (Streamlit)

**Objectivo:** interface para gestão — pendência prevista, prazo típico, faixa de celeridade — **sem re-treino**. Linguagem de gestão em todo o painel; nunca a notação $\\hat S(t\\mid x)$ do texto académico.

**App:** `dashboard/app.py`. Este caderno gera os catálogos (`tabelas/dash_*.csv`) e documenta a arquitectura. O Streamlit **não** corre dentro do Jupyter.

**Horizontes de exibição** (distintos da avaliação académica do Notebook 3): 6 meses, 1 ano, 2 anos, 5 anos e **mais de 7 anos** ($t=2555$ dias).

**Secção da dissertação:** Demonstração aplicada (Capítulo 5). **Não** faz parte da validação estatística formal."""
    ),
    md("## 0. Configuração"),
    code(
        """
from __future__ import annotations

import sys
import warnings
from pathlib import Path

from IPython.display import Markdown, display

warnings.filterwarnings("ignore")

HERE = Path.cwd().resolve()
ROOT = HERE if (HERE / "src").is_dir() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dash_aft import (
    caminho_booster, caminho_encoder, caminho_meta,
    carregar_modelo_aft, construir_artefactos_dash, dir_tab,
)
from src.data_utils import project_root

ROOT = project_root()
print("Booster:", caminho_booster(), caminho_booster().is_file())
print("Encoder:", caminho_encoder(), caminho_encoder().is_file())
print("Meta   :", caminho_meta(), caminho_meta().is_file())
pacote = carregar_modelo_aft()
print("Distribuição", pacote["dist"], "σ", round(pacote["sigma"], 3))
"""
    ),
    md(
        """## 1. Catálogos do painel

Gera `dash_catalogo_*.csv`, prazos por célula classe×órgão×ano, percentis de faixa, exemplos e anomalias. Uma previsão AFT por célula — barato, porque μ é constante na célula."""
    ),
    code(
        """
paths = construir_artefactos_dash(forcar=True)
for k, p in paths.items():
    print(f"{k:12s}  {p.name}  {p.stat().st_size/1e6:.2f} MB")
"""
    ),
    md(
        """## 2. Arquitectura da interface

Páginas em `dashboard/app.py`:

1. **Consulta por combinação** — dropdowns de órgão, classe e ano; curva com linhas verticais nos horizontes de exibição e rótulo percentual; cartões de pendência (6 meses, 1 ano, 2 anos, 5 anos, mais de 7 anos); semáforo de faixa.
2. **Comparar órgãos** — uma classe fixa, vários órgãos; overlay das curvas. Não se misturam classes.
3. **Tendência no tempo** — uma classe e um órgão fixos, 2015–2025, com a ressalva da Figura 18.
4. **Evolução no tempo (visão global)** — distingue coortes com follow-up suficiente (até 2023) das de 2024–2025.
5. **Consulta por número CNJ** — API DataJud em tempo real; ano fora de 2015–2025 usa o ano treinado mais próximo (2025) como *proxy*, com aviso visível; classes fora do top 95% mapeiam para «Outras classes».

Arranque, na raiz do repositório:

```text
streamlit run dashboard/app.py
```
"""
    ),
    code(
        """
display(Markdown('''
**Nota para o Capítulo 5.** O dashboard é uma demonstração de uso gerencial do *booster* já validado. Não gera métricas novas de C-index ou IBS. A consulta de processos de 2026 é uma **extrapolação** (dummy de ano = 2025) e deve ser descrita como tal na Discussão.
'''))
print("Tabelas do painel em", dir_tab())
"""
    ),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbformat.write(nb, OUT)
    print(f"Escrito {OUT} ({len(CELLS)} células)")


if __name__ == "__main__":
    main()
