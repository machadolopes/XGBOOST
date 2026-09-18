"""Estilo gráfico e tabular em conformidade com a APA 7.ª edição."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PALETTE = "colorblind"

# Mapas de calor só em preto e cinza (APA / impressão a preto).
CMAP_HEATMAP = LinearSegmentedColormap.from_list(
    "apa_preto_cinza",
    ["#ffffff", "#d0d0d0", "#9a9a9a", "#5c5c5c", "#111111"],
)
CMAP_HEATMAP_DIV = LinearSegmentedColormap.from_list(
    "apa_preto_cinza_div",
    ["#111111", "#6e6e6e", "#ffffff", "#6e6e6e", "#111111"],
)


def setup_apa_style() -> None:
    """Configura o estilo APA para todos os gráficos da sessão."""
    sns.set_style("white")
    sns.set_palette(PALETTE)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.3,
            "grid.color": "#cccccc",
            "grid.linestyle": "-",
        }
    )


def fmt_milhares(ax, eixo: str = "x") -> None:
    """Rótulos de eixo em milhares com espaço como separador (APA)."""
    fmt = ticker.FuncFormatter(lambda v, _p: f"{int(round(v)):,}".replace(",", " "))
    if eixo in {"x", "both"}:
        ax.xaxis.set_major_formatter(fmt)
    if eixo in {"y", "both"}:
        ax.yaxis.set_major_formatter(fmt)


def apa_heatmap(
    data,
    ax,
    *,
    cbar_label: str,
    vmin=None,
    vmax=None,
    annot: bool = False,
    fmt: str = ".2f",
    diverging: bool = False,
    xtick_rotation: int = 45,
    square: bool = False,
    ytick_size: int = 11,
    xtick_size: int = 11,
):
    """Heatmap APA: escala preto–cinza, sem grelha, eixos serifados."""
    ax.grid(False)
    cmap = CMAP_HEATMAP_DIV if diverging else CMAP_HEATMAP
    with plt.rc_context({"axes.grid": False}):
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0 if diverging else None,
            annot=annot,
            fmt=fmt,
            annot_kws={"fontsize": 9, "fontfamily": "serif"},
            linewidths=0.25,
            linecolor="#e6e6e6",
            cbar_kws={"label": cbar_label, "shrink": 0.82},
            square=square,
        )
    ax.grid(False)
    ax.tick_params(axis="x", labelrotation=xtick_rotation, labelsize=xtick_size)
    ax.tick_params(axis="y", labelsize=ytick_size)
    for lab in ax.get_xticklabels():
        lab.set_fontfamily("serif")
        lab.set_ha("right" if xtick_rotation else "center")
    for lab in ax.get_yticklabels():
        lab.set_fontfamily("serif")
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.yaxis.label.set_fontfamily("serif")
        cbar.ax.yaxis.label.set_fontsize(11)
        cbar.ax.tick_params(labelsize=10)


def apa_figure_title(fig, ax, number, title, note=None):
    """Adiciona título e nota em formato APA 7.ª edição."""
    fig.suptitle(
        f"Figura {number}",
        fontweight="bold",
        fontsize=13,
        x=0.01,
        ha="left",
        y=1.08,
    )
    ax.set_title(title, fontstyle="italic", fontsize=12, loc="left", pad=10)
    if note:
        fig.text(
            0.01,
            -0.08,
            f"Nota. {note}",
            fontsize=10,
            fontstyle="italic",
            ha="left",
            va="top",
            wrap=True,
        )
    fig.tight_layout()


def apa_multipanel_title(fig, number, title, note=None):
    """Título APA para figuras com vários eixos."""
    fig.suptitle(
        f"Figura {number}",
        fontweight="bold",
        fontsize=13,
        x=0.01,
        ha="left",
        y=1.06,
    )
    fig.text(
        0.01,
        0.995,
        title,
        fontstyle="italic",
        fontsize=12,
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    if note:
        fig.text(
            0.01,
            -0.04,
            f"Nota. {note}",
            fontsize=10,
            fontstyle="italic",
            ha="left",
            va="top",
            transform=fig.transFigure,
            wrap=True,
        )


def save_apa_figure(fig, path: str | Path, close: bool = False) -> Path:
    """Exporta a figura em PNG a 300 dpi, com fundo branco."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return path


def save_apa_table_png(
    df: pd.DataFrame,
    path: str | Path,
    number,
    title: str,
    note: str | None = None,
) -> tuple[Path, object]:
    """Renderiza a tabela em formato APA e guarda PNG a 300 dpi."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = df.shape
    fig_w = min(12.0, max(8.0, 1.1 * n_cols + 4.0))
    fig_h = min(18.0, 2.2 + 0.38 * (n_rows + 1) + (0.8 if note else 0.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    fig.suptitle(
        f"Tabela {number}",
        fontweight="bold",
        fontsize=13,
        x=0.01,
        ha="left",
        y=0.98,
    )
    ax.set_title(title, fontstyle="italic", fontsize=11, loc="left", pad=12)

    col_labels = [str(c) for c in df.columns]
    cell_text = df.astype(str).values.tolist()
    numeric_idx = {
        i
        for i, col in enumerate(df.columns)
        if pd.api.types.is_numeric_dtype(df[col])
    }

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.7)
        cell.get_text().set_fontfamily("serif")
        edges = ""
        if r == 0:
            edges = "TB"
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("left")
        else:
            if r == n_rows:
                edges = "B"
            if c in numeric_idx:
                cell.get_text().set_ha("right")
            else:
                cell.get_text().set_ha("left")
        cell.visible_edges = edges

    if note:
        fig.text(
            0.01,
            0.01,
            f"Nota. {note}",
            fontsize=9,
            fontstyle="italic",
            ha="left",
            va="bottom",
            wrap=True,
        )
    fig.tight_layout(rect=(0, 0.06 if note else 0.02, 1, 0.92))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    return path, fig


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Guarda uma tabela em CSV (UTF-8 com BOM, para Excel em Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def show_apa_table(df: pd.DataFrame, number, title: str, note: str | None = None):
    """Mostra uma tabela com cabeçalho APA no notebook."""
    from IPython.display import Markdown, display

    display(Markdown(f"**Tabela {number}**  \n*{title}*"))
    display(df)
    if note:
        display(Markdown(f"*Nota.* {note}"))


def exportar_tabela_apa(
    df: pd.DataFrame,
    number,
    title: str,
    note: str | None,
    dir_tab: str | Path,
    stem: str,
):
    """CSV + PNG 300 dpi + visualização no notebook."""
    from IPython.display import display

    dir_tab = Path(dir_tab)
    csv_path = save_table(df, dir_tab / f"{stem}.csv")
    png_path, fig = save_apa_table_png(
        df, dir_tab / f"{stem}.png", number, title, note
    )
    show_apa_table(df, number, title, note)
    display(fig)
    plt.close(fig)
    print(f"Exportado: {csv_path.name} | {png_path.name} (300 dpi)")
    return csv_path, png_path
