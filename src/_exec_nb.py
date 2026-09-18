"""Executa um notebook no interpretador do ambiente activo."""

from __future__ import annotations

import os
import sys
import traceback
import warnings
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import nbformat
from nbformat.v4 import new_output

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if len(sys.argv) < 2:
    print("Uso: python src/_exec_nb.py 01_exploracao_dados.ipynb")
    sys.exit(2)

nb_path = ROOT / "notebooks" / sys.argv[1]
nb = nbformat.read(nb_path, as_version=4)
globs = {"__name__": "__main__"}
n_code = 0

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    n_code += 1
    src = cell.source
    print(f"--- celula {i} ({n_code}) ---", flush=True)
    out, err = StringIO(), StringIO()
    try:
        from contextlib import redirect_stderr, redirect_stdout

        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(src, f"cell_{i}", "exec"), globs)
        stdout, stderr = out.getvalue(), err.getvalue()
        cell["outputs"] = []
        cell["execution_count"] = n_code
        if stdout:
            cell["outputs"].append(new_output("stream", name="stdout", text=stdout))
        if stderr:
            cell["outputs"].append(new_output("stream", name="stderr", text=stderr))
        print((stdout[-400:] if stdout else "(ok)"), flush=True)
    except Exception:
        tb = traceback.format_exc()
        stdout = out.getvalue()
        print("STDOUT (falha):", stdout[-2000:] if stdout else "(vazio)", flush=True)
        print(tb, flush=True)
        cell["outputs"] = []
        if stdout:
            cell["outputs"].append(new_output("stream", name="stdout", text=stdout))
        cell["outputs"].append(new_output("stream", name="stderr", text=tb))
        nbformat.write(nb, nb_path)
        sys.exit(1)

nbformat.write(nb, nb_path)
print("OK notebook executado", flush=True)
