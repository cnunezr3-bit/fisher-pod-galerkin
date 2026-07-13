
from pathlib import Path
import numpy as np

def to_sci(x, digits=3):
    if x is None:
        return "nan"
    try:
        if np.isnan(x) or np.isinf(x):
            return "nan"
    except Exception:
        pass
    return f"{x:.{digits}e}"

def write_latex_table(path: Path, caption: str, label: str, headers, rows, align=None):
    if align is None:
        align = "l" + "c" * (len(headers) - 1)

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")
PLOT_WITH_BOUNDARY = False

