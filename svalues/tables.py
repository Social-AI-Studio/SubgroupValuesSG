"""Helpers for writing paper tables as CSV (full precision) and LaTeX (paper formatting)."""

import pandas as pd

from svalues.config import TABLES_DIR


def fmt(x: float, digits: int = 3, signed: bool = False) -> str:
    """Paper number style: 0.512 -> '.512', -0.115 -> '-.115', signed=True -> '+.171'."""
    text = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return text.replace("0.", ".", 1) if text.lstrip("+-").startswith("0.") else text


def save_table(name: str, data: pd.DataFrame, latex: str) -> None:
    """Writes outputs/tables/<name>.csv and outputs/tables/<name>.tex."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    (TABLES_DIR / f"{name}.tex").write_text(latex.rstrip() + "\n", encoding="utf-8")
    print(f"  wrote outputs/tables/{name}.{{csv,tex}}")
