"""Output helpers: pretty ASCII tables (via tabulate), pickled database
save/load, and a centralized path-resolver for the `Intervention Results`
output folder.

All text output passes through `save_table` so the on-disk filenames have a
consistent style.
"""

from __future__ import annotations

import os
import pickle
from typing import Iterable

from tabulate import tabulate

from . import config


# ---------------------------------------------------------------------------
# Output dir
# ---------------------------------------------------------------------------
def ensure_output_dir(path: str = None) -> str:
    """Create the output dir if needed and return its absolute path."""
    path = path or config.OUTPUT_DIR
    os.makedirs(path, exist_ok=True)
    return path


def output_path(filename: str) -> str:
    """Full path to a file inside the output dir."""
    return os.path.join(ensure_output_dir(), filename)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def render_table(
    rows: Iterable[Iterable],
    headers: Iterable[str] | str = "firstrow",
    *,
    floatfmt: str = ".4f",
    title: str | None = None,
    notes: Iterable[str] | None = None,
    tablefmt: str = "github",
) -> str:
    """Render a table to a string. By default the first row in `rows` is
    treated as the header.

    `title` is added as a leading line; `notes` as trailing lines."""
    parts = []
    if title:
        parts.append(title)
        parts.append("=" * len(title))
        parts.append("")
    parts.append(tabulate(rows, headers=headers, floatfmt=floatfmt,
                          tablefmt=tablefmt))
    if notes:
        parts.append("")
        for note in notes:
            parts.append(note)
    return "\n".join(parts) + "\n"


def save_table(
    rows: Iterable[Iterable],
    filename: str,
    headers: Iterable[str] | str = "firstrow",
    *,
    floatfmt: str = ".4f",
    title: str | None = None,
    notes: Iterable[str] | None = None,
    tablefmt: str = "github",
    print_too: bool = True,
) -> str:
    """Render + save a table to the output dir. Returns the rendered string."""
    text = render_table(rows, headers=headers, floatfmt=floatfmt,
                        title=title, notes=notes, tablefmt=tablefmt)
    path = output_path(filename)
    with open(path, "w") as f:
        f.write(text)
    if print_too:
        print(text)
        print(f"(saved to {path})")
    return text


# ---------------------------------------------------------------------------
# Generic text writer (for outputs that aren't a table)
# ---------------------------------------------------------------------------
def save_text(text: str, filename: str, print_too: bool = True) -> str:
    path = output_path(filename)
    with open(path, "w") as f:
        f.write(text)
    if print_too:
        print(text)
        print(f"(saved to {path})")
    return text


# ---------------------------------------------------------------------------
# Database save/load (pickled). Records contain np arrays + torch tensors;
# pickle handles both. We strip `extras` of unpicklable items (e.g. CUDA
# tensors with device state) before saving.
# ---------------------------------------------------------------------------
def save_database(db: dict, filename: str = "database.pkl") -> str:
    """Save a database to the output dir. Strips runtime-only extras
    (everything cached for forward passes) since they're cheaply recomputed."""
    runtime_keys = {"prefix_acts", "clean_resid_at_probe", "spec",
                     "input_tokens"}
    stripped = {}
    for sq, by_cat in db.items():
        stripped[sq] = {}
        for cat, by_sub in by_cat.items():
            stripped[sq][cat] = {}
            for sub, recs in by_sub.items():
                new_recs = []
                for rec in recs:
                    # Shallow copy the record, then clean extras.
                    import copy
                    rec2 = copy.copy(rec)
                    rec2.extras = {
                        k: v for k, v in rec.extras.items()
                        if k not in runtime_keys
                    }
                    new_recs.append(rec2)
                stripped[sq][cat][sub] = new_recs
    path = output_path(filename)
    with open(path, "wb") as f:
        pickle.dump(stripped, f)
    return path


def load_database(filename: str = "database.pkl") -> dict:
    path = output_path(filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Convenience: format mean ± std
# ---------------------------------------------------------------------------
def fmt_mean_std(mean: float, std: float, prec: int = 2) -> str:
    """E.g. '4.10 ± 2.93'."""
    if mean != mean:   # NaN
        return "—"
    return f"{mean:.{prec}f} ± {std:.{prec}f}"
