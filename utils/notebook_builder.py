"""
Pundit Feynman Notebook Builder
Supports both structured JSON cells and legacy free-text → regex approach.
"""

import re
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


def build_notebook_from_cells(cells_json, output_path):
    """
    Build a .ipynb from a list of structured cell dicts.
    Each cell: {"cell_type": "code"|"markdown", "source": "..."}
    """
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.9",
    }

    for cell_data in cells_json:
        cell_type = cell_data.get("cell_type", "code")
        source = cell_data.get("source", "")

        if cell_type == "markdown":
            nb.cells.append(new_markdown_cell(source))
        elif cell_type == "code":
            nb.cells.append(new_code_cell(source))
        else:
            # Default to code for unknown types
            nb.cells.append(new_code_cell(source))

    # Fallback: if no cells, add a placeholder
    if not nb.cells:
        nb.cells.append(new_markdown_cell("# No cells were generated"))

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    md_cells = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(f"  📓 Notebook saved: {output_path} ({len(nb.cells)} cells: {code_cells} code, {md_cells} markdown)")
    return output_path


def build_notebook(full_text, output_path):
    """
    Legacy: Parses mixed markdown/code text into a Jupyter Notebook.
    Separates ```python code blocks into Code cells, everything else into Markdown cells.
    """
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    # Split on ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    parts = re.split(pattern, full_text, flags=re.DOTALL)

    for i, part in enumerate(parts):
        content = part.strip()
        if not content:
            continue

        if i % 2 == 0:
            nb.cells.append(new_markdown_cell(content))
        else:
            nb.cells.append(new_code_cell(content))

    if not nb.cells:
        nb.cells.append(new_markdown_cell(full_text))

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"  📓 Notebook saved: {output_path} ({len(nb.cells)} cells)")
    return output_path
