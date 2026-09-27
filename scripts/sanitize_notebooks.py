"""Clear machine-specific notebook output and make project discovery portable."""

from __future__ import annotations

import json
from pathlib import Path


def sanitize(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    for notebook_path in sorted(repository.glob("*.ipynb")):
        sanitize(notebook_path)
        print(f"sanitized {notebook_path.name}")
