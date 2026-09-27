#!/usr/bin/env python3
"""Execute Paper6_Colab_GeoAI_OSM.ipynb locally (GPU if free, else CPU)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper6_runtime import (  # noqa: E402
    RunStatus,
    apply_cuda_visible,
    describe_compute,
    project_root,
)


def main() -> int:
    root = project_root()
    nb = root / "Paper6_Colab_GeoAI_OSM.ipynb"
    if not nb.exists():
        raise FileNotFoundError(nb)
    device = describe_compute("torch")
    apply_cuda_visible(device)
    st = RunStatus("geoai", root)
    st.start(device=device, stage_total=1)
    st.update(stage="geoai_nb", stage_index=1, progress_pct=5,
              message=f"Executing GeoAI notebook ({device['kind']})")
    print(device["reason"], flush=True)

    out_dir = root / "derived" / "_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        notebook = nbformat.read(nb, as_version=4)
        import subprocess as _sp
        _sp.check_call([str(Path(sys.executable)), "-m", "ipykernel", "install", "--user",
                        "--name=paper6", "--display-name=Paper6"], stdout=_sp.DEVNULL)
        ep = ExecutePreprocessor(timeout=86400, kernel_name="paper6")
        ep.preprocess(notebook, {"metadata": {"path": str(root)}})
        dest = out_dir / "Paper6_Colab_GeoAI_OSM.executed.ipynb"
        nbformat.write(notebook, dest)
        st.done(f"GeoAI finished. Notebook copy: {dest}")
        print("wrote", dest, flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        st.fail(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
