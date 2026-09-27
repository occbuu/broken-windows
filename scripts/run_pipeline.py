#!/usr/bin/env python3
"""Full Paper 6 pipeline: prep → analysis → geoai, with stage checkpoints.

Launch detached:

    python scripts/run_job.py start all

Closing Cursor does not stop the job. Resume skips stages whose outputs exist.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper6_runtime import (  # noqa: E402
    RunStatus,
    atomic_json,
    cap_blas_threads,
    describe_compute,
    project_root,
    run_dir,
)

cap_blas_threads()


def pipeline_path(root: Path) -> Path:
    return run_dir(root) / "pipeline.json"


def load_pipe(root: Path) -> dict:
    p = pipeline_path(root)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"stages": {}}


def mark(root: Path, pipe: dict, stage: str, state: str, note: str = "") -> None:
    pipe.setdefault("stages", {})
    pipe["stages"][stage] = {"state": state, "note": note}
    pipe["current"] = stage
    atomic_json(pipeline_path(root), pipe)


def prep_done(root: Path) -> bool:
    d = root / "derived"
    return (d / "nyc_tract_month.parquet").exists() and (d / "chi_tract_month.parquet").exists()


def analysis_done(root: Path) -> bool:
    p = root / "tables" / "results_headline.json"
    if not p.exists():
        return False
    try:
        h = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return h.get("sample_city") == "NYC" and (root / "derived" / "nyc_tract_month.parquet").exists()


def geoai_done(root: Path) -> bool:
    osm = root / "derived" / "osm_tract_features.parquet"
    if not osm.exists():
        return False
    try:
        import pandas as pd
        d = pd.read_parquet(osm)
        if "city" in d.columns:
            return bool((d.city == "NYC").any())
    except Exception:
        pass
    # Illinois-only file from the Chicago-only run — redo once NYC panel exists.
    return False


def run_stage(name: str, fn, root: Path, pipe: dict, st: RunStatus, idx: int) -> int:
    st.update(stage=name, stage_index=idx, progress_pct=5 + 30 * (idx - 1),
              message=f"Starting {name}")
    mark(root, pipe, name, "running")
    try:
        rc = fn()
    except Exception as exc:
        traceback.print_exc()
        mark(root, pipe, name, "failed", repr(exc))
        raise
    if rc:
        mark(root, pipe, name, "failed", f"exit {rc}")
        return rc
    mark(root, pipe, name, "done")
    return 0


def main() -> int:
    root = project_root()
    os.environ.setdefault("PAPER6_ROOT", str(root))
    os.environ.setdefault("PAPER6_THREADS", "36")
    os.environ.setdefault("PAPER6_GPU_ALLOW", "0,1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    cap_blas_threads()

    device = describe_compute("prep")
    st = RunStatus("all", root)
    st.start(device=device, stage_total=3)
    st.close()  # children own status.json while they run; avoid heartbeat races
    pipe = load_pipe(root)
    print("pipeline checkpoints:", json.dumps(pipe.get("stages", {}), default=str), flush=True)
    print("device:", device.get("reason") or device.get("kind"), flush=True)
    print("threads:", os.environ.get("PAPER6_THREADS"), "gpus allow",
          os.environ.get("PAPER6_GPU_ALLOW"), flush=True)

    import run_analysis
    import run_geoai
    import run_prep

    if prep_done(root):
        print("[checkpoint] prep already complete — nyc_tract_month + chi_tract_month", flush=True)
        mark(root, pipe, "prep", "done", "outputs exist")
    else:
        rc = run_stage("prep", run_prep.main, root, pipe, st, 1)
        if rc:
            st.fail(RuntimeError(f"prep failed ({rc})"))
            return rc

    if analysis_done(root):
        print("[checkpoint] analysis already complete — NYC headline results", flush=True)
        mark(root, pipe, "analysis", "done", "NYC results_headline.json")
    else:
        rc = run_stage("analysis", run_analysis.main, root, pipe, st, 2)
        if rc:
            st.fail(RuntimeError(f"analysis failed ({rc})"))
            return rc

    if geoai_done(root):
        print("[checkpoint] geoai already complete — NYC OSM features", flush=True)
        mark(root, pipe, "geoai", "done", "osm NYC present")
    else:
        rc = run_stage("geoai", run_geoai.main, root, pipe, st, 3)
        if rc:
            st.fail(RuntimeError(f"geoai failed ({rc})"))
            return rc

    st.done("Full pipeline finished (prep + analysis + geoai).")
    print("DONE full pipeline", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
