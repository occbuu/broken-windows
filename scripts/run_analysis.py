#!/usr/bin/env python3
"""Run the restored NYC-primary + Chicago-replication analysis."""
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
    derived = root / "derived"
    if not (derived / "nyc_tract_month.parquet").exists():
        raise FileNotFoundError(
            "Missing nyc_tract_month.parquet. Run `python scripts/run_job.py start prep` first."
        )

    device = describe_compute("lgbm")
    apply_cuda_visible(device)
    st = RunStatus("analysis", root)
    st.start(device=device, stage_total=1)
    st.update(stage="analysis_nyc", stage_index=1, progress_pct=8,
              message=f"NYC + Chicago analysis ({device['kind']})")
    print(device["reason"], flush=True)

    try:
        import analysis_full
        rc = analysis_full.main()
        st.update(progress_pct=95, message="Writing headline results")
        if rc != 0:
            st.fail(RuntimeError("Verification checks failed — see tables/T00_verification.csv"))
            return rc
        st.done("Analysis finished. See tables/results_headline.json and figures/.")
        return 0
    except Exception as exc:
        traceback.print_exc()
        st.fail(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
