"""Shared Paper 6 runtime: paths, GPU/CPU selection, progress file, logging.

Prep is pandas / GeoPandas / disk I/O — it stays on CPU. LightGBM and the
optional GNN use a GPU only when one has enough free memory; otherwise CPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

N_THREADS_DEFAULT = 36
GPU_MIN_FREE_LGBM_MB = 2048
GPU_MIN_FREE_TORCH_MB = 3072
GPU_ALLOW_DEFAULT = "0,1"


def allowed_gpu_indices() -> set[int]:
    raw = os.environ.get("PAPER6_GPU_ALLOW", GPU_ALLOW_DEFAULT)
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out or {0, 1}


def project_root() -> Path:
    env = os.environ.get("PAPER6_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parents[1]
    if (here / "DataPaper6").exists() or (here / "Paper6_00_Prep_LocalData.ipynb").exists():
        return here
    cwd = Path.cwd()
    if (cwd / "DataPaper6").exists():
        return cwd
    return here


def run_dir(root: Path | None = None) -> Path:
    d = (root or project_root()) / "derived" / "_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def status_path(root: Path | None = None) -> Path:
    return run_dir(root) / "status.json"


def cap_blas_threads(n: int | None = None) -> int:
    """Limit BLAS/OpenMP to 30–40 threads so a 96-core node is not monopolised."""
    n = int(n or os.environ.get("PAPER6_THREADS", N_THREADS_DEFAULT))
    ncpu = os.cpu_count() or n
    n = max(1, min(n, 40, ncpu))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, str(n))
    return n


def _nvidia_gpus() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_total_mb": float(parts[2]),
                    "mem_used_mb": float(parts[3]),
                    "mem_free_mb": float(parts[4]),
                    "util_pct": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return gpus


def pick_gpu(min_free_mb: int) -> dict[str, Any] | None:
    allow = allowed_gpu_indices()
    gpus = [
        g for g in _nvidia_gpus()
        if g["index"] in allow and g["mem_free_mb"] >= min_free_mb
    ]
    if not gpus:
        return None
    gpus.sort(key=lambda g: (-g["mem_free_mb"], g["util_pct"]))
    return gpus[0]


def describe_compute(role: str = "prep") -> dict[str, Any]:
    """role: prep | lgbm | torch | auto"""
    gpus = _nvidia_gpus()
    info: dict[str, Any] = {
        "role": role,
        "kind": "cpu",
        "n_threads": cap_blas_threads(),
        "gpus": gpus,
        "gpu_index": None,
        "gpu_name": None,
        "reason": "",
    }
    if role == "prep":
        info["reason"] = (
            "Prep streams CSV and runs GeoPandas point-in-polygon — CPU/IO bound; GPU is unused."
        )
        return info

    min_free = GPU_MIN_FREE_TORCH_MB if role in {"torch", "geoai"} else GPU_MIN_FREE_LGBM_MB
    gpu = pick_gpu(min_free)
    if gpu is None:
        info["reason"] = (
            f"No GPU with ≥{min_free} MB free (this node’s GPUs are occupied); using CPU "
            f"({info['n_threads']} threads)."
        )
        return info

    if role == "lgbm":
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError:
            info["reason"] = "lightgbm not installed; CPU fallback."
            return info
        # Constructor accepts device='gpu' even on CPU wheels; only fit() tells the truth.
        try:
            X = np.random.default_rng(0).normal(size=(48, 3))
            y = X[:, 0] + 0.1
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            gpu_id = 0 if visible else gpu["index"]
            lgb.LGBMRegressor(
                n_estimators=1, max_depth=2, verbose=-1,
                device="gpu", gpu_device_id=gpu_id,
            ).fit(X, y)
            info.update(
                kind="gpu",
                gpu_index=gpu["index"],
                gpu_name=gpu["name"],
                reason=f"LightGBM GPU on cuda:{gpu['index']} ({gpu['name']}, {gpu['mem_free_mb']:.0f} MB free).",
            )
            return info
        except Exception as exc:
            info["reason"] = (
                f"LightGBM GPU build not available ({type(exc).__name__}: {exc}); "
                f"CPU with {info['n_threads']} threads. A free GPU ({gpu['name']} #{gpu['index']}) "
                "was present but the installed wheel cannot use it."
            )
            return info

    info.update(
        kind="gpu",
        gpu_index=gpu["index"],
        gpu_name=gpu["name"],
        reason=f"Using GPU {gpu['index']} ({gpu['name']}, {gpu['mem_free_mb']:.0f} MB free).",
    )
    return info


def apply_cuda_visible(info: dict[str, Any]) -> None:
    if info.get("kind") == "gpu" and info.get("gpu_index") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(info["gpu_index"])


def lgbm_fit_kwargs(info: dict[str, Any] | None = None) -> dict[str, Any]:
    info = info or describe_compute("lgbm")
    if info.get("kind") == "gpu":
        return {"device": "gpu", "gpu_device_id": 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else info["gpu_index"]}
    return {"device": "cpu", "n_jobs": info.get("n_threads", N_THREADS_DEFAULT)}


def torch_device_string(info: dict[str, Any] | None = None) -> str:
    info = info or describe_compute("torch")
    apply_cuda_visible(info)
    if info.get("kind") != "gpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


class RunStatus:
    """Live status.json + log so you can reopen Cursor and see how far a job got."""

    def __init__(self, job: str, root: Path | None = None):
        self.root = root or project_root()
        self.job = job
        self.path = status_path(self.root)
        self.log_path = run_dir(self.root) / f"{job}.log"
        self._lock = threading.Lock()
        self._stop_hb = threading.Event()
        self._hb: threading.Thread | None = None
        self.data: dict[str, Any] = {
            "job": job,
            "state": "starting",
            "pid": os.getpid(),
            "started_at": _now(),
            "updated_at": _now(),
            "elapsed_sec": 0,
            "t0": time.time(),
            "device": {},
            "stage": "init",
            "stage_index": 0,
            "stage_total": 7,
            "progress_pct": 0.0,
            "message": "starting",
            "detail": {},
            "error": None,
            "log": str(self.log_path),
            "python": sys.executable,
        }

    def start(self, device: dict[str, Any] | None = None, stage_total: int = 7) -> None:
        self.data["device"] = device or {}
        self.data["stage_total"] = stage_total
        self.data["state"] = "running"
        self.data["pid"] = os.getpid()
        self._flush()
        self._hb = threading.Thread(target=self._heartbeat, name="paper6-status-hb", daemon=True)
        self._hb.start()

    def update(
        self,
        *,
        stage: str | None = None,
        stage_index: int | None = None,
        progress_pct: float | None = None,
        message: str | None = None,
        detail: dict[str, Any] | None = None,
        state: str | None = None,
    ) -> None:
        with self._lock:
            if stage is not None:
                self.data["stage"] = stage
            if stage_index is not None:
                self.data["stage_index"] = stage_index
            if progress_pct is not None:
                self.data["progress_pct"] = round(float(progress_pct), 2)
            if message is not None:
                self.data["message"] = message
                print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
            if detail is not None:
                self.data["detail"] = detail
            if state is not None:
                self.data["state"] = state
            self._flush_unlocked()

    def done(self, message: str = "finished") -> None:
        self.update(state="done", progress_pct=100.0, message=message, stage="done")
        self.close()

    def fail(self, exc: BaseException) -> None:
        self.update(state="failed", message=f"{type(exc).__name__}: {exc}", detail={"error": repr(exc)})
        self.data["error"] = f"{type(exc).__name__}: {exc}"
        self._flush()
        self.close()

    def close(self) -> None:
        self._stop_hb.set()
        if self._hb and self._hb.is_alive():
            self._hb.join(timeout=2)

    def _heartbeat(self) -> None:
        while not self._stop_hb.wait(15):
            with self._lock:
                self._flush_unlocked()

    def _flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        self.data["updated_at"] = _now()
        self.data["elapsed_sec"] = round(time.time() - float(self.data.get("t0", time.time())), 1)
        self.data["pid_alive"] = True
        atomic_json(self.path, self.data)


def read_status(root: Path | None = None) -> dict[str, Any] | None:
    p = status_path(root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def format_status(st: dict[str, Any] | None) -> str:
    if not st:
        return "No status file yet. Start with:\n  python scripts/run_job.py start prep"
    alive = pid_alive(st.get("pid"))
    elapsed = st.get("elapsed_sec", 0)
    mins, secs = divmod(int(elapsed), 60)
    hours, mins = divmod(mins, 60)
    device = st.get("device") or {}
    lines = [
        f"job      : {st.get('job')}  [{st.get('state')}]",
        f"pid      : {st.get('pid')}  ({'running' if alive else 'not running'})",
        f"started  : {st.get('started_at')}",
        f"updated  : {st.get('updated_at')}  (elapsed {hours:d}:{mins:02d}:{secs:02d})",
        f"device   : {device.get('kind', '?')} — {device.get('reason', '')}",
        f"stage    : {st.get('stage_index')}/{st.get('stage_total')}  {st.get('stage')}",
        f"progress : {st.get('progress_pct')}%",
        f"message  : {st.get('message')}",
        f"log      : {st.get('log')}",
    ]
    detail = st.get("detail") or {}
    if detail:
        lines.append("detail   : " + json.dumps(detail, ensure_ascii=False, default=str))
    if st.get("error"):
        lines.append(f"error    : {st['error']}")
    if st.get("state") == "running" and not alive:
        lines.append("note     : status says running but the PID is gone — process was killed.")
    return "\n".join(lines)


def tail_log(path: Path, n: int = 40) -> str:
    if not path.exists():
        return "(no log yet)"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"(cannot read log: {exc})"
    return "\n".join(lines[-n:])
