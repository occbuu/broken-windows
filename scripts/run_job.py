#!/usr/bin/env python3
"""Start / watch / stop Paper 6 jobs so they survive closing Cursor.

Usage (from BehaveP6/):

    python scripts/run_job.py start prep
    python scripts/run_job.py status
    python scripts/run_job.py log
    python scripts/run_job.py stop

`start` detaches the process (systemd --user if available, otherwise setsid).
Closing Cursor does not send SIGHUP to that process. Reopen Cursor and run
`status` to see the current stage, percent, and last log lines.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper6_runtime import (  # noqa: E402
    format_status,
    pid_alive,
    project_root,
    read_status,
    run_dir,
    tail_log,
)

JOBS = {
    "prep": {
        "script": "run_prep.py",
        "unit": "paper6-prep",
        "desc": "Prep (NYC 311 monthly gz → derived/). CPU. Checkpoints per month.",
    },
    "analysis": {
        "script": "run_analysis.py",
        "unit": "paper6-analysis",
        "desc": "NYC primary + Chicago replication. LightGBM/GNN on GPU 0 or 1.",
    },
    "geoai": {
        "script": "run_geoai.py",
        "unit": "paper6-geoai",
        "desc": "OSM + GraphSAGE. GPU 0 or 1 if free. High RAM.",
    },
    "all": {
        "script": "run_pipeline.py",
        "unit": "paper6-pipeline",
        "desc": "Full pipeline: prep → analysis → geoai. Resumes finished stages.",
    },
}


def venv_python(root: Path) -> Path:
    cand = root / ".venv" / "bin" / "python"
    if cand.exists():
        return cand
    return Path(sys.executable)


def pid_file(root: Path, job: str) -> Path:
    return run_dir(root) / f"{job}.pid"


def log_file(root: Path, job: str) -> Path:
    return run_dir(root) / f"{job}.log"


def _stop_unit(unit: str) -> None:
    if not shutil.which("systemctl"):
        return
    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)


def start_job(job: str) -> int:
    if job not in JOBS:
        print("unknown job:", job, file=sys.stderr)
        return 2
    root = project_root()
    py = venv_python(root)
    script = HERE / JOBS[job]["script"]
    if not script.exists():
        print("missing", script, file=sys.stderr)
        return 1
    st = read_status(root)
    if st and st.get("job") == job and st.get("state") == "running" and pid_alive(st.get("pid")):
        print(f"{job} is already running (pid {st['pid']}).")
        print(format_status(st))
        return 0

    log = log_file(root, job)
    if log.exists():
        log.rename(log.with_suffix(".log.prev"))
    env = os.environ.copy()
    env["PAPER6_ROOT"] = str(root)
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = str(py.parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PAPER6_THREADS", "36")
    env.setdefault("PAPER6_GPU_ALLOW", "0,1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("OMP_NUM_THREADS", env["PAPER6_THREADS"])
    env.setdefault("OPENBLAS_NUM_THREADS", env["PAPER6_THREADS"])
    env.setdefault("MKL_NUM_THREADS", env["PAPER6_THREADS"])
    env.setdefault("NUMEXPR_NUM_THREADS", env["PAPER6_THREADS"])

    unit = JOBS[job]["unit"]
    started = False
    pid = None

    if shutil.which("systemd-run"):
        _stop_unit(unit)
        cmd = [
            "systemd-run", "--user",
            f"--unit={unit}",
            "--collect",
            f"--working-directory={root}",
            f"--setenv=PAPER6_ROOT={root}",
            "--setenv=PYTHONUNBUFFERED=1",
            f"--setenv=PATH={env['PATH']}",
            f"--setenv=PAPER6_THREADS={env['PAPER6_THREADS']}",
            f"--setenv=PAPER6_GPU_ALLOW={env['PAPER6_GPU_ALLOW']}",
            f"--setenv=CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
            f"--setenv=OMP_NUM_THREADS={env['OMP_NUM_THREADS']}",
            f"--setenv=OPENBLAS_NUM_THREADS={env['OPENBLAS_NUM_THREADS']}",
            f"--setenv=MKL_NUM_THREADS={env['MKL_NUM_THREADS']}",
            f"--setenv=NUMEXPR_NUM_THREADS={env['NUMEXPR_NUM_THREADS']}",
            f"--property=StandardOutput=append:{log}",
            f"--property=StandardError=append:{log}",
            f"--description=Paper6 {job} (survives Cursor close)",
            str(py), str(script),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            started = True
            # Resolve MainPID after a moment
            time.sleep(0.4)
            show = subprocess.run(
                ["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"],
                capture_output=True, text=True,
            )
            try:
                pid = int((show.stdout or "0").strip() or 0) or None
            except ValueError:
                pid = None
            print(r.stdout.strip() or f"started {unit} via systemd --user")
        else:
            print("systemd-run unavailable/failed, falling back to setsid:", r.stderr.strip(), file=sys.stderr)

    if not started:
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [str(py), str(script)],
            cwd=str(root),
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,  # setsid — ignore SIGHUP when Cursor/terminal closes
        )
        pid = proc.pid
        print(f"started {job} with setsid pid {pid}")

    if pid:
        pid_file(root, job).write_text(str(pid), encoding="utf-8")
    print(f"log: {log}")
    print("Close Cursor anytime. When you reopen:")
    print(f"  {py} scripts/run_job.py status")
    return 0


def stop_job(job: str | None) -> int:
    root = project_root()
    st = read_status(root)
    target = job or (st.get("job") if st else None)
    if not target:
        print("nothing to stop")
        return 0
    if target in JOBS:
        _stop_unit(JOBS[target]["unit"])
    pid = None
    pf = pid_file(root, target)
    if pf.exists():
        try:
            pid = int(pf.read_text().strip())
        except ValueError:
            pid = None
    if st and st.get("job") == target:
        pid = pid or st.get("pid")
    if pid and pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not pid_alive(pid):
                break
            time.sleep(0.2)
        if pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
        print(f"stopped pid {pid}")
    else:
        print("process already stopped")
    return 0


def _newest_log(root: Path, st: dict | None) -> Path:
    cands = [log_file(root, k) for k in JOBS]
    if st and st.get("log"):
        cands.append(Path(st["log"]))
    existing = [p for p in cands if p.exists()]
    if not existing:
        return log_file(root, "all")
    return max(existing, key=lambda p: p.stat().st_mtime)


def show_status() -> int:
    root = project_root()
    st = read_status(root)
    print(format_status(st))
    if st:
        log = _newest_log(root, st)
        print("\n--- last log lines ---")
        print(f"(from {log.name})")
        print(tail_log(log, 25))
    return 0


def show_log(n: int) -> int:
    root = project_root()
    st = read_status(root)
    path = _newest_log(root, st)
    print(tail_log(path, n))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("start", help="detach a job so it survives closing Cursor")
    sp.add_argument("job", choices=list(JOBS), help="all | prep | analysis | geoai")
    sub.add_parser("status", help="print live progress (safe to run after reopening Cursor)")
    lg = sub.add_parser("log", help="tail the current job log")
    lg.add_argument("-n", type=int, default=60)
    st = sub.add_parser("stop", help="stop the detached job")
    st.add_argument("job", nargs="?", choices=list(JOBS))
    ls = sub.add_parser("jobs", help="list jobs")
    args = p.parse_args()
    if args.cmd == "start":
        return start_job(args.job)
    if args.cmd == "status":
        return show_status()
    if args.cmd == "log":
        return show_log(args.n)
    if args.cmd == "stop":
        return stop_job(args.job)
    if args.cmd == "jobs":
        for k, v in JOBS.items():
            print(f"  {k:<10} {v['desc']}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
