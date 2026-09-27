#!/usr/bin/env bash
# Chicago V1 figures + Cities robustness.
# Writes ONLY tables/chicago_v1 and figures/chicago_v1.
# Does NOT run analysis_full.py / analysis_chicago.py / run_job.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export PAPER6_ROOT="${ROOT}"
export PYTHONUNBUFFERED=1
export PAPER6_THREADS="${PAPER6_THREADS:-36}"

PY="${ROOT}/.venv/bin/python"
JOB="scripts/analysis_chicago_v1_robust.py"
LOG="derived/_runs/chicago_v1_robust.log"
PIDF="derived/_runs/chicago_v1_robust.pid"
HEAD="tables/results_headline.json"
SNAP="derived/_runs/nyc_headline_md5_before.txt"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_chicago_v1_robust.sh preflight   # kiểm tra môi trường, không chạy
  bash scripts/run_chicago_v1_robust.sh start       # chạy nền (GPU nếu rảnh, không thì CPU)
  bash scripts/run_chicago_v1_robust.sh start-cpu   # ép CPU (khi GPU đang bị job khác chiếm)
  bash scripts/run_chicago_v1_robust.sh follow      # tail -f log
  bash scripts/run_chicago_v1_robust.sh status      # pid + 20 dòng log cuối
  bash scripts/run_chicago_v1_robust.sh check       # xong chưa? NYC headline còn nguyên?
  bash scripts/run_chicago_v1_robust.sh stop        # dừng job này (không đụng job khác)

Không dùng:
  python scripts/run_job.py start analysis
  python scripts/analysis_chicago.py
  python scripts/analysis_full.py
EOF
}

need() {
  local p="$1"
  if [[ ! -e "${p}" ]]; then
    echo "THIẾU: ${ROOT}/${p}" >&2
    exit 1
  fi
}

preflight() {
  echo "==== PREFLIGHT Chicago V1 ===="
  echo "host     : $(hostname)"
  echo "whoami   : $(whoami)"
  echo "root     : ${ROOT}"
  echo "date     : $(date -Is)"
  echo

  need ".venv/bin/python"
  need "scripts/analysis_chicago_v1_robust.py"
  need "scripts/analysis_chicago.py"
  need "scripts/paper6_runtime.py"
  need "derived/chi_tract_month.parquet"
  need "derived/chi311_events.parquet"
  need "derived/zhvi_zip_month.parquet"
  need "${HEAD}"

  echo "python   : ${PY}"
  "${PY}" -V
  echo
  echo "panel Chicago:"
  ls -lh derived/chi_tract_month.parquet derived/chi311_events.parquet derived/zhvi_zip_month.parquet
  echo
  echo "NYC headline hiện tại (PHẢI là sample_city=NYC):"
  "${PY}" - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("tables/results_headline.json").read_text())
print("  sample_city     =", h.get("sample_city"))
print("  n_tract_months  =", h.get("n_tract_months"))
print("  window          =", h.get("window") or h.get("years"))
if str(h.get("sample_city", "")).upper() != "NYC":
    raise SystemExit("ABORT: tables/results_headline.json không còn NYC. Dừng lại, đừng chạy thêm analysis.")
print("  OK — headline vẫn là NYC")
PY
  mkdir -p derived/_runs
  md5sum "${HEAD}" | tee "${SNAP}"
  echo

  echo "job analysis khác đang chạy? (nếu thấy analysis_full.py / analysis_chicago.py thì KHÔNG start job này trên cùng GPU nếu muốn GPU rảnh)"
  ps -ef | grep -E 'analysis_(full|chicago)\.py|run_pipeline\.py|run_job\.py' | grep -v grep || echo "  (không thấy analysis_full / analysis_chicago / pipeline)"
  echo

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
  else
    echo "nvidia-smi không có — sẽ chạy CPU."
  fi
  echo
  echo "ổ đĩa:"
  df -h "${ROOT}" | tail -n 1
  echo
  echo "PREFLIGHT OK. Tiếp: bash scripts/run_chicago_v1_robust.sh start"
}

force_cpu() {
  export PAPER6_GPU_ALLOW="none"
  export CUDA_VISIBLE_DEVICES=""
  echo "CPU-only: CUDA_VISIBLE_DEVICES empty"
}

start_job() {
  preflight
  mkdir -p derived/_runs tables/chicago_v1 figures/chicago_v1

  if [[ -f "${PIDF}" ]]; then
    old="$(cat "${PIDF}" || true)"
    if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
      echo "ĐANG CHẠY sẵn pid ${old}. Dùng: bash scripts/run_chicago_v1_robust.sh follow"
      exit 0
    fi
  fi

  if [[ -f "${LOG}" ]]; then
    mv "${LOG}" "${LOG}.prev.$(date +%Y%m%d_%H%M%S)"
  fi

  echo "==== START $(date -Is) ===="
  echo "log -> ${ROOT}/${LOG}"
  nohup "${PY}" -u "${JOB}" > "${LOG}" 2>&1 &
  echo $! | tee "${PIDF}"
  sleep 2
  if ! kill -0 "$(cat "${PIDF}")" 2>/dev/null; then
    echo "Job chết ngay. Log:"
    tail -n 80 "${LOG}"
    exit 1
  fi
  echo "pid $(cat "${PIDF}") đang chạy."
  echo "Theo dõi: bash scripts/run_chicago_v1_robust.sh follow"
  echo "Xong khi log có: DONE figures"
}

follow_log() {
  need "${LOG}"
  echo "Ctrl-C chỉ thoát tail, KHÔNG giết job."
  tail -n 30 -f "${LOG}"
}

status_job() {
  echo "root=${ROOT}"
  if [[ -f "${PIDF}" ]]; then
    pid="$(cat "${PIDF}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "RUNNING pid=${pid}"
      ps -p "${pid}" -o pid,etime,pcpu,pmem,cmd || true
    else
      echo "PID file ${pid} nhưng process đã chết."
    fi
  else
    echo "Không có pid file."
  fi
  echo "---- log (cuối) ----"
  if [[ -f "${LOG}" ]]; then
    tail -n 25 "${LOG}"
  else
    echo "(chưa có log)"
  fi
}

check_done() {
  echo "==== CHECK ===="
  if [[ -f "${PIDF}" ]] && kill -0 "$(cat "${PIDF}")" 2>/dev/null; then
    echo "VẪN ĐANG CHẠY pid=$(cat "${PIDF}"). Đợi DONE figures."
    tail -n 15 "${LOG}" || true
    exit 1
  fi
  if [[ ! -f "${LOG}" ]] || ! grep -q "DONE figures" "${LOG}"; then
    echo "CHƯA XONG hoặc fail. Log cuối:"
    tail -n 40 "${LOG}" || true
    exit 1
  fi
  grep -E "DONE figures|DONE tables|did NOT write" "${LOG}"
  echo
  echo "NYC headline sau job (vẫn phải NYC):"
  "${PY}" - <<'PY'
import json
from pathlib import Path
h = json.loads(Path("tables/results_headline.json").read_text())
print("  sample_city =", h.get("sample_city"))
if str(h.get("sample_city", "")).upper() != "NYC":
    raise SystemExit("LỖI: headline đã bị ghi đè, không còn NYC.")
print("  OK")
PY
  if [[ -f "${SNAP}" ]]; then
    echo "md5 trước :"
    cat "${SNAP}"
    echo "md5 sau   :"
    md5sum "${HEAD}"
  fi
  echo
  echo "figures/chicago_v1:"
  ls -lh figures/chicago_v1/*.png 2>/dev/null || echo "  (chưa có png)"
  echo
  echo "tables/chicago_v1:"
  ls -lh tables/chicago_v1/
  echo
  echo "CHECK OK — copy 2 thư mục chicago_v1 về máy local."
}

stop_job() {
  if [[ ! -f "${PIDF}" ]]; then
    echo "Không có pid file."
    exit 0
  fi
  pid="$(cat "${PIDF}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Kill ${pid}"
    kill "${pid}" || true
    sleep 2
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PIDF}"
  echo "stopped"
}

cmd="${1:-}"
case "${cmd}" in
  preflight) preflight ;;
  start-cpu) force_cpu; start_job ;;
  start) start_job ;;
  follow) follow_log ;;
  status) status_job ;;
  check) check_done ;;
  stop) stop_job ;;
  -h|--help|help|"") usage ;;
  *) echo "lệnh không rõ: ${cmd}"; usage; exit 2 ;;
esac
