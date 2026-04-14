#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/HPC_porepy_project"

cleanup() {
  if [[ -n "${POROUS_PID:-}" ]]; then
    kill "${POROUS_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[Docker] Starting PorousMedia participant..."
(
  cd "${ROOT}/porous_media_participant"
  source /opt/venvs/porous/bin/activate
  python main.py
) &
POROUS_PID=$!

echo "[Docker] Starting FreeFlow participant..."
(
  cd "${ROOT}/free_flow_participant"
  source /opt/venvs/freeflow/bin/activate
  python main.py
)

wait "${POROUS_PID}"

echo "[Docker] Coupled run finished."
