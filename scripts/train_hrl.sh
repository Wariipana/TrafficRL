#!/usr/bin/env bash
#
# TrafficRL — entrena únicamente el agente HRL (Manager + Worker jerárquico).
#
# Uso:
#     bash scripts/train_hrl.sh
#     bash scripts/train_hrl.sh --config config/city_medium.yaml --steps 500000
#
# Modelos resultantes:
#     rl/models/hrl/worker.pt
#     rl/models/hrl/manager.pt
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SERVER="$ROOT/simulation/build/trafficrl_server"
PYTHON="$ROOT/.venv/bin/python"

SHM_PREFIX="trafficrl_train_hrl_$$"
export TRAFFICRL_SHM_PREFIX="$SHM_PREFIX"

# --- argumentos --------------------------------------------------------------
CONFIG="config/city_small.yaml"
STEPS=500000
WIDTH=""
HEIGHT=""
WARMUP=""
SEED=42
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)  CONFIG="$2";  shift 2 ;;
    --steps)   STEPS="$2";   shift 2 ;;
    --width)   WIDTH="$2";   shift 2 ;;
    --height)  HEIGHT="$2";  shift 2 ;;
    --warmup)  WARMUP="$2";  shift 2 ;;
    --seed)    SEED="$2";    shift 2 ;;
    *) echo "Argumento desconocido: $1" >&2; exit 1 ;;
  esac
done

read_yaml_int() {
  grep -E "^[[:space:]]*$1:" "$CONFIG" | head -1 | grep -oE '[0-9]+' | head -1
}
[[ -z "$WIDTH"  ]] && WIDTH="$(read_yaml_int grid_width)"
[[ -z "$HEIGHT" ]] && HEIGHT="$(read_yaml_int grid_height)"
WIDTH="${WIDTH:-4}"; HEIGHT="${HEIGHT:-4}"

if [[ -z "$WARMUP" ]]; then
  WARMUP=$(( 31 * WIDTH * HEIGHT ))
  (( WARMUP < 1000 )) && WARMUP=1000
  (( WARMUP > 3000 )) && WARMUP=3000
fi

# --- comprobaciones previas --------------------------------------------------
if [[ ! -x "$SERVER" ]]; then
  echo "ERROR: no existe $SERVER — ejecuta primero: bash scripts/setup_wsl.sh" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: no existe .venv — ejecuta primero: bash scripts/setup_wsl.sh" >&2
  exit 1
fi

bash "$SCRIPT_DIR/cleanup_shm.sh" "$SHM_PREFIX" >/dev/null 2>&1 || true

# --- arrancar el motor C++ ---------------------------------------------------
echo "[train_hrl] Motor: ${WIDTH}x${HEIGHT} grid (de $CONFIG), warmup=$WARMUP, seed=$SEED"
"$SERVER" --width "$WIDTH" --height "$HEIGHT" --warmup "$WARMUP" \
          --seed "$SEED" --prefix "$SHM_PREFIX" &
SERVER_PID=$!

echo "[train_hrl] Esperando a la memoria compartida…"
for _ in $(seq 1 50); do
  [[ -e "/dev/shm/${SHM_PREFIX}_state" ]] && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: el motor terminó al arrancar." >&2; exit 1; }
  sleep 0.2
done
if [[ ! -e "/dev/shm/${SHM_PREFIX}_state" ]]; then
  echo "ERROR: el motor no creó la memoria compartida a tiempo." >&2
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

cleanup() {
  echo "[train_hrl] Deteniendo el motor…"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  bash "$SCRIPT_DIR/cleanup_shm.sh" "$SHM_PREFIX" >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

# --- entrenar HRL (Manager + Worker) -----------------------------------------
echo
echo "════════════════════════════════════════════════════════════"
echo "  HRL Manager + Worker  config=$CONFIG  steps=$STEPS"
echo "════════════════════════════════════════════════════════════"

"$PYTHON" -m rl.training.train_hrl \
  --config "$CONFIG" --steps "$STEPS" --seed "$SEED" \
  --save-dir rl/models/hrl

echo
echo "[train_hrl] ✅ Listo. Modelos guardados en:"
echo "    rl/models/hrl/worker.pt"
echo "    rl/models/hrl/manager.pt"
