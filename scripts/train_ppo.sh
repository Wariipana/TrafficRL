#!/usr/bin/env bash
#
# TrafficRL — entrena únicamente el agente PPO centralizado.
#
# PPO centralizado sirve como línea base de referencia: su naturaleza
# centralizada (56+ agentes aplanados en un vector) limita estructuralmente
# su capacidad de convergencia a gran escala, pero el modelo entrenado
# es útil para comparar contra IPPO y HRL en el benchmark.
#
# Uso:
#     bash scripts/train_ppo.sh
#     bash scripts/train_ppo.sh --config config/city_medium.yaml --steps 500000
#
# Modelo resultante:
#     rl/models/ppo_centralized.zip  (+ _vecnorm.pkl)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SERVER="$ROOT/simulation/build/trafficrl_server"
PYTHON="$ROOT/.venv/bin/python"

SHM_PREFIX="trafficrl_train_ppo_$$"
export TRAFFICRL_SHM_PREFIX="$SHM_PREFIX"

# --- argumentos --------------------------------------------------------------
CONFIG="config/city_small.yaml"
STEPS=300000
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
echo "[train_ppo] Motor: ${WIDTH}x${HEIGHT} grid (de $CONFIG), warmup=$WARMUP, seed=$SEED"
"$SERVER" --width "$WIDTH" --height "$HEIGHT" --warmup "$WARMUP" \
          --seed "$SEED" --prefix "$SHM_PREFIX" &
SERVER_PID=$!

echo "[train_ppo] Esperando a la memoria compartida…"
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
  echo "[train_ppo] Deteniendo el motor…"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  bash "$SCRIPT_DIR/cleanup_shm.sh" "$SHM_PREFIX" >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

# --- entrenar PPO ------------------------------------------------------------
echo
echo "════════════════════════════════════════════════════════════"
echo "  PPO centralizado (referencia)  config=$CONFIG  steps=$STEPS"
echo "════════════════════════════════════════════════════════════"

"$PYTHON" -m rl.training.train \
  --config "$CONFIG" --steps "$STEPS" --seed "$SEED" \
  --save-path rl/models/ppo_centralized

echo
echo "[train_ppo] ✅ Listo. Modelo guardado en:"
echo "    rl/models/ppo_centralized.zip  (+_vecnorm.pkl)"
