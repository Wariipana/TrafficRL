#!/usr/bin/env bash
#
# TrafficRL — ejecuta el benchmark comparando los 3 modelos entrenados
# contra las baselines (fixed_random).
#
# Requiere que los modelos ya existan en rl/models/ (generados por
# train_ppo.sh, train_ippo.sh, train_hrl.sh o train_all.sh).
#
# Uso:
#     bash scripts/benchmark.sh
#     bash scripts/benchmark.sh --config config/city_medium.yaml --episodes 30
#     bash scripts/benchmark.sh --mock                  # sin motor C++ (datos sintéticos)
#
# Opciones de modelos: si algún modelo no existe, se omite del benchmark.
#
# Reportes resultantes:
#     rl/results/benchmark_*.csv
#     rl/results/benchmark_*.json
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SERVER="$ROOT/simulation/build/trafficrl_server"
PYTHON="$ROOT/.venv/bin/python"

SHM_PREFIX="trafficrl_bench_$$"
export TRAFFICRL_SHM_PREFIX="$SHM_PREFIX"

# --- argumentos --------------------------------------------------------------
CONFIG="config/city_small.yaml"
EPISODES=20
WIDTH=""
HEIGHT=""
WARMUP=""
SEED=42
MOCK=0
PPO_MODEL="rl/models/ppo_centralized"
IPPO_MODEL="rl/models/ippo_gnn.pt"
HRL_WORKER="rl/models/hrl/worker.pt"
HRL_MANAGER="rl/models/hrl/manager.pt"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)       CONFIG="$2";       shift 2 ;;
    --episodes)     EPISODES="$2";     shift 2 ;;
    --width)        WIDTH="$2";        shift 2 ;;
    --height)       HEIGHT="$2";       shift 2 ;;
    --warmup)       WARMUP="$2";       shift 2 ;;
    --seed)         SEED="$2";         shift 2 ;;
    --mock)         MOCK=1;            shift 1 ;;
    --ppo-model)    PPO_MODEL="$2";    shift 2 ;;
    --ippo-model)   IPPO_MODEL="$2";   shift 2 ;;
    --hrl-worker)   HRL_WORKER="$2";   shift 2 ;;
    --hrl-manager)  HRL_MANAGER="$2";  shift 2 ;;
    *) echo "Argumento desconocido: $1" >&2; exit 1 ;;
  esac
done

# --- modo mock (sin motor C++) ------------------------------------------------
if [[ "$MOCK" -eq 1 ]]; then
  echo "[benchmark] Modo mock: datos sintéticos, sin motor C++"
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: no existe .venv — ejecuta primero: bash scripts/setup_wsl.sh" >&2
    exit 1
  fi
  "$PYTHON" -m rl.training.benchmark --mock --episodes "$EPISODES"
  echo "[benchmark] ✅ Listo. Reportes en rl/results/"
  exit 0
fi

# --- modo live: necesita motor C++ -------------------------------------------
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
echo "[benchmark] Motor: ${WIDTH}x${HEIGHT} grid (de $CONFIG), warmup=$WARMUP, seed=$SEED"
"$SERVER" --width "$WIDTH" --height "$HEIGHT" --warmup "$WARMUP" \
          --seed "$SEED" --prefix "$SHM_PREFIX" &
SERVER_PID=$!

echo "[benchmark] Esperando a la memoria compartida…"
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
  echo "[benchmark] Deteniendo el motor…"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  bash "$SCRIPT_DIR/cleanup_shm.sh" "$SHM_PREFIX" >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

# --- construir argumentos de modelos (solo los que existen) ------------------
MODEL_ARGS=()
if [[ -f "${PPO_MODEL}.zip" ]]; then
  MODEL_ARGS+=(--ppo-model "$PPO_MODEL")
  echo "[benchmark] PPO: $PPO_MODEL"
else
  echo "[benchmark] PPO: modelo no encontrado, se omite ($PPO_MODEL.zip)"
fi
if [[ -f "$IPPO_MODEL" ]]; then
  MODEL_ARGS+=(--ippo-model "$IPPO_MODEL")
  echo "[benchmark] IPPO: $IPPO_MODEL"
else
  echo "[benchmark] IPPO: modelo no encontrado, se omite ($IPPO_MODEL)"
fi
if [[ -f "$HRL_WORKER" && -f "$HRL_MANAGER" ]]; then
  MODEL_ARGS+=(--hrl-worker "$HRL_WORKER" --hrl-manager "$HRL_MANAGER")
  echo "[benchmark] HRL: $HRL_WORKER + $HRL_MANAGER"
else
  echo "[benchmark] HRL: modelos no encontrados, se omiten"
fi

# --- ejecutar benchmark ------------------------------------------------------
echo
echo "════════════════════════════════════════════════════════════"
echo "  Benchmark  config=$CONFIG  episodes=$EPISODES"
echo "════════════════════════════════════════════════════════════"

"$PYTHON" -m rl.training.benchmark \
  --config "$CONFIG" --episodes "$EPISODES" \
  --reference fixed_random \
  "${MODEL_ARGS[@]}"

echo
echo "[benchmark] ✅ Listo. Reportes en rl/results/"
