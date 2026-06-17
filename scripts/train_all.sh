#!/usr/bin/env bash
#
# TrafficRL — entrena los algoritmos de RL en secuencia y guarda cada modelo.
#
# Arranca el motor de simulación C++ una sola vez, entrena PPO, IPPO+GNN y HRL
# uno tras otro (cada uno guarda su modelo automáticamente) y limpia al terminar.
#
# Uso:
#     bash scripts/train_all.sh                       # 4x4 (city_small), pasos por defecto
#     bash scripts/train_all.sh --config config/city_medium.yaml   # 8x8, tamaño y warmup automáticos
#     bash scripts/train_all.sh --config config/city_small.yaml --steps 300000
#
# El tamaño de la ciudad (--width/--height) y el --warmup se derivan del --config
# automáticamente; puedes sobreescribir cualquiera pasándolo a mano.
#
# Modelos resultantes:
#     rl/models/ppo_centralized.zip      (+ _vecnorm.pkl)
#     rl/models/ippo_gnn.pt
#     rl/models/hrl/worker.pt  rl/models/hrl/manager.pt
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SERVER="$ROOT/simulation/build/trafficrl_server"
PYTHON="$ROOT/.venv/bin/python"

# Prefijo de memoria compartida único para esta corrida, así no choca con el
# dashboard u otra sesión que use el prefijo por defecto. El lado Python lo lee
# de TRAFFICRL_SHM_PREFIX y el motor C++ de --prefix.
SHM_PREFIX="trafficrl_train_$$"
export TRAFFICRL_SHM_PREFIX="$SHM_PREFIX"

# --- argumentos --------------------------------------------------------------
CONFIG="config/city_small.yaml"
STEPS=300000
WIDTH=""                   # vacío = derivar del CONFIG (grid_width/grid_height)
HEIGHT=""
WARMUP=""                  # vacío = derivar del tamaño de la grilla (ver abajo)
SEED=42
BENCH_EPISODES=20          # episodios de evaluación por algoritmo en el benchmark
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)          CONFIG="$2";          shift 2 ;;
    --steps)           STEPS="$2";           shift 2 ;;
    --width)           WIDTH="$2";           shift 2 ;;
    --height)          HEIGHT="$2";          shift 2 ;;
    --warmup)          WARMUP="$2";          shift 2 ;;
    --seed)            SEED="$2";            shift 2 ;;
    --bench-episodes)  BENCH_EPISODES="$2";  shift 2 ;;
    *) echo "Argumento desconocido: $1" >&2; exit 1 ;;
  esac
done

# La TOPOLOGÍA la define el motor C++ (--width/--height), no el YAML: el lado
# Python solo lee el grafo que el motor publica. Por eso debemos arrancar el motor
# con el tamaño del CONFIG; si no, entrenaríamos en 4x4 creyendo que es 8x8.
# Leemos grid_width/grid_height del YAML salvo que se pasen a mano.
read_yaml_int() {  # $1=clave  → primer entero tras "clave:"
  grep -E "^[[:space:]]*$1:" "$CONFIG" | head -1 | grep -oE '[0-9]+' | head -1
}
[[ -z "$WIDTH"  ]] && WIDTH="$(read_yaml_int grid_width)"
[[ -z "$HEIGHT" ]] && HEIGHT="$(read_yaml_int grid_height)"
WIDTH="${WIDTH:-4}"; HEIGHT="${HEIGHT:-4}"   # fallback si el YAML no lo trae

# Warmup = pasos para pre-llenar la ciudad en cada reset. Medido empíricamente
# (con los semáforos ciclando, como en el entrenamiento): el llenado satura a
# ~1000 pasos en 4x4 y a ~2000 en 8x8 — escala aprox. con el área (W·H). Usamos
# 31·(W·H) acotado a [1000, 3000]: da ≈1000 en 4x4 (16·31≈500→clamp 1000) y
# ≈2000 en 8x8 (64·31≈2000). Por encima el llenado marginal cae y sólo añade coste.
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
echo "[train_all] Motor: ${WIDTH}x${HEIGHT} grid (de $CONFIG), warmup=$WARMUP, seed=$SEED"
"$SERVER" --width "$WIDTH" --height "$HEIGHT" --warmup "$WARMUP" \
          --seed "$SEED" --prefix "$SHM_PREFIX" &
SERVER_PID=$!

echo "[train_all] Esperando a la memoria compartida…"
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

# --- limpieza al salir (también si un entrenamiento falla) -------------------
cleanup() {
  echo "[train_all] Deteniendo el motor…"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  bash "$SCRIPT_DIR/cleanup_shm.sh" "$SHM_PREFIX" >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

# --- entrenar cada algoritmo en secuencia ------------------------------------
echo
echo "════════════════════════════════════════════════════════════"
echo "  Entrenando con config=$CONFIG  steps=$STEPS"
echo "════════════════════════════════════════════════════════════"

echo; echo "──[1/3] PPO centralizado ───────────────────────────────────"
"$PYTHON" -m rl.training.train \
  --config "$CONFIG" --steps "$STEPS" --seed "$SEED" \
  --save-path rl/models/ppo_centralized

echo; echo "──[2/3] IPPO + GNN (multi-agente) ──────────────────────────"
"$PYTHON" -m rl.training.train_ippo \
  --config "$CONFIG" --steps "$STEPS" --seed "$SEED" \
  --save-path rl/models/ippo_gnn

echo; echo "──[3/3] HRL (Manager + Worker) ─────────────────────────────"
"$PYTHON" -m rl.training.train_hrl \
  --config "$CONFIG" --steps "$STEPS" --seed "$SEED" \
  --save-dir rl/models/hrl

# --- benchmark: registra y compara los 3 modelos contra las baselines --------
# Reusa el mismo motor (TRAFFICRL_SHM_PREFIX ya está exportado). Escribe los
# reportes en rl/results/ (la página "Comparar" del dashboard los lee de ahí).
echo; echo "──[benchmark] Evaluando los 3 modelos vs baselines ─────────"
"$PYTHON" -m rl.training.benchmark \
  --config "$CONFIG" --episodes "$BENCH_EPISODES" \
  --ppo-model   rl/models/ppo_centralized \
  --ippo-model  rl/models/ippo_gnn.pt \
  --hrl-worker  rl/models/hrl/worker.pt \
  --hrl-manager rl/models/hrl/manager.pt \
  --reference fixed_random \
  || echo "[train_all] (benchmark falló; los modelos sí quedaron guardados)"

echo
echo "[train_all] ✅ Listo. Modelos en rl/models/, reportes en rl/results/:"
echo "    ppo_centralized.zip  ippo_gnn.pt  hrl/worker.pt  hrl/manager.pt"
