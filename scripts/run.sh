#!/usr/bin/env bash
#
# TrafficRL — arranca el motor de simulación C++ + el dashboard web juntos.
#
# El dashboard (webviz) NO lanza el motor: se conecta por memoria compartida a
# un trafficrl_server que debe estar corriendo. Este script levanta ambos y los
# detiene limpiamente con Ctrl+C.
#
# Uso:
#     bash scripts/run.sh                 # ciudad 4x4 (rápida), puerto 8200
#     bash scripts/run.sh --width 8 --height 8
#
# Variables de entorno opcionales:
#     PORT=8200   puerto del dashboard
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SERVER="$ROOT/simulation/build/trafficrl_server"
PYTHON="$ROOT/.venv/bin/python"
PORT="${PORT:-8200}"

# Argumentos del motor: por defecto una ciudad 4x4 (la más ágil para entrenar).
SERVER_ARGS=("$@")
if [[ ${#SERVER_ARGS[@]} -eq 0 ]]; then
  SERVER_ARGS=(--width 4 --height 4 --seed 42)
fi

# --- comprobaciones previas --------------------------------------------------
if [[ ! -x "$SERVER" ]]; then
  echo "ERROR: no existe $SERVER" >&2
  echo "       Ejecuta primero:  bash scripts/setup_wsl.sh" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: no existe el entorno .venv ($PYTHON)" >&2
  echo "       Ejecuta primero:  bash scripts/setup_wsl.sh" >&2
  exit 1
fi

# Limpia segmentos de memoria compartida colgados de una ejecución anterior.
bash "$SCRIPT_DIR/cleanup_shm.sh" >/dev/null 2>&1 || true

# --- arrancar el motor C++ ---------------------------------------------------
echo "[run] Iniciando motor de simulación: $SERVER ${SERVER_ARGS[*]}"
"$SERVER" "${SERVER_ARGS[@]}" &
SERVER_PID=$!

# Esperar a que cree la memoria compartida antes de levantar el dashboard.
echo "[run] Esperando a la memoria compartida…"
for _ in $(seq 1 50); do
  [[ -e /dev/shm/trafficrl_state ]] && break
  # si el motor murió en el arranque, abortar
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: el motor terminó al arrancar." >&2; exit 1; }
  sleep 0.2
done
if [[ ! -e /dev/shm/trafficrl_state ]]; then
  echo "ERROR: el motor no creó la memoria compartida a tiempo." >&2
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

# --- limpieza al salir -------------------------------------------------------
cleanup() {
  echo
  echo "[run] Deteniendo…"
  kill "$SERVER_PID" "${DASH_PID:-}" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  bash "$SCRIPT_DIR/cleanup_shm.sh" >/dev/null 2>&1 || true
  echo "[run] Listo."
}
trap cleanup INT TERM EXIT

# --- arrancar el dashboard web -----------------------------------------------
echo "[run] Iniciando dashboard en http://localhost:${PORT}"
echo "[run] (Abre esa URL en tu navegador de Windows. Ctrl+C para detener todo.)"
WEBVIZ_PORT="$PORT" "$PYTHON" -m webviz.server &
DASH_PID=$!

# Mantener el script vivo mientras cualquiera de los dos siga corriendo.
wait -n "$SERVER_PID" "$DASH_PID"
