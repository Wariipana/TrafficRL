#!/usr/bin/env bash
#
# TrafficRL — configuración automática para WSL2 / Ubuntu (Linux x86-64).
#
# Instala dependencias del sistema, compila el motor de simulación en C++,
# crea un entorno virtual de Python e instala el proyecto con sus dependencias.
# Es idempotente: puedes volver a ejecutarlo sin problemas.
#
# Uso (dentro de WSL/Ubuntu, desde la raíz del repo):
#     bash scripts/setup_wsl.sh
#
# Después, para arrancar todo:
#     bash scripts/run.sh
#
set -euo pipefail

# Raíz del repo = carpeta padre de este script (funcione desde donde funcione).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

say() { printf "\n\033[1;36m[setup]\033[0m %s\n" "$*"; }
err() { printf "\n\033[1;31m[error]\033[0m %s\n" "$*" >&2; }

# --- 0) Aviso si parece Windows nativo en vez de WSL/Linux -------------------
if [[ "$(uname -s)" != "Linux" ]]; then
  err "Este script requiere Linux (WSL2). El motor usa memoria compartida POSIX"
  err "y no compila en Windows nativo. Abre Ubuntu (WSL) y ejecútalo ahí."
  exit 1
fi

# Recomendación de rendimiento: el repo debe vivir en el filesystem de Linux.
if [[ "$ROOT" == /mnt/* ]]; then
  err "ADVERTENCIA: el proyecto está en '$ROOT' (disco de Windows montado)."
  err "El acceso a /mnt/c es muy lento y degrada la simulación. Cópialo a tu"
  err "home de Linux, p. ej.:  cp -r \"$ROOT\" ~/TrafficRL && cd ~/TrafficRL"
  read -r -p "¿Continuar de todos modos? [y/N] " ans
  [[ "${ans,,}" == "y" ]] || exit 1
fi

# --- 1) Dependencias del sistema --------------------------------------------
say "Instalando dependencias del sistema (sudo)…"
sudo apt-get update -y
sudo apt-get install -y \
  build-essential cmake git \
  python3 python3-venv python3-pip

# --- 2) Compilar el motor de simulación en C++ ------------------------------
say "Compilando el motor C++ (Release)…"
cmake -S simulation -B simulation/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulation/build -j"$(nproc)"

if [[ ! -x simulation/build/trafficrl_server ]]; then
  err "No se generó simulation/build/trafficrl_server. Revisa la salida de cmake."
  exit 1
fi
say "Motor compilado: simulation/build/trafficrl_server"

# --- 3) Entorno virtual de Python + dependencias ----------------------------
if [[ ! -d .venv ]]; then
  say "Creando entorno virtual en .venv…"
  python3 -m venv .venv
fi

say "Instalando el proyecto y sus dependencias (incluye dashboard web)…"
# torch se instala en su variante CPU por defecto; aquí la GPU no acelera nada,
# así que no se configura CUDA a propósito.
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[viz]"

# --- 4) Listo ----------------------------------------------------------------
say "Configuración completa ✅"
cat <<EOF

Para arrancar el simulador + dashboard:

    bash scripts/run.sh

Luego abre en tu navegador de Windows:  http://localhost:8200

EOF
