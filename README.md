# TrafficRL

Simulación de tráfico urbano para aprendizaje por refuerzo: control de semáforos
con un motor de física en C++, entorno Gymnasium/PettingZoo, varios algoritmos de
RL y un dashboard web 3D para entrenar y visualizar en vivo.

## Arquitectura

```
┌──────────────────┐  memoria compartida POSIX  ┌─────────────────────┐
│  trafficrl_server │ ◄────────────────────────► │  Python (rl/ webviz/)│
│  (motor C++)      │   /dev/shm/trafficrl_*      │  entorno + RL + web  │
└──────────────────┘                             └─────────────────────┘
                                                          │ HTTP :8200
                                                          ▼
                                                   navegador (3D + métricas)
```

El motor C++ y el código Python corren en el **mismo entorno Linux** y se
comunican por memoria compartida. El dashboard se conecta a un `trafficrl_server`
que ya está corriendo (no lo lanza por sí mismo) — por eso `run.sh` levanta los dos.

## Requisitos

- **Linux** (en Windows: **WSL2** con Ubuntu — el motor usa memoria compartida
  POSIX y no compila en Windows nativo).
- Python 3.11+, CMake, un compilador C++20 (`build-essential`).
- GPU: **no es necesaria**. El cuello de botella es el motor de simulación, no
  la red neuronal; `torch` se instala en su variante CPU.

## Instalación

En Windows, primero instala WSL2 (PowerShell como administrador):

```powershell
wsl --install -d Ubuntu
```

Luego, **dentro de Ubuntu/WSL**, clona el repo en el filesystem de Linux
(no en `/mnt/c`, que es mucho más lento) y ejecuta el setup:

```bash
git clone <URL-del-repo> ~/TrafficRL
cd ~/TrafficRL
bash scripts/setup_wsl.sh
```

`setup_wsl.sh` instala dependencias del sistema, compila el motor C++, crea
`.venv` e instala el proyecto. Es idempotente.

## Uso

```bash
bash scripts/run.sh                      # ciudad 4x4 (ágil), dashboard en :8200
bash scripts/run.sh --width 8 --height 8 # ciudad más grande
```

Abre **http://localhost:8200** en tu navegador (en Windows funciona directo:
WSL2 reenvía el puerto). Desde el panel puedes elegir algoritmo y parámetros,
iniciar/pausar/parar el entrenamiento y ver la ciudad en 3D con métricas en vivo.
`Ctrl+C` en la terminal detiene el motor y el dashboard.

La página **Comparar** (`http://localhost:8200/compare`, enlace en la cabecera)
muestra una tabla con los resultados del benchmark de cada algoritmo —espera
media, throughput, congestión, velocidad y % de mejora frente al semáforo de
tiempo fijo—. Se llena con los reportes de `rl/results/` que genera
`train_all.sh` o el benchmark por CLI.

## Algoritmos de RL

| Algoritmo            | Descripción                                            |
|----------------------|--------------------------------------------------------|
| PPO centralizado     | Una política sobre la observación global (SB3).        |
| A2C                  | Variante on-policy de SB3.                              |
| IPPO + GNN           | Multi-agente: un agente por semáforo + canal GNN.      |
| HRL (Manager-Worker) | Jerárquico: el Manager fija metas por zona al Worker.  |
| Fijo / Aleatorio     | Baselines sin aprendizaje, para comparar.              |

## Entrenar por consola

La forma más simple: entrenar los tres algoritmos RL en secuencia, guardando
cada modelo automáticamente. Arranca el motor C++ por su cuenta y limpia al final
(usa un prefijo de memoria compartida propio, así que puede correr a la vez que el
dashboard sin chocar). Al terminar corre el benchmark y deja los reportes en
`rl/results/` (visibles en la página **Comparar**):

```bash
bash scripts/train_all.sh                                  # 4x4, 300k pasos c/u
bash scripts/train_all.sh --config config/city_small.yaml --steps 500000
bash scripts/train_all.sh --bench-episodes 30              # más episodios de eval
```

Modelos resultantes en `rl/models/`:

| Algoritmo  | Archivos                                       |
|------------|------------------------------------------------|
| PPO        | `ppo_centralized.zip` + `ppo_centralized_vecnorm.pkl` |
| IPPO + GNN | `ippo_gnn.pt`                                  |
| HRL        | `hrl/worker.pt` + `hrl/manager.pt`             |

### O cada algoritmo por separado

Requieren un `trafficrl_server` ya corriendo (p. ej. `bash scripts/run.sh` en otra
terminal, o el propio `train_all.sh`). Cada uno guarda su modelo automáticamente:

```bash
.venv/bin/python -m rl.training.train       --config config/city_small.yaml --steps 300000
.venv/bin/python -m rl.training.train_ippo  --config config/city_small.yaml --steps 300000
.venv/bin/python -m rl.training.train_hrl   --config config/city_small.yaml --steps 300000
```

## Comandos útiles

```bash
# Benchmark comparativo entre algoritmos
.venv/bin/python -m rl.training.benchmark  --config config/city_small.yaml --episodes 30

# Tests
.venv/bin/python -m pytest tests/python/ -v
ctest --test-dir simulation/build

# Limpiar memoria compartida colgada (prefijo opcional)
bash scripts/cleanup_shm.sh
```

## Estructura

```
simulation/   motor de simulación en C++ (CMake) + servidor standalone
rl/           entorno Gymnasium/PettingZoo, agentes RL, benchmark
webviz/       dashboard web (FastAPI + Three.js)
config/       configuraciones de ciudad (YAML)
scripts/      setup_wsl.sh, run.sh, train_all.sh, cleanup_shm.sh
tests/        tests C++ y Python
```
