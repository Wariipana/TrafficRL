# TrafficRL — Visión General

## Objetivo

Comparar algoritmos de aprendizaje por refuerzo para el control adaptativo de
semáforos en una ciudad simulada. La hipótesis central: una política aprendida
puede reducir tiempos de espera y congestión frente a un semáforo de ciclo fijo.

## Simulador

Motor de física en C++ (compilado con CMake, comunicación por memoria compartida
POSIX) que expone un entorno Gymnasium. Simula intersecciones en grilla, vehículos
con origen/destino y rutas por pathfinding, y ciclos de semáforo con fases. Corre
en Linux a ~3 400 pasos/segundo sin GPU.

## Algoritmos

| Algoritmo | Tipo | Detalle |
|---|---|---|
| **PPO centralizado** | On-policy, agente único | Observación global aplanada, implementado con Stable-Baselines3. |
| **A2C** | On-policy, agente único | Variante síncrona de SB3; baseline más simple que PPO. |
| **IPPO + GNN** | Multi-agente descentralizado | Un agente por semáforo con política compartida; las observaciones locales se enriquecen con un Graph Neural Network que propaga información entre intersecciones vecinas. |
| **HRL (Manager-Worker)** | Jerárquico, dos niveles | El Manager (nivel alto) fija metas de throughput/espera por zona cada 20 pasos; el Worker (nivel bajo) controla cada semáforo para alcanzarlas. Ambos se entrenan con PPO. |
| **Fijo / Aleatorio** | Baselines | Ciclo de tiempo fijo y cambio aleatorio; referencia mínima de comparación. |

## Validación

El benchmark corre N episodios por algoritmo y mide, independientemente de la
recompensa de entrenamiento:

- **Espera media** (segundos por vehículo) con IC 95 % por bootstrap.
- **Throughput** (vehículos completados por paso).
- **Congestión** (fracción de intersecciones congestionadas).
- **Velocidad media** (m/s en carriles activos).

Los resultados se guardan en `rl/results/` y se visualizan en la página
**Comparar** del dashboard (`http://localhost:8200/compare`).

## Stack

- Motor: C++20, CMake, memoria compartida POSIX.
- Entorno RL: Python 3.12, Gymnasium, PettingZoo.
- Redes: PyTorch (CPU).
- Algoritmos base: Stable-Baselines3 (PPO, A2C).
- Dashboard: FastAPI + WebSockets + Three.js.
