# Cómo funciona la simulación

Este documento explica cómo funciona el simulador de tráfico de TrafficRL y cómo
están definidas las recompensas que guían a los algoritmos de RL. El objetivo del
proyecto es controlar semáforos para reducir esperas y congestión, comparando
varios algoritmos contra un baseline de semáforos mal configurados.

---

## 1. Arquitectura en dos procesos

La simulación vive en C++ y el RL en Python. Se comunican por **memoria
compartida POSIX** (`/dev/shm/trafficrl_*`):

```
┌──────────────────┐  memoria compartida  ┌──────────────────────┐
│ trafficrl_server │ ◄──────────────────► │ Python (entorno + RL) │
│   (motor C++)    │   /dev/shm/...        │                       │
└──────────────────┘                      └──────────────────────┘
```

El ciclo por paso es siempre el mismo:

1. Python escribe la **acción** (las fases que quiere cada semáforo).
2. El motor C++ avanza la física un paso y escribe el **estado** resultante.
3. Python lee el estado, calcula la **recompensa** y la observación, y vuelve a 1.

Un paso de simulación equivale a `dt = 0.1 s` de tiempo simulado. Un episodio
dura `episode_length_steps` pasos (1000 en la ciudad pequeña ≈ 100 s simulados).

---

## 2. La ciudad

La ciudad es una **grilla** de calles (p. ej. 4×4) generada a partir del YAML de
configuración (`config/city_small.yaml`). Sus piezas:

- **Nodos / intersecciones:** los cruces de la grilla. Algunos tienen semáforo
  (`traffic_light_density` controla cuántos); el resto son cruces sin señal que
  se resuelven por prioridad.
- **Aristas / segmentos de calle:** los tramos de vía entre dos nodos. Cada uno
  tiene longitud, número de carriles, límite de velocidad y una **dirección**
  (Norte, Sur, Este u Oeste).
- **Nodos de entrada/salida (gateways):** los bordes de la ciudad por donde
  entran y salen los autos.

### Vehículos: origen, destino y ruta

Los autos **no aparecen en cualquier sitio**: entran por un gateway del borde, se
les asigna un **destino** (otro gateway alcanzable) y siguen la **ruta más corta**
hacia él (pathfinding precalculado, `next_edge`). Cuando llegan al nodo destino,
abandonan el mapa (despawn). El ritmo de aparición lo fija `spawn_rate_base` y
puede subir durante eventos (ver §5).

---

## 3. El paso de simulación, por dentro

Cada llamada a `step()` ejecuta esta secuencia (ver
`simulation/src/simulation/simulation_loop.cpp`):

```
events_.update()        → eventos dinámicos (lluvia, incidentes, picos de demanda)
check_spawn_despawn()   → entran autos nuevos, salen los que llegaron, navegación entre cruces
rebuild_spatial_hash()  → reindexa posiciones para búsquedas rápidas de vecinos
update_vehicles()       → física de movimiento (IDM) de cada auto
update_traffic_lights() → avanza los temporizadores de los semáforos
```

### Movimiento de los autos: modelo IDM

Cada auto se mueve con el **Intelligent Driver Model** (`vehicles/idm.hpp`), el
modelo estándar de seguimiento de vehículos. En cada paso decide su aceleración
mirando dos cosas:

- **Velocidad deseada** (término de vía libre): acelera hasta su velocidad
  objetivo cuando tiene la vía despejada.
- **Distancia al de adelante** (término de interacción): frena suavemente para
  mantener una separación segura con el vehículo líder, respetando su tiempo de
  reacción y su distancia mínima.

El frenado nunca supera un tope de emergencia. Sobre el IDM hay una **red de
seguridad dura** ("Layer B"): garantiza que dos autos en el mismo carril nunca se
solapen, aunque la física numérica fallara.

### Personalidad del conductor

Cada auto tiene una **personalidad** (`personality_generator.hpp`,
`personality_dynamics.hpp`): velocidad deseada, aceleración, agresividad, tiempo
de reacción, distancia mínima y **cumplimiento del rojo** (`red_light_compliance`).
La personalidad es **dinámica**: cambia un poco según el contexto (p. ej. cerca de
un incidente). Esto da heterogeneidad realista en vez de autos idénticos.

### Qué pasa en un cruce

El cruce es la parte delicada. La lógica evita choques y bloqueos:

- **Semáforo en rojo:** el auto frena en la línea de parada (situada en el borde
  del cruce, no dentro). Un conductor con `red_light_compliance < 1` puede,
  con cierta probabilidad, **saltarse el rojo** (decisión que se fija una vez por
  cruce). Sin esta lógica, un auto que seguía a otro cruzaba el rojo "de
  arrastre" — bug ya corregido.
- **Cruce sin semáforo / conflicto de caja:** dos flujos solo chocan si van en
  **ejes perpendiculares** (uno N–S y otro E–W). Dos autos del mismo eje (incluso
  en sentidos opuestos) usan carriles separados y nunca se cruzan. Un auto cede el
  paso si hay un flujo perpendicular ocupando la caja del cruce.
- **Anti-bloqueo (anti-deadlock):** para que nadie ceda eternamente, hay dos
  reglas:
  - **Desempate por id:** ante un cruce simultáneo, el auto de **menor id** pasa y
    el otro cede. Es asimétrico, así que exactamente uno espera (no hay choque ni
    abrazo mortal).
  - **Anti-inanición:** si un auto lleva esperando más de **8 s**, deja de ceder y
    fuerza su entrada, garantizando progreso.
- **"No bloquees la caja":** un auto no entra al cruce si el segmento de salida no
  tiene sitio para recibirlo; espera fuera. Evita que se quede atascado en medio
  del cruce bloqueando a todos.

El movimiento por el cruce se dibuja como un **arco circular** tangente a los
carriles de entrada y salida, así el giro es suave (el auto reduce un poco la
velocidad al girar, sin saltos).

---

## 4. Los semáforos

Cada semáforo (`traffic_light_system.cpp`) tiene **dos fases**:

- **Fase 0 — NS_GREEN:** verde para Norte–Sur, rojo para Este–Oeste.
- **Fase 1 — EW_GREEN:** verde para Este–Oeste, rojo para Norte–Sur.

Reglas de cambio de fase:

- **Verde mínimo (`min_green`):** una acción del agente solo se acepta si la fase
  actual ya cumplió su tiempo mínimo. Antes de eso, la petición de cambio se
  ignora. Esto impide parpadeos imposibles.
- **Verde máximo (`max_green`):** si una fase dura demasiado, el semáforo cambia
  solo (respaldo autónomo aunque el agente no actúe).
- **Todo-rojo (`ALL_RED_DURATION`):** entre fase y fase hay una ventana de todo-rojo
  (el ámbar) para vaciar el cruce antes de dar el verde cruzado.

**La acción del RL** es justamente elegir, en cada paso y para cada semáforo, qué
fase quiere (0 o 1). Lo único que cambia entre algoritmos es *cómo* deciden y
*qué información* usan.

---

## 5. Eventos dinámicos

Para que el tráfico no sea siempre igual, el motor inyecta eventos
(`events/`):

- **Lluvia fuerte:** reduce la velocidad (hasta −40 %) y alarga el tiempo de
  reacción (hasta +60 %) de todos los autos.
- **Incidentes / obras:** bloquean parcialmente un carril; los autos pasan
  despacio (≈9 km/h) en vez de detenerse en seco, para no bloquear toda la cola.
- **Picos de demanda (mass event):** suben el ritmo de aparición de autos en
  ciertos segmentos.

---

## 6. La observación (lo que ve el agente)

Tras cada paso, el motor escribe el estado de cada intersección. Por intersección
el agente recibe:

| Campo | Qué es |
|-------|--------|
| `vehicles_per_lane` | autos por carril de entrada |
| `queue_length` | autos detenidos por carril (cola) |
| `avg_speed` | velocidad media por carril |
| `avg_wait_time` | espera media de los autos en ese cruce |
| `current_phase` | fase actual del semáforo (0/1) |
| `phase_timer` | cuánto lleva la fase actual |

Un auto cuenta como **en cola** cuando su velocidad es casi nula (`< 0.5 m/s`).

- En **PPO/A2C centralizados**, todas estas observaciones se concatenan en un
  único vector plano (estado global).
- En **IPPO+GNN y HRL** (multi-agente), cada semáforo ve **solo su intersección**
  más un **resumen de sus vecinos** (`neighbor_summary`: cola, espera y throughput
  medios de los cruces adyacentes), y la GNN propaga información por la topología.

---

## 7. Sistema de recompensas

> Punto clave: la **recompensa de entrenamiento** está deliberadamente **separada
> de las métricas de evaluación** del benchmark. Así se evita la *ley de Goodhart*
> (que el agente optimice justo la métrica con la que luego se le juzga). El
> benchmark mide espera, throughput, congestión y velocidad por su cuenta.

La función vive en `rl/env/reward.py` y sus pesos en `config/reward_default.yaml`.

### 7.1 Recompensa global (PPO/A2C y base de los demás)

Tiene **dos componentes** que se combinan con pesos `local_weight` y
`global_weight`:

**Componente local** — promediado sobre todas las intersecciones:

```
local_r = − alpha · espera_media
          − beta  · autos_detenidos
          − gamma · cola_maxima
          + delta · throughput_local
```

**Componente global** — a nivel de toda la ciudad/zona:

```
global_r = + eta  · throughput_total
           − zeta · propagacion_de_congestion
```

**Recompensa final:**

```
reward = local_weight · local_r + global_weight · global_r
```

### 7.2 Qué significa cada término y su peso por defecto

| Símbolo | Peso | Signo | Qué penaliza / premia |
|---------|------|-------|------------------------|
| `alpha` | 0.4 | − | **Espera media** por vehículo en los cruces. El término que más pesa. |
| `beta`  | 0.3 | − | **Autos detenidos** (suma de colas). Castiga acumular vehículos parados. |
| `gamma` | 0.2 | − | **Cola máxima** por carril. Castiga colas muy largas y desbalance. |
| `delta` | 0.1 | + | **Throughput local**: premia que pasen autos por cada cruce. |
| `eta`   | 0.3 | + | **Throughput global**: premia que la ciudad mueva tráfico en conjunto. |
| `zeta`  | 0.2 | − | **Propagación de la congestión**: castiga que la congestión se extienda. |
| `local_weight` | 0.7 | | Peso de lo local (cada cruce). |
| `global_weight` | 0.3 | | Peso de lo global (toda la ciudad). |

En resumen: el agente gana recompensa **moviendo tráfico** (throughput) y la pierde
**dejando autos esperando, parados, en colas largas o congestionando** la red. Los
pesos están elegidos para que la espera sea la prioridad y la coordinación global
sume un 30 %.

### 7.3 Recompensa multi-agente (IPPO+GNN)

En el entorno multi-agente (`rl/env/marl_env.py`) cada semáforo recibe **su
propia** recompensa: una parte **local de su cruce** más una fracción de la
recompensa **global compartida** (señal de cooperación):

```
recompensa_agente_i = local_weight · ( − alpha·espera_i
                                        − beta ·colas_i
                                        + delta·throughput_i )
                    + global_weight · recompensa_global_compartida
```

Así cada agente cuida su intersección pero también es premiado/castigado por cómo
le va a la ciudad entera — incentivo para coordinarse (tipo "onda verde").

### 7.4 Recompensa jerárquica (HRL: Manager–Worker)

El HRL añade una capa más (`rl/agents/hrl/worker.py`, `train_hrl.py`):

- El **Manager** divide la ciudad en zonas y, cada `decision_interval` pasos
  (20 por defecto), fija una **meta por zona**: un objetivo normalizado de
  *throughput* y de *espera* (`GOAL_DIM = 2`).
- El **Worker** (un IPPO+GNN condicionado a esas metas) controla los semáforos y
  recibe una **recompensa intrínseca** por **acercarse** a las metas del Manager:

  ```
  recompensa_worker = recompensa_del_entorno + weight · recompensa_intrinseca
  ```

  donde la intrínseca premia superar el throughput objetivo y quedar por debajo de
  la espera objetivo de su zona (estilo Feudal/Manager-Worker, `weight = 0.3`).

- El **Manager** se entrena con su propia recompensa: las métricas agregadas de
  cada zona (cuán bien le fue a esa zona), descontadas con `goal_horizon = 0.95`.

Esto separa la **estrategia** (Manager, lento, fija metas) de la **ejecución**
(Worker, rápido, mueve semáforos).

---

## 8. Cómo se evalúa (benchmark)

Independiente de la recompensa, el benchmark corre N episodios con las **mismas
semillas** para todos los algoritmos y mide:

| Métrica | Mejor cuando |
|---------|--------------|
| Espera media (s) | menor ↓ |
| Throughput / paso | mayor ↑ |
| Congestión (fracción de cruces congestionados) | menor ↓ |
| Velocidad media (m/s) | mayor ↑ |
| Mejora vs `fixed_random` | mayor reducción de espera ↑ |

Con intervalos de confianza al 95 % (bootstrap) y tests pareados (Wilcoxon +
Cohen) para distinguir mejoras reales del ruido. Lo esperado tras entrenar lo
suficiente:

```
fixed_random  <  PPO  ≲  IPPO+GNN  ≈/≲  HRL
   (peor)                              (mejor)
```

Lo esencial: los tres algoritmos de RL **deben batir al baseline de semáforos mal
configurados**. Ahí es donde el control adaptativo demuestra su valor.

---

## Archivos clave

| Tema | Archivo |
|------|---------|
| Bucle de simulación | `simulation/src/simulation/simulation_loop.cpp` |
| Modelo de conducción (IDM) | `simulation/src/vehicles/idm.hpp` |
| Semáforos | `simulation/src/simulation/traffic_light_system.cpp` |
| Entrada/salida y rutas de autos | `simulation/src/city/spawn_manager.cpp`, `city_graph.cpp` |
| Entorno Gymnasium (single-agent) | `rl/env/traffic_env.py` |
| Entorno PettingZoo (multi-agente) | `rl/env/marl_env.py` |
| Recompensa | `rl/env/reward.py`, `config/reward_default.yaml` |
| Recompensa intrínseca HRL | `rl/agents/hrl/worker.py` |
| Detalle de los algoritmos | `docs/ALGORITMOS.md` |
