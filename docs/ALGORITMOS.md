# Algoritmos de RL — qué hace cada uno, parámetros y qué esperar

Este proyecto compara tres algoritmos de aprendizaje por refuerzo para el control
adaptativo de semáforos, frente a un baseline de "semáforos mal configurados"
(`fixed_random`). Todos controlan las fases de los semáforos a partir del estado
del tráfico (colas, esperas, velocidades) que reporta el motor de simulación.

> **Acción de control:** en cada paso, cada semáforo elige una de dos fases
> (verde Norte-Sur o verde Este-Oeste). La diferencia entre algoritmos está en
> *cómo* deciden y *qué información* usan.

---

## Baseline — `fixed_random` (semáforos mal configurados)

No aprende. Cada semáforo cicla con un **periodo fijo aleatorio** (15–60 pasos) y
un **desfase aleatorio**, constante durante el episodio. Modela una ciudad real
con semáforos instalados sin optimizar: ritmos estables pero descoordinados, sin
"onda verde". Es el punto de partida realista que el RL debe superar.

- **Parámetros:** periodo mínimo/máximo del ciclo, desfase (aleatorios por episodio).
- **Qué esperar:** esperas medias-altas y throughput bajo. Es la referencia (0 %).

---

## 1. PPO centralizado

**Idea:** un único agente observa el estado de **toda** la ciudad (concatenado en
un vector plano) y decide las fases de todos los semáforos a la vez. Usa PPO
(Proximal Policy Optimization) de Stable-Baselines3 sobre una red MLP.

- **Observación:** estado global aplanado de todas las intersecciones.
- **Red:** MLP `[256, 256, 128]`.
- **Fortaleza:** simple y estable; buena primera referencia de RL. Entrena rápido.
- **Debilidad:** no escala bien a ciudades grandes (el vector de observación crece
  con el número de semáforos) y no modela explícitamente la estructura de la red.

### Parámetros principales (defaults)
| Parámetro        | Valor   | Qué controla |
|------------------|---------|--------------|
| `learning_rate`  | 3e-4    | Velocidad de aprendizaje |
| `n_steps`        | 2048    | Pasos por rollout antes de actualizar |
| `batch_size`     | 64      | Tamaño de minibatch |
| `n_epochs`       | 10      | Pasadas de optimización por rollout |
| `ent_coef`       | 0.01    | Exploración (entropía) |
| `gamma`          | 0.99    | Descuento de recompensas futuras |

**Qué esperar:** mejora clara sobre `fixed_random`, especialmente en ciudades
pequeñas (4×4). Es el algoritmo más rápido de entrenar.

---

## 2. IPPO + GNN (multi-agente)

**Idea:** cada semáforo es un **agente independiente** con su propia política
(Independent PPO), pero comparten una **red neuronal de grafos (GNN)** que les
deja "comunicarse" con sus vecinos. Cada agente observa solo su intersección + un
resumen de los vecinos, y la GNN propaga información por la topología de calles.

- **Observación:** local por intersección + resumen de vecinos (vía GNN).
- **Red:** codificador local → GNN (atención sobre vecinos) → cabezas actor/crítico.
- **Fortaleza:** escala mejor (cada agente ve solo su entorno) y modela la
  estructura de la red — puede aprender coordinación tipo "onda verde".
- **Debilidad:** más lento de entrenar (el paso por la GNN es costoso en CPU).

### Parámetros principales (defaults)
| Parámetro        | Valor   | Qué controla |
|------------------|---------|--------------|
| `learning_rate`  | 3e-4    | Velocidad de aprendizaje |
| `n_steps`        | 1024    | Pasos por rollout y por agente |
| `batch_size`     | 256     | Tamaño de minibatch |
| `n_epochs`       | 8       | Pasadas de optimización por rollout |
| `gamma`          | 0.99    | Descuento |
| `gae_lambda`     | 0.95    | Suavizado de la estimación de ventaja (GAE) |
| `clip_eps`       | 0.2     | Recorte de PPO |
| `k_hops`         | 1       | Saltos de vecindario que ve la GNN |
| `gnn_hidden`     | 128     | Dimensión oculta de la GNN |
| `gnn_embed`      | 64      | Dimensión del embedding de comunicación |

**Qué esperar:** debería igualar o superar a PPO centralizado, con mayor ventaja
cuando hay más semáforos o demanda asimétrica, gracias a la coordinación.

---

## 3. HRL — jerárquico (Manager + Worker)

**Idea:** dos niveles. Un **Manager** divide la ciudad en zonas y, cada cierto
número de pasos, fija una *meta* por zona (p. ej. objetivo de throughput y de
espera). Un **Worker** (un IPPO+GNN condicionado a esas metas) controla los
semáforos para cumplirlas. El Worker recibe una recompensa intrínseca por
acercarse a las metas del Manager (estilo Feudal/Manager-Worker).

- **Observación:** el Worker ve lo mismo que IPPO + el vector de meta de su zona;
  el Manager ve features agregadas por zona.
- **Fortaleza:** separa la estrategia (Manager, lenta) de la ejecución (Worker,
  rápida); pensado para coordinación a gran escala y demanda variable.
- **Debilidad:** el más complejo y lento de entrenar; necesita más pasos para que
  ambos niveles converjan de forma coherente.

### Parámetros principales (defaults)
| Parámetro            | Valor | Qué controla |
|----------------------|-------|--------------|
| Worker `learning_rate` | 3e-4 | Aprendizaje del Worker |
| Manager `learning_rate`| 1e-4 | Aprendizaje del Manager (más lento) |
| `decision_interval`  | 20    | Pasos del Worker por decisión del Manager |
| `goal_horizon` (γ)   | 0.95  | Descuento del Manager |
| `GOAL_DIM`           | 2     | Dimensiones de la meta (throughput, espera) |
| `max_zones`          | 8     | Zonas en que se divide la ciudad |
| `n_steps` / `gamma`  | 1024 / 0.99 | Igual que IPPO para el Worker |

**Qué esperar:** en ciudades pequeñas puede no superar a IPPO (la jerarquía aporta
poco cuando hay pocas zonas); su ventaja aparece en ciudades grandes y escenarios
con demanda cambiante. Requiere el entrenamiento más largo.

---

## Métricas del benchmark

Cada algoritmo se evalúa sobre N episodios (mismas semillas para todos) y se mide:

| Métrica | Qué mide | Mejor cuando |
|---------|----------|--------------|
| **Espera media (s)** | Tiempo medio de espera de los vehículos en los cruces | menor ↓ |
| **Throughput/paso**  | Vehículos que completan su ruta por paso | mayor ↑ |
| **Congestión**       | Fracción de intersecciones congestionadas (0–1) | menor ↓ |
| **Vel. media (m/s)** | Velocidad media de los vehículos | mayor ↑ |
| **Mejora vs fijo**   | % de reducción de espera respecto a `fixed_random` | negativo = mejor |

Además se calculan intervalos de confianza 95 % (bootstrap) y tests pareados
(Wilcoxon + tamaño de efecto de Cohen) para saber si las diferencias entre
algoritmos son estadísticamente significativas o ruido.

## Expectativa general (orden esperado tras entrenamiento suficiente)

```
fixed_random   <   PPO   ≲   IPPO+GNN   ≈/≲   HRL
   (peor)                                      (mejor)
```

En una grilla 4×4 perfectamente simétrica las diferencias entre los RL pueden ser
pequeñas; la ventaja de IPPO/HRL crece con el tamaño de la ciudad y la asimetría
de la demanda. Lo importante es que **los tres deberían batir a `fixed_random`**:
ahí es donde el control adaptativo demuestra su valor frente a semáforos mal
configurados.

> ⚠️ Con pocos pasos de entrenamiento (p. ej. `--steps 1000`) los modelos salen
> casi sin entrenar y pueden aparecer *peor* que el baseline en el benchmark. Eso
> no es un fallo: necesitan entrenamiento suficiente (≥ 50 000 pasos, idealmente
> los 300 000 por defecto) para reflejar su rendimiento real.
