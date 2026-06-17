"""
TrainingSession — owns the RL training run and exposes it to the web layer.

The whole point of this object is to dissolve the "two processes fighting over
the simulation" problem: the training loop (Stable-Baselines3 PPO) and the
visualizer live in the SAME Python process. The web server controls the run by
flipping flags this object reads between steps; the render data is whatever the
agent itself saw on the last step. No second connection to shared memory, no
competing writers.

State machine:  idle → running ⇄ paused → stopped
A SB3 callback (StreamingCallback) runs once per environment step. It:
  * blocks while paused (so the sim genuinely freezes),
  * aborts training when stop is requested,
  * throttles to the requested steps/second,
  * captures the latest StateSnapshot + scalar metrics into `latest`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.env.traffic_env import TrafficEnv
from rl.env.data_types import EnvConfig, GraphData, StateSnapshot


@dataclass
class Metrics:
    """Scalar time-series point pushed to the browser each step."""
    step: int = 0
    sim_tick: int = 0
    episode_step: int = 0
    reward: float = 0.0
    avg_wait: float = 0.0
    max_wait: float = 0.0
    throughput: float = 0.0
    congestion: float = 0.0
    num_vehicles: int = 0
    fps: float = 0.0


@dataclass
class SessionStatus:
    state: str = "idle"            # idle | running | paused | stopped | error
    algo: str = "ppo"              # active algorithm (ppo|ippo_gnn|hrl|fixed_random)
    total_timesteps: int = 0
    current_step: int = 0
    speed_hz: float = 30.0         # target steps/second (0 = unthrottled)
    config_path: str = ""
    error: str = ""
    metrics: dict = field(default_factory=lambda: asdict(Metrics()))
    summary: dict | None = None


class TrainingSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        # Control flags read by the callback between steps.
        self._pause_evt = threading.Event()   # set => paused
        self._stop_evt  = threading.Event()   # set => abort training
        self._speed_hz  = 30.0

        self.status = SessionStatus()

        # Latest render payload (graph sent once on connect, frame each step).
        self._graph: GraphData | None = None
        self.latest_frame: dict | None = None
        self._frame_lock = threading.Lock()

        # tiny rolling history for the LIVE metric charts (kept small for perf)
        self.history: list[dict] = []
        self._max_history = 600

        # full (sub-sampled) history kept for the end-of-run summary and the
        # complete historical charts; rollout_marks are the PPO rollout
        # boundaries (vertical lines in the charts).
        self.full_history: list[dict] = []
        self._full_stride = 1          # 1 point kept every _full_stride steps
        self._full_cap = 4000          # cap points; raise stride when exceeded
        self.rollout_marks: list[int] = []
        self.summary: dict | None = None
        self._run_started_at: float = 0.0

        # Trained artefacts kept after learn() so they can be saved on demand.
        self._model: Any = None
        self._venv: Any = None

        self._bridge_ref: Any = None  # the env's BridgeClient, for vehicle reads

    # ---- public control surface (called from FastAPI handlers) ----

    # Algorithms the dashboard can launch:
    #   ppo          — SB3 PPO over the centralized (flattened) env
    #   ippo_gnn     — Independent PPO with a GNN comm channel (rl.agents.marl)
    #   hrl          — hierarchical Manager + Worker (rl.agents.hrl)
    #   fixed_random — "badly configured city" baseline (no learning); the
    #                  realistic starting point RL must beat in the benchmark
    # ippo_gnn and hrl are hand-written PyTorch loops over MARLTrafficEnv; ppo and
    # the baseline run over the centralized TrafficEnv.
    ALGORITHMS = ("ppo", "ippo_gnn", "hrl", "fixed_random")

    def start(self, config_path: str, total_timesteps: int, algo: str = "ppo",
              params: dict | None = None) -> None:
        algo = (algo or "ppo").lower()
        if algo not in self.ALGORITHMS:
            raise RuntimeError(f"Algoritmo desconocido: {algo}")
        params = self._sanitize_params(params or {})
        with self._lock:
            if self.status.state in ("running", "paused"):
                raise RuntimeError("Ya hay un entrenamiento en curso")
            self._pause_evt.clear()
            self._stop_evt.clear()
            self.history.clear()
            self.full_history.clear()
            self.rollout_marks.clear()
            self.summary = None
            self._model = None
            self._venv = None
            self._algo = algo
            self._params = params
            self._run_started_at = time.time()
            self.status = SessionStatus(
                state="running",
                total_timesteps=total_timesteps,
                speed_hz=self._speed_hz,
                config_path=config_path,
            )
            self.status.algo = algo
            self._thread = threading.Thread(
                target=self._run, args=(config_path, total_timesteps, algo, params),
                daemon=True,
            )
            self._thread.start()

    # Bounds for the optional training hyperparameters the dashboard exposes.
    # Anything missing or out of range falls back to each algorithm's default.
    _PARAM_BOUNDS = {
        "learning_rate":        (1e-6, 1e-1),
        "ent_coef":             (0.0,  0.5),
        "gamma":                (0.5,  0.9999),
        "n_steps":              (16,   8192),
        "seed":                 (0,    2**31 - 1),
        "episode_length_steps": (100,  100_000),
    }

    def _sanitize_params(self, params: dict) -> dict:
        """Clamp/validate the user-supplied hyperparameters; drop unknown or
        non-numeric entries so a bad value can never crash a run."""
        clean: dict = {}
        int_keys = {"n_steps", "seed", "episode_length_steps"}
        for key, (lo, hi) in self._PARAM_BOUNDS.items():
            if key not in params or params[key] in (None, ""):
                continue
            try:
                val = float(params[key])
            except (TypeError, ValueError):
                continue
            val = max(lo, min(hi, val))
            clean[key] = int(round(val)) if key in int_keys else val
        return clean

    def pause(self) -> None:
        if self.status.state in ("running", "inference"):
            self._pause_evt.set()
            self._prev_state = self.status.state
            self.status.state = "paused"

    def resume(self) -> None:
        if self.status.state == "paused":
            self._pause_evt.clear()
            self.status.state = getattr(self, "_prev_state", "running")

    def stop(self) -> None:
        if self.status.state in ("running", "paused", "inference"):
            self._stop_evt.set()
            self._pause_evt.clear()  # unblock the loop so it can see the stop

    def set_speed(self, hz: float) -> None:
        hz = max(0.0, min(240.0, float(hz)))
        self._speed_hz = hz
        self.status.speed_hz = hz

    # ---- inference (load a saved model and let it drive the lights) ----

    def start_inference(self, model_name: str, config_path: str) -> None:
        with self._lock:
            if self.status.state in ("running", "paused", "inference"):
                raise RuntimeError("Ya hay una sesión en curso")
            det = self.detect_model(model_name)
            if det is None:
                raise RuntimeError(f"Modelo no encontrado: {model_name}")
            self._pause_evt.clear()
            self._stop_evt.clear()
            self.history.clear()
            self.full_history.clear()
            self.rollout_marks.clear()
            self.summary = None
            self._model = None
            self._venv = None
            self._run_started_at = time.time()
            self.status = SessionStatus(
                state="inference",
                speed_hz=self._speed_hz,
                config_path=config_path,
            )
            self.status.algo = det["algo"]
            self._thread = threading.Thread(
                target=self._run_inference, args=(det, config_path), daemon=True
            )
            self._thread.start()

    def _run_inference(self, det: dict, config_path: str) -> None:
        """Drive the lights with a saved model. Dispatches by algorithm: SB3/PPO
        over the centralized env, IPPO+GNN / HRL over the MARL env."""
        try:
            algo = det["algo"]
            if algo == "ppo":
                self._infer_sb3(det, config_path)
            elif algo == "ippo_gnn":
                self._infer_ippo(det, config_path)
            elif algo == "hrl":
                self._infer_hrl(det, config_path)
            else:
                raise RuntimeError(f"Inferencia no soportada para algoritmo: {algo}")
            self.status.state = "stopped"
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.status.state = "error"
            self.status.error = f"{type(exc).__name__}: {exc}"

    def _infer_sb3(self, det: dict, config_path: str) -> None:
        from rl.agents.centralized.ppo_agent import load_model
        cfg = EnvConfig.from_yaml(config_path)
        env_raw = TrafficEnv(cfg)
        self._graph = env_raw._graph
        self._bridge_ref = env_raw._bridge

        def _make():
            return gymnasium.wrappers.FlattenObservation(env_raw)

        venv = DummyVecEnv([_make])
        # VecNormalize stats live next to the model: "<dir>/vecnorm.pkl" for
        # dashboard saves, "<name>_vecnorm.pkl" for flat CLI saves.
        model_base = det["model"]
        for vnpath in (os.path.join(os.path.dirname(model_base), "vecnorm.pkl"),
                       model_base + "_vecnorm.pkl"):
            if os.path.exists(vnpath):
                venv = VecNormalize.load(vnpath, venv)
                venv.training = False          # freeze stats; just normalise
                venv.norm_reward = False
                break
        model = load_model(model_base, venv)

        last_wall = time.perf_counter()
        obs = venv.reset()
        while not self._stop_evt.is_set():
            while self._pause_evt.is_set():
                if self._stop_evt.is_set():
                    break
                time.sleep(0.05)
            if self._stop_evt.is_set():
                break
            # throttle to target steps/second
            hz = self._speed_hz
            if hz > 0:
                target_dt = 1.0 / hz
                el = time.perf_counter() - last_wall
                if el < target_dt:
                    time.sleep(target_dt - el)
            now = time.perf_counter()
            fps = 1.0 / max(now - last_wall, 1e-6)
            last_wall = now

            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = venv.step(action)

            state = env_raw._last_state
            if state is not None:
                m = Metrics(
                    step=self.status.current_step + 1,
                    sim_tick=int(state.sim_tick),
                    episode_step=int(state.episode_step),
                    reward=0.0,
                    avg_wait=round(float(state.avg_wait_global), 3),
                    max_wait=round(float(state.max_wait_global), 3),
                    throughput=round(float(state.total_throughput), 3),
                    congestion=round(float(state.congestion_spread), 4),
                    num_vehicles=int(state.num_vehicles),
                    fps=round(fps, 1),
                )
                self.publish_frame(state, m)
        venv.close()

    def _infer_ippo(self, det: dict, config_path: str) -> None:
        """Drive the lights with a saved IPPO+GNN model over the MARL env."""
        import torch
        from rl.env.marl_env import MARLTrafficEnv
        from rl.agents.marl.ippo_agent import IPPOActorCritic, flatten_obs
        from rl.models.gnn import build_adjacency_mask

        cfg = EnvConfig.from_yaml(config_path)
        env = MARLTrafficEnv(cfg)
        obs_d, _ = env.reset()
        self._graph = env._graph
        self._bridge_ref = env._bridge
        n = env._graph.num_lights

        model = IPPOActorCritic()
        model.load_state_dict(torch.load(det["model"], map_location="cpu"))
        model.eval()
        adj = build_adjacency_mask(env._graph.light_node_ids, env._graph.edges, k_hops=1)

        last_wall = time.perf_counter()
        step = 0
        while not self._stop_evt.is_set():
            stop, fps, last_wall = self._marl_gate(last_wall)
            if stop:
                break
            flat = torch.tensor(
                np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(n)]),
                dtype=torch.float32)
            with torch.no_grad():
                actions, _, _ = model.get_action(flat, adj, deterministic=True)
            action_dict = {f"light_{i}": int(actions[i]) for i in range(n)}
            obs_d, _, _, _, _ = env.step(action_dict)
            step += 1
            self._publish_marl_step(env, step, 0.0, fps)
            if not env.agents:
                obs_d, _ = env.reset()
        env.close()

    def _infer_hrl(self, det: dict, config_path: str) -> None:
        """Drive the lights with a saved HRL Manager + Worker over the MARL env."""
        import torch
        from rl.env.marl_env import MARLTrafficEnv
        from rl.agents.hrl.worker import HRLWorkerActorCritic
        from rl.agents.hrl.manager import HRLManager, ManagerConfig
        from rl.agents.marl.ippo_agent import flatten_obs
        from rl.models.gnn import build_adjacency_mask

        cfg = EnvConfig.from_yaml(config_path)
        env = MARLTrafficEnv(cfg)
        obs_d, _ = env.reset()
        self._graph = env._graph
        self._bridge_ref = env._bridge
        n = env._graph.num_lights
        num_zones = min(8, max(1, n // 4))

        worker = HRLWorkerActorCritic()
        worker.load_state_dict(torch.load(det["worker"], map_location="cpu"))
        worker.eval()
        manager = HRLManager(ManagerConfig(), num_zones=num_zones)
        manager.load(det["manager"])
        adj = build_adjacency_mask(env._graph.light_node_ids, env._graph.edges, k_hops=1)

        last_wall = time.perf_counter()
        step = 0
        while not self._stop_evt.is_set():
            stop, fps, last_wall = self._marl_gate(last_wall)
            if stop:
                break
            state = env._last_state
            goals_np = manager.get_goals(state, step)
            goals_per = np.stack([
                goals_np[int(env._graph.light_node_ids[i]) % num_zones] for i in range(n)])
            flat = torch.tensor(
                np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(n)]),
                dtype=torch.float32)
            goals_t = torch.tensor(goals_per, dtype=torch.float32)
            with torch.no_grad():
                actions, _, _ = worker.get_action(flat, goals_t, adj, deterministic=True)
            action_dict = {f"light_{i}": int(actions[i]) for i in range(n)}
            obs_d, _, _, _, _ = env.step(action_dict)
            step += 1
            self._publish_marl_step(env, step, 0.0, fps)
            if not env.agents:
                obs_d, _ = env.reset()
        env.close()

    # ---- persistence ----

    MODELS_DIR = "rl/models"

    def save_model(self, name: str) -> str:
        """Persist the trained model + VecNormalize stats + run.json (summary and
        a benchmark-comparable AlgorithmResult) under rl/models/<name>/."""
        if self._model is None:
            raise RuntimeError("No hay modelo entrenado para guardar")
        safe = "".join(c for c in name if c.isalnum() or c in "-_") or "model"
        out_dir = os.path.join(self.MODELS_DIR, safe)
        os.makedirs(out_dir, exist_ok=True)

        algo = getattr(self, "_algo", "ppo")
        if algo in ("ippo_gnn", "hrl"):
            # Hand-written PyTorch models: persist state dicts. These don't load
            # into the SB3 inference path (no model.zip), so they won't appear in
            # the inference dropdown — they are kept for the benchmark/eval CLIs.
            import torch
            if algo == "hrl":
                torch.save(self._model.state_dict(), os.path.join(out_dir, "worker.pt"))
                mgr = getattr(self, "_hrl_manager", None)
                if mgr is not None:
                    mgr.save(os.path.join(out_dir, "manager.pt"))
            else:
                torch.save(self._model.state_dict(), os.path.join(out_dir, "model.pt"))
        else:
            self._model.save(os.path.join(out_dir, "model"))      # → model.zip
            if self._venv is not None:
                try:
                    self._venv.save(os.path.join(out_dir, "vecnorm.pkl"))
                except Exception:
                    pass
        run = {
            "name": safe,
            "algo": algo,
            "params": getattr(self, "_params", {}),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self.summary,
        }
        with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2, ensure_ascii=False)
        return out_dir

    def detect_model(self, name: str) -> dict | None:
        """Resolve a saved model into {algo, paths} by inspecting its files.

        Supports both layouts:
          - dashboard saves:  rl/models/<name>/  with model.zip | model.pt |
                              worker.pt+manager.pt  (+ run.json giving 'algo')
          - CLI saves:        rl/models/<name>.zip            (PPO)
                              rl/models/<name>.pt             (IPPO)
                              rl/models/<name>/worker.pt+manager.pt  (HRL)
        Returns None if nothing recognisable is found.
        """
        base = self.MODELS_DIR
        safe = "".join(c for c in name if c.isalnum() or c in "-_")
        d = os.path.join(base, safe)

        # Prefer the explicit algo from run.json when present (dashboard saves).
        declared = None
        rj = os.path.join(d, "run.json")
        if os.path.exists(rj):
            try:
                with open(rj, encoding="utf-8") as f:
                    declared = (json.load(f).get("algo") or "").lower()
            except Exception:
                declared = None

        # --- directory layouts ---
        if os.path.isdir(d):
            worker, manager = os.path.join(d, "worker.pt"), os.path.join(d, "manager.pt")
            if os.path.exists(worker) and os.path.exists(manager):
                return {"algo": "hrl", "worker": worker, "manager": manager}
            if os.path.exists(os.path.join(d, "model.zip")):
                return {"algo": "ppo", "model": os.path.join(d, "model")}
            if os.path.exists(os.path.join(d, "model.pt")):
                return {"algo": declared or "ippo_gnn", "model": os.path.join(d, "model.pt")}

        # --- flat-file layouts (CLI / train_all.sh) ---
        if os.path.exists(os.path.join(base, safe + ".zip")):
            return {"algo": "ppo", "model": os.path.join(base, safe)}  # SB3 adds .zip
        if os.path.exists(os.path.join(base, safe + ".pt")):
            return {"algo": "ippo_gnn", "model": os.path.join(base, safe + ".pt")}
        return None

    def list_models(self) -> list[dict]:
        """List saved models runnable in inference, across both file layouts."""
        out: list[dict] = []
        base = self.MODELS_DIR
        if not os.path.isdir(base):
            return out
        seen: set[str] = set()
        # candidate names: subdirectories + flat .zip/.pt files (deduped)
        names: list[str] = []
        for entry in sorted(os.listdir(base)):
            full = os.path.join(base, entry)
            if os.path.isdir(full):
                names.append(entry)
            elif entry.endswith((".zip", ".pt")):
                names.append(os.path.splitext(entry)[0])
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            det = self.detect_model(name)
            if det is None:
                continue
            info = {"name": name, "algo": det["algo"], "saved_at": "", "config": ""}
            rj = os.path.join(base, name, "run.json")
            if os.path.exists(rj):
                try:
                    with open(rj, encoding="utf-8") as f:
                        run = json.load(f)
                    info["saved_at"] = run.get("saved_at", "")
                    info["config"] = (run.get("summary") or {}).get("config", "")
                except Exception:
                    pass
            out.append(info)
        return out

    # Note: episodes reset automatically at their boundary (SB3 calls env.reset
    # when an episode truncates), so the visualizer shows fresh episodes without
    # an explicit reset control. A manual reset button can be added later by
    # routing a flag through StreamingCallback.

    # ---- snapshot accessors for the web layer ----

    def graph_payload(self) -> dict | None:
        if self._graph is None:
            return None
        g = self._graph
        return {
            "nodes": [
                {"id": n.id, "x": n.x, "y": n.y, "zone": n.zone,
                 "has_light": bool(n.has_light), "light_id": n.light_id,
                 "num_outgoing": n.num_outgoing}
                for n in g.nodes
            ],
            "edges": [
                {"from": e.from_node, "to": e.to_node,
                 "lanes": e.num_lanes, "dir": e.direction}
                for e in g.edges
            ],
            "light_node_ids": list(g.light_node_ids),
            "num_lights": g.num_lights,
        }

    def frame_payload(self) -> dict | None:
        with self._frame_lock:
            return self.latest_frame

    # ---- training thread ----

    def _run(self, config_path: str, total_timesteps: int, algo: str = "ppo",
             params: dict | None = None) -> None:
        params = params or {}
        try:
            cfg = EnvConfig.from_yaml(config_path)
            # env-level override applies to every algorithm (incl. baselines)
            if "episode_length_steps" in params:
                cfg.episode_length_steps = params["episode_length_steps"]

            if algo in ("ippo_gnn", "hrl"):
                # multi-agent PyTorch loops over the PettingZoo-style MARL env;
                # they build and own their env (and set _graph/_bridge_ref).
                if algo == "ippo_gnn":
                    self._run_ippo(cfg, total_timesteps)
                else:
                    self._run_hrl(cfg, total_timesteps)
            else:
                env_raw = TrafficEnv(cfg)
                self._graph = env_raw._graph
                self._bridge_ref = env_raw._bridge
                if algo == "ppo":
                    self._run_sb3(env_raw, config_path, total_timesteps, algo)
                else:  # fixed_random — rule-based baseline (no learning)
                    self._run_baseline(env_raw, config_path, total_timesteps, algo)

            self._build_summary(config_path, total_timesteps)
            self.status.state = "stopped"
        except Exception as exc:  # surface the error to the UI instead of dying silently
            import traceback
            traceback.print_exc()
            self.status.state = "error"
            self.status.error = f"{type(exc).__name__}: {exc}"

    def _run_sb3(self, env_raw, config_path: str, total_timesteps: int, algo: str) -> None:
        """Train the centralized SB3 PPO policy and stream it live."""
        def _make():
            return gymnasium.wrappers.FlattenObservation(env_raw)

        venv = DummyVecEnv([_make])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

        p = getattr(self, "_params", {})
        ppo_n_steps = int(p.get("n_steps", 2048))
        # SB3 needs the rollout buffer (n_steps * 1 env) ≥ batch_size, so shrink
        # the batch for tiny user-chosen rollouts instead of crashing.
        ppo_batch = min(64, ppo_n_steps)
        model = PPO("MlpPolicy", venv,
                    n_steps=ppo_n_steps, batch_size=ppo_batch, n_epochs=10,
                    learning_rate=p.get("learning_rate", 3e-4),
                    ent_coef=p.get("ent_coef", 0.01),
                    verbose=0, policy_kwargs={"net_arch": [256, 256, 128]})
        # keep the artefacts so the model can be saved on demand afterward
        self._model = model
        self._venv  = venv

        cb = StreamingCallback(self, env_raw)
        model.learn(total_timesteps=total_timesteps, callback=cb)
        venv.close()

    def _run_baseline(self, env_raw, config_path: str, total_timesteps: int, algo: str) -> None:
        """Run the 'badly configured city' baseline (fixed_random): each light
        cycles on a FIXED random period + offset, held constant for the run.
        Streams each step like training so the dashboard shows live traffic and a
        start-vs-end summary — the realistic starting point RL must beat. No model
        is produced, so it can't be saved (can_save stays false)."""
        env = gymnasium.wrappers.FlattenObservation(env_raw)
        n_lights = env_raw._graph.num_lights
        rng = np.random.default_rng(0)
        # per-light fixed period (15-60 steps) + offset — stable but uncoordinated
        periods = rng.integers(15, 61, size=n_lights)
        offsets = np.array([rng.integers(0, p) for p in periods], dtype=np.int64)

        last_wall = time.perf_counter()
        obs, _ = env.reset()
        step = 0
        while not self._stop_evt.is_set() and step < total_timesteps:
            while self._pause_evt.is_set():
                if self._stop_evt.is_set():
                    break
                time.sleep(0.05)
            if self._stop_evt.is_set():
                break

            hz = self._speed_hz
            if hz > 0:
                target_dt = 1.0 / hz
                el = time.perf_counter() - last_wall
                if el < target_dt:
                    time.sleep(target_dt - el)
            now = time.perf_counter()
            fps = 1.0 / max(now - last_wall, 1e-6)
            last_wall = now

            action = (((step + offsets) // periods) % 2).astype(np.int64)
            obs, reward, terminated, truncated, _ = env.step(action)

            state = env_raw._last_state
            if state is not None:
                m = Metrics(
                    step=step + 1,
                    sim_tick=int(state.sim_tick),
                    episode_step=int(state.episode_step),
                    reward=round(float(reward), 4),
                    avg_wait=round(float(state.avg_wait_global), 3),
                    max_wait=round(float(state.max_wait_global), 3),
                    throughput=round(float(state.total_throughput), 3),
                    congestion=round(float(state.congestion_spread), 4),
                    num_vehicles=int(state.num_vehicles),
                    fps=round(fps, 1),
                )
                self.publish_frame(state, m)
            step += 1
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    # ---- live-streaming control shared by the MARL (PyTorch) loops ----

    def _marl_gate(self, last_wall: float) -> tuple[bool, float, float]:
        """Honour pause/stop/speed between env steps for the hand-written MARL
        loops, mirroring StreamingCallback. Returns (stop, fps, new_last_wall);
        callers must abort their rollout when stop is True."""
        if self._stop_evt.is_set():
            return True, 0.0, last_wall
        while self._pause_evt.is_set():
            if self._stop_evt.is_set():
                return True, 0.0, last_wall
            time.sleep(0.05)
        hz = self._speed_hz
        if hz > 0:
            target_dt = 1.0 / hz
            el = time.perf_counter() - last_wall
            if el < target_dt:
                time.sleep(target_dt - el)
        now = time.perf_counter()
        fps = 1.0 / max(now - last_wall, 1e-6)
        return False, fps, now

    def _publish_marl_step(self, env, step: int, reward: float, fps: float) -> None:
        """Build + publish a frame from a MARLTrafficEnv's latest snapshot."""
        state = env._last_state
        if state is None:
            return
        m = Metrics(
            step=step,
            sim_tick=int(state.sim_tick),
            episode_step=int(state.episode_step),
            reward=round(float(reward), 4),
            avg_wait=round(float(state.avg_wait_global), 3),
            max_wait=round(float(state.max_wait_global), 3),
            throughput=round(float(state.total_throughput), 3),
            congestion=round(float(state.congestion_spread), 4),
            num_vehicles=int(state.num_vehicles),
            fps=round(fps, 1),
        )
        self.publish_frame(state, m)

    def _run_ippo(self, cfg: EnvConfig, total_timesteps: int) -> None:
        """Train Independent PPO + GNN over MARLTrafficEnv, streaming each step.

        Same rollout/GAE/PPO-update logic as rl.agents.marl.ippo_agent.train_ippo,
        but the rollout collection honours pause/stop/speed and publishes a frame
        per env step so the dashboard renders the live city."""
        import torch
        import torch.nn.functional as F
        from rl.env.marl_env import MARLTrafficEnv
        from rl.agents.marl.ippo_agent import (
            IPPOActorCritic, IPPOConfig, RolloutBuffer, flatten_obs, LOCAL_FEAT_DIM,
        )
        from rl.models.gnn import build_adjacency_mask

        # Smaller rollout than the CLI default: the dashboard updates per-timestep
        # through the GNN, so a 1024-step rollout would freeze the live view for a
        # long stretch on each PPO update. 256 keeps the city moving.
        p = getattr(self, "_params", {})
        icfg = IPPOConfig(
            env=cfg, total_timesteps=total_timesteps,
            n_steps=int(p.get("n_steps", 256)),
            learning_rate=p.get("learning_rate", IPPOConfig.learning_rate),
            ent_coef=p.get("ent_coef", IPPOConfig.ent_coef),
            gamma=p.get("gamma", IPPOConfig.gamma),
            seed=int(p.get("seed", IPPOConfig.seed)),
        )
        device = torch.device(icfg.device)
        torch.manual_seed(icfg.seed)

        env = MARLTrafficEnv(cfg)
        obs_d, _ = env.reset(seed=icfg.seed)
        self._graph = env._graph
        self._bridge_ref = env._bridge

        n_agents = env._graph.num_lights
        adj_mask = build_adjacency_mask(
            env._graph.light_node_ids, env._graph.edges, k_hops=icfg.k_hops, device=device,
        )
        model = IPPOActorCritic(gnn_hidden=icfg.gnn_hidden, gnn_embed=icfg.gnn_embed).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=icfg.learning_rate)
        buffer = RolloutBuffer(icfg.n_steps, n_agents, device)
        self._model = model          # so save_model() can persist it (torch path)
        self._venv = None

        total_steps = 0
        step = 0
        last_wall = time.perf_counter()

        while total_steps < icfg.total_timesteps and not self._stop_evt.is_set():
            buffer.reset()
            model.eval()
            with torch.no_grad():
                while not buffer.is_full():
                    stop, fps, last_wall = self._marl_gate(last_wall)
                    if stop:
                        env.close()
                        return

                    flat_all = torch.tensor(
                        np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(n_agents)]),
                        dtype=torch.float32, device=device,
                    )
                    actions_t, log_probs_t, values_t = model.get_action(flat_all, adj_mask)
                    action_dict = {f"light_{i}": int(actions_t[i]) for i in range(n_agents)}
                    obs_d, rews_d, terms_d, trunc_d, _ = env.step(action_dict)

                    rewards_t = torch.tensor(
                        [rews_d.get(f"light_{i}", 0.0) for i in range(n_agents)],
                        dtype=torch.float32, device=device,
                    )
                    dones_t = torch.tensor(
                        [float(terms_d.get(f"light_{i}", False) or trunc_d.get(f"light_{i}", False))
                         for i in range(n_agents)],
                        dtype=torch.float32, device=device,
                    )
                    buffer.add(flat_all, adj_mask, actions_t, log_probs_t, rewards_t, values_t, dones_t)
                    total_steps += n_agents
                    step += 1
                    self._publish_marl_step(env, step, float(rewards_t.mean()), fps)

                    if not env.agents:
                        obs_d, _ = env.reset()

                flat_last = torch.tensor(
                    np.stack([flatten_obs(obs_d.get(f"light_{i}", obs_d[list(obs_d.keys())[0]]))
                              for i in range(n_agents)]),
                    dtype=torch.float32, device=device,
                )
                _, last_vals = model.forward(flat_last, adj_mask)

            returns, advantages = buffer.compute_returns_and_advantages(
                last_vals, icfg.gamma, icfg.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            model.train()
            self.rollout_marks.append(total_steps)
            T = len(buffer.obs)
            flat_obs_all = torch.stack(buffer.obs)
            actions_all = torch.stack(buffer.actions)
            old_lp_all = torch.stack(buffer.log_probs)
            # The GNN forward is defined for a single graph of N nodes against an
            # (N,N) adjacency, so we evaluate each rollout timestep separately and
            # average the PPO loss over a minibatch of timesteps before stepping.
            for _ in range(icfg.n_epochs):
                perm = torch.randperm(T)
                bs = max(1, icfg.batch_size // n_agents)
                for start in range(0, T, bs):
                    idx = perm[start:start + bs]
                    losses = []
                    for t in idx.tolist():
                        new_lp, entropy, vals = model.evaluate_actions(
                            flat_obs_all[t], adj_mask, actions_all[t])
                        adv = advantages[t]
                        ratio = torch.exp(new_lp - old_lp_all[t])
                        pg_loss = torch.max(
                            -adv * ratio,
                            -adv * ratio.clamp(1 - icfg.clip_eps, 1 + icfg.clip_eps),
                        ).mean()
                        vf_loss = F.mse_loss(vals, returns[t])
                        losses.append(pg_loss + icfg.vf_coef * vf_loss
                                      - icfg.ent_coef * entropy.mean())
                    loss = torch.stack(losses).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), icfg.max_grad_norm)
                    optimizer.step()
        env.close()

    def _run_hrl(self, cfg: EnvConfig, total_timesteps: int) -> None:
        """Train hierarchical Manager + Worker over MARLTrafficEnv, streaming each
        step. Mirrors rl.training.train_hrl with pause/stop/speed + live frames."""
        import torch
        from rl.env.marl_env import MARLTrafficEnv
        from rl.agents.hrl.manager import HRLManager, ManagerConfig, aggregate_zones
        from rl.agents.hrl.worker import HRLWorkerActorCritic, goal_reward_shaping
        from rl.agents.marl.ippo_agent import flatten_obs
        from rl.models.gnn import build_adjacency_mask

        p = getattr(self, "_params", {})
        seed = int(p.get("seed", 42))
        device = torch.device("cpu")
        torch.manual_seed(seed)

        env = MARLTrafficEnv(cfg)
        obs_d, _ = env.reset(seed=seed)
        self._graph = env._graph
        self._bridge_ref = env._bridge

        n_agents = env._graph.num_lights
        num_zones = min(8, max(1, n_agents // 4))
        adj_mask = build_adjacency_mask(
            env._graph.light_node_ids, env._graph.edges, k_hops=1, device=device,
        )
        worker = HRLWorkerActorCritic().to(device)
        manager = HRLManager(ManagerConfig(), num_zones=num_zones)
        w_opt = torch.optim.Adam(worker.parameters(), lr=p.get("learning_rate", 3e-4))
        # keep both artefacts so save_model() can persist the HRL pair
        self._model = worker
        self._hrl_manager = manager
        self._venv = None

        total_steps = 0
        step = 0
        w_rollout: list[dict] = []
        last_wall = time.perf_counter()

        while total_steps < total_timesteps and not self._stop_evt.is_set():
            stop, fps, last_wall = self._marl_gate(last_wall)
            if stop:
                break

            state = env._last_state
            goals_np = manager.get_goals(state, total_steps)
            zone_feats = aggregate_zones(state, num_zones)
            flat_local = np.stack([flatten_obs(obs_d[f"light_{i}"]) for i in range(n_agents)])
            goals_per_agent = np.stack([
                goals_np[int(env._graph.light_node_ids[i]) % num_zones]
                for i in range(n_agents)
            ])
            flat_t = torch.tensor(flat_local, dtype=torch.float32, device=device)
            goals_t = torch.tensor(goals_per_agent, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions_t, log_probs_t, values_t = worker.get_action(flat_t, goals_t, adj_mask)

            action_dict = {f"light_{i}": int(actions_t[i]) for i in range(n_agents)}
            obs_d, rews_d, terms_d, trunc_d, _ = env.step(action_dict)
            total_steps += n_agents
            step += 1

            intrinsic = goal_reward_shaping(env._last_state, goals_np, num_zones)
            rewards_np = np.array([rews_d.get(f"light_{i}", 0.0) + intrinsic[i]
                                   for i in range(n_agents)], dtype=np.float32)
            dones_np = np.array([
                float(terms_d.get(f"light_{i}", False) or trunc_d.get(f"light_{i}", False))
                for i in range(n_agents)], dtype=np.float32)

            w_rollout.append({
                "flat": flat_t, "goals": goals_t, "actions": actions_t,
                "log_probs": log_probs_t,
                "rewards": torch.tensor(rewards_np, device=device),
                "values": values_t,
                "dones": torch.tensor(dones_np, device=device),
            })
            manager.record_transition(
                zone_feats=zone_feats, goals=goals_np,
                zone_reward=float(np.mean(list(rews_d.values()))),
                done=bool(np.any(dones_np)),
            )
            self._publish_marl_step(env, step, float(rewards_np.mean()), fps)

            if not env.agents:
                obs_d, _ = env.reset()
                manager.update()

            if len(w_rollout) >= max(1, 1024 // n_agents):
                self._update_hrl_worker(
                    worker, w_opt, w_rollout, adj_mask, device,
                    gamma=p.get("gamma", 0.99), ent_coef=p.get("ent_coef", 0.01))
                w_rollout.clear()
                self.rollout_marks.append(total_steps)
        env.close()

    def _update_hrl_worker(self, worker, optimizer, rollout, adj_mask, device,
                           gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                           vf_coef=0.5, ent_coef=0.01, n_epochs=4) -> None:
        """PPO update for the HRL Worker. Same GAE/clip objective as
        rl.training.train_hrl._update_worker, but evaluates each timestep's graph
        separately so the GNN sees a single (N,N) adjacency per call (the upstream
        helper reshapes timesteps into one batch, which breaks the GNN contract
        when n_agents != batch granularity)."""
        import torch
        import torch.nn.functional as F

        T = len(rollout)
        rewards = torch.stack([r["rewards"] for r in rollout])   # (T, N)
        values  = torch.stack([r["values"]  for r in rollout])   # (T, N)
        dones   = torch.stack([r["dones"]   for r in rollout])   # (T, N)

        advantages = torch.zeros_like(rewards, device=device)
        gae = torch.zeros(rewards.shape[1], device=device)
        for t in reversed(range(T)):
            nv = values[t + 1] if t < T - 1 else torch.zeros_like(values[0])
            delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        worker.train()
        bs = max(1, T // 4)
        for _ in range(n_epochs):
            perm = torch.randperm(T)
            for start in range(0, T, bs):
                idx = perm[start:start + bs]
                losses = []
                for t in idx.tolist():
                    new_lp, entropy, vals = worker.evaluate_actions(
                        rollout[t]["flat"], rollout[t]["goals"], adj_mask,
                        rollout[t]["actions"])
                    adv = advantages[t]
                    ratio = torch.exp(new_lp - rollout[t]["log_probs"])
                    pg_loss = torch.max(
                        -adv * ratio,
                        -adv * ratio.clamp(1 - clip_eps, 1 + clip_eps),
                    ).mean()
                    vf_loss = F.mse_loss(vals, returns[t])
                    losses.append(pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean())
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(worker.parameters(), 0.5)
                optimizer.step()
        worker.eval()

    def _build_summary(self, config_path: str, total_timesteps: int) -> None:
        """Aggregate the run into a summary: totals/averages, start-vs-end deltas,
        and an AlgorithmResult (rl/benchmark schema) so runs are comparable across
        algorithms via rl/benchmark/report.py."""
        h = self.full_history
        if not h:
            self.summary = None
            return

        def col(key):
            return [p[key] for p in h]
        import numpy as _np
        rwd, wait = col("reward"), col("avg_wait")
        tp, cong  = col("throughput"), col("congestion")
        veh = col("num_vehicles")

        # start-vs-end: mean of first vs last 10% of the run (≥1 sample each)
        n = len(h); k = max(1, n // 10)
        def delta(arr):
            start = float(_np.mean(arr[:k])); end = float(_np.mean(arr[-k:]))
            return {"start": round(start, 3), "end": round(end, 3),
                    "delta": round(end - start, 3)}

        # AlgorithmResult from final-window samples → comparable with benchmark.
        algo_result = None
        try:
            from rl.benchmark.metrics import EpisodeMetrics, AlgorithmResult
            episodes = []
            for i, p in enumerate(h[-k:]):
                episodes.append(EpisodeMetrics(
                    total_vehicles_served=float(p["throughput"]) * total_timesteps,
                    throughput_per_step=float(p["throughput"]),
                    avg_wait_time_s=float(p["avg_wait"]),
                    max_wait_time_s=float(p["avg_wait"]),
                    congestion_spread=float(p["congestion"]),
                    avg_speed_ms=0.0,
                    steps_completed=int(p["step"]),
                    episode_id=i,
                ))
            ar = AlgorithmResult(name=getattr(self, "_algo", "ppo"),
                                 config_label=config_path.split("/")[-1].replace(".yaml", ""),
                                 episodes=episodes)
            ar.compute_summary()
            algo_result = asdict(ar)
        except Exception:
            algo_result = None

        self.summary = {
            "config": config_path,
            "total_timesteps": total_timesteps,
            "steps_recorded": n,
            "rollouts": len(self.rollout_marks),
            "duration_s": round(time.time() - self._run_started_at, 1),
            "aggregate": {
                "reward_mean": round(float(_np.mean(rwd)), 4),
                "reward_final": round(float(rwd[-1]), 4),
                "avg_wait_mean": round(float(_np.mean(wait)), 3),
                "avg_wait_min": round(float(_np.min(wait)), 3),
                "throughput_mean": round(float(_np.mean(tp)), 3),
                "congestion_mean": round(float(_np.mean(cong)), 4),
                "vehicles_mean": round(float(_np.mean(veh)), 1),
            },
            "start_vs_end": {
                "reward": delta(rwd), "avg_wait": delta(wait),
                "throughput": delta(tp), "congestion": delta(cong),
            },
            "algorithm_result": algo_result,
        }
        self.status.summary = self.summary

    def publish_frame(self, state: StateSnapshot, metrics: Metrics) -> None:
        """Build the per-step render+metric payload (called by the callback)."""
        # Light phases keyed by light index (matches graph light order).
        lights = [
            {"id": s.id, "phase": s.phase, "amber": bool(s.in_all_red),
             "timer": round(s.phase_timer_s, 1)}
            for s in state.intersections
        ]
        vehicles: list[dict] = []
        if self._bridge_ref is not None:
            try:
                raw = self._bridge_ref.parse_vehicles()
                vehicles = [
                    {"id": v["id"], "x": round(v["x"], 2), "y": round(v["y"], 2),
                     "v": round(v["vel"], 2)}
                    for v in raw
                ]
            except Exception:
                vehicles = []

        events: list[dict] = []
        if self._bridge_ref is not None:
            try:
                events = [
                    {"x": round(e["x"], 2), "y": round(e["y"], 2), "type": e["type"]}
                    for e in self._bridge_ref.parse_events()
                ]
            except Exception:
                events = []

        m = asdict(metrics)
        frame = {"type": "frame", "lights": lights,
                 "vehicles": vehicles, "events": events, "metrics": m}
        with self._frame_lock:
            self.latest_frame = frame
        self.status.current_step = metrics.step
        self.status.metrics = m
        point = {
            "step": metrics.step, "reward": m["reward"],
            "avg_wait": m["avg_wait"], "throughput": m["throughput"],
            "congestion": m["congestion"], "num_vehicles": m["num_vehicles"],
        }
        self.history.append(point)
        if len(self.history) > self._max_history:
            self.history.pop(0)

        # Full (sub-sampled) history for the summary and complete charts.
        if metrics.step % self._full_stride == 0:
            self.full_history.append(point)
            if len(self.full_history) > self._full_cap:
                # halve resolution: keep every other point and double the stride
                self.full_history = self.full_history[::2]
                self._full_stride *= 2


class StreamingCallback(BaseCallback):
    """Runs once per env step inside model.learn(). Honours pause/stop/speed
    and publishes the frame the agent just produced."""

    def __init__(self, session: TrainingSession, env: TrafficEnv) -> None:
        super().__init__()
        self._session = session
        self._env = env
        self._last_wall = time.perf_counter()

    def _on_step(self) -> bool:
        sess = self._session

        # 1) abort if stop requested
        if sess._stop_evt.is_set():
            return False

        # 2) block while paused (poll the stop flag so we can still bail out)
        while sess._pause_evt.is_set():
            if sess._stop_evt.is_set():
                return False
            time.sleep(0.05)

        # 3) throttle to target steps/second
        hz = sess._speed_hz
        if hz > 0:
            target_dt = 1.0 / hz
            elapsed = time.perf_counter() - self._last_wall
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
        now = time.perf_counter()
        fps = 1.0 / max(now - self._last_wall, 1e-6)
        self._last_wall = now

        # 4) capture state + metrics from what the agent just saw
        state = self._env._last_state
        if state is not None:
            # reward of the most recent step from the SB3 rollout buffer
            rewards = self.locals.get("rewards")
            reward = float(np.mean(rewards)) if rewards is not None else 0.0
            m = Metrics(
                step=int(self.num_timesteps),
                sim_tick=int(state.sim_tick),
                episode_step=int(state.episode_step),
                reward=round(reward, 4),
                avg_wait=round(float(state.avg_wait_global), 3),
                max_wait=round(float(state.max_wait_global), 3),
                throughput=round(float(state.total_throughput), 3),
                congestion=round(float(state.congestion_spread), 4),
                num_vehicles=int(state.num_vehicles),
                fps=round(fps, 1),
            )
            sess.publish_frame(state, m)
        return True

    def _on_rollout_end(self) -> None:
        # Called by SB3 after each rollout (every n_steps), right before the
        # optimisation epochs. Mark the boundary for the historical charts.
        self._session.rollout_marks.append(int(self.num_timesteps))
