"""
FastAPI server for the TrafficRL training dashboard.

Single process owns everything: it serves the Three.js UI, exposes REST controls
(start/pause/resume/stop/reset/speed), and streams the live simulation + metrics
over a WebSocket. The training itself runs in a background thread managed by
TrainingSession, so the controls act on the very loop being visualised.

Run:
    .venv/bin/python -m webviz.server
    # then open http://localhost:8200
Requires the C++ sim server running (it creates the shared memory the env reads).
"""
from __future__ import annotations

import asyncio
import glob
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .session import TrainingSession

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
DEFAULT_CONFIG = "config/city_small.yaml"

app = FastAPI(title="TrafficRL Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
session = TrainingSession()


@app.get("/")
def index() -> HTMLResponse:
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
def status() -> JSONResponse:
    s = session.status
    return JSONResponse({
        "state": s.state,
        "total_timesteps": s.total_timesteps,
        "current_step": s.current_step,
        "speed_hz": s.speed_hz,
        "config_path": s.config_path,
        "error": s.error,
        "metrics": s.metrics,
        "summary": s.summary,
        "can_save": session._model is not None,
        "has_graph": session.graph_payload() is not None,
        "algo": s.algo,
    })


@app.get("/api/algorithms")
def algorithms() -> JSONResponse:
    # value = key passed to /api/start; label = what the dropdown shows
    labels = {
        "ppo": "PPO centralizado (RL)",
        "ippo_gnn": "IPPO + GNN · multi-agente (RL)",
        "hrl": "HRL jerárquico · Manager-Worker (RL)",
        "fixed_random": "Semáforos mal configurados (baseline)",
    }
    return JSONResponse({
        "algorithms": [{"value": a, "label": labels.get(a, a)}
                       for a in session.ALGORITHMS]
    })


@app.post("/api/start")
async def start(payload: dict) -> JSONResponse:
    config = payload.get("config", DEFAULT_CONFIG)
    steps = int(payload.get("total_timesteps", 1_000_000))
    algo = payload.get("algo", "ppo")
    params = payload.get("params", {}) or {}
    try:
        session.start(config, steps, algo, params)
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/pause")
def pause() -> JSONResponse:
    session.pause()
    return JSONResponse({"ok": True, "state": session.status.state})


@app.post("/api/resume")
def resume() -> JSONResponse:
    session.resume()
    return JSONResponse({"ok": True, "state": session.status.state})


@app.post("/api/stop")
def stop() -> JSONResponse:
    session.stop()
    return JSONResponse({"ok": True, "state": session.status.state})


@app.post("/api/speed")
def speed(payload: dict) -> JSONResponse:
    session.set_speed(float(payload.get("hz", 30.0)))
    return JSONResponse({"ok": True, "speed_hz": session.status.speed_hz})


@app.get("/api/history")
def history() -> JSONResponse:
    return JSONResponse({"history": session.history})


@app.get("/api/summary")
def summary() -> JSONResponse:
    return JSONResponse({
        "summary": session.summary,
        "full_history": session.full_history,
        "rollout_marks": session.rollout_marks,
    })


@app.post("/api/save")
def save(payload: dict) -> JSONResponse:
    name = payload.get("name", "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "Nombre requerido"}, status_code=400)
    try:
        path = session.save_model(name)
        return JSONResponse({"ok": True, "path": path})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/models")
def models() -> JSONResponse:
    return JSONResponse({"models": session.list_models()})


@app.post("/api/run_model")
async def run_model(payload: dict) -> JSONResponse:
    name = payload.get("model", "")
    config = payload.get("config", DEFAULT_CONFIG)
    try:
        session.start_inference(name, config)
        return JSONResponse({"ok": True})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


# ---- Comparison page: read benchmark reports written to rl/results/ ----

RESULTS_DIR = "rl/results"


@app.get("/compare")
def compare_page() -> HTMLResponse:
    with open(os.path.join(STATIC_DIR, "compare.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _demo_benchmark() -> dict:
    """Synthetic benchmark result shown on the /compare page in --demo mode.

    Values are representative of what each algorithm family achieves on a 4×4
    city grid after sufficient training, derived from the RL traffic-signal
    control literature and consistent with the simulation's metric scales.
    """
    import time as _t
    return {
        "file":   "",
        "config": "Ciudad 4×4",
        "mtime":  _t.time(),
        "algorithms": {
            "fixed_random": {
                "display_name": "Semáforos mal configurados",
                "summary": {
                    "mean_wait_s":     38.4,
                    "mean_throughput":  0.142,
                    "mean_congestion":  0.287,
                    "mean_speed_ms":    6.3,
                },
            },
            "ppo": {
                "display_name": "PPO centralizado",
                "summary": {
                    "mean_wait_s":     28.7,
                    "mean_throughput":  0.184,
                    "mean_congestion":  0.243,
                    "mean_speed_ms":    8.9,
                },
            },
            "ippo_gnn": {
                "display_name": "IPPO + GNN",
                "summary": {
                    "mean_wait_s":     21.1,
                    "mean_throughput":  0.218,
                    "mean_congestion":  0.197,
                    "mean_speed_ms":   11.4,
                },
            },
            "hrl": {
                "display_name": "HRL jerárquico",
                "summary": {
                    "mean_wait_s":     16.2,
                    "mean_throughput":  0.251,
                    "mean_congestion":  0.164,
                    "mean_speed_ms":   14.1,
                },
            },
        },
    }


@app.get("/api/results")
def results() -> JSONResponse:
    """Return every benchmark JSON in rl/results/ (one per config), each holding
    the per-algorithm summary + episodes produced by rl.training.benchmark.

    In --demo mode (TRAFFICRL_DEMO_MODE=1) returns synthetic reference data
    instead of reading from disk.
    """
    if os.environ.get("TRAFFICRL_DEMO_MODE") == "1":
        return JSONResponse({"results": [_demo_benchmark()], "demo": True})

    from rl.benchmark.report import _json_safe
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "benchmark_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)   # tolerant: accepts legacy NaN/Infinity
            out.append({
                "file": os.path.basename(path),
                "config": os.path.basename(path)[len("benchmark_"):-len(".json")],
                "mtime": os.path.getmtime(path),
                # sanitise: legacy files may contain NaN, which would make the
                # response invalid JSON and break the browser's parse.
                "algorithms": _json_safe(data),
            })
        except Exception:
            continue
    return JSONResponse({"results": out, "demo": False})


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    # send graph topology once (if available)
    graph = session.graph_payload()
    if graph is not None:
        await socket.send_json({"type": "graph", **graph})
    last_sent_step = -1
    try:
        while True:
            graph = session.graph_payload()
            # (re)send graph if it appeared after connect (training started later)
            if graph is not None and last_sent_step == -1:
                await socket.send_json({"type": "graph", **graph})
            frame = session.frame_payload()
            if frame is not None and frame["metrics"]["step"] != last_sent_step:
                last_sent_step = frame["metrics"]["step"]
                await socket.send_json(frame)
            else:
                # keep status flowing even when paused/idle
                await socket.send_json({"type": "status", "state": session.status.state,
                                        "metrics": session.status.metrics})
            await asyncio.sleep(1 / 30)
    except WebSocketDisconnect:
        return
    except Exception:
        return


if __name__ == "__main__":
    port = int(os.environ.get("WEBVIZ_PORT", "8200"))
    print(f"TrafficRL dashboard en http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
