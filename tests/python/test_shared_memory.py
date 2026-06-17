"""
Shared memory integration tests.
These tests require a running trafficrl_server process.
Skip automatically if the server is not available.
"""
from __future__ import annotations
import os
import subprocess
import time
import signal
import pytest
import numpy as np

SERVER_BINARY = os.path.join(
    os.path.dirname(__file__), "../../simulation/build/trafficrl_server"
)


def _server_available() -> bool:
    return os.path.isfile(SERVER_BINARY)


@pytest.fixture(scope="module")
def running_server():
    if not _server_available():
        pytest.skip("trafficrl_server binary not found — run cmake build first")
    proc = subprocess.Popen(
        [SERVER_BINARY, "--width", "4", "--height", "4", "--seed", "42", "--steps", "10000"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(1.0)  # let server initialize shm
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.mark.skipif(not _server_available(), reason="Server binary not built")
def test_bridge_connect(running_server):
    from rl.env.bridge_client import BridgeClient
    client = BridgeClient("trafficrl")
    graph = client.connect()
    assert graph.num_lights > 0
    assert graph.num_nodes == 28   # 4x4 grid (16) + 12 exterior gateways
    assert graph.num_edges == 72   # 48 grid + 24 access-road edges
    client.disconnect()


@pytest.mark.skipif(not _server_available(), reason="Server binary not built")
def test_bridge_roundtrip(running_server):
    from rl.env.bridge_client import BridgeClient
    client = BridgeClient("trafficrl")
    graph = client.connect()

    n = graph.num_lights
    actions = np.zeros(n, dtype=np.uint8)
    client.send_action(actions)

    t0 = time.perf_counter()
    state = client.wait_for_state(timeout_ms=500)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert state.sim_tick >= 0
    assert len(state.intersections) == n
    assert elapsed_ms < 500, f"Roundtrip took {elapsed_ms:.1f}ms, expected < 500ms"
    client.disconnect()


@pytest.mark.skipif(not _server_available(), reason="Server binary not built")
def test_reset_protocol(running_server):
    from rl.env.bridge_client import BridgeClient
    client = BridgeClient("trafficrl")
    client.connect()

    graph1 = client.reset_episode(seed=42)
    state1 = client.wait_for_state()

    graph2 = client.reset_episode(seed=42)
    state2 = client.wait_for_state()

    assert graph1.num_lights == graph2.num_lights, "Same seed must give same graph"
    assert state1.num_intersections == state2.num_intersections
    client.disconnect()
