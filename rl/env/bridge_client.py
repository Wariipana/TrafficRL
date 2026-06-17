from __future__ import annotations
import ctypes
import mmap
import os
import struct
import time
import numpy as np
from .data_types import (
    GraphData, NodeRecord, EdgeRecord,
    IntersectionSnapshot, StateSnapshot,
    MAX_LANES, MAX_LIGHTS, MAX_VEHICLES_EXPORT,
)

# ---- ctypes layout mirrors shared_memory_layout.hpp ----

class _AtomicU32(ctypes.Structure):
    _fields_ = [("value", ctypes.c_uint32)]


class _ShmStateHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic",              ctypes.c_uint32),
        ("version",            ctypes.c_uint32),
        ("sim_tick",           ctypes.c_uint64),
        ("num_intersections",  ctypes.c_uint32),
        ("num_vehicles",       ctypes.c_uint32),
        ("sim_time_ms",        ctypes.c_uint32),
        ("episode_step",       ctypes.c_uint32),
        ("flags",              ctypes.c_uint32),
        ("num_vehicles_export", ctypes.c_uint32),  # vehicles written to the shm vehicle array
        ("num_events_export",  ctypes.c_uint32),   # active incidents written after the vehicles
        ("write_lock",         ctypes.c_uint32),   # atomic
        ("state_generation",   ctypes.c_uint32),
        ("total_throughput",   ctypes.c_float),
        ("avg_wait_global",    ctypes.c_float),
        ("max_wait_global",    ctypes.c_float),
        ("congestion_spread",  ctypes.c_float),
        ("reserved1",          ctypes.c_uint8 * 8),
    ]


class _ShmIntersectionState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("id",               ctypes.c_uint32),
        ("phase",            ctypes.c_uint8),
        ("in_all_red",       ctypes.c_uint8),    # 1 during inter-phase transition
        ("num_lanes",        ctypes.c_uint8),
        ("reserved0",        ctypes.c_uint8),
        ("phase_timer_ms",   ctypes.c_uint16),
        ("reserved1",        ctypes.c_uint16),
        ("vehicles_per_lane", ctypes.c_float * MAX_LANES),
        ("queue_length",     ctypes.c_float * MAX_LANES),
        ("avg_speed",        ctypes.c_float * MAX_LANES),
        ("avg_wait_time",    ctypes.c_float),
        ("throughput",       ctypes.c_float),
    ]


class _ShmVehicle(ctypes.Structure):
    # Mirrors ShmVehicle in shared_memory_layout.hpp — per-vehicle render state.
    _pack_ = 1
    _fields_ = [
        ("id",       ctypes.c_uint32),
        ("x",        ctypes.c_float),
        ("y",        ctypes.c_float),
        ("velocity", ctypes.c_float),
        ("lane",     ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class _ShmEvent(ctypes.Structure):
    # Mirrors ShmEvent in shared_memory_layout.hpp — one active incident.
    _pack_ = 1
    _fields_ = [
        ("x",        ctypes.c_float),
        ("y",        ctypes.c_float),
        ("type",     ctypes.c_uint8),   # 0=collision, 1=road_works, 2=breakdown
        ("reserved", ctypes.c_uint8 * 3),
    ]


MAX_EVENTS_EXPORT = 64


class _ShmCmdHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic",          ctypes.c_uint32),
        ("version",        ctypes.c_uint32),
        ("write_lock",     ctypes.c_uint32),    # atomic
        ("cmd_generation", ctypes.c_uint32),
        ("num_actions",    ctypes.c_uint32),
        ("step_ready",     ctypes.c_uint32),    # atomic
        ("reset_flag",     ctypes.c_uint32),
        ("reset_seed",     ctypes.c_uint64),
        ("phase_actions",  ctypes.c_uint8 * MAX_LIGHTS),
        ("reserved",       ctypes.c_uint8 * 8),
    ]


# Graph segment layout
_GRAPH_HEADER_FMT = "<IIIII12x"  # magic, version, num_nodes, num_edges, num_lights + 12 reserved
_GRAPH_HEADER_SIZE = struct.calcsize(_GRAPH_HEADER_FMT)

_NODE_FMT  = "<IffBBIBx"     # id, x, y, zone, has_light, light_id, num_outgoing, pad
_NODE_SIZE  = struct.calcsize(_NODE_FMT)

_EDGE_FMT  = "<IIIfBfBxx"    # id, from, to, length, num_lanes, speed_limit, direction, pad
_EDGE_SIZE  = struct.calcsize(_EDGE_FMT)


class BridgeClient:
    SHM_STATE_MAGIC = 0x54524C73
    SHM_CMD_MAGIC   = 0x54524C63
    SHM_GRAPH_MAGIC = 0x54524C67

    SHM_STATE_SIZE = 4 * 1024 * 1024
    SHM_CMD_SIZE   = 4 * 1024
    SHM_GRAPH_SIZE = 1 * 1024 * 1024

    def __init__(self, shm_prefix: str = "trafficrl", timeout_ms: int = 5000):
        self._prefix      = shm_prefix
        self._timeout_ms  = timeout_ms
        self._state_mm: mmap.mmap | None = None
        self._cmd_mm:   mmap.mmap | None = None
        self._graph_mm: mmap.mmap | None = None
        self._state_hdr: _ShmStateHeader | None = None
        self._cmd_hdr:   _ShmCmdHeader | None   = None
        self._last_generation = -1

    def connect(self) -> GraphData:
        """Open shm segments created by the C++ server and return graph topology."""
        self._state_mm = self._open_shm(self._state_name(), self.SHM_STATE_SIZE)
        self._cmd_mm   = self._open_shm(self._cmd_name(),   self.SHM_CMD_SIZE)
        self._graph_mm = self._open_shm(self._graph_name(), self.SHM_GRAPH_SIZE)

        self._state_hdr = _ShmStateHeader.from_buffer(self._state_mm)
        self._cmd_hdr   = _ShmCmdHeader.from_buffer(self._cmd_mm)

        return self._parse_graph()

    def send_action(self, phases: np.ndarray) -> None:
        """Write phase actions and signal the C++ server to step."""
        hdr = self._cmd_hdr
        n   = min(len(phases), MAX_LIGHTS)

        # Acquire spinlock
        self._spinlock_acquire_cmd()
        try:
            for i in range(n):
                hdr.phase_actions[i] = int(phases[i]) & 0xFF
            hdr.num_actions    = n
            hdr.cmd_generation += 1
        finally:
            self._spinlock_release_cmd()

        # Signal C++ to process
        ctypes.c_uint32.from_address(
            ctypes.addressof(hdr) + _ShmCmdHeader.step_ready.offset
        ).value = 1

    def wait_for_state(self, timeout_ms: int | None = None) -> StateSnapshot:
        """Spin until state_generation changes, then parse and return state."""
        timeout_ms = timeout_ms or self._timeout_ms
        deadline   = time.perf_counter() + timeout_ms / 1000.0
        hdr        = self._state_hdr

        while time.perf_counter() < deadline:
            gen = hdr.state_generation
            if gen != self._last_generation:
                self._last_generation = gen
                return self._parse_state()
            # Yield to avoid burning 100% CPU on this core
            time.sleep(0)

        # Timeout: return last known state rather than raising
        return self._parse_state()

    def reset_episode(self, seed: int) -> GraphData:
        """Request the C++ server to reset and wait for the graph to be refreshed."""
        hdr = self._cmd_hdr
        hdr.reset_seed = seed & 0xFFFFFFFFFFFFFFFF
        hdr.reset_flag = 1
        # Wait for server to clear reset_flag and write new state
        deadline = time.perf_counter() + self._timeout_ms / 1000.0
        while hdr.reset_flag != 0 and time.perf_counter() < deadline:
            time.sleep(0.001)
        self._last_generation = -1
        self.wait_for_state()
        return self._parse_graph()

    def disconnect(self) -> None:
        # Release ctypes views before closing mmap (they hold exported pointers)
        self._state_hdr = None
        self._cmd_hdr   = None
        for mm in (self._state_mm, self._cmd_mm, self._graph_mm):
            if mm is not None:
                mm.close()
        self._state_mm = self._cmd_mm = self._graph_mm = None

    # ---- Internal helpers ----

    def _state_name(self) -> str: return f"/dev/shm/{self._prefix}_state"
    def _cmd_name(self)   -> str: return f"/dev/shm/{self._prefix}_cmd"
    def _graph_name(self) -> str: return f"/dev/shm/{self._prefix}_graph"

    def _open_shm(self, path: str, size: int) -> mmap.mmap:
        fd = os.open(path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        return mm

    def _spinlock_acquire_cmd(self) -> None:
        lock_addr = ctypes.addressof(self._cmd_hdr) + _ShmCmdHeader.write_lock.offset
        lock      = ctypes.c_uint32.from_address(lock_addr)
        deadline  = time.perf_counter() + 0.1
        while time.perf_counter() < deadline:
            if lock.value == 0:
                lock.value = 1
                return
        # Timeout: proceed anyway (better than deadlock)

    def _spinlock_release_cmd(self) -> None:
        lock_addr = ctypes.addressof(self._cmd_hdr) + _ShmCmdHeader.write_lock.offset
        ctypes.c_uint32.from_address(lock_addr).value = 0

    def _parse_state(self) -> StateSnapshot:
        hdr  = self._state_hdr
        mm   = self._state_mm
        n    = hdr.num_intersections

        hdr_size   = ctypes.sizeof(_ShmStateHeader)
        int_size   = ctypes.sizeof(_ShmIntersectionState)
        int_offset = hdr_size

        intersections = []
        for i in range(min(n, MAX_LIGHTS)):
            offset = int_offset + i * int_size
            mm.seek(offset)
            raw_bytes = mm.read(int_size)
            raw = _ShmIntersectionState.from_buffer_copy(raw_bytes)
            intersections.append(IntersectionSnapshot(
                id            = raw.id,
                phase         = raw.phase,
                in_all_red    = bool(raw.in_all_red),
                num_lanes     = raw.num_lanes,
                phase_timer_s = raw.phase_timer_ms / 1000.0,
                vehicles_per_lane = np.array(list(raw.vehicles_per_lane), dtype=np.float32),
                queue_length      = np.array(list(raw.queue_length),      dtype=np.float32),
                avg_speed         = np.array(list(raw.avg_speed),         dtype=np.float32),
                avg_wait_time = raw.avg_wait_time,
                throughput    = raw.throughput,
            ))

        return StateSnapshot(
            sim_tick         = hdr.sim_tick,
            num_intersections = hdr.num_intersections,
            num_vehicles     = hdr.num_vehicles,
            sim_time_s       = hdr.sim_time_ms / 1000.0,
            episode_step     = hdr.episode_step,
            terminated       = bool(hdr.flags & 0x1),
            truncated        = bool(hdr.flags & 0x2),
            intersections    = intersections,
            total_throughput = hdr.total_throughput,
            avg_wait_global  = hdr.avg_wait_global,
            max_wait_global  = hdr.max_wait_global,
            congestion_spread = hdr.congestion_spread,
        )

    def parse_vehicles(self) -> list[dict]:
        """Read per-vehicle render state (id, x, y, velocity, lane) for the web
        visualizer. The vehicle array lives immediately after the intersection
        array in the state segment; its length is num_vehicles_export."""
        hdr = self._state_hdr
        mm  = self._state_mm

        hdr_size  = ctypes.sizeof(_ShmStateHeader)
        int_size  = ctypes.sizeof(_ShmIntersectionState)
        veh_size  = ctypes.sizeof(_ShmVehicle)
        veh_base  = hdr_size + int(hdr.num_intersections) * int_size
        count     = min(int(hdr.num_vehicles_export), MAX_VEHICLES_EXPORT)

        out: list[dict] = []
        for i in range(count):
            mm.seek(veh_base + i * veh_size)
            raw = _ShmVehicle.from_buffer_copy(mm.read(veh_size))
            out.append({
                "id":  raw.id,
                "x":   raw.x,
                "y":   raw.y,
                "vel": raw.velocity,
                "lane": raw.lane,
            })
        return out

    def parse_events(self) -> list[dict]:
        """Read active incidents (x, y, type) for the visualizer. The event array
        lives immediately after the vehicle array in the state segment."""
        hdr = self._state_hdr
        mm  = self._state_mm

        hdr_size = ctypes.sizeof(_ShmStateHeader)
        int_size = ctypes.sizeof(_ShmIntersectionState)
        veh_size = ctypes.sizeof(_ShmVehicle)
        evt_size = ctypes.sizeof(_ShmEvent)
        veh_count = min(int(hdr.num_vehicles_export), MAX_VEHICLES_EXPORT)
        evt_base  = hdr_size + int(hdr.num_intersections) * int_size \
                    + veh_count * veh_size
        count = min(int(hdr.num_events_export), MAX_EVENTS_EXPORT)

        kinds = {0: "collision", 1: "road_works", 2: "breakdown"}
        out: list[dict] = []
        for i in range(count):
            mm.seek(evt_base + i * evt_size)
            raw = _ShmEvent.from_buffer_copy(mm.read(evt_size))
            out.append({"x": raw.x, "y": raw.y,
                        "type": kinds.get(raw.type, "incident")})
        return out

    def _parse_graph(self) -> GraphData:
        mm = self._graph_mm
        mm.seek(0)
        hdr_bytes = mm.read(_GRAPH_HEADER_SIZE)
        magic, version, num_nodes, num_edges, num_lights = struct.unpack(_GRAPH_HEADER_FMT, hdr_bytes)

        nodes = []
        for _ in range(num_nodes):
            raw = mm.read(_NODE_SIZE)
            node_id, x, y, zone, has_light, light_id, num_out = struct.unpack(_NODE_FMT, raw)
            nodes.append(NodeRecord(
                id=node_id, x=x, y=y, zone=zone,
                has_light=bool(has_light), light_id=light_id,
                num_outgoing=num_out,
            ))

        edges = []
        for _ in range(num_edges):
            raw = mm.read(_EDGE_SIZE)
            eid, from_n, to_n, length, nlanes, speed, direction = struct.unpack(_EDGE_FMT, raw)
            edges.append(EdgeRecord(
                id=eid, from_node=from_n, to_node=to_n,
                length=length, num_lanes=nlanes,
                speed_limit=speed, direction=direction,
            ))

        light_node_ids_bytes = mm.read(num_lights * 4)
        light_node_ids = list(struct.unpack(f"<{num_lights}I", light_node_ids_bytes))

        return GraphData(
            nodes=nodes,
            edges=edges,
            light_node_ids=light_node_ids,
            num_lights=num_lights,
        )
