#pragma once
#include "core/types.hpp"
#include <atomic>
#include <cstdint>

// Binary contract between C++ server and Python client.
// All structs must be identical on both sides (ctypes mirrors these).

static constexpr uint32_t SHM_STATE_MAGIC = 0x54524C73;  // 'TRLs'
static constexpr uint32_t SHM_CMD_MAGIC   = 0x54524C63;  // 'TRLc'
static constexpr uint32_t SHM_GRAPH_MAGIC = 0x54524C67;  // 'TRLg'
static constexpr uint32_t SHM_VERSION     = 1;

static constexpr size_t SHM_STATE_SIZE = 4 * 1024 * 1024;  // 4 MB
static constexpr size_t SHM_CMD_SIZE   = 4 * 1024;          // 4 KB
static constexpr size_t SHM_GRAPH_SIZE = 1 * 1024 * 1024;  // 1 MB

#pragma pack(push, 1)

// ---- IntersectionState as stored in shm ----
struct ShmIntersectionState {
    uint32_t id;
    uint8_t  phase;
    uint8_t  in_all_red;                  // 1 during inter-phase transition (amber render)
    uint8_t  num_lanes;
    uint8_t  reserved0;                   // keep phase_timer_ms 2-byte aligned within the packed struct
    uint16_t phase_timer_ms;              // phase_timer in milliseconds (saves space)
    uint16_t reserved1;                   // pad so the float array stays 4-aligned
    float    vehicles_per_lane[MAX_LANES];
    float    queue_length[MAX_LANES];
    float    avg_speed[MAX_LANES];
    float    avg_wait_time;
    float    throughput;
};

static_assert(sizeof(ShmIntersectionState) == 4 + 1 + 1 + 1 + 1 + 2 + 2 + 3*8*4 + 4 + 4, "Layout check");

// ---- Per-vehicle state as stored in shm (for visualization) ----
// Written after the intersection array. Lets the visualizer render real,
// individually-moving vehicles instead of synthesising them from lane counts.
struct ShmVehicle {
    uint32_t id;       // stable id — same vehicle keeps the same id across steps
    float    x;        // world position (metres)
    float    y;
    float    velocity; // m/s
    uint8_t  lane;
    uint8_t  reserved[3];
};

static_assert(sizeof(ShmVehicle) == 4 + 4 + 4 + 4 + 1 + 3, "Vehicle layout check");

// ---- Active incident (collision / road works / breakdown) for the viz ----
// Exported so the dashboard can drop an icon on the spot — otherwise a stalled
// queue behind an incident looks like a bug. Lives after the vehicle array.
struct ShmEvent {
    float    x;        // world position (metres)
    float    y;
    uint8_t  type;     // 0=collision, 1=road_works, 2=breakdown
    uint8_t  reserved[3];
};

static_assert(sizeof(ShmEvent) == 4 + 4 + 1 + 3, "Event layout check");

static constexpr uint32_t MAX_EVENTS_EXPORT = 64;

// ---- State segment header ----
struct ShmStateHeader {
    uint32_t magic;                        // SHM_STATE_MAGIC
    uint32_t version;
    uint64_t sim_tick;
    uint32_t num_intersections;
    uint32_t num_vehicles;
    uint32_t sim_time_ms;
    uint32_t episode_step;
    uint32_t flags;                        // bit0=terminated, bit1=truncated
    uint32_t num_vehicles_export;          // vehicles written to the shm vehicle array
    uint32_t num_events_export;            // active incidents written after the vehicle array
    // Atomic spinlock: 0=free, 1=locked
    std::atomic<uint32_t> write_lock;
    // Monotonic counter — Python spins until this changes
    uint32_t state_generation;
    float    total_throughput;
    float    avg_wait_global;
    float    max_wait_global;
    float    congestion_spread;
    uint8_t  reserved1[8];
    // Followed by: ShmIntersectionState[num_intersections]
};

// ---- Command segment header ----
struct ShmCmdHeader {
    uint32_t magic;                        // SHM_CMD_MAGIC
    uint32_t version;
    std::atomic<uint32_t> write_lock;
    uint32_t cmd_generation;               // incremented by Python on each write
    uint32_t num_actions;
    std::atomic<uint32_t> step_ready;      // 1=Python wrote action, C++ sets to 0 after processing
    uint32_t reset_flag;                   // 1=Python requests episode reset
    uint64_t reset_seed;
    uint8_t  phase_actions[MAX_LIGHTS];    // one uint8 per traffic light
    uint8_t  reserved[8];
};

#pragma pack(pop)

// Pointer arithmetic helpers (inlined, no overhead)
inline ShmIntersectionState* shm_intersections(ShmStateHeader* hdr) {
    return reinterpret_cast<ShmIntersectionState*>(
        reinterpret_cast<uint8_t*>(hdr) + sizeof(ShmStateHeader));
}

// Vehicle array lives immediately after the intersection array.
inline ShmVehicle* shm_vehicles(ShmStateHeader* hdr) {
    return reinterpret_cast<ShmVehicle*>(
        reinterpret_cast<uint8_t*>(shm_intersections(hdr)) +
        static_cast<size_t>(hdr->num_intersections) * sizeof(ShmIntersectionState));
}

inline const ShmVehicle* shm_vehicles(const ShmStateHeader* hdr) {
    return reinterpret_cast<const ShmVehicle*>(
        reinterpret_cast<const uint8_t*>(hdr) + sizeof(ShmStateHeader) +
        static_cast<size_t>(hdr->num_intersections) * sizeof(ShmIntersectionState));
}

// Event array lives immediately after the vehicle array.
inline ShmEvent* shm_events(ShmStateHeader* hdr) {
    return reinterpret_cast<ShmEvent*>(
        reinterpret_cast<uint8_t*>(shm_vehicles(hdr)) +
        static_cast<size_t>(hdr->num_vehicles_export) * sizeof(ShmVehicle));
}
