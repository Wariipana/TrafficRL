#pragma once
#include "core/types.hpp"
#include "core/spatial_hash.hpp"
#include "city/city_graph.hpp"
#include "city/spawn_manager.hpp"
#include "vehicles/vehicle_pool.hpp"
#include "simulation/traffic_light_system.hpp"
#include "events/event_scheduler.hpp"
#include <vector>
#include <cstdint>

// Per-step state snapshot written to the shared memory bridge.
struct SimStateBuffer {
    uint64_t            sim_tick;
    uint32_t            num_intersections;
    uint32_t            num_vehicles;
    uint32_t            sim_time_ms;
    uint32_t            episode_step;
    uint32_t            flags;             // bit0=terminated, bit1=truncated
    IntersectionState   intersections[MAX_LIGHTS];
    GlobalMetrics       metrics;
};

// Central simulation orchestrator.
// Call reset() to start a new episode, then step() repeatedly.
class SimulationLoop {
public:
    SimulationLoop(const CityConfig&  city_cfg,
                   const EventConfig& event_cfg           = EventConfig{},
                   int                episode_length_steps = 2000,
                   float              dt                   = 0.1f,
                   float              spawn_rate_mult      = 1.0f);

    // Reset the episode. warmup_steps > 0 advances the simulation that many steps
    // with fixed-time lights right after clearing, so the city is already at a
    // realistic traffic level when the episode (and any RL measurement) begins —
    // otherwise every episode starts from an empty grid and the "before" metrics
    // look artificially good, hiding the agent's improvement.
    void reset(uint64_t seed, int warmup_steps = 0);
    void step();

    // Apply phase actions from agent before calling step().
    void apply_light_actions(const uint8_t* phases, uint32_t count);

    // Fill a SimStateBuffer for the shared memory bridge.
    void snapshot(SimStateBuffer& buf) const;

    uint64_t tick()          const { return tick_; }
    float    sim_time()      const { return sim_time_; }
    bool     is_terminated() const { return terminated_; }
    bool     is_truncated()  const { return truncated_; }

    const CityGraph&          city()    const { return city_; }
    const VehiclePool&        pool()    const { return pool_; }
    const TrafficLightSystem& lights()  const { return lights_; }
    const EventScheduler&     events()  const { return events_; }

private:
    CityGraph          city_;
    VehiclePool        pool_;
    SpawnManager       spawner_;
    TrafficLightSystem lights_;
    EventScheduler     events_;
    SpatialHash<Vehicle> spatial_;

    float    dt_;
    int      episode_length_;
    int      step_count_   = 0;
    uint64_t tick_         = 0;
    float    sim_time_     = 0.0f;
    bool     terminated_   = false;
    bool     truncated_    = false;
    uint64_t rng_          = 42;
    EventConfig event_cfg_;

    // Per-step pipeline
    void update_vehicles();
    void update_traffic_lights();
    void check_spawn_despawn();
    void rebuild_spatial_hash();

    // Compute per-intersection stats for the snapshot
    void compute_intersection_state(uint32_t light_idx, IntersectionState& out) const;
    void compute_global_metrics(GlobalMetrics& out) const;

    // Vehicle position in world coords from segment + position
    void update_world_position(Vehicle& v) const;
    // Lane-offset world point of a segment at parameter t∈[0,1] along its length.
    void segment_point(uint32_t segment_id, uint8_t lane, float t,
                       float& out_x, float& out_y) const;
    // Unit heading vector of a segment (from_node → to_node).
    void lane_heading(uint32_t segment_id, float& hx, float& hy) const;
};
