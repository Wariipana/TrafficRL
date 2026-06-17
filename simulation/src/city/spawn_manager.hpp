#pragma once
#include "city_graph.hpp"
#include "core/math_utils.hpp"

// Determines per-segment vehicle spawn rates based on zone type and simulation time.
// Phase 1: constant rates, no diurnal pattern yet.
class SpawnManager {
public:
    // Per-segment base rates (vehicles/second). Tuned so the city does not
    // saturate: with ~12 entry gateways and ~5-hop routes, higher rates flooded
    // the grid faster than cars could drain (spawn >> despawn).
    static constexpr float BASE_RATE_RESIDENTIAL = 0.08f;
    static constexpr float BASE_RATE_COMMERCIAL  = 0.20f;
    static constexpr float BASE_RATE_INDUSTRIAL  = 0.10f;

    explicit SpawnManager(const CityGraph& graph, float global_rate_multiplier = 1.0f);

    // Returns spawn rate (vehicles/s) for a segment given current sim time.
    float spawn_rate(uint32_t segment_id, float sim_time_s) const;

    // Probabilistic tick: returns true if a vehicle should spawn this dt.
    bool should_spawn(uint32_t segment_id, float sim_time_s, float dt, uint64_t& rng) const;

    // A transient event surge multiplies ON TOP of the fixed density multiplier
    // (set once at construction). Resetting the surge does NOT wipe the density.
    void set_rate_multiplier(float m) { surge_multiplier_ = m; }

private:
    const CityGraph& graph_;
    float base_multiplier_;     // fixed traffic density (from --spawn-rate)
    float surge_multiplier_;    // transient event surge (resets to 1.0)

    ZoneType zone_for_segment(uint32_t segment_id) const;
};
