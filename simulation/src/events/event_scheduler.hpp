#pragma once
#include "event_types.hpp"
#include "city/city_graph.hpp"
#include <span>

// Generates and manages dynamic events deterministically from a seed.
// Domain randomization mechanism: ensures the agent learns robust policies
// rather than memorizing fixed-pattern environments.
class EventScheduler {
public:
    explicit EventScheduler(const CityGraph& city, const EventConfig& cfg);

    void reset(uint64_t seed);
    void update(float dt);

    // Currently active events (read by SimulationLoop to apply effects)
    std::span<const SimEvent> active_events() const;
    uint32_t active_count() const { return active_count_; }

    // Query helpers for SimulationLoop
    bool    has_heavy_rain()   const;
    float   rain_factor()      const;  // 0 if no rain, otherwise the current factor
    bool    segment_blocked(uint32_t seg_id, uint8_t lane) const;
    float   surge_factor_for(uint32_t seg_id) const;
    bool    near_incident(float x, float y, float radius = 50.0f) const;

    // For serialization to shared memory / debug
    uint32_t total_events_fired() const { return total_fired_; }

private:
    const CityGraph& city_;
    EventConfig      cfg_;
    uint64_t         rng_;
    float            sim_time_ = 0.0f;

    SimEvent events_[MAX_ACTIVE_EVENTS];
    uint32_t active_count_ = 0;
    uint32_t total_fired_  = 0;

    // Time until next candidate event of each type (Poisson process via exponential)
    float    next_event_time_[5] = {};

    void try_spawn_events();
    void spawn_event(EventType type);
    float sample_duration(EventType type);
    float next_exp(float rate_per_min);  // exponential inter-arrival time in seconds

    uint32_t random_segment() const;
    uint32_t random_node() const;
};
