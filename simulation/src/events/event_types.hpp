#pragma once
#include <cstdint>
#include <cstring>

enum class EventType : uint8_t {
    COLLISION         = 0,  // blocks one lane for duration_s; creates near_incident zone
    ROAD_WORKS        = 1,  // partial lane closure for duration_s
    HEAVY_RAIN        = 2,  // global: reduce speed_limit by rain_factor, increase reaction_time
    MASS_EVENT        = 3,  // directional traffic surge toward a target node
    VEHICLE_BREAKDOWN = 4,  // single blocked vehicle on segment (like COLLISION but smaller)
};

static constexpr uint32_t MAX_ACTIVE_EVENTS = 32;

struct SimEvent {
    EventType type         = EventType::COLLISION;
    bool      active       = false;

    uint32_t  segment_id   = 0;      // affected segment (for per-segment events)
    uint8_t   lane         = 0;      // affected lane (COLLISION / ROAD_WORKS / BREAKDOWN)
    float     position     = 0.0f;   // metres along segment (COLLISION / BREAKDOWN)

    float     duration_s   = 0.0f;   // total planned duration
    float     elapsed_s    = 0.0f;   // time active so far

    // HEAVY_RAIN
    float     rain_factor  = 0.0f;   // [0,1] 0=dry, 1=heavy

    // MASS_EVENT
    uint32_t  target_node  = 0;      // vehicles bias routing toward this node
    float     surge_factor = 0.0f;   // spawn rate multiplier for nearby segments

    bool is_expired() const { return elapsed_s >= duration_s; }
};

struct EventConfig {
    float collision_prob_per_min         = 0.5f;
    float road_works_prob_per_min        = 0.2f;
    float heavy_rain_prob_per_min        = 0.1f;
    float mass_event_prob_per_min        = 0.05f;
    float vehicle_breakdown_prob_per_min = 0.4f;

    float collision_duration_min         = 2.0f;    // minutes
    float collision_duration_max         = 8.0f;
    float road_works_duration_min        = 10.0f;
    float road_works_duration_max        = 30.0f;
    float heavy_rain_duration_min        = 5.0f;
    float heavy_rain_duration_max        = 20.0f;
    float mass_event_duration_min        = 15.0f;
    float mass_event_duration_max        = 45.0f;
    float breakdown_duration_min         = 1.0f;
    float breakdown_duration_max         = 5.0f;

    float rain_factor_min                = 0.3f;
    float rain_factor_max                = 0.9f;
    float mass_surge_min                 = 1.5f;
    float mass_surge_max                 = 4.0f;

    uint64_t seed                        = 42;
};
