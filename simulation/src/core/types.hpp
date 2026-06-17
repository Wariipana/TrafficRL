#pragma once
#include <cstdint>
#include <cmath>

// ---- Enums ----

enum class BrakingStyle : uint8_t { SMOOTH = 0, ABRUPT = 1 };
enum class ZoneType     : uint8_t { RESIDENTIAL = 0, COMMERCIAL = 1, INDUSTRIAL = 2 };
enum class LaneDir      : uint8_t { NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3 };

// Traffic light phases: green for North-South, green for East-West, all red (transition)
enum class TrafficPhase : uint8_t { NS_GREEN = 0, EW_GREEN = 1, ALL_RED = 2 };

static constexpr uint32_t MAX_LANES        = 8;
static constexpr uint32_t MAX_VEHICLES     = 4096;
static constexpr uint32_t MAX_NODES        = 512;
static constexpr uint32_t MAX_EDGES        = 2048;
static constexpr uint32_t MAX_LIGHTS       = 256;
static constexpr float    VEHICLE_LENGTH   = 4.5f;   // metres
static constexpr float    ALL_RED_DURATION = 2.0f;   // seconds
// Half-width of an intersection in metres. Cars must stop BEFORE this radius so
// they never sit inside the junction box (where crossing traffic would drive
// over them). Matches the visual INT_HALF (1.4 world units * 10 m/wu).
static constexpr float    INTERSECTION_RADIUS_M = 14.0f;

// ---- City configuration ----

struct CityConfig {
    int      grid_width          = 4;
    int      grid_height         = 4;
    float    block_size          = 100.0f;  // metres per block
    float    avenue_probability  = 0.3f;
    float    street_probability  = 0.7f;
    float    residential_ratio   = 0.6f;
    float    commercial_ratio    = 0.3f;
    float    industrial_ratio    = 0.1f;
    float    traffic_light_density = 0.8f;
    bool     access_roads        = true;   // exterior gateway roads (cars enter/leave rolling)
    uint64_t seed                = 42;
};

// ---- Driver personality (sampled at spawn) ----

struct DriverPersonality {
    // IDM longitudinal parameters
    float desired_speed         = 50.0f / 3.6f;  // m/s (~50 km/h)
    float minimum_gap           = 1.5f;           // metres
    float max_acceleration      = 2.5f;           // m/s²
    float comfort_deceleration  = 3.0f;           // m/s²
    float reaction_time         = 0.8f;           // seconds

    // Extended behaviour
    float red_light_compliance  = 0.97f;  // 1.0 = always stops
    float lane_change_propensity = 0.1f;
    float distraction_factor    = 0.05f;  // multiplied onto reaction_time randomly
    float frustration_rate      = 0.02f;  // aggressiveness accumulation rate
    BrakingStyle braking_style  = BrakingStyle::SMOOTH;
};

// ---- Road network ----

struct Intersection {
    uint32_t id        = 0;
    float    x         = 0.0f;
    float    y         = 0.0f;
    ZoneType zone      = ZoneType::RESIDENTIAL;
    bool     has_light = false;
    uint32_t light_id  = UINT32_MAX;  // index into TrafficLightSystem
    uint8_t  num_incoming = 0;
    uint8_t  num_outgoing = 0;
};

struct RoadSegment {
    uint32_t id          = 0;
    uint32_t from_node   = 0;
    uint32_t to_node     = 0;
    float    length      = 0.0f;   // metres
    uint8_t  num_lanes   = 1;
    float    speed_limit = 50.0f / 3.6f;  // m/s
    LaneDir  direction   = LaneDir::NORTH;
};

// ---- Vehicle (runtime state) ----

struct Vehicle {
    uint32_t         id          = 0;
    uint32_t         segment_id  = 0;
    uint32_t         dest_node   = 0;   // pathfinding destination
    float            position    = 0.0f; // metres along segment
    float            velocity    = 0.0f; // m/s
    float            acceleration = 0.0f;
    float            x           = 0.0f; // world position
    float            y           = 0.0f;
    uint8_t          lane        = 0;
    float            wait_time   = 0.0f; // accumulated seconds stopped
    bool             active      = false;
    DriverPersonality personality;

    // Dynamic context — updated each step, drives personality changes
    float frustration        = 0.0f;  // [0,1] accumulated in-episode
    float route_familiarity  = 0.0f;  // [0,1] increases with time on network
    bool  near_incident      = false; // true when within 50m of collision/breakdown

    // Red-light decision, latched per crossing so the compliance roll is stable
    // while the vehicle approaches the same red (avoids per-step flicker).
    bool  red_decided        = false; // a decision has been made for the current red
    bool  run_red            = false; // this vehicle chose to ignore the current red

    // Turn rendering: the physics is 1D (position along a segment), so when a car
    // crosses a junction its centre would teleport from the end of the old lane to
    // the start of the new one — different points at a 90° turn, which reads as a
    // "slide". To render a real arc we remember where the car came from and curve
    // its world position through the junction for the first few metres of the new
    // segment. prev_segment_id == UINT32_MAX means "no turn in progress".
    uint32_t prev_segment_id = UINT32_MAX; // segment occupied before the current junction
    uint8_t  prev_lane       = 0;          // lane on that previous segment
};

// ---- Traffic light ----

struct TrafficLight {
    uint32_t     id           = 0;
    uint32_t     node_id      = 0;
    TrafficPhase phase        = TrafficPhase::NS_GREEN;
    float        phase_timer  = 0.0f;   // time elapsed in current phase
    float        min_green    = 10.0f;  // seconds
    float        max_green    = 60.0f;
    bool         in_all_red   = false;
    float        all_red_timer = 0.0f;
};

// ---- Intersection state (exported to RL agent) ----

struct IntersectionState {
    uint32_t id                              = 0;
    uint8_t  phase                           = 0;
    bool     in_all_red                      = false;  // inter-phase transition (for amber render)
    float    phase_timer                     = 0.0f;
    uint8_t  num_lanes                       = 0;
    float    vehicles_per_lane[MAX_LANES]    = {};
    float    queue_length[MAX_LANES]         = {};
    float    avg_speed[MAX_LANES]            = {};
    float    avg_wait_time                   = 0.0f;
    float    throughput                      = 0.0f;  // vehicles that passed last step
};

// ---- Global metrics ----

struct GlobalMetrics {
    float total_throughput    = 0.0f;
    float avg_wait_global     = 0.0f;
    float max_wait_global     = 0.0f;
    float congestion_spread   = 0.0f;  // fraction of segments over capacity
    uint32_t active_vehicles  = 0;
    uint32_t completed_trips  = 0;
};
