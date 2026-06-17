#include "event_scheduler.hpp"
#include "core/math_utils.hpp"
#include <cstring>
#include <cmath>

EventScheduler::EventScheduler(const CityGraph& city, const EventConfig& cfg)
    : city_(city), cfg_(cfg), rng_(cfg.seed)
{
    reset(cfg.seed);
}

void EventScheduler::reset(uint64_t seed) {
    rng_          = seed;
    sim_time_     = 0.0f;
    active_count_ = 0;
    total_fired_  = 0;
    for (auto& e : events_) e = SimEvent{};

    // Stagger initial event timers so they don't all fire simultaneously
    float rates[5] = {
        cfg_.collision_prob_per_min,
        cfg_.road_works_prob_per_min,
        cfg_.heavy_rain_prob_per_min,
        cfg_.mass_event_prob_per_min,
        cfg_.vehicle_breakdown_prob_per_min,
    };
    for (int i = 0; i < 5; ++i) {
        next_event_time_[i] = sim_time_ + next_exp(rates[i]);
    }
}

void EventScheduler::update(float dt) {
    sim_time_ += dt;

    // Age active events; deactivate expired ones
    uint32_t new_count = 0;
    for (uint32_t i = 0; i < active_count_; ++i) {
        SimEvent& e = events_[i];
        e.elapsed_s += dt;
        if (!e.is_expired()) {
            if (i != new_count) events_[new_count] = e;
            ++new_count;
        }
    }
    active_count_ = new_count;

    try_spawn_events();
}

void EventScheduler::try_spawn_events() {
    if (active_count_ >= MAX_ACTIVE_EVENTS) return;

    float rates[5] = {
        cfg_.collision_prob_per_min,
        cfg_.road_works_prob_per_min,
        cfg_.heavy_rain_prob_per_min,
        cfg_.mass_event_prob_per_min,
        cfg_.vehicle_breakdown_prob_per_min,
    };

    for (int i = 0; i < 5; ++i) {
        if (sim_time_ >= next_event_time_[i]) {
            spawn_event(static_cast<EventType>(i));
            next_event_time_[i] = sim_time_ + next_exp(rates[i]);
        }
    }
}

void EventScheduler::spawn_event(EventType type) {
    if (active_count_ >= MAX_ACTIVE_EVENTS) return;

    // Only one HEAVY_RAIN at a time (global effect)
    if (type == EventType::HEAVY_RAIN) {
        for (uint32_t i = 0; i < active_count_; ++i) {
            if (events_[i].type == EventType::HEAVY_RAIN) return;
        }
    }

    SimEvent e{};
    e.type       = type;
    e.active     = true;
    e.elapsed_s  = 0.0f;
    e.duration_s = sample_duration(type) * 60.0f;  // minutes → seconds

    switch (type) {
        case EventType::COLLISION:
        case EventType::ROAD_WORKS:
        case EventType::VEHICLE_BREAKDOWN: {
            // Incidents can land on any street: they no longer fully block a lane
            // (the engine has no lane-changing, so a hard block would deadlock the
            // queue). Instead vehicles crawl past — see update_vehicles. We still
            // place the incident on a real lane of the chosen segment.
            uint32_t seg = random_segment();
            e.segment_id = seg;
            uint8_t lanes = city_.edge(seg).num_lanes;
            if (lanes < 1) lanes = 1;
            e.lane       = static_cast<uint8_t>(math::lcg_next(rng_) % lanes);
            e.position   = math::lcg_float(rng_) * city_.edge(seg).length;
            break;
        }

        case EventType::HEAVY_RAIN:
            e.rain_factor = cfg_.rain_factor_min
                + math::lcg_float(rng_) * (cfg_.rain_factor_max - cfg_.rain_factor_min);
            break;

        case EventType::MASS_EVENT:
            e.target_node  = random_node();
            e.surge_factor = cfg_.mass_surge_min
                + math::lcg_float(rng_) * (cfg_.mass_surge_max - cfg_.mass_surge_min);
            break;
    }

    events_[active_count_++] = e;
    ++total_fired_;
}

float EventScheduler::sample_duration(EventType type) {
    float lo, hi;
    switch (type) {
        case EventType::COLLISION:         lo = cfg_.collision_duration_min;    hi = cfg_.collision_duration_max;    break;
        case EventType::ROAD_WORKS:        lo = cfg_.road_works_duration_min;   hi = cfg_.road_works_duration_max;   break;
        case EventType::HEAVY_RAIN:        lo = cfg_.heavy_rain_duration_min;   hi = cfg_.heavy_rain_duration_max;   break;
        case EventType::MASS_EVENT:        lo = cfg_.mass_event_duration_min;   hi = cfg_.mass_event_duration_max;   break;
        case EventType::VEHICLE_BREAKDOWN: lo = cfg_.breakdown_duration_min;    hi = cfg_.breakdown_duration_max;    break;
        default:                           lo = 1.0f; hi = 5.0f; break;
    }
    return lo + math::lcg_float(rng_) * (hi - lo);
}

float EventScheduler::next_exp(float rate_per_min) {
    if (rate_per_min <= 0.0f) return 1e9f;
    // Exponential inter-arrival: -ln(U) / λ  (λ in events/second)
    float u = math::lcg_float(rng_);
    if (u < 1e-7f) u = 1e-7f;
    return -std::log(u) / (rate_per_min / 60.0f);
}

uint32_t EventScheduler::random_segment() const {
    if (city_.edge_count() == 0) return 0;
    return static_cast<uint32_t>(math::lcg_next(const_cast<uint64_t&>(rng_)) % city_.edge_count());
}

uint32_t EventScheduler::random_node() const {
    if (city_.node_count() == 0) return 0;
    return static_cast<uint32_t>(math::lcg_next(const_cast<uint64_t&>(rng_)) % city_.node_count());
}

// ---- Query helpers ----

std::span<const SimEvent> EventScheduler::active_events() const {
    return { events_, active_count_ };
}

bool EventScheduler::has_heavy_rain() const {
    for (uint32_t i = 0; i < active_count_; ++i)
        if (events_[i].type == EventType::HEAVY_RAIN) return true;
    return false;
}

float EventScheduler::rain_factor() const {
    for (uint32_t i = 0; i < active_count_; ++i)
        if (events_[i].type == EventType::HEAVY_RAIN)
            return events_[i].rain_factor;
    return 0.0f;
}

bool EventScheduler::segment_blocked(uint32_t seg_id, uint8_t lane) const {
    for (uint32_t i = 0; i < active_count_; ++i) {
        const SimEvent& e = events_[i];
        if (e.segment_id == seg_id && e.lane == lane) {
            if (e.type == EventType::COLLISION ||
                e.type == EventType::ROAD_WORKS ||
                e.type == EventType::VEHICLE_BREAKDOWN) return true;
        }
    }
    return false;
}

float EventScheduler::surge_factor_for(uint32_t seg_id) const {
    for (uint32_t i = 0; i < active_count_; ++i) {
        const SimEvent& e = events_[i];
        if (e.type == EventType::MASS_EVENT) {
            // Surge affects segments heading toward target node
            if (city_.edge(seg_id).to_node == e.target_node) return e.surge_factor;
        }
    }
    return 1.0f;
}

bool EventScheduler::near_incident(float x, float y, float radius) const {
    float r2 = radius * radius;
    for (uint32_t i = 0; i < active_count_; ++i) {
        const SimEvent& e = events_[i];
        if (e.type != EventType::COLLISION && e.type != EventType::VEHICLE_BREAKDOWN) continue;
        // Approximate incident position from segment midpoint
        if (e.segment_id >= city_.edge_count()) continue;
        const RoadSegment& seg = city_.edge(e.segment_id);
        const Intersection& fn = city_.node(seg.from_node);
        const Intersection& tn = city_.node(seg.to_node);
        float t   = e.position / (seg.length + 1e-4f);
        float ix  = fn.x + (tn.x - fn.x) * t;
        float iy  = fn.y + (tn.y - fn.y) * t;
        float dx  = x - ix, dy = y - iy;
        if (dx * dx + dy * dy <= r2) return true;
    }
    return false;
}
