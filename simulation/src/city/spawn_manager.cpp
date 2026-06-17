#include "spawn_manager.hpp"

SpawnManager::SpawnManager(const CityGraph& graph, float global_rate_multiplier)
    : graph_(graph), base_multiplier_(global_rate_multiplier), surge_multiplier_(1.0f) {}

ZoneType SpawnManager::zone_for_segment(uint32_t segment_id) const {
    if (segment_id >= graph_.edge_count()) return ZoneType::RESIDENTIAL;
    const RoadSegment& seg  = graph_.edge(segment_id);
    const Intersection& src = graph_.node(seg.from_node);
    return src.zone;
}

float SpawnManager::spawn_rate(uint32_t segment_id, float /*sim_time_s*/) const {
    float base = BASE_RATE_RESIDENTIAL;
    switch (zone_for_segment(segment_id)) {
        case ZoneType::COMMERCIAL:  base = BASE_RATE_COMMERCIAL;  break;
        case ZoneType::INDUSTRIAL:  base = BASE_RATE_INDUSTRIAL;  break;
        default:                    base = BASE_RATE_RESIDENTIAL; break;
    }
    return base * base_multiplier_ * surge_multiplier_;
}

bool SpawnManager::should_spawn(uint32_t segment_id, float sim_time_s, float dt, uint64_t& rng) const {
    float rate = spawn_rate(segment_id, sim_time_s);
    // Probability of at least one spawn in this dt = 1 - e^(-rate*dt) ≈ rate*dt for small dt
    float prob = 1.0f - std::exp(-rate * dt);
    return math::lcg_float(rng) < prob;
}
