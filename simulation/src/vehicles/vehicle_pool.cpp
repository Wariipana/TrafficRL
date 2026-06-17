#include "vehicle_pool.hpp"
#include <cstring>
#include <stdexcept>

VehiclePool::VehiclePool(uint64_t seed) {
    reset(seed);
}

void VehiclePool::reset(uint64_t seed) {
    active_count_ = 0;
    next_id_      = 1;
    rng_state_    = seed;
    std::memset(id_to_index_, 0xFF, sizeof(id_to_index_));
    for (auto& v : vehicles_) v = Vehicle{};
}

Vehicle* VehiclePool::spawn(uint32_t segment_id, uint8_t lane, float position, uint32_t dest_node) {
    if (active_count_ >= MAX_VEHICLES) return nullptr;

    uint32_t idx = active_count_++;
    uint32_t id  = next_id_++;
    if (next_id_ >= MAX_VEHICLES) next_id_ = 1;

    Vehicle& v       = vehicles_[idx];
    v                = Vehicle{};
    v.id             = id;
    v.segment_id     = segment_id;
    v.lane           = lane;
    v.position       = position;
    v.velocity       = 0.0f;
    v.dest_node      = dest_node;
    v.active         = true;
    v.wait_time      = 0.0f;
    v.personality    = personality::generate(rng_state_);

    id_to_index_[id % MAX_VEHICLES] = idx;
    return &v;
}

void VehiclePool::despawn(uint32_t vehicle_id) {
    uint32_t key = vehicle_id % MAX_VEHICLES;
    uint32_t idx = id_to_index_[key];
    if (idx >= active_count_) return;

    // Swap with last active vehicle
    uint32_t last_idx = active_count_ - 1;
    if (idx != last_idx) {
        vehicles_[idx] = vehicles_[last_idx];
        id_to_index_[vehicles_[idx].id % MAX_VEHICLES] = idx;
    }

    vehicles_[last_idx].active = false;
    id_to_index_[key] = UINT32_MAX;
    --active_count_;
}
