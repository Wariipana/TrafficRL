#pragma once
#include "core/types.hpp"
#include "personality_generator.hpp"
#include <cstdint>
#include <cstring>

// Flat array vehicle pool with O(1) spawn/despawn (swap-and-pop).
// Vehicles are stored contiguously for cache-friendly IDM iteration.
class VehiclePool {
public:
    explicit VehiclePool(uint64_t seed = 42);

    // Spawn a vehicle on segment at position. Returns pointer to the new vehicle or nullptr if full.
    Vehicle* spawn(uint32_t segment_id, uint8_t lane, float position, uint32_t dest_node);

    // Mark vehicle as inactive and swap with last active vehicle.
    void despawn(uint32_t vehicle_id);

    // Reset pool entirely for new episode.
    void reset(uint64_t seed);

    // Cache-friendly iteration over active vehicles only.
    Vehicle* data()       { return vehicles_; }
    uint32_t active_count() const { return active_count_; }

    Vehicle*       begin()       { return vehicles_; }
    Vehicle*       end()         { return vehicles_ + active_count_; }
    const Vehicle* begin() const { return vehicles_; }
    const Vehicle* end()   const { return vehicles_ + active_count_; }

private:
    Vehicle  vehicles_[MAX_VEHICLES];
    uint32_t id_to_index_[MAX_VEHICLES];  // vehicle_id → array index
    uint32_t active_count_ = 0;
    uint32_t next_id_      = 1;  // 0 is reserved as null/invalid
    uint64_t rng_state_    = 42;
};
