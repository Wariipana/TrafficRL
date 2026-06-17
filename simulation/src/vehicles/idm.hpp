#pragma once
#include "core/types.hpp"
#include "core/math_utils.hpp"
#include <cmath>

namespace idm {

static constexpr float MAX_EMERGENCY_DECEL = 9.0f;   // m/s² hard cap
static constexpr float FREE_ROAD_THRESHOLD = 200.0f;  // metres — treat as open road

// Compute IDM acceleration for vehicle v given its immediate leader.
// Pass leader=nullptr when there is no vehicle ahead (open road).
inline float compute_acceleration(const Vehicle& v, const Vehicle* leader) {
    const float v0  = v.personality.desired_speed;
    const float a   = v.personality.max_acceleration;
    const float b   = v.personality.comfort_deceleration;
    const float T   = v.personality.reaction_time;
    const float s0  = v.personality.minimum_gap;

    // Free-road term
    const float free_term = 1.0f - std::pow(v.velocity / (v0 + 1e-4f), 4.0f);

    if (leader == nullptr) {
        // No leader: only free-road deceleration applies
        return a * free_term;
    }

    float gap = leader->position - v.position - VEHICLE_LENGTH;

    // Prevent division by zero / overlap
    if (gap < 0.1f) gap = 0.1f;

    if (gap > FREE_ROAD_THRESHOLD) {
        return a * free_term;
    }

    const float dv     = v.velocity - leader->velocity;
    const float s_star = s0 + v.velocity * T
                       + (v.velocity * dv)
                         / (2.0f * std::sqrt(a * b));

    const float interaction_term = std::pow(math::clamp(s_star / gap, 0.0f, 10.0f), 2.0f);

    float acc = a * (free_term - interaction_term);

    // Hard cap: never decelerate harder than emergency braking
    return math::clamp(acc, -MAX_EMERGENCY_DECEL, a);
}

// Compute acceleration when stopped at a red light (virtual stopped vehicle at stop_distance ahead).
inline float compute_at_stopline(const Vehicle& v, float stop_distance) {
    Vehicle ghost{};
    ghost.position = v.position + stop_distance;
    ghost.velocity = 0.0f;
    return compute_acceleration(v, &ghost);
}

} // namespace idm
