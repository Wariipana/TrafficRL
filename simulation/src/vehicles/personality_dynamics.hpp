#pragma once
#include "core/types.hpp"
#include "core/math_utils.hpp"

// Updates a vehicle's dynamic context and adjusts its effective personality
// parameters in real time based on:
//   - Accumulated wait time (frustration buildup)
//   - Proximity to incidents (alertness / caution)
//   - Route familiarity (confidence on the network)
//
// These produce emergent phenomena without explicit programming:
//   - Accordion effect:  frustrated drivers accelerate hard after a queue clears
//   - Self-sustaining jams: frustrated drivers leave smaller gaps → density wave
//   - Density waves: alternating compression/expansion propagating upstream
namespace personality_dynamics {

// How fast frustration builds up per second of waiting (seconds⁻¹)
static constexpr float FRUSTRATION_BUILD_RATE    = 0.003f;
// How fast frustration decays per second of free movement
static constexpr float FRUSTRATION_DECAY_RATE    = 0.005f;
// Familiarity accumulates slowly over the episode
static constexpr float FAMILIARITY_RATE          = 0.0002f;
// Max speedup from high familiarity (confident drivers push desired_speed up)
static constexpr float FAMILIARITY_SPEED_BOOST   = 0.10f;
// Max reaction_time reduction from high familiarity
static constexpr float FAMILIARITY_RT_REDUCTION  = 0.15f;

// Update context fields and return effective (modified) personality for this step.
// Does NOT permanently alter personality — the base values are preserved.
// Only frustration and familiarity are persistent state.
inline DriverPersonality update(Vehicle& v, float dt, bool near_incident_override = false) {
    v.near_incident = near_incident_override;

    // ---- Frustration dynamics ----
    if (v.velocity < 0.5f) {
        // Stopped or nearly stopped: frustration builds
        float rate = FRUSTRATION_BUILD_RATE * (1.0f + v.personality.frustration_rate * 10.0f);
        v.frustration = math::clamp(v.frustration + rate * dt, 0.0f, 1.0f);
    } else {
        // Moving: frustration decays
        v.frustration = math::clamp(v.frustration - FRUSTRATION_DECAY_RATE * dt, 0.0f, 1.0f);
    }

    // ---- Route familiarity ----
    v.route_familiarity = math::clamp(
        v.route_familiarity + FAMILIARITY_RATE * dt, 0.0f, 1.0f);

    // ---- Build effective personality ----
    DriverPersonality eff = v.personality;

    // Frustrated drivers: smaller gap, harder acceleration (accordion source)
    float frust_gap_factor  = 1.0f - 0.35f * v.frustration;  // up to 35% shorter gap
    float frust_accel_boost = 1.0f + 0.40f * v.frustration;  // up to 40% harder accel
    eff.minimum_gap     = math::clamp(eff.minimum_gap * frust_gap_factor, 0.3f, 5.0f);
    eff.max_acceleration = math::clamp(eff.max_acceleration * frust_accel_boost, 1.0f, 6.0f);

    // Frustrated drivers tolerate red less
    eff.red_light_compliance = math::clamp(
        eff.red_light_compliance - 0.10f * v.frustration, 0.5f, 1.0f);

    // Familiar drivers: slightly faster desired speed, shorter reaction time
    eff.desired_speed  *= 1.0f + FAMILIARITY_SPEED_BOOST  * v.route_familiarity;
    eff.reaction_time  *= 1.0f - FAMILIARITY_RT_REDUCTION * v.route_familiarity;
    eff.reaction_time   = math::clamp(eff.reaction_time, 0.4f, 2.5f);

    // Near incident: cautious drivers slow down
    if (v.near_incident) {
        eff.desired_speed       *= 0.70f;
        eff.reaction_time       *= 1.25f;
        eff.red_light_compliance = 1.0f;  // everyone stops near an accident
    }

    return eff;
}

} // namespace personality_dynamics
