#pragma once
#include "core/types.hpp"
#include "core/math_utils.hpp"

namespace personality {

// Generate a correlated driver personality from a seeded LCG state.
// Correlations:
//   high aggressiveness → higher acceleration, shorter gap, shorter reaction time
//   high distraction → longer reaction time
inline DriverPersonality generate(uint64_t& rng) {
    DriverPersonality p;

    // Desired speed: normal(50 km/h, 15 km/h) clamped to [20, 130] km/h
    float speed_kmh = math::normal(rng, 50.0f, 15.0f);
    speed_kmh = math::clamp(speed_kmh, 20.0f, 130.0f);
    p.desired_speed = speed_kmh / 3.6f;

    // Aggressiveness factor [0, 1] — beta-like via two uniforms
    float u1 = math::lcg_float(rng);
    float u2 = math::lcg_float(rng);
    float aggressiveness = (u1 + u2) * 0.5f;  // approx triangular [0,1], peak at 0.5

    // min gap: lognormal, mean=1.5m, std=0.5m, modulated by aggressiveness
    p.minimum_gap = math::lognormal(rng, 1.5f, 0.5f) * (1.0f - 0.4f * aggressiveness);
    p.minimum_gap = math::clamp(p.minimum_gap, 0.5f, 5.0f);

    // max_acceleration: base 2.5, aggressive drivers up to 4.0
    p.max_acceleration = math::normal(rng, 2.5f, 0.4f) * (1.0f + 0.6f * aggressiveness);
    p.max_acceleration = math::clamp(p.max_acceleration, 1.0f, 5.0f);

    // comfort_deceleration correlated with acceleration
    p.comfort_deceleration = p.max_acceleration * math::normal(rng, 1.2f, 0.15f);
    p.comfort_deceleration = math::clamp(p.comfort_deceleration, 1.0f, 6.0f);

    // reaction time: base 0.8s, distracted drivers up to 2.0s
    float distraction = math::lcg_float(rng);
    distraction = distraction * distraction;  // skew toward low distraction
    p.distraction_factor = distraction;
    p.reaction_time = math::lognormal(rng, 0.8f, 0.2f) * (1.0f + 1.2f * distraction);
    p.reaction_time = math::clamp(p.reaction_time, 0.5f, 2.5f);

    // Red light compliance: most drivers comply; aggressive ones sometimes don't
    p.red_light_compliance = 1.0f - aggressiveness * 0.15f * math::lcg_float(rng);
    p.red_light_compliance = math::clamp(p.red_light_compliance, 0.7f, 1.0f);

    p.lane_change_propensity = 0.05f + aggressiveness * 0.25f;
    p.frustration_rate       = 0.01f + aggressiveness * 0.05f;

    float u_style = math::lcg_float(rng);
    p.braking_style = (u_style < 0.3f + aggressiveness * 0.4f)
                      ? BrakingStyle::ABRUPT
                      : BrakingStyle::SMOOTH;

    return p;
}

} // namespace personality
