#pragma once
#include <cmath>
#include <algorithm>

namespace math {

template<typename T>
inline T clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(hi, v));
}

template<typename T>
inline T lerp(T a, T b, float t) {
    return a + static_cast<T>((b - a) * t);
}

// LCG pseudo-random (fast, seeded, deterministic)
inline uint64_t lcg_next(uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state;
}

// [0, 1)
inline float lcg_float(uint64_t& state) {
    return static_cast<float>(lcg_next(state) >> 11) / static_cast<float>(1ULL << 53);
}

// Normal distribution via Box-Muller (returns one sample, discards second)
inline float normal(uint64_t& state, float mean, float std_dev) {
    float u1 = lcg_float(state);
    float u2 = lcg_float(state);
    if (u1 < 1e-7f) u1 = 1e-7f;
    float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    return mean + std_dev * z;
}

// Log-normal: e^normal(ln_mean, ln_std)
inline float lognormal(uint64_t& state, float mean, float std_dev) {
    float ln_mean = std::log(mean * mean / std::sqrt(std_dev * std_dev + mean * mean));
    float ln_std  = std::sqrt(std::log(1.0f + (std_dev * std_dev) / (mean * mean)));
    return std::exp(normal(state, ln_mean, ln_std));
}

// Simple distance squared (avoids sqrt when just comparing)
inline float dist2(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1, dy = y2 - y1;
    return dx * dx + dy * dy;
}

} // namespace math
