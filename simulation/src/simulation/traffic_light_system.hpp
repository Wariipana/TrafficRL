#pragma once
#include "core/types.hpp"
#include <vector>
#include <span>

// Manages all traffic lights in the simulation.
// Supports agent-driven phase control and autonomous fallback timing.
class TrafficLightSystem {
public:
    static constexpr float DEFAULT_MIN_GREEN = 10.0f;  // seconds
    // 120 s matches the Python normalization (phase_timer / 120.0 in flatten_obs).
    // The old 60 s cap only filled half the [0,1] observation range and could
    // override the agent before long unidirectional green phases were useful.
    static constexpr float DEFAULT_MAX_GREEN = 120.0f;

    void init(const std::vector<uint32_t>& light_node_ids);
    void reset();

    // Called each simulation step; advances timers and auto-switches phases if needed.
    void update(float dt);

    // Apply agent action: set desired phase for a light.
    // Returns false if phase change is blocked (still in ALL_RED transition).
    bool apply_action(uint32_t light_id, uint8_t desired_phase);

    // Apply all actions from a packed array (one byte per light, indexed by light_id).
    void apply_actions(const uint8_t* actions, uint32_t count);

    uint32_t count() const { return static_cast<uint32_t>(lights_.size()); }

    const TrafficLight& light(uint32_t id) const { return lights_[id]; }
    std::span<const TrafficLight> lights() const { return lights_; }

    // Returns true if the given direction has green at this light.
    bool is_green(uint32_t light_id, LaneDir dir) const;

private:
    std::vector<TrafficLight> lights_;

    void switch_phase(TrafficLight& tl, TrafficPhase new_phase);
    TrafficPhase next_auto_phase(const TrafficLight& tl) const;
};
