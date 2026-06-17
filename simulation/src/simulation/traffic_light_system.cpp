#include "traffic_light_system.hpp"
#include <stdexcept>

void TrafficLightSystem::init(const std::vector<uint32_t>& light_node_ids) {
    lights_.clear();
    lights_.reserve(light_node_ids.size());
    for (uint32_t i = 0; i < light_node_ids.size(); ++i) {
        TrafficLight tl{};
        tl.id         = i;
        tl.node_id    = light_node_ids[i];
        tl.phase      = TrafficPhase::NS_GREEN;
        tl.phase_timer = 0.0f;
        tl.min_green  = DEFAULT_MIN_GREEN;
        tl.max_green  = DEFAULT_MAX_GREEN;
        tl.in_all_red = false;
        lights_.push_back(tl);
    }
}

void TrafficLightSystem::reset() {
    for (auto& tl : lights_) {
        tl.phase       = TrafficPhase::NS_GREEN;
        tl.phase_timer = 0.0f;
        tl.in_all_red  = false;
        tl.all_red_timer = 0.0f;
    }
}

void TrafficLightSystem::update(float dt) {
    for (auto& tl : lights_) {
        if (tl.in_all_red) {
            tl.all_red_timer += dt;
            if (tl.all_red_timer >= ALL_RED_DURATION) {
                tl.in_all_red    = false;
                tl.all_red_timer = 0.0f;
                // Transition to the pending phase (stored in phase)
            }
            continue;
        }

        tl.phase_timer += dt;

        // Auto-switch when max_green exceeded (autonomous fallback)
        if (tl.phase_timer >= tl.max_green) {
            switch_phase(tl, next_auto_phase(tl));
        }
    }
}

bool TrafficLightSystem::apply_action(uint32_t light_id, uint8_t desired_phase) {
    if (light_id >= lights_.size()) return false;
    TrafficLight& tl = lights_[light_id];
    if (tl.in_all_red) return false;

    // Require minimum green time before allowing phase change
    if (tl.phase_timer < tl.min_green) return false;

    auto new_phase = static_cast<TrafficPhase>(desired_phase % 2);  // only NS/EW
    if (new_phase == tl.phase) return true;  // already in requested phase

    switch_phase(tl, new_phase);
    return true;
}

void TrafficLightSystem::apply_actions(const uint8_t* actions, uint32_t count) {
    uint32_t n = (count < lights_.size()) ? count : static_cast<uint32_t>(lights_.size());
    for (uint32_t i = 0; i < n; ++i) {
        apply_action(i, actions[i]);
    }
}

void TrafficLightSystem::switch_phase(TrafficLight& tl, TrafficPhase new_phase) {
    tl.in_all_red    = true;
    tl.all_red_timer = 0.0f;
    tl.phase         = new_phase;  // will activate after ALL_RED_DURATION
    tl.phase_timer   = 0.0f;
}

TrafficPhase TrafficLightSystem::next_auto_phase(const TrafficLight& tl) const {
    return (tl.phase == TrafficPhase::NS_GREEN) ? TrafficPhase::EW_GREEN : TrafficPhase::NS_GREEN;
}

bool TrafficLightSystem::is_green(uint32_t light_id, LaneDir dir) const {
    if (light_id >= lights_.size()) return true;  // no light = always green
    const TrafficLight& tl = lights_[light_id];
    if (tl.in_all_red) return false;
    if (tl.phase == TrafficPhase::NS_GREEN)
        return (dir == LaneDir::NORTH || dir == LaneDir::SOUTH);
    return (dir == LaneDir::EAST || dir == LaneDir::WEST);
}
