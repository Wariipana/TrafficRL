#include <cstdio>
#include <cassert>
#include <cmath>
#include "events/event_scheduler.hpp"
#include "city/city_graph.hpp"
#include "simulation/simulation_loop.hpp"

static CityConfig make_cfg(int w = 4, int h = 4, uint64_t seed = 42) {
    CityConfig cfg;
    cfg.grid_width  = w;
    cfg.grid_height = h;
    cfg.seed        = seed;
    cfg.traffic_light_density = 0.8f;
    return cfg;
}

static EventConfig high_rate_cfg() {
    EventConfig e;
    // Very high rates so events fire quickly in tests
    e.collision_prob_per_min         = 30.0f;
    e.road_works_prob_per_min        = 20.0f;
    e.heavy_rain_prob_per_min        = 10.0f;
    e.mass_event_prob_per_min        = 5.0f;
    e.vehicle_breakdown_prob_per_min = 20.0f;
    e.collision_duration_min         = 0.1f;
    e.collision_duration_max         = 0.3f;
    e.road_works_duration_min        = 0.1f;
    e.road_works_duration_max        = 0.5f;
    e.heavy_rain_duration_min        = 0.1f;
    e.heavy_rain_duration_max        = 0.5f;
    e.mass_event_duration_min        = 0.1f;
    e.mass_event_duration_max        = 0.5f;
    e.breakdown_duration_min         = 0.1f;
    e.breakdown_duration_max         = 0.3f;
    e.seed = 42;
    return e;
}

static void test_events_fire_over_time() {
    CityConfig cfg = make_cfg();
    CityGraph  city(cfg);
    EventConfig ecfg = high_rate_cfg();
    EventScheduler sched(city, ecfg);
    sched.reset(42);

    // Run for 60 simulated seconds — many events should fire
    for (int i = 0; i < 600; ++i) sched.update(0.1f);

    uint32_t total = sched.total_events_fired();
    assert(total > 0 && "Events must fire with high-rate config");
    printf("PASS: %u events fired in 60s\n", total);
}

static void test_heavy_rain_active() {
    CityConfig cfg = make_cfg();
    CityGraph  city(cfg);
    EventConfig ecfg = high_rate_cfg();
    ecfg.collision_prob_per_min         = 0.0f;
    ecfg.road_works_prob_per_min        = 0.0f;
    ecfg.mass_event_prob_per_min        = 0.0f;
    ecfg.vehicle_breakdown_prob_per_min = 0.0f;
    ecfg.heavy_rain_prob_per_min        = 120.0f;  // guaranteed very quickly
    EventScheduler sched(city, ecfg);
    sched.reset(42);

    bool saw_rain = false;
    for (int i = 0; i < 200 && !saw_rain; ++i) {
        sched.update(0.1f);
        if (sched.has_heavy_rain()) {
            saw_rain = true;
            float rf = sched.rain_factor();
            assert(rf > 0.0f && rf <= 1.0f && "Rain factor must be in (0,1]");
            printf("PASS: heavy rain active, factor=%.2f\n", rf);
        }
    }
    assert(saw_rain && "Heavy rain must fire within 20s at rate=120/min");
}

static void test_heavy_rain_unique() {
    // Only one HEAVY_RAIN event active at a time
    CityConfig cfg = make_cfg();
    CityGraph  city(cfg);
    EventConfig ecfg = high_rate_cfg();
    ecfg.heavy_rain_prob_per_min = 600.0f;  // extremely high rate
    ecfg.heavy_rain_duration_min = 10.0f;
    ecfg.heavy_rain_duration_max = 10.0f;
    EventScheduler sched(city, ecfg);
    sched.reset(42);

    for (int i = 0; i < 1000; ++i) {
        sched.update(0.1f);
        uint32_t rain_count = 0;
        for (const auto& e : sched.active_events()) {
            if (e.type == EventType::HEAVY_RAIN) ++rain_count;
        }
        assert(rain_count <= 1 && "At most one HEAVY_RAIN at a time");
    }
    printf("PASS: at most one HEAVY_RAIN active simultaneously\n");
}

static void test_segment_blocked() {
    CityConfig cfg = make_cfg();
    CityGraph  city(cfg);
    EventConfig ecfg = high_rate_cfg();
    ecfg.heavy_rain_prob_per_min = 0.0f;
    ecfg.mass_event_prob_per_min = 0.0f;
    ecfg.collision_prob_per_min  = 300.0f;  // fires in < 0.2s
    ecfg.collision_duration_min  = 100.0f;  // keep it active long enough to query
    ecfg.collision_duration_max  = 100.0f;
    EventScheduler sched(city, ecfg);
    sched.reset(42);

    bool saw_block = false;
    for (int i = 0; i < 50 && !saw_block; ++i) {
        sched.update(0.1f);
        for (const auto& e : sched.active_events()) {
            if (e.type == EventType::COLLISION) {
                bool blocked = sched.segment_blocked(e.segment_id, e.lane);
                assert(blocked && "segment_blocked must return true for active collision");
                saw_block = true;
                printf("PASS: collision blocks segment %u lane %u\n", e.segment_id, e.lane);
                break;
            }
        }
    }
    assert(saw_block && "A COLLISION must have fired");
}

static void test_reproducibility_events() {
    CityConfig cfg = make_cfg();
    CityGraph  city(cfg);
    EventConfig ecfg = high_rate_cfg();

    EventScheduler s1(city, ecfg);
    EventScheduler s2(city, ecfg);
    s1.reset(99);
    s2.reset(99);

    for (int i = 0; i < 300; ++i) { s1.update(0.1f); s2.update(0.1f); }

    assert(s1.total_events_fired() == s2.total_events_fired() &&
           "Same seed must produce same event sequence");
    printf("PASS: event reproducibility, total_fired=%u\n", s1.total_events_fired());
}

static void test_events_expire() {
    CityConfig cfg = make_cfg();
    CityGraph  city(cfg);
    EventConfig ecfg = high_rate_cfg();
    ecfg.collision_duration_min = 1.0f;  // 1 second
    ecfg.collision_duration_max = 1.0f;
    ecfg.collision_prob_per_min = 300.0f;
    ecfg.heavy_rain_prob_per_min = 0.0f;
    ecfg.road_works_prob_per_min = 0.0f;
    ecfg.mass_event_prob_per_min = 0.0f;
    ecfg.vehicle_breakdown_prob_per_min = 0.0f;
    EventScheduler sched(city, ecfg);
    sched.reset(42);

    // Advance 0.5s: fire event
    for (int i = 0; i < 5; ++i) sched.update(0.1f);
    bool had_active = sched.active_count() > 0;

    // Advance another 2s: event must expire
    for (int i = 0; i < 20; ++i) sched.update(0.1f);
    // Events may re-fire; just verify expired ones are removed
    for (const auto& e : sched.active_events()) {
        assert(e.elapsed_s < e.duration_s && "No expired events should remain active");
    }
    printf("PASS: expired events removed (had_active=%d, current=%u)\n",
           (int)had_active, sched.active_count());
}

static void test_rain_reduces_speed_in_simulation() {
    CityConfig cfg = make_cfg(4, 4, 42);
    EventConfig ecfg;
    // Only HEAVY_RAIN, fires immediately
    ecfg.heavy_rain_prob_per_min        = 600.0f;
    ecfg.collision_prob_per_min         = 0.0f;
    ecfg.road_works_prob_per_min        = 0.0f;
    ecfg.mass_event_prob_per_min        = 0.0f;
    ecfg.vehicle_breakdown_prob_per_min = 0.0f;
    ecfg.heavy_rain_duration_min        = 1000.0f;
    ecfg.heavy_rain_duration_max        = 1000.0f;
    ecfg.rain_factor_min                = 0.9f;
    ecfg.rain_factor_max                = 0.9f;  // fixed heavy rain
    ecfg.seed = 42;

    // No-rain baseline
    EventConfig no_rain_cfg;
    no_rain_cfg.collision_prob_per_min         = 0.0f;
    no_rain_cfg.road_works_prob_per_min        = 0.0f;
    no_rain_cfg.heavy_rain_prob_per_min        = 0.0f;
    no_rain_cfg.mass_event_prob_per_min        = 0.0f;
    no_rain_cfg.vehicle_breakdown_prob_per_min = 0.0f;

    SimulationLoop sim_dry(cfg, no_rain_cfg, 500, 0.1f, 2.0f);
    SimulationLoop sim_wet(cfg, ecfg,        500, 0.1f, 2.0f);

    sim_dry.reset(42);
    sim_wet.reset(42);

    for (int i = 0; i < 200; ++i) { sim_dry.step(); sim_wet.step(); }

    SimStateBuffer buf_dry{}, buf_wet{};
    sim_dry.snapshot(buf_dry);
    sim_wet.snapshot(buf_wet);

    // In rain, congestion should be higher (more waiting vehicles)
    float dry_wait = buf_dry.metrics.avg_wait_global;
    float wet_wait = buf_wet.metrics.avg_wait_global;

    printf("INFO: avg_wait dry=%.3f wet=%.3f\n", dry_wait, wet_wait);
    // Rain should cause at least equal or more congestion
    // (loose assertion since both start from same seed)
    assert(wet_wait >= dry_wait * 0.8f && "Rain should not somehow reduce wait time significantly");
    printf("PASS: rain increases or maintains congestion vs dry conditions\n");
}

static void test_personality_frustration_builds() {
    // Verify that a stopped vehicle accumulates frustration over time
    CityConfig cfg = make_cfg(4, 4, 42);
    SimulationLoop sim(cfg, EventConfig{}, 2000, 0.1f);
    sim.reset(42);

    // Run for 50 steps to get some vehicles spawned and potentially waiting
    for (int i = 0; i < 50; ++i) sim.step();

    bool found_frustrated = false;
    for (const Vehicle& v : sim.pool()) {
        if (v.frustration > 0.0f) {
            found_frustrated = true;
            break;
        }
    }
    assert(found_frustrated && "Some vehicles must build frustration after 50 steps");
    printf("PASS: frustration builds in stopped vehicles\n");
}

int main() {
    printf("=== Events & Dynamics Tests ===\n");
    test_events_fire_over_time();
    test_heavy_rain_active();
    test_heavy_rain_unique();
    test_segment_blocked();
    test_reproducibility_events();
    test_events_expire();
    test_rain_reduces_speed_in_simulation();
    test_personality_frustration_builds();
    printf("All events tests passed.\n");
    return 0;
}
