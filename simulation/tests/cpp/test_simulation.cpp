#include <cstdio>
#include <cassert>
#include <cmath>
#include <chrono>
#include "simulation/simulation_loop.hpp"

static void test_reset_and_initial_state() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;
    SimulationLoop sim(cfg, EventConfig{}, 1000, 0.1f);
    sim.reset(42);

    assert(sim.tick() == 0);
    assert(!sim.is_terminated());
    assert(!sim.is_truncated());
    printf("PASS: reset produces clean initial state\n");
}

static void test_step_advances_tick() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;
    SimulationLoop sim(cfg, EventConfig{}, 1000, 0.1f);
    sim.reset(42);

    sim.step();
    assert(sim.tick() == 1);
    sim.step();
    assert(sim.tick() == 2);
    printf("PASS: step advances tick correctly\n");
}

static void test_truncation_at_episode_end() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;
    SimulationLoop sim(cfg, EventConfig{}, 10, 0.1f);  // short episode
    sim.reset(42);

    for (int i = 0; i < 10; ++i) sim.step();
    assert(sim.is_truncated() && "Must truncate after episode_length steps");
    printf("PASS: truncation after episode_length steps\n");
}

static void test_snapshot_fields() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;
    SimulationLoop sim(cfg, EventConfig{}, 1000, 0.1f);
    sim.reset(42);
    sim.step();

    SimStateBuffer buf{};
    sim.snapshot(buf);
    assert(buf.sim_tick == 1);
    assert(buf.episode_step == 1);
    assert(buf.flags == 0 && "No termination after 1 step");
    printf("PASS: snapshot fields correct after 1 step\n");
}

static void test_reproducibility() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;

    SimulationLoop sim1(cfg, EventConfig{}, 100, 0.1f);
    SimulationLoop sim2(cfg, EventConfig{}, 100, 0.1f);
    sim1.reset(42);
    sim2.reset(42);

    for (int i = 0; i < 50; ++i) { sim1.step(); sim2.step(); }

    SimStateBuffer buf1{}, buf2{};
    sim1.snapshot(buf1);
    sim2.snapshot(buf2);

    assert(buf1.num_vehicles == buf2.num_vehicles && "Same seed must produce same vehicle count");
    printf("PASS: reproducibility — same vehicle count after 50 steps\n");
}

static void test_performance() {
    CityConfig cfg;
    cfg.grid_width  = 8;
    cfg.grid_height = 8;
    cfg.seed        = 42;
    SimulationLoop sim(cfg, EventConfig{}, 100000, 0.1f, 3.0f);  // high spawn rate to fill city
    sim.reset(42);

    // Warm up
    for (int i = 0; i < 100; ++i) sim.step();

    auto t0 = std::chrono::high_resolution_clock::now();
    int N = 1000;
    for (int i = 0; i < N; ++i) sim.step();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double steps_per_sec = N / elapsed_s;
    uint32_t active = sim.pool().active_count();
    printf("INFO: 8x8 city, %u active vehicles, %.0f steps/sec\n", active, steps_per_sec);
    // Soft warning, not hard fail (server hardware varies)
    if (steps_per_sec < 1000.0) {
        printf("WARN: steps/sec below 1000 target — consider profiling\n");
    } else {
        printf("PASS: performance target met (%.0f steps/sec >= 1000)\n", steps_per_sec);
    }
}

static void test_no_vehicle_overlap() {
    // Invariant: two vehicles in the same (segment, lane) never have centres
    // closer than VEHICLE_LENGTH. Regression guard for the stacking bug.
    CityConfig cfg;
    cfg.grid_width  = 8;
    cfg.grid_height = 8;
    cfg.seed        = 42;
    SimulationLoop sim(cfg, EventConfig{}, 100000, 0.1f, 3.0f);  // high spawn to stress queues
    sim.reset(42);

    long violations = 0;
    for (int s = 0; s < 2000; ++s) {
        sim.step();
        for (const Vehicle& a : sim.pool()) {
            for (const Vehicle& b : sim.pool()) {
                if (a.id >= b.id) continue;
                if (a.segment_id == b.segment_id && a.lane == b.lane) {
                    if (std::fabs(a.position - b.position) < VEHICLE_LENGTH - 0.01f)
                        ++violations;
                }
            }
        }
    }
    assert(violations == 0 && "No two vehicles may overlap in the same segment+lane");
    printf("PASS: no vehicle overlap over 2000 steps (8x8, high spawn)\n");
}

static void test_no_junction_drive_through() {
    // Cars must brake for traffic already in the intersection instead of driving
    // through it. We allow a small transient residual but assert the world-space
    // overlap count stays far below the pre-fix level (~thousands per 1500 steps).
    CityConfig cfg;
    cfg.grid_width  = 8;
    cfg.grid_height = 8;
    cfg.seed        = 42;
    SimulationLoop sim(cfg, EventConfig{}, 100000, 0.1f, 3.0f);
    sim.reset(42);

    long world_overlaps = 0;
    std::vector<const Vehicle*> vs;
    for (int s = 0; s < 1500; ++s) {
        sim.step();
        vs.clear();
        for (const Vehicle& v : sim.pool()) vs.push_back(&v);
        for (size_t i = 0; i < vs.size(); ++i)
            for (size_t j = i + 1; j < vs.size(); ++j) {
                float dx = vs[i]->x - vs[j]->x, dy = vs[i]->y - vs[j]->y;
                if (std::sqrt(dx * dx + dy * dy) < VEHICLE_LENGTH - 0.01f) ++world_overlaps;
            }
    }
    // Pre-fix this was ~14000+; with junction braking it is a couple hundred.
    assert(world_overlaps < 600 && "Cars must not drive through junction traffic");
    printf("PASS: junction drive-through bounded (%ld world overlaps in 1500 steps)\n", world_overlaps);
}

static void test_spawn_and_dest_on_gateway() {
    // With access roads on, vehicles enter from exterior gateways and target a
    // gateway (they roll in from outside the map and leave through one).
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.access_roads = true;
    cfg.seed        = 42;
    SimulationLoop sim(cfg, EventConfig{}, 100000, 0.1f);
    sim.reset(42);
    const CityGraph& city = sim.city();

    for (int s = 0; s < 1000; ++s) {
        sim.step();
        for (const Vehicle& v : sim.pool()) {
            assert(city.is_gateway(v.dest_node) && "every destination must be a gateway");
        }
    }
    printf("PASS: all destinations are exterior gateways\n");
}

int main() {
    printf("=== Simulation Tests ===\n");
    test_reset_and_initial_state();
    test_step_advances_tick();
    test_truncation_at_episode_end();
    test_snapshot_fields();
    test_reproducibility();
    test_no_vehicle_overlap();
    test_no_junction_drive_through();
    test_spawn_and_dest_on_gateway();
    test_performance();
    printf("All simulation tests passed.\n");
    return 0;
}
