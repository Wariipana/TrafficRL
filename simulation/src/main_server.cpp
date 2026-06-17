#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <chrono>
#include <thread>
#include <string>
#include "simulation/simulation_loop.hpp"
#include "bridge/bridge_server.hpp"

// Signal handler for graceful shutdown
static volatile bool g_running = true;
static void on_signal(int) { g_running = false; }

static void print_usage(const char* prog) {
    std::printf("Usage: %s [--width W] [--height H] [--seed S] [--steps N] [--prefix P] [--spawn-rate M] [--warmup K]\n", prog);
    std::printf("  --width  W   grid width  (default 8)\n");
    std::printf("  --height H   grid height (default 8)\n");
    std::printf("  --seed   S   city seed   (default 42)\n");
    std::printf("  --steps  N   episode length in steps (default 2000)\n");
    std::printf("  --prefix P   shm prefix (default trafficrl)\n");
    std::printf("  --warmup K   steps to pre-fill the city on each reset (default 1000)\n");
}

int main(int argc, char** argv) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Defaults
    CityConfig cfg;
    cfg.grid_width  = 8;
    cfg.grid_height = 8;
    cfg.seed        = 42;
    int   episode_steps = 2000;
    float dt            = 0.1f;
    float spawn_mult    = 1.0f;   // scales the base spawn rates (traffic density)
    int   warmup_steps  = 1000;   // steps run on each reset to pre-fill the city.
                                  // Measured: the 4x4 grid keeps filling roughly
                                  // linearly until ~1000-1200 steps before the
                                  // inflow/outflow balance saturates (~80-100
                                  // vehicles); 300 left the city at ~25% load and
                                  // the episode trained on the fill-up transient.
    std::string prefix  = "trafficrl";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--width"  && i+1 < argc) cfg.grid_width     = std::atoi(argv[++i]);
        else if (arg == "--height" && i+1 < argc) cfg.grid_height = std::atoi(argv[++i]);
        else if (arg == "--seed"   && i+1 < argc) cfg.seed        = std::stoull(argv[++i]);
        else if (arg == "--steps"  && i+1 < argc) episode_steps   = std::atoi(argv[++i]);
        else if (arg == "--prefix" && i+1 < argc) prefix          = argv[++i];
        else if (arg == "--spawn-rate" && i+1 < argc) spawn_mult  = std::stof(argv[++i]);
        else if (arg == "--warmup" && i+1 < argc) warmup_steps    = std::atoi(argv[++i]);
        else if (arg == "--help")  { print_usage(argv[0]); return 0; }
    }

    std::printf("[server] Starting TrafficRL server (%dx%d grid, seed=%llu, steps=%d)\n",
        cfg.grid_width, cfg.grid_height, (unsigned long long)cfg.seed, episode_steps);

    EventConfig event_cfg;
    event_cfg.seed = cfg.seed;

    SimulationLoop sim(cfg, event_cfg, episode_steps, dt, spawn_mult);
    BridgeServer   bridge(prefix);

    if (!bridge.init()) {
        std::fprintf(stderr, "[server] Failed to initialize shared memory.\n");
        return 1;
    }

    uint64_t episode_seed = cfg.seed;
    sim.reset(episode_seed, warmup_steps);
    bridge.write_graph(sim.city());
    bridge.write_state(sim);

    std::printf("[server] Ready. Listening for Python client on shm/%s_*\n", prefix.c_str());

    uint8_t actions[MAX_LIGHTS] = {};
    auto last_status = std::chrono::steady_clock::now();

    while (g_running) {
        // Check for reset request
        uint64_t new_seed;
        if (bridge.consume_reset(new_seed)) {
            episode_seed = new_seed;
            sim.reset(episode_seed, warmup_steps);
            bridge.write_graph(sim.city());
            bridge.write_state(sim);
            std::printf("[server] Episode reset with seed=%llu (warmup=%d)\n",
                (unsigned long long)new_seed, warmup_steps);
            continue;
        }

        // Wait for Python action
        if (!bridge.has_pending_action()) {
            struct timespec ts = {0, 100000};  // 100 µs
            nanosleep(&ts, nullptr);
            continue;
        }

        bridge.read_actions(actions, MAX_LIGHTS);
        sim.apply_light_actions(actions, sim.lights().count());
        sim.step();
        bridge.write_state(sim);
        bridge.ack_step();

        // Auto-reset on episode end
        if (sim.is_terminated() || sim.is_truncated()) {
            // Python will send a reset_flag; just wait
        }

        // Print status every 5 seconds
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_status).count() >= 5.0) {
            std::printf("[server] tick=%llu vehicles=%u\n",
                (unsigned long long)sim.tick(), sim.pool().active_count());
            last_status = now;
        }
    }

    std::printf("[server] Shutting down.\n");
    return 0;
}
