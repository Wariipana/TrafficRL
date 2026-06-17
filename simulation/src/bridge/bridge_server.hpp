#pragma once
#include "shared_memory_layout.hpp"
#include "simulation/simulation_loop.hpp"
#include <string>
#include <cstdint>

// Manages the three POSIX shared memory segments and the step protocol.
// Owner of the shm segments: creates them on init, unlinks on destruction.
class BridgeServer {
public:
    explicit BridgeServer(std::string prefix = "trafficrl");
    ~BridgeServer();

    // Create/reopen shm segments. Unlinks any stale segments first.
    bool init();

    // Write current simulation state to trafficrl_state.
    void write_state(const SimulationLoop& sim);

    // Write graph topology to trafficrl_graph (call once after reset).
    void write_graph(const CityGraph& city);

    // Returns true if Python posted a new action (step_ready==1).
    // Copies phase actions into actions_out (count = min(src, MAX_LIGHTS)).
    bool read_actions(uint8_t* actions_out, uint32_t count);

    // Returns true if Python requested an episode reset.
    // Fills seed_out with the requested seed and clears the flag.
    bool consume_reset(uint64_t& seed_out);

    // Block-free poll: returns true when step_ready transitions from 0→1.
    // Internally just reads the atomic without spinning.
    bool has_pending_action() const;

    // Acknowledge the step: sets step_ready back to 0.
    void ack_step();

private:
    std::string prefix_;
    int         state_fd_ = -1;
    int         cmd_fd_   = -1;
    int         graph_fd_ = -1;
    void*       state_ptr_ = nullptr;
    void*       cmd_ptr_   = nullptr;
    void*       graph_ptr_ = nullptr;

    ShmStateHeader* state_hdr()       { return static_cast<ShmStateHeader*>(state_ptr_); }
    ShmCmdHeader*   cmd_hdr()         { return static_cast<ShmCmdHeader*>(cmd_ptr_); }
    const ShmCmdHeader* cmd_hdr() const { return static_cast<const ShmCmdHeader*>(cmd_ptr_); }

    bool open_segment(const char* name, size_t size, int& fd_out, void*& ptr_out);
    void close_all();

    static bool spinlock_acquire(std::atomic<uint32_t>& lock, int max_spins = 100000);
    static void spinlock_release(std::atomic<uint32_t>& lock);

    std::string state_name() const { return "/" + prefix_ + "_state"; }
    std::string cmd_name()   const { return "/" + prefix_ + "_cmd";   }
    std::string graph_name() const { return "/" + prefix_ + "_graph"; }
};
