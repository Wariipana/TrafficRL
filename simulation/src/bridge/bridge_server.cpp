#include "bridge_server.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <cstdio>

BridgeServer::BridgeServer(std::string prefix)
    : prefix_(std::move(prefix)) {}

BridgeServer::~BridgeServer() {
    close_all();
    // Unlink segments so stale files don't accumulate
    shm_unlink(state_name().c_str());
    shm_unlink(cmd_name().c_str());
    shm_unlink(graph_name().c_str());
}

bool BridgeServer::init() {
    return open_segment(state_name().c_str(), SHM_STATE_SIZE, state_fd_, state_ptr_)
        && open_segment(cmd_name().c_str(),   SHM_CMD_SIZE,   cmd_fd_,   cmd_ptr_)
        && open_segment(graph_name().c_str(), SHM_GRAPH_SIZE, graph_fd_, graph_ptr_);
}

bool BridgeServer::open_segment(const char* name, size_t size, int& fd_out, void*& ptr_out) {
    // Unlink any stale segment with different size
    shm_unlink(name);

    fd_out = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (fd_out == -1) {
        std::perror("shm_open");
        return false;
    }
    if (ftruncate(fd_out, static_cast<off_t>(size)) == -1) {
        std::perror("ftruncate");
        return false;
    }
    ptr_out = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);
    if (ptr_out == MAP_FAILED) {
        std::perror("mmap");
        ptr_out = nullptr;
        return false;
    }
    std::memset(ptr_out, 0, size);
    return true;
}

void BridgeServer::close_all() {
    if (state_ptr_) { munmap(state_ptr_, SHM_STATE_SIZE); state_ptr_ = nullptr; }
    if (cmd_ptr_)   { munmap(cmd_ptr_,   SHM_CMD_SIZE);   cmd_ptr_   = nullptr; }
    if (graph_ptr_) { munmap(graph_ptr_, SHM_GRAPH_SIZE); graph_ptr_ = nullptr; }
    if (state_fd_ >= 0) { close(state_fd_); state_fd_ = -1; }
    if (cmd_fd_   >= 0) { close(cmd_fd_);   cmd_fd_   = -1; }
    if (graph_fd_ >= 0) { close(graph_fd_); graph_fd_ = -1; }
}

bool BridgeServer::spinlock_acquire(std::atomic<uint32_t>& lock, int max_spins) {
    uint32_t expected = 0;
    for (int i = 0; i < max_spins; ++i) {
        if (lock.compare_exchange_weak(expected, 1,
                                       std::memory_order_acquire,
                                       std::memory_order_relaxed)) {
            return true;
        }
        expected = 0;
    }
    return false;
}

void BridgeServer::spinlock_release(std::atomic<uint32_t>& lock) {
    lock.store(0, std::memory_order_release);
}

void BridgeServer::write_state(const SimulationLoop& sim) {
    if (!state_ptr_) return;

    SimStateBuffer buf{};
    sim.snapshot(buf);

    if (!spinlock_acquire(state_hdr()->write_lock)) {
        // If Python is stuck holding the lock, bail — next step will retry
        return;
    }

    ShmStateHeader* hdr = state_hdr();
    hdr->magic            = SHM_STATE_MAGIC;
    hdr->version          = SHM_VERSION;
    hdr->sim_tick         = buf.sim_tick;
    hdr->num_intersections = buf.num_intersections;
    hdr->num_vehicles     = buf.num_vehicles;
    hdr->sim_time_ms      = buf.sim_time_ms;
    hdr->episode_step     = buf.episode_step;
    hdr->flags            = buf.flags;
    hdr->total_throughput = buf.metrics.total_throughput;
    hdr->avg_wait_global  = buf.metrics.avg_wait_global;
    hdr->max_wait_global  = buf.metrics.max_wait_global;
    hdr->congestion_spread = buf.metrics.congestion_spread;

    ShmIntersectionState* shm_ints = shm_intersections(hdr);
    uint32_t n = buf.num_intersections < MAX_LIGHTS ? buf.num_intersections : MAX_LIGHTS;
    for (uint32_t i = 0; i < n; ++i) {
        const IntersectionState& src = buf.intersections[i];
        ShmIntersectionState& dst    = shm_ints[i];
        dst.id              = src.id;
        dst.phase           = src.phase;
        dst.in_all_red      = src.in_all_red ? 1 : 0;
        dst.num_lanes       = src.num_lanes;
        dst.reserved0       = 0;
        dst.phase_timer_ms  = static_cast<uint16_t>(src.phase_timer * 1000.0f);
        dst.reserved1       = 0;
        dst.avg_wait_time   = src.avg_wait_time;
        dst.throughput      = src.throughput;
        for (uint32_t l = 0; l < MAX_LANES; ++l) {
            dst.vehicles_per_lane[l] = src.vehicles_per_lane[l];
            dst.queue_length[l]      = src.queue_length[l];
            dst.avg_speed[l]         = src.avg_speed[l];
        }
    }

    // Export per-vehicle positions so the visualizer renders real, moving cars.
    // hdr->num_intersections must already be set: shm_vehicles() uses it to find
    // the start of the vehicle array (right after the intersection array).
    const VehiclePool& pool = sim.pool();
    ShmVehicle* shm_veh = shm_vehicles(hdr);
    uint32_t vn = pool.active_count();
    if (vn > MAX_VEHICLES) vn = MAX_VEHICLES;
    uint32_t written = 0;
    const Vehicle* vbegin = pool.begin();
    for (uint32_t i = 0; i < vn; ++i) {
        const Vehicle& v = vbegin[i];
        if (!v.active) continue;
        ShmVehicle& d = shm_veh[written];
        d.id       = v.id;
        d.x        = v.x;
        d.y        = v.y;
        d.velocity = v.velocity;
        d.lane     = v.lane;
        d.reserved[0] = 0; d.reserved[1] = 0; d.reserved[2] = 0;
        ++written;
    }
    hdr->num_vehicles_export = written;

    // Export active lane-blocking incidents so the visualizer can mark them — a
    // queue stalled behind a collision/road-works should read as an incident, not
    // a bug. Convert the event's (segment, position) to a world point.
    ShmEvent* shm_evt = shm_events(hdr);
    uint32_t ev_written = 0;
    const CityGraph& city = sim.city();
    for (const SimEvent& ev : sim.events().active_events()) {
        if (ev_written >= MAX_EVENTS_EXPORT) break;
        uint8_t kind;
        switch (ev.type) {
            case EventType::COLLISION:         kind = 0; break;
            case EventType::ROAD_WORKS:        kind = 1; break;
            case EventType::VEHICLE_BREAKDOWN: kind = 2; break;
            default: continue;  // HEAVY_RAIN / MASS_EVENT have no point location
        }
        if (ev.segment_id >= city.edge_count()) continue;
        const RoadSegment& s = city.edge(ev.segment_id);
        const Intersection& a = city.node(s.from_node);
        const Intersection& b = city.node(s.to_node);
        float t = (s.length > 1e-4f) ? (ev.position / s.length) : 0.0f;
        t = (t < 0.0f) ? 0.0f : (t > 1.0f ? 1.0f : t);
        ShmEvent& d = shm_evt[ev_written];
        d.x = a.x + (b.x - a.x) * t;
        d.y = a.y + (b.y - a.y) * t;
        d.type = kind;
        d.reserved[0] = 0; d.reserved[1] = 0; d.reserved[2] = 0;
        ++ev_written;
    }
    hdr->num_events_export = ev_written;

    hdr->state_generation++;  // signal Python that new data is ready
    spinlock_release(hdr->write_lock);
}

void BridgeServer::write_graph(const CityGraph& city) {
    if (!graph_ptr_) return;
    city.serialize_to(graph_ptr_, SHM_GRAPH_SIZE);
}

bool BridgeServer::read_actions(uint8_t* actions_out, uint32_t count) {
    if (!cmd_ptr_) return false;
    ShmCmdHeader* hdr = cmd_hdr();
    if (!spinlock_acquire(hdr->write_lock)) return false;

    uint32_t n = (count < MAX_LIGHTS) ? count : MAX_LIGHTS;
    std::memcpy(actions_out, hdr->phase_actions, n);

    spinlock_release(hdr->write_lock);
    return true;
}

bool BridgeServer::consume_reset(uint64_t& seed_out) {
    if (!cmd_ptr_) return false;
    ShmCmdHeader* hdr = cmd_hdr();
    if (hdr->reset_flag == 0) return false;

    seed_out = hdr->reset_seed;
    hdr->reset_flag = 0;
    return true;
}

bool BridgeServer::has_pending_action() const {
    if (!cmd_ptr_) return false;
    return cmd_hdr()->step_ready.load(std::memory_order_acquire) == 1;
}

void BridgeServer::ack_step() {
    if (!cmd_ptr_) return;
    cmd_hdr()->step_ready.store(0, std::memory_order_release);
}
