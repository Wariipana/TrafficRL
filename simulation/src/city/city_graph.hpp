#pragma once
#include "core/types.hpp"
#include <vector>
#include <span>
#include <cstdint>

// City road network generated procedurally from a CityConfig.
// Grid topology for Phase 1: intersections at every (i,j) position,
// bidirectional segments on horizontal and vertical axes.
// CSR (Compressed Sparse Row) format for O(1) neighbor lookup.
class CityGraph {
public:
    explicit CityGraph(const CityConfig& cfg);

    // Accessors
    const Intersection& node(uint32_t id) const { return nodes_[id]; }
    const RoadSegment&  edge(uint32_t id) const { return edges_[id]; }

    std::span<const Intersection> nodes() const { return nodes_; }
    std::span<const RoadSegment>  edges() const { return edges_; }
    std::span<const uint32_t>     light_ids() const { return light_ids_; }

    uint32_t node_count()  const { return static_cast<uint32_t>(nodes_.size()); }
    uint32_t edge_count()  const { return static_cast<uint32_t>(edges_.size()); }
    uint32_t light_count() const { return static_cast<uint32_t>(light_ids_.size()); }

    // Outgoing edges from a node (CSR)
    std::span<const uint32_t> outgoing_edges(uint32_t node_id) const;

    // --- Routing (precomputed shortest-path next-hop table) ---
    // Edge to take from `from_node` to make shortest progress toward `to_dest`.
    // Returns UINT32_MAX if unreachable or from_node == to_dest.
    uint32_t next_edge(uint32_t from_node, uint32_t to_dest) const;

    // True if the node sits on the outer border of the grid.
    bool is_perimeter(uint32_t node_id) const;

    // Grid border node ids (used internally to attach gateways).
    std::span<const uint32_t> perimeter_nodes() const { return perimeter_nodes_; }

    // True if the node is an exterior gateway (outside the grid; cars spawn here
    // and roll in along an access road, and leave the map through one).
    bool is_gateway(uint32_t node_id) const;

    // Gateway node ids (the map's entry/exit points when access_roads is on).
    std::span<const uint32_t> gateway_nodes() const { return gateway_nodes_; }

    // Serialize topology to a flat buffer for shared memory (trafficrl_graph segment)
    size_t serialize_size() const;
    void   serialize_to(void* buffer, size_t buffer_size) const;

    const CityConfig& config() const { return cfg_; }

private:
    CityConfig cfg_;

    std::vector<Intersection> nodes_;
    std::vector<RoadSegment>  edges_;
    std::vector<uint32_t>     light_ids_;

    // CSR adjacency: csr_offsets_[i]..csr_offsets_[i+1] are edge IDs from node i
    std::vector<uint32_t> csr_offsets_;
    std::vector<uint32_t> csr_edges_;

    // Routing: next_hop_[src*N + dst] = edge id to take from src toward dst
    // (UINT32_MAX if unreachable or src==dst). Precomputed once per episode.
    std::vector<uint32_t> next_hop_;
    std::vector<uint32_t> perimeter_nodes_;
    std::vector<uint32_t> gateway_nodes_;

    void generate_grid(uint64_t seed);
    void build_routing_table();
    uint32_t node_index(int col, int row) const { return row * cfg_.grid_width + col; }
    ZoneType assign_zone(int col, int row) const;
    uint8_t  lane_count_for(float avenue_prob, uint64_t& rng) const;
};
