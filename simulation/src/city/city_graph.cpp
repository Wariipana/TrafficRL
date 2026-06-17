#include "city_graph.hpp"
#include "core/math_utils.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <queue>

// ---- Serialization header (mirrors Python ctypes struct) ----
#pragma pack(push, 1)
struct GraphShmHeader {
    uint32_t magic;          // 0x54524C67 ('TRLg')
    uint32_t version;        // 1
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t num_lights;
    uint8_t  reserved[12];
};

struct NodeShmRecord {
    uint32_t id;
    float    x, y;
    uint8_t  zone;
    uint8_t  has_light;
    uint32_t light_id;
    uint8_t  num_outgoing;
    uint8_t  reserved[1];
};

struct EdgeShmRecord {
    uint32_t id;
    uint32_t from_node, to_node;
    float    length;
    uint8_t  num_lanes;
    float    speed_limit;
    uint8_t  direction;
    uint8_t  reserved[2];
};
#pragma pack(pop)

// ---- CityGraph implementation ----

CityGraph::CityGraph(const CityConfig& cfg) : cfg_(cfg) {
    generate_grid(cfg.seed);
}

ZoneType CityGraph::assign_zone(int col, int row) const {
    int cx = cfg_.grid_width / 2;
    int cy = cfg_.grid_height / 2;
    float dx = static_cast<float>(col - cx) / cx;
    float dy = static_cast<float>(row - cy) / cy;
    float dist = std::sqrt(dx * dx + dy * dy);

    if (dist < 0.4f)  return ZoneType::COMMERCIAL;
    if (dist < 0.75f) return ZoneType::RESIDENTIAL;
    return ZoneType::INDUSTRIAL;
}

uint8_t CityGraph::lane_count_for(float prob, uint64_t& rng) const {
    return (math::lcg_float(rng) < prob) ? 2 : 1;
}

void CityGraph::generate_grid(uint64_t seed) {
    uint64_t rng = seed;
    const int W = cfg_.grid_width;
    const int H = cfg_.grid_height;

    nodes_.clear();
    edges_.clear();
    light_ids_.clear();

    // --- Create nodes ---
    nodes_.reserve(W * H);
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            Intersection node{};
            node.id        = node_index(col, row);
            node.x         = col * cfg_.block_size;
            node.y         = row * cfg_.block_size;
            node.zone      = assign_zone(col, row);
            node.has_light = (math::lcg_float(rng) < cfg_.traffic_light_density);
            node.light_id  = UINT32_MAX;
            nodes_.push_back(node);
        }
    }

    // --- Assign traffic light IDs ---
    uint32_t light_id = 0;
    for (auto& n : nodes_) {
        if (n.has_light) {
            n.light_id = light_id++;
            light_ids_.push_back(n.id);
        }
    }

    // --- Create road segments (bidirectional grid edges) ---
    uint32_t edge_id = 0;

    auto add_edge = [&](uint32_t from, uint32_t to, LaneDir dir) {
        const Intersection& fn = nodes_[from];
        const Intersection& tn = nodes_[to];
        float dx = tn.x - fn.x;
        float dy = tn.y - fn.y;
        float len = std::sqrt(dx * dx + dy * dy);

        RoadSegment seg{};
        seg.id          = edge_id++;
        seg.from_node   = from;
        seg.to_node     = to;
        seg.length      = len;
        seg.direction   = dir;
        seg.num_lanes   = lane_count_for(cfg_.avenue_probability, rng);
        seg.speed_limit = 50.0f / 3.6f;
        edges_.push_back(seg);
    };

    // Horizontal edges
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W - 1; ++col) {
            uint32_t a = node_index(col, row);
            uint32_t b = node_index(col + 1, row);
            add_edge(a, b, LaneDir::EAST);
            add_edge(b, a, LaneDir::WEST);
        }
    }

    // Vertical edges
    for (int row = 0; row < H - 1; ++row) {
        for (int col = 0; col < W; ++col) {
            uint32_t a = node_index(col, row);
            uint32_t b = node_index(col, row + 1);
            add_edge(a, b, LaneDir::NORTH);
            add_edge(b, a, LaneDir::SOUTH);
        }
    }

    // --- Exterior access roads (gateways) ---
    // One gateway per border node, placed one block outside the grid, joined by
    // a bidirectional access road. Cars spawn at the gateway and roll into the
    // city, and leave the map through a gateway. Gateways get ids contiguous
    // after the grid (>= W*H) so node_index/is_perimeter stay valid; they carry
    // no traffic light. Created AFTER the grid edges so grid num_lanes (which
    // consume the rng) are unchanged → determinism preserved.
    gateway_nodes_.clear();
    if (cfg_.access_roads) {
        const float bs = cfg_.block_size;
        for (int row = 0; row < H; ++row) {
            for (int col = 0; col < W; ++col) {
                bool left = (col == 0), right = (col == W - 1);
                bool top = (row == 0),  bottom = (row == H - 1);
                if (!(left || right || top || bottom)) continue;  // interior

                uint32_t border = node_index(col, row);
                // Orthogonal placement; corners use the horizontal axis so the
                // access road keeps length=block_size and a valid LaneDir.
                float gx = nodes_[border].x, gy = nodes_[border].y;
                LaneDir inward;   // direction of gateway -> border (entering the map)
                if (left)        { gx -= bs; inward = LaneDir::EAST; }
                else if (right)  { gx += bs; inward = LaneDir::WEST; }
                else if (top)    { gy -= bs; inward = LaneDir::SOUTH; }
                else             { gy += bs; inward = LaneDir::NORTH; }  // bottom
                LaneDir outward = (inward == LaneDir::EAST)  ? LaneDir::WEST  :
                                  (inward == LaneDir::WEST)  ? LaneDir::EAST  :
                                  (inward == LaneDir::SOUTH) ? LaneDir::NORTH : LaneDir::SOUTH;

                Intersection g{};
                g.id        = static_cast<uint32_t>(nodes_.size());
                g.x         = gx;
                g.y         = gy;
                g.zone      = nodes_[border].zone;
                g.has_light = false;
                g.light_id  = UINT32_MAX;
                nodes_.push_back(g);
                gateway_nodes_.push_back(g.id);

                add_edge(g.id, border, inward);    // entering the city
                add_edge(border, g.id, outward);   // leaving the city
            }
        }
    }

    // --- Update node outgoing count ---
    std::vector<uint32_t> out_count(nodes_.size(), 0);
    for (const auto& e : edges_) out_count[e.from_node]++;
    for (auto& n : nodes_) n.num_outgoing = static_cast<uint8_t>(out_count[n.id]);

    // --- Build CSR adjacency ---
    csr_offsets_.assign(nodes_.size() + 1, 0);
    for (const auto& e : edges_) csr_offsets_[e.from_node + 1]++;
    for (size_t i = 1; i < csr_offsets_.size(); ++i)
        csr_offsets_[i] += csr_offsets_[i - 1];

    csr_edges_.resize(edges_.size());
    std::vector<uint32_t> tmp_cursor(csr_offsets_.begin(), csr_offsets_.end());
    for (const auto& e : edges_) {
        csr_edges_[tmp_cursor[e.from_node]++] = e.id;
    }

    // Routing table + perimeter list depend on the CSR being ready.
    build_routing_table();
}

void CityGraph::build_routing_table() {
    const uint32_t N = static_cast<uint32_t>(nodes_.size());
    next_hop_.assign(static_cast<size_t>(N) * N, UINT32_MAX);

    // All-pairs shortest next-hop via per-source BFS. The grid edges share the
    // same length, so BFS by hop count yields the shortest route. The first hop
    // is propagated during the BFS so no path reconstruction is needed.
    std::vector<uint32_t> first_hop(N);   // first_hop[u] = edge to take from src to reach u
    std::vector<uint8_t>  visited(N);
    for (uint32_t src = 0; src < N; ++src) {
        std::fill(visited.begin(), visited.end(), 0);
        visited[src] = 1;
        std::queue<uint32_t> q;
        q.push(src);
        while (!q.empty()) {
            uint32_t cur = q.front(); q.pop();
            for (uint32_t eid : outgoing_edges(cur)) {
                uint32_t nxt = edges_[eid].to_node;
                if (visited[nxt]) continue;
                visited[nxt] = 1;
                // first hop toward nxt: the edge from src if cur==src, else inherit
                uint32_t fh = (cur == src) ? eid : first_hop[cur];
                first_hop[nxt] = fh;
                next_hop_[static_cast<size_t>(src) * N + nxt] = fh;
                q.push(nxt);
            }
        }
    }

    // Perimeter nodes: on the outer border of the grid.
    perimeter_nodes_.clear();
    for (uint32_t id = 0; id < N; ++id) {
        if (is_perimeter(id)) perimeter_nodes_.push_back(id);
    }
}

uint32_t CityGraph::next_edge(uint32_t from_node, uint32_t to_dest) const {
    const uint32_t N = static_cast<uint32_t>(nodes_.size());
    if (from_node >= N || to_dest >= N) return UINT32_MAX;
    return next_hop_[static_cast<size_t>(from_node) * N + to_dest];
}

bool CityGraph::is_perimeter(uint32_t node_id) const {
    const int W = cfg_.grid_width;
    const int H = cfg_.grid_height;
    // Only grid nodes can be perimeter; gateways (id >= W*H) are not.
    if (node_id >= static_cast<uint32_t>(W * H)) return false;
    int col = static_cast<int>(node_id) % W;
    int row = static_cast<int>(node_id) / W;
    return col == 0 || col == W - 1 || row == 0 || row == H - 1;
}

bool CityGraph::is_gateway(uint32_t node_id) const {
    const int W = cfg_.grid_width;
    const int H = cfg_.grid_height;
    return node_id >= static_cast<uint32_t>(W * H) && node_id < nodes_.size();
}

std::span<const uint32_t> CityGraph::outgoing_edges(uint32_t node_id) const {
    if (node_id >= nodes_.size()) return {};
    uint32_t begin = csr_offsets_[node_id];
    uint32_t end   = csr_offsets_[node_id + 1];
    return { csr_edges_.data() + begin, end - begin };
}

size_t CityGraph::serialize_size() const {
    return sizeof(GraphShmHeader)
         + nodes_.size() * sizeof(NodeShmRecord)
         + edges_.size() * sizeof(EdgeShmRecord)
         + light_ids_.size() * sizeof(uint32_t);
}

void CityGraph::serialize_to(void* buffer, size_t buffer_size) const {
    if (buffer_size < serialize_size()) {
        throw std::runtime_error("serialize_to: buffer too small");
    }

    uint8_t* ptr = static_cast<uint8_t*>(buffer);

    GraphShmHeader hdr{};
    hdr.magic      = 0x54524C67;
    hdr.version    = 1;
    hdr.num_nodes  = static_cast<uint32_t>(nodes_.size());
    hdr.num_edges  = static_cast<uint32_t>(edges_.size());
    hdr.num_lights = static_cast<uint32_t>(light_ids_.size());
    std::memcpy(ptr, &hdr, sizeof(hdr));
    ptr += sizeof(hdr);

    for (const auto& n : nodes_) {
        NodeShmRecord r{};
        r.id           = n.id;
        r.x            = n.x;
        r.y            = n.y;
        r.zone         = static_cast<uint8_t>(n.zone);
        r.has_light    = n.has_light ? 1 : 0;
        r.light_id     = n.light_id;
        r.num_outgoing = n.num_outgoing;
        std::memcpy(ptr, &r, sizeof(r));
        ptr += sizeof(r);
    }

    for (const auto& e : edges_) {
        EdgeShmRecord r{};
        r.id          = e.id;
        r.from_node   = e.from_node;
        r.to_node     = e.to_node;
        r.length      = e.length;
        r.num_lanes   = e.num_lanes;
        r.speed_limit = e.speed_limit;
        r.direction   = static_cast<uint8_t>(e.direction);
        std::memcpy(ptr, &r, sizeof(r));
        ptr += sizeof(r);
    }

    std::memcpy(ptr, light_ids_.data(), light_ids_.size() * sizeof(uint32_t));
}
