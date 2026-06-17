#include <cstdio>
#include <cassert>
#include "city/city_graph.hpp"
#include "city/spawn_manager.hpp"

static void test_grid_node_count() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.access_roads = false;  // pure grid topology
    cfg.seed        = 42;
    CityGraph g(cfg);
    assert(g.node_count() == 16 && "4x4 grid must have 16 nodes");
    printf("PASS: 4x4 node count = %u\n", g.node_count());
}

static void test_grid_edge_count() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.access_roads = false;  // pure grid topology
    cfg.seed        = 42;
    CityGraph g(cfg);
    // Horizontal: (W-1)*H*2 + Vertical: W*(H-1)*2 = 3*4*2 + 4*3*2 = 24+24 = 48
    assert(g.edge_count() == 48 && "4x4 grid must have 48 directed edges");
    printf("PASS: 4x4 edge count = %u\n", g.edge_count());
}

static void test_traffic_lights_assigned() {
    CityConfig cfg;
    cfg.grid_width         = 4;
    cfg.grid_height        = 4;
    cfg.traffic_light_density = 1.0f;  // all intersections get a light
    cfg.seed               = 42;
    CityGraph g(cfg);
    assert(g.light_count() == 16 && "All nodes should have lights at density=1.0");
    printf("PASS: all nodes have lights = %u\n", g.light_count());
}

static void test_no_lights_at_zero_density() {
    CityConfig cfg;
    cfg.grid_width         = 4;
    cfg.grid_height        = 4;
    cfg.traffic_light_density = 0.0f;
    cfg.seed               = 42;
    CityGraph g(cfg);
    assert(g.light_count() == 0 && "No lights at density=0.0");
    printf("PASS: no lights at density=0.0\n");
}

static void test_outgoing_edges_interior_node() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;
    CityGraph g(cfg);
    // Interior node (col=1, row=1) -> id=5 in 4-wide grid
    uint32_t interior_id = 1 * 4 + 1;  // row*width + col
    auto out = g.outgoing_edges(interior_id);
    assert(out.size() == 4 && "Interior node must have 4 outgoing edges (N/S/E/W)");
    printf("PASS: interior node outgoing edges = %zu\n", out.size());
}

static void test_corner_node_outgoing_edges() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.access_roads = false;  // pure grid: corner has only the 2 grid edges
    cfg.seed        = 42;
    CityGraph g(cfg);
    // Corner node (col=0, row=0) -> id=0
    auto out = g.outgoing_edges(0);
    assert(out.size() == 2 && "Corner node must have 2 outgoing edges");
    printf("PASS: corner node outgoing edges = %zu\n", out.size());
}

static void test_reproducibility() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 12345;
    CityGraph g1(cfg);
    CityGraph g2(cfg);
    assert(g1.light_count() == g2.light_count() && "Same seed must produce same light count");
    for (uint32_t i = 0; i < g1.light_count(); ++i)
        assert(g1.light_ids()[i] == g2.light_ids()[i] && "Same seed must produce same light positions");
    printf("PASS: reproducibility with same seed\n");
}

static void test_serialization_size() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;
    CityGraph g(cfg);
    size_t sz = g.serialize_size();
    assert(sz > 0 && sz < 1024 * 1024 && "Serialization size must be > 0 and < 1 MB");
    printf("PASS: serialize_size = %zu bytes\n", sz);
}

static void test_spawn_rate_positive() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;
    CityGraph g(cfg);
    SpawnManager sm(g, 1.0f);
    for (uint32_t i = 0; i < g.edge_count(); ++i) {
        float rate = sm.spawn_rate(i, 0.0f);
        assert(rate > 0.0f && "Spawn rate must be positive");
    }
    printf("PASS: all segment spawn rates > 0\n");
}

static void test_perimeter_classification() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.access_roads = false;  // count only grid nodes
    cfg.seed        = 42;
    CityGraph g(cfg);
    // 4x4: 12 border nodes, 4 interior (ids 5,6,9,10)
    uint32_t perim = 0, interior = 0;
    for (uint32_t id = 0; id < g.node_count(); ++id)
        g.is_perimeter(id) ? ++perim : ++interior;
    assert(perim == 12 && "4x4 must have 12 perimeter nodes");
    assert(interior == 4 && "4x4 must have 4 interior nodes");
    assert(!g.is_perimeter(5) && !g.is_perimeter(6) &&
           !g.is_perimeter(9) && !g.is_perimeter(10) && "interior ids must be 5,6,9,10");
    assert(g.perimeter_nodes().size() == 12 && "perimeter_nodes() size");
    printf("PASS: perimeter classification (12 border, 4 interior)\n");
}

static void test_next_edge_routing() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.seed        = 42;
    CityGraph g(cfg);
    // src==dst → no hop
    assert(g.next_edge(0, 0) == UINT32_MAX && "src==dst gives no edge");
    // For every reachable pair, next_edge must start at src and reduce hop distance.
    for (uint32_t src = 0; src < g.node_count(); ++src) {
        for (uint32_t dst = 0; dst < g.node_count(); ++dst) {
            if (src == dst) continue;
            uint32_t e = g.next_edge(src, dst);
            assert(e != UINT32_MAX && "grid is connected: every pair routable");
            assert(g.edge(e).from_node == src && "next_edge must leave from src");
        }
    }
    printf("PASS: next_edge routing valid for all pairs\n");
}

static void test_gateway_count() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.access_roads = true;
    cfg.seed        = 42;
    CityGraph g(cfg);
    // 12 border nodes → 12 gateways. Nodes: 16 + 12 = 28. Edges: 48 + 12*2 = 72.
    assert(g.gateway_nodes().size() == 12 && "4x4 must have 12 gateways");
    assert(g.node_count() == 28 && "16 grid + 12 gateway nodes");
    assert(g.edge_count() == 72 && "48 grid + 24 access edges");
    printf("PASS: gateway count (12 gateways, 28 nodes, 72 edges)\n");
}

static void test_gateway_classification() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.access_roads = true;
    cfg.seed        = 42;
    CityGraph g(cfg);
    for (uint32_t gid : g.gateway_nodes()) {
        assert(g.is_gateway(gid) && "gateway must classify as gateway");
        assert(!g.is_perimeter(gid) && "gateway is not a grid perimeter node");
        assert(!g.node(gid).has_light && "gateways carry no traffic light");
        assert(g.node(gid).num_outgoing == 1 && "gateway has exactly one outgoing edge");
    }
    printf("PASS: gateway classification (no light, 1 outgoing, not perimeter)\n");
}

static void test_gateway_routing() {
    CityConfig cfg;
    cfg.grid_width  = 4;
    cfg.grid_height = 4;
    cfg.access_roads = true;
    cfg.seed        = 42;
    CityGraph g(cfg);
    // Every distinct gateway pair must be routable (city is connected).
    auto gws = g.gateway_nodes();
    for (uint32_t a : gws) {
        for (uint32_t b : gws) {
            if (a == b) continue;
            assert(g.next_edge(a, b) != UINT32_MAX && "gateway pair must be routable");
        }
    }
    printf("PASS: all gateway pairs routable\n");
}

int main() {
    printf("=== City Tests ===\n");
    test_grid_node_count();
    test_grid_edge_count();
    test_traffic_lights_assigned();
    test_no_lights_at_zero_density();
    test_outgoing_edges_interior_node();
    test_corner_node_outgoing_edges();
    test_reproducibility();
    test_serialization_size();
    test_spawn_rate_positive();
    test_perimeter_classification();
    test_next_edge_routing();
    test_gateway_count();
    test_gateway_classification();
    test_gateway_routing();
    printf("All city tests passed.\n");
    return 0;
}
