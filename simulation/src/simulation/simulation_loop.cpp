#include "simulation_loop.hpp"
#include "vehicles/idm.hpp"
#include "vehicles/personality_dynamics.hpp"
#include "core/math_utils.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

// Travel axis of a lane direction: 0 = N–S (vertical), 1 = E–W (horizontal).
// Two flows only conflict in a junction box when their axes differ; same-axis
// flows (incl. opposing-but-collinear) ride separate lanes and never cross.
static inline int8_t dir_axis(LaneDir d) {
    return (d == LaneDir::EAST || d == LaneDir::WEST) ? 1 : 0;
}

SimulationLoop::SimulationLoop(const CityConfig&  city_cfg,
                               const EventConfig& event_cfg,
                               int                episode_length_steps,
                               float              dt,
                               float              spawn_rate_mult)
    : city_(city_cfg)
    , pool_(city_cfg.seed)
    , spawner_(city_, spawn_rate_mult)
    , events_(city_, event_cfg)
    , dt_(dt)
    , episode_length_(episode_length_steps)
    , event_cfg_(event_cfg)
{
    std::vector<uint32_t> light_nodes(city_.light_ids().begin(), city_.light_ids().end());
    lights_.init(light_nodes);
}

void SimulationLoop::reset(uint64_t seed, int warmup_steps) {
    pool_.reset(seed);
    lights_.reset();
    events_.reset(seed ^ 0xDEADBEEF);  // different seed offset so events ≠ city layout
    spatial_.clear();
    rng_        = seed;
    tick_       = 0;
    step_count_ = 0;
    sim_time_   = 0.0f;
    terminated_ = false;
    truncated_  = false;

    // Warm the city up to a steady traffic level before the episode starts. We run
    // the normal per-step pipeline (spawning, IDM, junction logic) but let the
    // traffic lights cycle on their own; the agent's actions only matter once the
    // measured episode begins. Afterwards we zero the episode counters so the
    // warmup is invisible to the RL loop — it just inherits a populated grid.
    for (int i = 0; i < warmup_steps && !truncated_; ++i) {
        events_.update(dt_);
        check_spawn_despawn();
        rebuild_spatial_hash();
        update_vehicles();
        update_traffic_lights();
        sim_time_ += dt_;
    }
    tick_       = 0;
    step_count_ = 0;
    sim_time_   = 0.0f;
    terminated_ = false;
    truncated_  = false;
}

void SimulationLoop::apply_light_actions(const uint8_t* phases, uint32_t count) {
    lights_.apply_actions(phases, count);
}

void SimulationLoop::step() {
    if (terminated_ || truncated_) return;

    events_.update(dt_);
    check_spawn_despawn();
    rebuild_spatial_hash();
    update_vehicles();
    update_traffic_lights();

    sim_time_   += dt_;
    tick_       += 1;
    step_count_ += 1;

    if (step_count_ >= episode_length_) truncated_ = true;
}

// ---- Private methods ----

void SimulationLoop::update_traffic_lights() {
    lights_.update(dt_);
}

void SimulationLoop::rebuild_spatial_hash() {
    spatial_.clear();
    for (Vehicle& v : pool_) {
        spatial_.insert(&v, v.x, v.y);
    }
}

void SimulationLoop::check_spawn_despawn() {
    const uint32_t E = city_.edge_count();

    // Position of the vehicle closest to the entry (position 0) of each
    // (segment, lane). A new entry is only admitted if this is >= VEHICLE_LENGTH
    // so cars never stack on top of each other when crossing or spawning. The
    // slot is "reserved" (set to 0) once admitted so two vehicles in the same
    // step can't both claim an empty segment.
    std::vector<float> min_entry(static_cast<size_t>(E) * MAX_LANES, 1e30f);
    for (const Vehicle& v : pool_) {
        if (v.segment_id < E && v.lane < MAX_LANES) {
            float& m = min_entry[static_cast<size_t>(v.segment_id) * MAX_LANES + v.lane];
            if (v.position < m) m = v.position;
        }
    }
    auto entry_free = [&](uint32_t edge, uint8_t lane) -> bool {
        if (edge >= E || lane >= MAX_LANES) return false;
        return min_entry[static_cast<size_t>(edge) * MAX_LANES + lane] >= VEHICLE_LENGTH;
    };
    auto reserve_entry = [&](uint32_t edge, uint8_t lane) {
        if (edge < E && lane < MAX_LANES)
            min_entry[static_cast<size_t>(edge) * MAX_LANES + lane] = 0.0f;
    };

    // --- Point-conflict detection at junctions ---
    // A junction box conflict only happens between PERPENDICULAR flows — two cars
    // on the same axis (both N–S, or both E–W) never cross paths in the centre,
    // even when they head in opposite directions (they ride separate lanes). So we
    // track the occupying AXIS (0 = N–S vertical, 1 = E–W horizontal), not the
    // 4-way LaneDir. Using the full direction made opposing-but-collinear flows
    // (e.g. NORTH vs SOUTH) falsely conflict, which needlessly stalled cars and
    // congested intersections. crossing_axis[node] = occupying axis, or -1 if free.
    const uint32_t Nn = city_.node_count();
    std::vector<int8_t> crossing_axis(Nn, -1);
    for (const Vehicle& v : pool_) {
        if (v.segment_id >= E) continue;
        const RoadSegment& s = city_.edge(v.segment_id);
        // A car holds a junction box from the moment it ENTERS the box (last
        // INTERSECTION_RADIUS_M of its segment, approaching to_node) until it has
        // CLEARED it (first INTERSECTION_RADIUS_M of the next segment, leaving
        // from_node). Marking the entering side too — not just the leaving side —
        // closes the window in which two perpendicular cars could both commit into
        // the box and overlap in the centre.
        if (s.from_node < Nn && v.position < INTERSECTION_RADIUS_M) {
            crossing_axis[s.from_node] = dir_axis(s.direction);   // leaving from_node
        }
        if (s.to_node < Nn && (s.length - v.position) < INTERSECTION_RADIUS_M) {
            crossing_axis[s.to_node] = dir_axis(s.direction);     // entering to_node
        }
    }
    auto crossing_free = [&](uint32_t node, LaneDir dir) -> bool {
        if (node >= Nn) return true;
        int8_t occ = crossing_axis[node];
        return occ < 0 || occ == dir_axis(dir);
    };
    auto reserve_crossing = [&](uint32_t node, LaneDir dir) {
        if (node < Nn) crossing_axis[node] = dir_axis(dir);
    };

    // --- Navigation + despawn ---
    // Process in id order so junction priority is deterministic (lowest id wins
    // a contested box), which also breaks any 4-way yield cycle.
    std::vector<uint32_t> to_despawn;
    for (Vehicle& v : pool_) {
        const RoadSegment& seg = city_.edge(v.segment_id);
        if (v.position < seg.length) continue;

        // Reached the destination border node → leave the map.
        if (seg.to_node == v.dest_node) {
            to_despawn.push_back(v.id);
            continue;
        }

        // Shortest-path next hop toward the destination. Unreachable should not
        // happen (validated at spawn) but despawn as a safeguard if it does.
        uint32_t next = city_.next_edge(seg.to_node, v.dest_node);
        if (next == UINT32_MAX) {
            to_despawn.push_back(v.id);
            continue;
        }

        // Yield if a conflicting flow holds this junction box — unless we've been
        // starved waiting (same anti-deadlock rule as the approach check), in which
        // case we take our turn so a busy cross-axis can't block us forever.
        if (v.wait_time <= 8.0f && !crossing_free(seg.to_node, seg.direction)) {
            v.position = seg.length;
            v.velocity = 0.0f;
            continue;
        }

        // Only cross if there is room at the start of the next segment; otherwise
        // wait at the intersection (stay put, stopped) and retry next step.
        if (!entry_free(next, v.lane)) {
            v.position = seg.length;
            v.velocity = 0.0f;
            continue;
        }
        reserve_entry(next, v.lane);
        reserve_crossing(seg.to_node, seg.direction);
        // Remember the segment/lane we are leaving so update_world_position can
        // draw a curved arc through the junction instead of teleporting the centre.
        v.prev_segment_id = v.segment_id;
        v.prev_lane       = v.lane;
        v.segment_id  = next;
        v.position    = 0.0f;
        v.red_decided = false;  // judge the next intersection's red afresh
        v.run_red     = false;
    }
    for (uint32_t id : to_despawn) pool_.despawn(id);

    // --- Spawn: vehicles enter from exterior gateways and roll into the city ---
    // (Falls back to grid-perimeter entry if access roads are disabled.)
    // MASS_EVENT can boost spawn rate on target segments.
    auto entries = city_.gateway_nodes();
    bool use_gateways = !entries.empty();
    if (!use_gateways) entries = city_.perimeter_nodes();
    if (entries.empty()) { spawner_.set_rate_multiplier(1.0f); return; }

    for (uint32_t eid = 0; eid < E; ++eid) {
        // Only spawn on segments leaving an entry node (gateway, or perimeter in
        // fallback mode). A car born here rolls the access road into the city.
        uint32_t from = city_.edge(eid).from_node;
        bool is_entry = use_gateways ? city_.is_gateway(from) : city_.is_perimeter(from);
        if (!is_entry) continue;

        float surge = events_.surge_factor_for(eid);
        spawner_.set_rate_multiplier(surge);
        bool do_spawn = spawner_.should_spawn(eid, sim_time_, dt_, rng_);
        spawner_.set_rate_multiplier(1.0f);
        if (!do_spawn) continue;

        // Need room at the segment entry (Layer A also applies to spawns).
        if (!entry_free(eid, 0)) continue;

        // Pick a reachable destination (another gateway / perimeter node),
        // different from the entry node. The car reaches `to_node` first, from
        // where it must be able to route to dest.
        uint32_t to_node = city_.edge(eid).to_node;
        uint32_t dest = UINT32_MAX;
        for (int attempt = 0; attempt < 4; ++attempt) {
            uint32_t cand = entries[math::lcg_next(rng_) % entries.size()];
            if (cand == from) continue;  // don't immediately turn around
            // Reachable if cand is the immediate to_node, or routable from it.
            if (cand != to_node && city_.next_edge(to_node, cand) == UINT32_MAX) continue;
            dest = cand;
            break;
        }
        if (dest == UINT32_MAX) continue;  // no reachable destination this step

        reserve_entry(eid, 0);
        pool_.spawn(eid, 0, 0.0f, dest);
    }
    spawner_.set_rate_multiplier(1.0f);
}

void SimulationLoop::update_vehicles() {
    // Global weather effects
    float weather_speed_mult = 1.0f;
    float weather_rt_mult    = 1.0f;
    if (events_.has_heavy_rain()) {
        float rf = events_.rain_factor();
        weather_speed_mult = 1.0f - rf * 0.40f;  // up to 40% speed reduction
        weather_rt_mult    = 1.0f + rf * 0.60f;  // up to 60% longer reaction time
    }

    // Build per-segment sorted vehicle lists for IDM
    std::vector<std::vector<Vehicle*>> seg_vehicles(city_.edge_count());
    for (Vehicle& v : pool_) {
        if (v.segment_id < city_.edge_count())
            seg_vehicles[v.segment_id].push_back(&v);
    }

    for (auto& svec : seg_vehicles) {
        for (size_t i = 1; i < svec.size(); ++i) {
            Vehicle* key = svec[i];
            int j = static_cast<int>(i) - 1;
            while (j >= 0 && svec[j]->position > key->position) {
                svec[j + 1] = svec[j];
                --j;
            }
            svec[j + 1] = key;
        }
    }

    for (uint32_t eid = 0; eid < city_.edge_count(); ++eid) {
        auto& svec = seg_vehicles[eid];
        if (svec.empty()) continue;

        const RoadSegment& seg  = city_.edge(eid);
        const Intersection& dst = city_.node(seg.to_node);

        bool red_at_end = false;
        if (dst.has_light && dst.light_id < lights_.count()) {
            red_at_end = !lights_.is_green(dst.light_id, seg.direction);
        }

        // Effective speed limit: reduced by rain and road works
        float eff_speed_limit = seg.speed_limit * weather_speed_mult;
        if (events_.segment_blocked(eid, 0) || events_.segment_blocked(eid, 1)) {
            eff_speed_limit *= 0.5f;  // slow past blocked lane
        }

        for (size_t i = 0; i < svec.size(); ++i) {
            Vehicle& v = *svec[i];

            // Update dynamic personality state; get effective parameters for this step
            bool near_inc = events_.near_incident(v.x, v.y, 50.0f);
            DriverPersonality eff = personality_dynamics::update(v, dt_, near_inc);

            // Apply weather onto effective personality
            eff.reaction_time  *= weather_rt_mult;
            eff.desired_speed   = std::min(eff.desired_speed, eff_speed_limit);

            // Temporarily substitute effective personality for IDM calculation
            DriverPersonality saved = v.personality;
            v.personality = eff;

            Vehicle* leader = (i + 1 < svec.size()) ? svec[i + 1] : nullptr;

            // Incident on this lane: the car squeezes PAST it slowly rather than
            // stopping dead. The engine has no lane-changing, so a fully-stopped
            // virtual blocker would trap the whole queue behind it forever (the
            // deadlock we saw). Instead the incident just caps speed hard near the
            // obstacle — drivers crawl past it, as in real life — and the dashboard
            // shows an icon there so the slow-down reads as an incident, not a bug.
            float incident_speed_cap = 1e9f;
            if (events_.segment_blocked(eid, v.lane)) {
                for (const SimEvent& e : events_.active_events()) {
                    if (e.segment_id == eid && e.lane == v.lane &&
                        std::fabs(e.position - v.position) < 20.0f) {
                        incident_speed_cap = 2.5f;  // ~9 km/h crawl past the incident
                        break;
                    }
                }
            }

            // A red light at the end of the segment is a stopped obstacle on the
            // stop line. The vehicle must brake for whichever is closer: its real
            // leader or the stop line. Previously the red light was only honoured
            // when there was no leader, so any car following another car would
            // sail through the intersection on red — that is the bug being fixed.
            //
            // red_light_compliance models drivers that occasionally run reds: a
            // value < 1.0 means the vehicle ignores the stop line with some
            // probability (sampled once per crossing, latched in run_red).
            Vehicle stopline{};
            bool honour_red = red_at_end;
            if (red_at_end) {
                if (!v.red_decided) {
                    float roll = math::lcg_float(rng_);
                    v.run_red     = (roll > v.personality.red_light_compliance);
                    v.red_decided = true;
                }
                honour_red = !v.run_red;
            } else {
                // Reset the per-crossing decision once the light is green again so
                // the next red is judged afresh.
                v.red_decided = false;
                v.run_red     = false;
            }

            // Hold at the stop line for a red light. The stop line sits at the
            // junction box edge (INTERSECTION_RADIUS_M back) so waiting cars never
            // sit inside the intersection where crossing traffic would hit them.
            if (honour_red) {
                float setback  = std::max(VEHICLE_LENGTH, INTERSECTION_RADIUS_M);
                float stop_pos = seg.length - setback;
                if (stop_pos < 0.0f) stop_pos = 0.0f;
                if (stop_pos > v.position &&
                    (leader == nullptr || stop_pos < leader->position))
                {
                    stopline.position = stop_pos;
                    stopline.velocity = 0.0f;
                    leader = &stopline;
                }
            }

            // Cross-traffic obstacle: when approaching the junction, brake only for
            // a car genuinely CROSSING our path — one inside the box travelling on
            // the PERPENDICULAR axis. We must not brake for cars on our own axis
            // (a leader ahead, or opposing traffic on its own lane): they never
            // cross our trajectory, and yielding to them made cars refuse to enter
            // junctions that actually had room, congesting the intersections. The
            // blocker is a virtual stopped leader at the near edge of the box so the
            // IDM brakes smoothly and respects each driver's minimum_gap. A car
            // already committed inside the box keeps going — it must clear.
            Vehicle crossblock{};
            float dist_to_node = seg.length - v.position;
            // Keep checking for cross traffic until the car has reached the stop
            // line at the box edge (dist_to_node ≈ INTERSECTION_RADIUS_M). Crucially
            // we no longer let a car "commit" the moment it touches the box edge —
            // it must find the box clear of PERPENDICULAR traffic right up to the
            // stop line, which closes the window where two perpendicular cars both
            // rolled in and met in the centre. Once past the stop line (inside the
            // box) it keeps going — it must clear, not freeze mid-junction.
            bool approaching = dist_to_node >= INTERSECTION_RADIUS_M;
            // Anti-starvation: at an unsignalled junction a car can yield forever if
            // the crossing axis keeps a steady stream of traffic. Once it has waited
            // past a threshold, it stops yielding and forces its way in — this
            // guarantees progress and breaks any mutual-yield deadlock. (Signalled
            // junctions don't need this; the lights arbitrate.)
            bool starved = v.wait_time > 8.0f;
            if (approaching && !starved && dist_to_node < INTERSECTION_RADIUS_M + 25.0f) {
                const Intersection& node = city_.node(seg.to_node);
                int8_t my_axis = dir_axis(seg.direction);
                std::vector<Vehicle*> near;
                // Look a bit beyond the box so we also yield to a perpendicular car
                // that is itself approaching fast and about to enter — not only one
                // already inside. This catches simultaneous commits.
                float scan_r = INTERSECTION_RADIUS_M * 1.5f;
                spatial_.query_radius(node.x, node.y, scan_r, near);
                bool box_occupied = false;
                for (Vehicle* o : near) {
                    if (o == &v || o->segment_id == v.segment_id) continue;
                    if (o->segment_id >= city_.edge_count()) continue;
                    // only cars on the perpendicular axis actually cross our path
                    const RoadSegment& os = city_.edge(o->segment_id);
                    if (dir_axis(os.direction) == my_axis) continue;
                    float od = std::sqrt((o->x - node.x) * (o->x - node.x) +
                                         (o->y - node.y) * (o->y - node.y));
                    // Inside the box → definitely yield. Just outside but heading
                    // into THIS node and closer than we are → yield to break the tie
                    // deterministically (the nearer car goes first).
                    // A car already inside the box always has priority — yield.
                    bool o_inside = od < INTERSECTION_RADIUS_M;
                    // Simultaneous commit: a perpendicular car that has also reached
                    // the box edge (about to enter). Break the tie by a STABLE key
                    // (id) so it never flips between steps: the lower id goes, the
                    // higher id yields — asymmetric, so exactly one waits (no
                    // deadlock) and they never both enter (no collision). Kept to the
                    // box edge only (not the whole approach) so cars don't over-yield
                    // and re-congest; the box reservation handles the rest.
                    float o_dist_to_node = os.length - o->position;
                    bool o_entering = os.to_node == seg.to_node &&
                                      o_dist_to_node < INTERSECTION_RADIUS_M &&
                                      o->id < v.id;
                    if (o_inside || o_entering) { box_occupied = true; break; }
                }
                if (box_occupied) {
                    // virtual stopped car at the near edge of the box
                    float obstacle_pos = seg.length - INTERSECTION_RADIUS_M;
                    if (obstacle_pos < 0.0f) obstacle_pos = 0.0f;
                    if (obstacle_pos > v.position &&
                        (leader == nullptr || obstacle_pos < leader->position)) {
                        crossblock.position = obstacle_pos;
                        crossblock.velocity = 0.0f;
                        leader = &crossblock;
                    }
                }

                // "Don't block the box": never roll into the junction unless the
                // OUTGOING segment has room to receive us. Otherwise a car enters,
                // can't leave (its exit is full), and sits stalled inside the box
                // forever — permanently blocking every perpendicular flow and
                // deadlocking the whole intersection. We brake at the stop line and
                // wait outside the box until the exit clears, exactly like a real
                // driver who won't enter a junction they can't clear.
                if (seg.to_node != v.dest_node) {
                    uint32_t nxt = city_.next_edge(seg.to_node, v.dest_node);
                    if (nxt != UINT32_MAX) {
                        const RoadSegment& nx = city_.edge(nxt);
                        // Is the entry of the next segment (our lane) occupied within
                        // a vehicle length? Query near its start point.
                        const Intersection& nf = city_.node(nx.from_node);
                        const Intersection& nt = city_.node(nx.to_node);
                        float ex = nf.x, ey = nf.y;  // start of next segment ≈ node
                        std::vector<Vehicle*> en;
                        spatial_.query_radius(ex, ey, VEHICLE_LENGTH * 2.0f, en);
                        bool exit_full = false;
                        for (Vehicle* o : en) {
                            if (o == &v || o->segment_id != nxt) continue;
                            if (o->position < VEHICLE_LENGTH * 1.5f) { exit_full = true; break; }
                        }
                        (void)nt;
                        if (exit_full) {
                            float stop_pos = seg.length - INTERSECTION_RADIUS_M;
                            if (stop_pos < 0.0f) stop_pos = 0.0f;
                            if (stop_pos > v.position &&
                                (leader == nullptr || stop_pos < leader->position)) {
                                crossblock.position = stop_pos;
                                crossblock.velocity = 0.0f;
                                leader = &crossblock;
                            }
                        }
                    }
                }
            }

            float acc = idm::compute_acceleration(v, leader);

            v.personality = saved;  // restore base personality

            v.velocity   += acc * dt_;
            v.velocity    = math::clamp(v.velocity, 0.0f, eff_speed_limit * 1.5f);
            if (v.velocity > incident_speed_cap) v.velocity = incident_speed_cap;
            v.position   += v.velocity * dt_;
            v.acceleration = acc;

            if (v.velocity < 0.1f) v.wait_time += dt_;
        }

        // --- Layer B: hard anti-penetration safety net over the IDM ---
        // Guarantee that no two vehicles in the same (segment, lane) ever have
        // their centres closer than VEHICLE_LENGTH. Process front-to-back (the
        // leader is already finalized before its follower) so a dense queue
        // resolves in a single pass. svec is sorted by ascending position; the
        // leader of svec[i] is the next vehicle ahead in the SAME lane.
        for (int i = static_cast<int>(svec.size()) - 1; i >= 0; --i) {
            Vehicle& v = *svec[i];
            // find nearest leader ahead in the same lane
            Vehicle* leader = nullptr;
            for (size_t k = i + 1; k < svec.size(); ++k) {
                if (svec[k]->lane == v.lane) { leader = svec[k]; break; }
            }
            if (leader != nullptr) {
                float max_pos = leader->position - VEHICLE_LENGTH;
                if (v.position > max_pos) {
                    v.position = max_pos < 0.0f ? 0.0f : max_pos;
                    v.velocity = 0.0f;
                }
            }
            update_world_position(v);
        }
    }
}

// Lane-offset world point of a segment at parameter t∈[0,1] along its length.
// Offset perpendicular to the direction of travel so opposing flows and separate
// lanes don't overlap on the centre line. Right-hand driving: every vehicle sits
// to the RIGHT of its heading, which puts the two directions of a street on
// opposite sides automatically.
//
// Coordinate frame: x = east, y = south (y grows downward on the top-down map,
// north = up). In that frame the RIGHT-hand normal of heading (dx,dy) is
// (-dy, dx): e.g. heading east (+x) → right is south (+y).
void SimulationLoop::segment_point(uint32_t segment_id, uint8_t lane, float t,
                                   float& out_x, float& out_y) const {
    const RoadSegment& seg   = city_.edge(segment_id);
    const Intersection& from = city_.node(seg.from_node);
    const Intersection& to   = city_.node(seg.to_node);
    t = math::clamp(t, 0.0f, 1.0f);
    float cx = math::lerp(from.x, to.x, t);
    float cy = math::lerp(from.y, to.y, t);

    constexpr float LANE_W_M = 6.0f;   // metres per lane (wide enough that the
                                       // scaled-down visual cars don't overlap)
    float dx = to.x - from.x;
    float dy = to.y - from.y;
    float len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-3f) {
        // right-hand normal of (dx,dy) in this frame is (-dy, dx)
        float rx = -dy / len;
        float ry =  dx / len;
        // first lane sits half a lane off the centre line; extra lanes stack outward
        float off = LANE_W_M * (0.5f + static_cast<float>(lane));
        cx += rx * off;
        cy += ry * off;
    }
    out_x = cx;
    out_y = cy;
}

// Unit heading vector of a segment (from_node → to_node), in the same x=east,
// y=south frame as segment_point. Falls back to +x for a degenerate segment.
void SimulationLoop::lane_heading(uint32_t segment_id, float& hx, float& hy) const {
    const RoadSegment& seg   = city_.edge(segment_id);
    const Intersection& from = city_.node(seg.from_node);
    const Intersection& to   = city_.node(seg.to_node);
    float dx = to.x - from.x, dy = to.y - from.y;
    float len = std::sqrt(dx * dx + dy * dy);
    if (len > 1e-4f) { hx = dx / len; hy = dy / len; }
    else             { hx = 1.0f;     hy = 0.0f; }
}

void SimulationLoop::update_world_position(Vehicle& v) const {
    const RoadSegment& seg = city_.edge(v.segment_id);

    // Straight-line lane point for the current segment position (the default).
    float cx, cy;
    segment_point(v.segment_id, v.lane, v.position / (seg.length + 1e-4f), cx, cy);

    // --- Turn arc (circular fillet, tangent to both lanes) ----------------
    // A real turn is a circular arc tangent to the incoming and outgoing lanes.
    // Tangency is what keeps the straight→arc→straight handoffs perfectly smooth
    // (no kink, no jump): the arc leaves each lane along that lane's own direction.
    // For a circle of radius R turning by angle θ, the tangent points sit a fixed
    // distance  d = R·tan(θ/2)  from the node along each lane — these are A and B.
    //
    // The car drives `position` linearly, so it covers 2d metres of road crossing
    // the zone while the arc itself is only R·θ metres long (R·θ < 2d always). We
    // map the car's signed distance-from-node linearly onto the arc angle, which
    // means the car eases off to ~θ/(2·tan(θ/2)) of its speed mid-turn (≈0.79× for
    // 90°) and back — a smooth, realistic "slow down to turn", NOT the abrupt
    // frame-1 speed drop the earlier mismatched parametrisation produced.
    constexpr float TURN_R = 0.65f * INTERSECTION_RADIUS_M;  // turn radius (metres)

    // Resolve which junction (if any) the car is turning at.
    uint32_t in_seg = UINT32_MAX, out_seg = UINT32_MAX;
    uint8_t  in_lane = 0, out_lane = 0;
    bool leaving = false;

    float dist_to_node = seg.length - v.position;  // metres until current segment ends
    // Lookahead window must cover the largest possible tangent distance d. For a
    // U-turn (θ→π) tan(θ/2)→∞, but real grid turns are ≤90° (d = R). Cap at 2R.
    constexpr float LOOKAHEAD_M = 2.0f * (0.65f * INTERSECTION_RADIUS_M);
    if (v.prev_segment_id != UINT32_MAX && v.position < LOOKAHEAD_M) {
        in_seg = v.prev_segment_id; in_lane = v.prev_lane;
        out_seg = v.segment_id;     out_lane = v.lane;
        leaving = true;
    } else if (dist_to_node < LOOKAHEAD_M && seg.to_node != v.dest_node) {
        uint32_t next = city_.next_edge(seg.to_node, v.dest_node);
        if (next != UINT32_MAX) {
            in_seg = v.segment_id;  in_lane = v.lane;
            out_seg = next;         out_lane = v.lane;
        }
    }

    if (in_seg != UINT32_MAX) {
        // Heading change at the junction.
        float hinx, hiny, houtx, houty;
        lane_heading(in_seg,  hinx,  hiny);
        lane_heading(out_seg, houtx, houty);
        float cross = hinx * houty - hiny * houtx;
        float dot   = hinx * houtx + hiny * houty;
        float turn  = std::atan2(cross, dot);   // (-π, π]; 0 = straight through

        if (std::fabs(turn) >= 0.05f) {
            const RoadSegment& sin  = city_.edge(in_seg);
            const RoadSegment& sout = city_.edge(out_seg);

            // Build the fillet from the two LANE lines (each offset sideways from
            // the centre line), not from the node — the offset lanes don't meet at
            // the node, so the tangent points are NOT R·tan(θ/2) from it. The
            // centre C is the intersection of the two lane lines each shifted by R
            // toward the inside of the turn; A and B are C projected back onto each
            // lane line, which are then true tangent points (|C-A| = |C-B| = R), so
            // the arc connects A→B exactly with no end jump.
            //
            // A point on each lane line at the node end, plus its unit direction:
            float pinx, piny, poutx, pouty;
            segment_point(in_seg,  in_lane,  1.0f, pinx, piny);  // incoming lane @ node
            segment_point(out_seg, out_lane, 0.0f, poutx, pouty); // outgoing lane @ node
            // Inside-of-turn normals (left normal if turning left, else right).
            float sgn = (turn > 0.0f) ? 1.0f : -1.0f;
            float ninx = -hiny  * sgn, niny = hinx  * sgn;   // normal of incoming lane
            float noutx = -houty * sgn, nouty = houtx * sgn; // normal of outgoing lane
            // Offset lane lines by R toward the inside.
            float a1x = pinx  + TURN_R * ninx,  a1y = piny  + TURN_R * niny;
            float a2x = poutx + TURN_R * noutx, a2y = pouty + TURN_R * nouty;
            // Intersect line1 (a1 + t·hin) with line2 (a2 + u·hout): solve for t.
            float det = hinx * (-houty) - (-houtx) * hiny;
            float ccx, ccy;
            if (std::fabs(det) > 1e-5f) {
                float t = ((a2x - a1x) * (-houty) - (-houtx) * (a2y - a1y)) / det;
                ccx = a1x + t * hinx;  ccy = a1y + t * hiny;
            } else {                       // near-parallel: fall back to node-based
                ccx = pinx + TURN_R * ninx;  ccy = piny + TURN_R * niny;
            }
            // A = projection of C onto the incoming lane line (the tangent point);
            // the outgoing tangent point B is symmetric and the arc reaches it by
            // construction (we rotate A around C), so we only need A here.
            float ta = (ccx - pinx) * hinx + (ccy - piny) * hiny;
            float ax = pinx + ta * hinx, ay = piny + ta * hiny;

            // Turn-zone half-length = how far A sits BEFORE the node along the
            // incoming lane (ta is negative, since A precedes the node). Symmetric
            // with B by construction. Clamp so we never reach past a short segment.
            float d = -ta;                                   // metres before node
            d = std::min(d, 0.45f * std::min(sin.length, sout.length));

            // Signed distance of the car from the node (negative before, positive
            // after), along the path it actually drives.
            float dist_from_node = leaving ? v.position
                                           : -(seg.length - v.position);

            if (dist_from_node > -d && dist_from_node < d) {
                // Fraction along the arc, linear in distance-from-node. The arc is
                // shorter than 2d, so the car eases to ~θ/(2tan(θ/2))× speed mid-
                // turn and back — a smooth, continuous slow-down, no jumps.
                float f = (dist_from_node + d) / (2.0f * d);
                f = math::clamp(f, 0.0f, 1.0f);
                float ang = f * turn;

                float rx = ax - ccx, ry = ay - ccy;
                float ca = std::cos(ang), sa = std::sin(ang);
                cx = ccx + (rx * ca - ry * sa);
                cy = ccy + (rx * sa + ry * ca);
            }
        }
    }

    if (v.prev_segment_id != UINT32_MAX && v.position >= LOOKAHEAD_M) {
        // Finished the exit half of the arc — clear the turn state.
        v.prev_segment_id = UINT32_MAX;
    }

    v.x = cx;
    v.y = cy;
}

void SimulationLoop::compute_intersection_state(uint32_t light_idx, IntersectionState& out) const {
    if (light_idx >= lights_.count()) return;
    const TrafficLight& tl = lights_.light(light_idx);
    out.id          = tl.node_id;
    // phase stays in {0,1} so it matches the RL observation space (MultiDiscrete
    // [2]). The inter-phase transition is signalled separately via in_all_red so
    // the visualizer can render the amber warning without polluting what the
    // agent observes.
    out.phase       = static_cast<uint8_t>(tl.phase);
    out.phase_timer = tl.phase_timer;
    out.in_all_red  = tl.in_all_red;

    std::memset(out.vehicles_per_lane, 0, sizeof(out.vehicles_per_lane));
    std::memset(out.queue_length,      0, sizeof(out.queue_length));
    std::memset(out.avg_speed,         0, sizeof(out.avg_speed));

    uint32_t lane_idx    = 0;
    float    total_wait  = 0.0f;
    uint32_t total_count = 0;

    for (const auto& edge : city_.edges()) {
        if (edge.to_node != tl.node_id) continue;
        if (lane_idx >= MAX_LANES) break;

        uint32_t count = 0, queued = 0;
        float speed_sum = 0.0f, wait_sum = 0.0f;

        for (const Vehicle& v : pool_) {
            if (v.segment_id != edge.id) continue;
            ++count;
            speed_sum += v.velocity;
            wait_sum  += v.wait_time;
            if (v.velocity < 0.5f) ++queued;
        }

        out.vehicles_per_lane[lane_idx] = static_cast<float>(count);
        out.queue_length[lane_idx]      = static_cast<float>(queued);
        out.avg_speed[lane_idx]         = (count > 0) ? speed_sum / count : 0.0f;
        total_wait  += wait_sum;
        total_count += count;
        ++lane_idx;
    }

    out.num_lanes     = static_cast<uint8_t>(lane_idx);
    out.avg_wait_time = (total_count > 0) ? total_wait / total_count : 0.0f;
}

void SimulationLoop::compute_global_metrics(GlobalMetrics& out) const {
    out.active_vehicles = pool_.active_count();
    float wait_sum = 0.0f, max_wait = 0.0f;

    for (const Vehicle& v : pool_) {
        wait_sum += v.wait_time;
        if (v.wait_time > max_wait) max_wait = v.wait_time;
    }

    uint32_t count = pool_.active_count();
    out.avg_wait_global = (count > 0) ? wait_sum / count : 0.0f;
    out.max_wait_global = max_wait;

    uint32_t congested = 0;
    for (const auto& edge : city_.edges()) {
        uint32_t n = 0;
        for (const Vehicle& v : pool_) {
            if (v.segment_id == edge.id) ++n;
        }
        float density = static_cast<float>(n) / (edge.length / 10.0f + 1.0f);
        if (density > 0.5f) ++congested;
    }
    out.congestion_spread = (city_.edge_count() > 0)
        ? static_cast<float>(congested) / city_.edge_count()
        : 0.0f;
    out.total_throughput  = events_.total_events_fired() > 0
        ? out.active_vehicles * 0.1f  // rough proxy for phase 1/2
        : out.active_vehicles * 0.1f;
}

void SimulationLoop::snapshot(SimStateBuffer& buf) const {
    buf.sim_tick          = tick_;
    buf.num_intersections = lights_.count();
    buf.num_vehicles      = pool_.active_count();
    buf.sim_time_ms       = static_cast<uint32_t>(sim_time_ * 1000.0f);
    buf.episode_step      = static_cast<uint32_t>(step_count_);
    buf.flags             = (terminated_ ? 0x1 : 0) | (truncated_ ? 0x2 : 0);

    for (uint32_t i = 0; i < lights_.count() && i < MAX_LIGHTS; ++i) {
        compute_intersection_state(i, buf.intersections[i]);
    }
    compute_global_metrics(buf.metrics);
}
