#include <cstdio>
#include <cmath>
#include <cassert>
#include "vehicles/idm.hpp"

static Vehicle make_vehicle(float pos, float vel, float desired_speed = 50.0f / 3.6f) {
    Vehicle v{};
    v.position                     = pos;
    v.velocity                     = vel;
    v.personality.desired_speed    = desired_speed;
    v.personality.minimum_gap      = 1.5f;
    v.personality.max_acceleration = 2.5f;
    v.personality.comfort_deceleration = 3.0f;
    v.personality.reaction_time    = 0.8f;
    return v;
}

static void test_free_road() {
    // Vehicle well below desired speed with no leader → positive acceleration
    Vehicle v = make_vehicle(0.0f, 5.0f);
    float acc = idm::compute_acceleration(v, nullptr);
    assert(acc > 0.0f && "Expected positive acceleration on free road");
    assert(acc <= v.personality.max_acceleration + 1e-4f && "Acceleration must not exceed max");
    printf("PASS: free road acceleration = %.3f m/s^2\n", acc);
}

static void test_at_desired_speed() {
    // Vehicle at desired speed with no leader → acceleration ~0
    float v0 = 50.0f / 3.6f;
    Vehicle v = make_vehicle(0.0f, v0);
    float acc = idm::compute_acceleration(v, nullptr);
    assert(std::fabs(acc) < 0.1f && "Expected ~0 acceleration at desired speed on free road");
    printf("PASS: acceleration at desired speed = %.4f m/s^2\n", acc);
}

static void test_stopped_leader_close() {
    // Leader stopped 3m ahead → strong deceleration
    Vehicle v      = make_vehicle(0.0f, 10.0f);
    Vehicle leader = make_vehicle(3.0f + VEHICLE_LENGTH, 0.0f);
    float acc = idm::compute_acceleration(v, &leader);
    assert(acc < -v.personality.comfort_deceleration && "Expected hard deceleration near stopped leader");
    assert(acc >= -idm::MAX_EMERGENCY_DECEL - 1e-4f && "Must not exceed emergency decel cap");
    printf("PASS: deceleration near stopped leader = %.3f m/s^2\n", acc);
}

static void test_distant_leader() {
    // Leader very far ahead → behaves like free road
    Vehicle v      = make_vehicle(0.0f, 5.0f);
    Vehicle leader = make_vehicle(idm::FREE_ROAD_THRESHOLD + 10.0f, 5.0f);
    float acc_free    = idm::compute_acceleration(v, nullptr);
    float acc_distant = idm::compute_acceleration(v, &leader);
    float diff = std::fabs(acc_free - acc_distant);
    assert(diff < 1e-3f && "Distant leader should give same result as free road");
    printf("PASS: distant leader ≈ free road, diff = %.5f\n", diff);
}

static void test_stop_at_redlight() {
    // Approaching stopline at 10 m away → should produce deceleration
    Vehicle v = make_vehicle(0.0f, 8.0f);
    float acc = idm::compute_at_stopline(v, 10.0f);
    assert(acc < 0.0f && "Should decelerate when approaching red light");
    printf("PASS: stopline deceleration = %.3f m/s^2\n", acc);
}

static void test_acceleration_bounds() {
    // Acceleration must always be within [-MAX_DECEL, max_acceleration]
    Vehicle v = make_vehicle(0.0f, 20.0f);
    Vehicle leader = make_vehicle(VEHICLE_LENGTH + 0.1f, 0.0f);  // almost touching
    float acc = idm::compute_acceleration(v, &leader);
    assert(acc >= -idm::MAX_EMERGENCY_DECEL - 1e-4f);
    assert(acc <= v.personality.max_acceleration + 1e-4f);
    printf("PASS: acceleration bounds respected, value = %.3f m/s^2\n", acc);
}

int main() {
    printf("=== IDM Tests ===\n");
    test_free_road();
    test_at_desired_speed();
    test_stopped_leader_close();
    test_distant_leader();
    test_stop_at_redlight();
    test_acceleration_bounds();
    printf("All IDM tests passed.\n");
    return 0;
}
