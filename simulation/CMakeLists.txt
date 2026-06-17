cmake_minimum_required(VERSION 3.20)
project(TrafficRL CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Core simulation library ---
add_library(trafficrl_sim STATIC
    src/city/city_graph.cpp
    src/city/spawn_manager.cpp
    src/vehicles/vehicle_pool.cpp
    src/simulation/traffic_light_system.cpp
    src/simulation/simulation_loop.cpp
    src/events/event_scheduler.cpp
    src/bridge/bridge_server.cpp
)
target_include_directories(trafficrl_sim PUBLIC src/)
target_compile_options(trafficrl_sim PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)
target_link_libraries(trafficrl_sim PUBLIC rt pthread)

# --- Standalone server (drives training + the web dashboard) ---
add_executable(trafficrl_server src/main_server.cpp)
target_link_libraries(trafficrl_server PRIVATE trafficrl_sim)

# --- Tests ---
enable_testing()

add_executable(test_idm tests/cpp/test_idm.cpp)
target_link_libraries(test_idm PRIVATE trafficrl_sim)
target_include_directories(test_idm PRIVATE src/)
add_test(NAME IDMTest COMMAND test_idm)

add_executable(test_city tests/cpp/test_city.cpp)
target_link_libraries(test_city PRIVATE trafficrl_sim)
target_include_directories(test_city PRIVATE src/)
add_test(NAME CityTest COMMAND test_city)

add_executable(test_simulation tests/cpp/test_simulation.cpp)
target_link_libraries(test_simulation PRIVATE trafficrl_sim)
target_include_directories(test_simulation PRIVATE src/)
add_test(NAME SimulationTest COMMAND test_simulation)

add_executable(test_events tests/cpp/test_events.cpp)
target_link_libraries(test_events PRIVATE trafficrl_sim)
target_include_directories(test_events PRIVATE src/)
add_test(NAME EventsTest COMMAND test_events)
