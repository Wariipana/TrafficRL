#!/usr/bin/env bash
# Remove any stale TrafficRL shared memory segments
rm -f /dev/shm/trafficrl_state /dev/shm/trafficrl_cmd /dev/shm/trafficrl_graph
echo "Shared memory segments cleaned."
