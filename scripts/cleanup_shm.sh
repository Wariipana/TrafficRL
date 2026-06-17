#!/usr/bin/env bash
# Remove stale TrafficRL shared memory segments.
# Usage: cleanup_shm.sh [prefix]   (default prefix: trafficrl)
PREFIX="${1:-trafficrl}"
rm -f "/dev/shm/${PREFIX}_state" "/dev/shm/${PREFIX}_cmd" "/dev/shm/${PREFIX}_graph"
echo "Shared memory segments cleaned (prefix: ${PREFIX})."
