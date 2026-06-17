#!/usr/bin/env bash
# Abre el dashboard de TrafficRL en Chromium con WebGL habilitado por software.
#
# Este escritorio usa renderizado por software (llvmpipe, sin GPU), así que
# Chromium bloquea WebGL por su blocklist. Estos flags fuerzan el fallback
# SwiftShader para que la vista 3D del dashboard funcione.
#
# Uso:  ./scripts/open_dashboard.sh   (o doble clic si lo haces ejecutable)
set -e

URL="${1:-http://localhost:8200}"
PORT="${URL##*:}"; PORT="${PORT%%/*}"

# Esperar a que el dashboard responda (por si lo lanzaste recién)
for i in $(seq 1 20); do
  if curl -s -o /dev/null "http://localhost:${PORT}/" 2>/dev/null; then break; fi
  sleep 0.5
done

exec /usr/bin/chromium \
  --enable-unsafe-swiftshader \
  --ignore-gpu-blocklist \
  --enable-webgl \
  --no-first-run \
  --new-window \
  "$URL"
