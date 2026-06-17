import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── constants (mirror city_grid.gd scaling) ────────────────────────────────
const M_TO_WU = 0.1;          // metres → world units
const LANE_W  = 0.6;          // wu (engine LANE_W_M 6.0m * M_TO_WU)
const INT_HALF = 1.4;
// Cars are drawn at their REAL engine dimensions so the visual spacing matches
// the physics. The engine keeps centres at least VEHICLE_LENGTH(4.5m)+min_gap
// apart along a lane, so a 4.5 m car leaves a real gap and never overlaps —
// an earlier 0.16 scale inflated them ~50% and made them appear to touch.
const VEH_LEN_M = 4.5, VEH_WID_M = 1.9;            // metres (engine VEHICLE_LENGTH + typical width)
const VEH_LEN = VEH_LEN_M * M_TO_WU;               // 0.45 wu
const VEH_WID = VEH_WID_M * M_TO_WU;               // 0.19 wu
const MAX_CARS = 4096;
// Real traffic-light semantics: green = go, amber = about to change, red = stop.
const COL_GREEN = 0x1ae626, COL_AMBER = 0xf0a020, COL_RED = 0xf01e1e;

// ── three.js setup ──────────────────────────────────────────────────────────
// The 3D view is best-effort: if WebGL is unavailable (e.g. a headless browser
// with no GPU) we keep the control panel, metrics and WebSocket fully working
// and just show a notice instead of crashing the whole module.
const wrap = document.getElementById("canvas-wrap");
let renderer = null, scene = null, camera = null, controls = null;
let render3DEnabled = false;

try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  wrap.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1419);

  camera = new THREE.PerspectiveCamera(50, 1, 0.1, 2000);
  camera.position.set(0, 60, 0.01);
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;

  // Flat, bright top-down lighting so the saturated car colours read at a glance
  // and don't fall into shadow against the dark asphalt.
  scene.add(new THREE.AmbientLight(0xffffff, 1.35));
  const dir = new THREE.DirectionalLight(0xffffff, 0.5);
  dir.position.set(10, 60, 10);
  scene.add(dir);

  const resize = () => {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(wrap);
  resize();
  render3DEnabled = true;
} catch (err) {
  console.warn("WebGL no disponible; el panel sigue activo sin vista 3D.", err);
  const o = document.getElementById("overlay");
  o.textContent = "Vista 3D no disponible (WebGL deshabilitado). Métricas y controles activos.";
}

// ── world containers ─────────────────────────────────────────────────────────
const cityGroup = new THREE.Group();
if (scene) scene.add(cityGroup);

let nodePos = {};                 // id -> THREE.Vector3
let lightMeshes = {};             // node_id -> {ns: mesh, ew: mesh}  (frame keys by node_id)
let gatewaySet = new Set();       // node ids that are exterior gateways (num_outgoing===1)
let carMesh = null;               // InstancedMesh (bodies)
let carOutline = null;            // InstancedMesh (dark borders)
let carColors = null;
const carState = new Map();       // engine id -> {pos: Vector3, target: Vector3, yaw}
let freeSlots = [];
const idToSlot = new Map();
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(),
      _e = new THREE.Euler(), _s = new THREE.Vector3(1, 1, 1), _c = new THREE.Color(),
      _bs = new THREE.Vector3(1, 1, 1), _bp = new THREE.Vector3();

// Visual size of the cars. Default 1× draws them at their real proportions
// (0.19 wide × 0.45 long → properly elongated). carScale scales the car
// UNIFORMLY so it always keeps a car shape (longer than wide); the LENGTH factor
// is capped so queued cars never visually overlap (engine packs centres ≥0.6 wu
// apart). Earlier this scaled width/height only, which at 3× made the car wider
// than long — a tile, not a car — and looked wrong when turning.
let carScale = 1.0;
const VEH_LEN_CAP = 0.6 / VEH_LEN;   // max length factor before queued cars touch

function clearCity() {
  cityGroup.clear();
  nodePos = {}; lightMeshes = {}; gatewaySet = new Set();
  carState.clear(); idToSlot.clear(); freeSlots = [];
  carMesh = null;
  eventMarkers = [];   // cleared with the group; rebuilt on demand
}

function buildCity(graph) {
  if (!render3DEnabled) {  // still update the overlay so the user sees topology info
    document.getElementById("overlay").textContent =
      `${graph.nodes.length} nodos · ${graph.edges.length} calles · ${graph.num_lights} semáforos · (vista 3D off)`;
    return;
  }
  clearCity();
  for (const n of graph.nodes) {
    nodePos[n.id] = new THREE.Vector3(n.x * M_TO_WU, 0, n.y * M_TO_WU);
    // A gateway (exterior access node) has exactly one outgoing edge; track it
    // so fitCamera can frame the city tightly without the outside roads.
    if (n.num_outgoing === 1) gatewaySet.add(n.id);
  }

  // roads (dedup undirected) + intersections with traffic-light bulbs
  const drawn = new Set();
  for (const e of graph.edges) {
    const a = Math.min(e.from, e.to), b = Math.max(e.from, e.to), key = a + "_" + b;
    if (drawn.has(key)) continue; drawn.add(key);
    const p0 = nodePos[e.from], p1 = nodePos[e.to];
    if (!p0 || !p1) continue;
    spawnRoad(p0, p1, (e.lanes || 1) * 2);
  }

  const lightSet = new Set(graph.light_node_ids);
  for (const n of graph.nodes) {
    const p = nodePos[n.id]; if (!p) continue;
    const isLight = lightSet.has(n.id) || n.has_light;
    // Key the light meshes by NODE id, because the per-step frame identifies
    // each light by its node_id (IntersectionState.id = tl.node_id), not by the
    // dense light_id. Using light_id here desynced the last lights whenever a
    // node had no light (node_id and light_id diverge after the gap).
    spawnIntersection(p, isLight, n.id);
  }

  // instanced cars: body + outline as two instanced meshes sharing transforms.
  // Height ~1.5 m (0.15 wu). Low roughness + a touch of emissive so the bright
  // body colour pops against the dark asphalt and the dark page background.
  const bodyGeo = new THREE.BoxGeometry(VEH_WID, 0.15, VEH_LEN);
  // InstancedMesh.instanceColor already tints each instance — do NOT set
  // vertexColors (the BoxGeometry has no per-vertex colour attribute, which
  // made the bodies render grey). Low roughness keeps the colour vivid.
  const bodyMat = new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.0 });
  carMesh = new THREE.InstancedMesh(bodyGeo, bodyMat, MAX_CARS);
  carMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  carColors = new Float32Array(MAX_CARS * 3);
  carMesh.instanceColor = new THREE.InstancedBufferAttribute(carColors, 3);
  carMesh.count = MAX_CARS;
  cityGroup.add(carMesh);

  // differentiating dark border: a slightly larger, lower instanced box per car
  // so each vehicle reads as a distinct chip even when traffic is dense.
  const outGeo = new THREE.BoxGeometry(VEH_WID + 0.06, 0.12, VEH_LEN + 0.06);
  const outMat = new THREE.MeshStandardMaterial({ color: 0x05060a, roughness: 1.0 });
  carOutline = new THREE.InstancedMesh(outGeo, outMat, MAX_CARS);
  carOutline.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  carOutline.count = MAX_CARS;
  cityGroup.add(carOutline);

  for (let i = 0; i < MAX_CARS; i++) { freeSlots.push(i); hideSlot(i); }

  fitCamera();
  document.getElementById("overlay").textContent =
    `${graph.nodes.length} nodos · ${graph.edges.length} calles · ${graph.num_lights} semáforos`;
}

function spawnRoad(p0, p1, lanes) {
  const diff = new THREE.Vector3().subVectors(p1, p0);
  const span = diff.length();
  if (span < 0.05) return;
  const length = Math.max(span - 2 * INT_HALF, 0.4);
  const roadW = LANE_W * Math.max(lanes, 1);
  const rotY = -Math.atan2(diff.x, diff.z);
  const mid = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);

  const road = new THREE.Mesh(
    new THREE.BoxGeometry(roadW, 0.1, length),
    new THREE.MeshStandardMaterial({ color: 0x21211f, roughness: 1.0 }));
  road.position.set(mid.x, 0.02, mid.z); road.rotation.y = rotY;
  cityGroup.add(road);
}

function spawnIntersection(p, isLight, nodeId) {
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(INT_HALF * 2, 0.12, INT_HALF * 2),
    new THREE.MeshStandardMaterial({ color: 0x2b2b30, roughness: 1.0 }));
  base.position.set(p.x, 0.03, p.z);
  cityGroup.add(base);
  if (!isLight || nodeId == null || nodeId < 0) return;

  // two bulbs: NS-facing and EW-facing, so you can read the phase at a glance
  const mkBulb = (dx, dz) => {
    const b = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 12, 12),
      new THREE.MeshStandardMaterial({ color: 0x444, emissive: 0x000000, emissiveIntensity: 1.5 }));
    b.position.set(p.x + dx, 0.9, p.z + dz);
    cityGroup.add(b);
    return b;
  };
  lightMeshes[nodeId] = { ns: mkBulb(0, -INT_HALF), ew: mkBulb(INT_HALF, 0) };
}

function fitCamera() {
  // Frame the city itself, excluding the exterior gateway nodes so the access
  // roads outside the grid don't force the camera to zoom out.
  const entries = Object.entries(nodePos).filter(([id]) => !gatewaySet.has(Number(id)));
  const vals = (entries.length ? entries.map(([, p]) => p) : Object.values(nodePos));
  if (!vals.length) return;
  let minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
  for (const p of vals) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z); }
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  // Margin leaves the access roads (one block outside the grid) partly visible
  // so you can see cars rolling in from beyond the city.
  const margin = 2.2 * LANE_W + 6;
  const spanX = (maxX - minX) + margin;
  const spanZ = (maxZ - minZ) + margin;
  // Perspective fit: pick the camera height so the larger span fits the viewport
  // given the vertical FOV and aspect ratio (looking straight down).
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const aspect = camera.aspect || 1;
  const hForZ = (spanZ / 2) / Math.tan(vFov / 2);            // fit vertical span
  const hForX = (spanX / 2) / (Math.tan(vFov / 2) * aspect); // fit horizontal span
  const height = Math.max(hForZ, hForX) * 1.05;              // 5% padding
  controls.target.set(cx, 0, cz);
  camera.position.set(cx, height, cz + 0.01);
  camera.updateProjectionMatrix();
}

// ── per-frame car + light updates ────────────────────────────────────────────
function hideSlot(i) {
  _m.makeScale(0, 0, 0);
  carMesh.setMatrixAt(i, _m);
  if (carOutline) carOutline.setMatrixAt(i, _m);
}

function updateLights(lights) {
  if (!render3DEnabled) return;
  for (const l of lights) {
    const lm = lightMeshes[l.id]; if (!lm) continue;
    // Real semaphore behaviour. Steady phase: the axis with right-of-way is
    // green, the other red. During the inter-phase transition (amber=true) the
    // engine has already advanced `phase` to the INCOMING axis, so the axis that
    // is LOSING the green (the opposite one) shows amber while the incoming axis
    // waits on red — exactly like a real light going green→amber→red.
    let nsCol, ewCol;
    if (l.amber) {
      // phase = incoming axis (0 = NS incoming, 1 = EW incoming)
      nsCol = l.phase === 1 ? COL_AMBER : COL_RED;  // NS losing green if EW is incoming
      ewCol = l.phase === 0 ? COL_AMBER : COL_RED;  // EW losing green if NS is incoming
    } else {
      nsCol = l.phase === 0 ? COL_GREEN : COL_RED;
      ewCol = l.phase === 1 ? COL_GREEN : COL_RED;
    }
    lm.ns.material.color.setHex(nsCol); lm.ns.material.emissive.setHex(nsCol);
    lm.ew.material.color.setHex(ewCol); lm.ew.material.emissive.setHex(ewCol);
  }
}

// ── incident markers ─────────────────────────────────────────────────────────
// A small floating icon over each active collision / road-works / breakdown so a
// queue stalled behind one reads as an incident, not a simulation bug. Colour by
// type. Markers are pooled and reused; extras are hidden.
let eventMarkers = [];
const EVENT_COLORS = { collision: 0xff2d2d, road_works: 0xffa500, breakdown: 0xffe14d };
function makeEventMarker() {
  // A cone (like a traffic cone / warning marker) bobbing above the road.
  const m = new THREE.Mesh(
    new THREE.ConeGeometry(0.5, 1.1, 4),
    new THREE.MeshStandardMaterial({ color: 0xffa500, emissive: 0x331a00,
                                     emissiveIntensity: 1.0, roughness: 0.5 }));
  m.visible = false;
  if (scene) cityGroup.add(m);
  return m;
}
function updateEvents(events) {
  if (!render3DEnabled) return;
  for (let i = 0; i < events.length; i++) {
    if (i >= eventMarkers.length) eventMarkers.push(makeEventMarker());
    const e = events[i], mk = eventMarkers[i];
    mk.position.set(e.x * M_TO_WU, 1.6, e.y * M_TO_WU);
    const col = EVENT_COLORS[e.type] ?? 0xffa500;
    mk.material.color.setHex(col);
    mk.visible = true;
  }
  for (let i = events.length; i < eventMarkers.length; i++) eventMarkers[i].visible = false;
}

function ingestVehicles(vehicles) {
  if (!render3DEnabled || !carMesh) return;
  const seen = new Set();
  for (const v of vehicles) {
    seen.add(v.id);
    let slot = idToSlot.get(v.id);
    const target = new THREE.Vector3(v.x * M_TO_WU, 0.2, v.y * M_TO_WU);
    if (slot === undefined) {
      if (!freeSlots.length) continue;
      slot = freeSlots.pop();
      idToSlot.set(v.id, slot);
      carState.set(v.id, { pos: target.clone(), target: target.clone(), yaw: 0, slot });
    } else {
      carState.get(v.id).target.copy(target);
    }
    // colour by speed: stopped red, slow amber, moving blue
    const cs = carState.get(v.id);
    cs.vel = v.v;
  }
  // recycle vehicles that vanished this step
  for (const [id, cs] of carState) {
    if (!seen.has(id)) {
      hideSlot(cs.slot); freeSlots.push(cs.slot);
      idToSlot.delete(id); carState.delete(id);
    }
  }
  carMesh.instanceMatrix.needsUpdate = true;
}

const MAX_TURN_RATE = 9.0;   // rad/s — cap so turns sweep as an arc, not a snap
function glide(dt) {
  if (!carMesh) return;
  const a = Math.min(dt * 9.0, 1.0);
  for (const cs of carState.values()) {
    // Move toward the target at a steady rate (no braking on junction jumps —
    // that was what made cars "stall and spin" at corners). Track the ACTUAL
    // movement and derive heading from it. The engine now emits a clean turn
    // arc, so the heading should follow the path TIGHTLY — heavy smoothing here
    // just lags the body behind the curve, leaving the nose pointing too "open"
    // mid-turn. So we smooth only enough to kill jitter at very low speed.
    const bx = cs.pos.x, bz = cs.pos.z;
    cs.pos.lerp(cs.target, a);
    const mvx = cs.pos.x - bx, mvz = cs.pos.z - bz;
    const mv2 = mvx * mvx + mvz * mvz;
    if (mv2 > 1e-7) {
      // Adaptive smoothing: when the car is clearly moving, trust the heading
      // almost fully (s≈0.85) so the nose tracks the arc; only at near-stop
      // (tiny, noisy movement vectors) fall back to heavier smoothing.
      if (cs.hx === undefined) { cs.hx = mvx; cs.hz = mvz; }
      const moving = mv2 > 4e-5;             // ~ >0.006 wu/frame
      const s = moving ? 0.85 : 0.2;
      cs.hx += (mvx - cs.hx) * s;
      cs.hz += (mvz - cs.hz) * s;
      const yaw = Math.atan2(cs.hx, cs.hz);
      let dyaw = (((yaw - cs.yaw) % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
      const maxStep = MAX_TURN_RATE * dt;
      if (dyaw >  maxStep) dyaw =  maxStep;
      if (dyaw < -maxStep) dyaw = -maxStep;
      cs.yaw += dyaw;
    }
    _e.set(0, cs.yaw, 0); _q.setFromEuler(_e);
    // Cap the LENGTH factor so queued cars never overlap (centres ≥0.6 wu). Cap
    // the WIDTH so the car never becomes wider than it is long (would look like a
    // tile, especially mid-turn). Height follows width. Result: always car-shaped.
    const lenF = Math.min(carScale, VEH_LEN_CAP);
    const maxWidF = (VEH_LEN * lenF * 0.7) / VEH_WID;   // keep width ≤ 0.7·length
    const widF = Math.min(carScale, maxWidF);
    _s.set(widF, widF, lenF);
    cs.pos.y = 0.15 + 0.05 * widF;
    _m.compose(cs.pos, _q, _s);
    carMesh.setMatrixAt(cs.slot, _m);
    if (carOutline) {
      _bs.set(widF, widF, lenF);
      _bp.set(cs.pos.x, cs.pos.y - 0.05, cs.pos.z);
      _m.compose(_bp, _q, _bs);
      carOutline.setMatrixAt(cs.slot, _m);
    }
    // Bright, saturated colours so each car stands out from the dark asphalt
    // (0x21211f) and the dark page background (0x0f1419):
    //   parado = rojo vivo · lento = ámbar · en movimiento = cian brillante
    const col = cs.vel < 0.5 ? 0xff3b3b : (cs.vel < 5.0 ? 0xffb020 : 0x29d4ff);
    _c.setHex(col); _c.toArray(carColors, cs.slot * 3);
  }
  carMesh.instanceMatrix.needsUpdate = true;
  carMesh.instanceColor.needsUpdate = true;
  if (carOutline) carOutline.instanceMatrix.needsUpdate = true;
}

// ── render loop ───────────────────────────────────────────────────────────────
let lastT = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now(), dt = (now - lastT) / 1000; lastT = now;
  if (!render3DEnabled) return;
  glide(dt);
  // spin + bob the incident markers so they catch the eye
  const t = now / 1000;
  for (const mk of eventMarkers) {
    if (!mk.visible) continue;
    mk.rotation.y = t * 2.0;
    mk.position.y = 1.6 + 0.15 * Math.sin(t * 4.0);
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ── websocket stream ──────────────────────────────────────────────────────────
let ws = null;
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  const conn = document.getElementById("conn");
  ws.onopen = () => { conn.textContent = "conectado"; conn.className = "badge b-running"; };
  ws.onclose = () => { conn.textContent = "desconectado"; conn.className = "badge b-error"; setTimeout(connectWS, 1500); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "graph") {
      buildCity(msg);
    } else if (msg.type === "frame") {
      if (msg.lights) updateLights(msg.lights);
      if (msg.vehicles) ingestVehicles(msg.vehicles);
      if (msg.events) updateEvents(msg.events);
      if (msg.metrics) updateMetrics(msg.metrics);
    } else if (msg.type === "status") {
      if (msg.metrics) updateMetrics(msg.metrics, true);
    }
  };
}
connectWS();

// ── metrics + charts ──────────────────────────────────────────────────────────
const rewardHist = [], waitHist = [];
function updateMetrics(m, statusOnly) {
  document.getElementById("m-reward").textContent = (m.reward ?? 0).toFixed(2);
  document.getElementById("m-wait").textContent   = (m.avg_wait ?? 0).toFixed(2);
  document.getElementById("m-thru").textContent   = (m.throughput ?? 0).toFixed(1);
  document.getElementById("m-cong").textContent   = ((m.congestion ?? 0) * 100).toFixed(0) + "%";
  document.getElementById("m-veh").textContent    = m.num_vehicles ?? 0;
  document.getElementById("m-fps").textContent    = (m.fps ?? 0).toFixed(0);
  document.getElementById("progress").textContent = `paso ${m.step ?? 0} / ${totalSteps}`;
  if (!statusOnly) {
    pushHist(rewardHist, m.reward ?? 0);
    pushHist(waitHist, m.avg_wait ?? 0);
    drawChart("chart-reward", rewardHist, 0x3b82f6);
    drawChart("chart-wait", waitHist, 0xf0a020);
  }
}
function pushHist(arr, v) { arr.push(v); if (arr.length > 240) arr.shift(); }
function drawChart(id, data, color) {
  const cv = document.getElementById(id);
  const w = cv.width = cv.clientWidth, h = cv.height = cv.clientHeight;
  const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;
  let lo = Math.min(...data), hi = Math.max(...data); if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  ctx.strokeStyle = "#" + color.toString(16).padStart(6, "0"); ctx.lineWidth = 1.5; ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w, y = h - ((v - lo) / (hi - lo)) * (h - 6) - 3;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

// ── controls ──────────────────────────────────────────────────────────────────
let totalSteps = 0;
const $ = (id) => document.getElementById(id);
async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}) });
  return r.json();
}

// Hyperparameters the dashboard sends to /api/start. Each maps to an input
// #p-<key>; the backend clamps/validates and falls back to defaults.
const PARAM_KEYS = ["seed", "episode_length_steps", "learning_rate",
                    "ent_coef", "gamma", "n_steps"];

function collectParams() {
  const params = {};
  for (const key of PARAM_KEYS) {
    const el = $("p-" + key);
    // skip params hidden for the current algorithm (let the backend default them)
    if (!el || el.offsetParent === null || el.value === "") continue;
    params[key] = parseFloat(el.value);
  }
  return params;
}

// Show only the hyperparameters relevant to the selected algorithm:
//  - baselines (fixed_time/random) hide every RL-only param
//  - data-algos on an element restricts it to a whitelist of algorithms
function syncParamVisibility() {
  const algo = $("algo").value || "ppo";
  const isRL = ["ppo", "a2c", "ippo_gnn", "hrl"].includes(algo);
  for (const el of document.querySelectorAll(".rl-param")) {
    const only = el.getAttribute("data-algos");
    const ok = isRL && (!only || only.split(" ").includes(algo));
    el.style.display = ok ? "" : "none";
  }
}

// Disable/enable every run-configuration control. Called from the status poll so
// parameters can't be changed mid-run (the backend ignores them anyway, but the
// UI should make the lock obvious). idempotent — safe to call every poll tick.
function setConfigLocked(locked) {
  const ids = ["config", "algo", "steps"];
  for (const key of PARAM_KEYS) ids.push("p-" + key);
  for (const id of ids) {
    const el = $(id);
    if (el) el.disabled = locked;
  }
}

$("btn-start").onclick = async function () {
  totalSteps = parseInt($("steps").value, 10);
  const res = await api("/api/start", {
    config: $("config").value,
    total_timesteps: totalSteps,
    algo: $("algo").value || "ppo",
    params: collectParams(),
  });
  if (!res.ok) { $("err").textContent = res.error || "error al iniciar"; }
};

// populate the algorithm selector from the server
async function loadAlgorithms() {
  try {
    const { algorithms } = await (await fetch("/api/algorithms")).json();
    const sel = $("algo");
    sel.innerHTML = algorithms
      .map(function (a) { return `<option value="${a.value}">${a.label}</option>`; })
      .join("");
    syncParamVisibility();
  } catch (e) {}
}
loadAlgorithms();
$("algo").onchange = syncParamVisibility;
$("btn-pause").onclick  = () => api("/api/pause");
$("btn-resume").onclick = () => api("/api/resume");
$("btn-stop").onclick   = () => api("/api/stop");
$("speed").oninput = (e) => { $("hz-label").textContent = e.target.value; };
$("speed").onchange = (e) => api("/api/speed", { hz: parseFloat(e.target.value) });
$("carscale").oninput = (e) => {
  carScale = parseFloat(e.target.value);
  $("carscale-label").textContent = carScale.toFixed(1) + "×";
};

// save trained model
$("btn-save").onclick = async () => {
  const name = $("save-name").value.trim();
  if (!name) { $("save-msg").textContent = "Escribe un nombre"; return; }
  const res = await api("/api/save", { name });
  $("save-msg").textContent = res.ok ? `Guardado en ${res.path}` : (res.error || "error");
  $("save-msg").style.color = res.ok ? "#1ae626" : "#f88";
  if (res.ok) loadModelList();
};

// inference: run a saved model
$("btn-infer").onclick = async () => {
  const model = $("model-select").value;
  if (!model) return;
  const res = await api("/api/run_model", { model, config: $("config").value });
  if (!res.ok) $("err").textContent = res.error || "error al ejecutar modelo";
};

async function loadModelList() {
  try {
    const { models } = await (await fetch("/api/models")).json();
    const sel = $("model-select");
    sel.innerHTML = models.length
      ? models.map(function (m) {
          const tag = m.algo ? ` (${m.algo})` : "";
          return `<option value="${m.name}">${m.name}${tag}</option>`;
        }).join("")
      : '<option value="">— sin modelos —</option>';
    $("btn-infer").disabled = models.length === 0;
  } catch (e) {}
}
loadModelList();

// ── summary panel (shown when a run finishes) ─────────────────────────────────
let summaryShown = false;
async function showSummary() {
  const { summary } = await (await fetch("/api/summary")).json();
  if (!summary) return;
  $("summary-card").style.display = "block";
  const a = summary.aggregate;
  $("summary-agg").innerHTML =
    `${summary.steps_recorded} muestras · ${summary.rollouts} rollouts · ${summary.duration_s}s<br>` +
    `reward media ${a.reward_mean} · espera mín ${a.avg_wait_min}s · throughput ${a.throughput_mean}`;
  // start→end table with coloured deltas (lower wait/congestion is better)
  const sv = summary.start_vs_end;
  const rows = [
    ["Reward", sv.reward, true],
    ["Espera media (s)", sv.avg_wait, false],
    ["Throughput", sv.throughput, true],
    ["Congestión", sv.congestion, false],
  ];
  $("summary-table").innerHTML =
    `<tr style="color:#8b98a8"><th align="left">Métrica</th><th>Inicio</th><th>Final</th><th>Δ</th></tr>` +
    rows.map(([label, d, higherBetter]) => {
      const improved = higherBetter ? d.delta > 0 : d.delta < 0;
      const col = Math.abs(d.delta) < 1e-6 ? "#aab" : (improved ? "#1ae626" : "#f0a020");
      const sign = d.delta > 0 ? "+" : "";
      return `<tr><td>${label}</td><td align="right">${d.start}</td>` +
             `<td align="right">${d.end}</td>` +
             `<td align="right" style="color:${col}">${sign}${d.delta}</td></tr>`;
    }).join("");
  // full history charts with rollout marks
  const { full_history, rollout_marks } = await (await fetch("/api/summary")).json();
  if (full_history && full_history.length > 1) {
    drawFullChart("chart-reward-full", full_history, "reward", 0x3b82f6, rollout_marks);
    drawFullChart("chart-wait-full", full_history, "avg_wait", 0xf0a020, rollout_marks);
  }
}
function drawFullChart(id, hist, key, color, marks) {
  const cv = $(id);
  const w = cv.width = cv.clientWidth, h = cv.height = cv.clientHeight;
  const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, w, h);
  const data = hist.map(p => p[key]);
  const steps = hist.map(p => p.step);
  if (data.length < 2) return;
  const s0 = steps[0], s1 = steps[steps.length - 1] || 1;
  let lo = Math.min(...data), hi = Math.max(...data); if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  // rollout marks (vertical lines)
  ctx.strokeStyle = "rgba(139,152,168,0.3)"; ctx.lineWidth = 1;
  for (const mk of (marks || [])) {
    const x = ((mk - s0) / (s1 - s0)) * w;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  ctx.strokeStyle = "#" + color.toString(16).padStart(6, "0"); ctx.lineWidth = 1.5; ctx.beginPath();
  data.forEach((v, i) => {
    const x = ((steps[i] - s0) / (s1 - s0)) * w, y = h - ((v - lo) / (hi - lo)) * (h - 6) - 3;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

// poll status to drive button states + badge
async function pollStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    totalSteps = s.total_timesteps || totalSteps;
    const badge = $("state-badge");
    badge.textContent = s.state;
    badge.className = "badge b-" + s.state;
    $("err").textContent = s.error || "";
    const busy = s.state === "running" || s.state === "paused" || s.state === "inference";
    const paused = s.state === "paused";
    $("btn-start").disabled  = busy;
    $("btn-infer").disabled  = busy || !$("model-select").value;
    $("btn-pause").disabled  = !(s.state === "running" || s.state === "inference");
    $("btn-resume").disabled = !paused;
    $("btn-stop").disabled   = !busy;
    $("btn-save").disabled   = !s.can_save || busy;
    // lock the run configuration (city/algo/steps/hyperparameters) while a run
    // is active; re-enable once it stops so the next run can be reconfigured.
    setConfigLocked(busy);
    // when a run finishes, reveal the summary once
    if (s.state === "stopped" && s.summary && !summaryShown) {
      summaryShown = true; showSummary();
    }
    if (busy) summaryShown = false;  // reset for the next run
  } catch (e) {}
  setTimeout(pollStatus, 700);
}
pollStatus();
