// Interactive demo: load a trained NCA, run it every animation frame, let the user damage it.
const MODELS = [
  { name: "lizard", emoji: "🦎" },
  { name: "mushroom", emoji: "🍄" },
  { name: "sunflower", emoji: "🌻" },
];

const $ = id => document.getElementById(id);
const canvas = $("canvas");
const status = $("status");

let nca = null;
let state = null;
let size = 72;
let paused = false;
let steps = 0;
let brush = 8;
let speed = 2;
let showHidden = false;
let grid = null;           // [1, size, size, 2] cell coordinates, for circular masks
let pendingCuts = [];      // damage queued by pointer events, applied between steps
let loading = null;

function setState(next) {
  if (state) state.dispose();
  state = next;
}

async function loadModel(name) {
  const token = (loading = {});
  status.hidden = false;
  status.textContent = "Loading…";
  const res = await fetch(`models/${name}.json`);
  if (!res.ok) { status.textContent = `Model "${name}" not found`; return; }
  const model = await res.json();
  if (token !== loading) return;  // a newer selection won the race
  if (nca) nca.dispose();
  nca = createNCA(tf, model);
  nca.setAngle((+$("angle").value * Math.PI) / 180);
  size = model.grid;
  canvas.width = canvas.height = size;
  if (grid) grid.dispose();
  grid = tf.tidy(() => {
    const r = tf.range(0, size).cast("float32");
    const yy = r.reshape([size, 1]).tile([1, size]);
    const xx = r.reshape([1, size]).tile([size, 1]);
    return tf.stack([yy, xx], -1).expandDims(0);
  });
  setState(seedState(tf, size, nca.channels));
  steps = 0;
  status.hidden = true;
  document.querySelectorAll("#picker button").forEach(b =>
    b.setAttribute("aria-checked", String(b.dataset.name === name)));
}

// --- damage --------------------------------------------------------------------------------------

function cellFromEvent(e) {
  const rect = canvas.getBoundingClientRect();
  return [((e.clientY - rect.top) / rect.height) * size, ((e.clientX - rect.left) / rect.width) * size];
}

function applyDamage() {
  if (!pendingCuts.length || !state) return;
  const cuts = pendingCuts;
  pendingCuts = [];
  setState(tf.tidy(() => {
    let keep = tf.onesLike(grid.slice([0, 0, 0, 0], [-1, -1, -1, 1]));
    for (const cut of cuts) keep = keep.mul(cut());
    return state.mul(keep);
  }));
}

function circle(cy, cx, r) {
  return () => {
    const d = grid.sub(tf.tensor1d([cy, cx])).square().sum(-1, true);
    return d.greater(r * r).cast("float32");
  };
}

let dragging = false;
let last = null;
canvas.addEventListener("pointerdown", e => {
  dragging = true;
  canvas.setPointerCapture(e.pointerId);
  last = cellFromEvent(e);
  pendingCuts.push(circle(...last, brush / 2));
});
canvas.addEventListener("pointermove", e => {
  if (!dragging) return;
  const p = cellFromEvent(e);
  // Interpolate so fast drags still cut a continuous line.
  const n = Math.max(1, Math.ceil(Math.hypot(p[0] - last[0], p[1] - last[1]) / (brush / 4)));
  for (let i = 1; i <= n; i++)
    pendingCuts.push(circle(last[0] + ((p[0] - last[0]) * i) / n, last[1] + ((p[1] - last[1]) * i) / n, brush / 2));
  last = p;
});
["pointerup", "pointercancel"].forEach(t => canvas.addEventListener(t, () => (dragging = false)));

// --- render loop ---------------------------------------------------------------------------------

async function render() {
  const img = tf.tidy(() => {
    if (showHidden) {
      // Hidden channels 4-6 as colour, masked to living cells.
      const h = state.slice([0, 0, 0, 4], [-1, -1, -1, 3]);
      const a = state.slice([0, 0, 0, 3], [-1, -1, -1, 1]).clipByValue(0, 1);
      return h.mul(0.5).add(0.5).clipByValue(0, 1).mul(a).add(tf.scalar(1).sub(a)).squeeze([0]);
    }
    // Premultiplied RGBA composited onto white.
    const rgb = state.slice([0, 0, 0, 0], [-1, -1, -1, 3]);
    const a = state.slice([0, 0, 0, 3], [-1, -1, -1, 1]).clipByValue(0, 1);
    return tf.scalar(1).sub(a).add(rgb).clipByValue(0, 1).squeeze([0]);
  });
  await tf.browser.toPixels(img, canvas);
  img.dispose();
}

let frames = 0;
let fpsClock = performance.now();
async function loop() {
  if (nca && state) {
    applyDamage();
    if (!paused) {
      for (let i = 0; i < speed; i++) setState(nca.step(state));
      steps += speed;
    }
    await render();
    $("steps").textContent = steps.toLocaleString();
    frames++;
    const now = performance.now();
    if (now - fpsClock > 1000) {
      $("fps").textContent = Math.round((frames * 1000) / (now - fpsClock));
      frames = 0;
      fpsClock = now;
    }
  }
  requestAnimationFrame(loop);
}

// --- controls ------------------------------------------------------------------------------------

function bindRange(id, outId, fmt, onChange) {
  const input = $(id);
  const update = () => { $(outId).textContent = fmt(+input.value); onChange(+input.value); };
  input.addEventListener("input", update);
  update();
}

function init() {
  const picker = $("picker");
  for (const m of MODELS) {
    const b = document.createElement("button");
    b.textContent = m.emoji;
    b.dataset.name = m.name;
    b.setAttribute("role", "radio");
    b.setAttribute("aria-label", m.name);
    b.addEventListener("click", () => loadModel(m.name));
    picker.appendChild(b);
  }

  $("reset").addEventListener("click", () => { if (nca) { setState(seedState(tf, size, nca.channels)); steps = 0; } });
  $("halve").addEventListener("click", () =>
    pendingCuts.push(() => grid.slice([0, 0, 0, 1], [-1, -1, -1, 1]).less(size / 2).cast("float32")));
  $("blast").addEventListener("click", () => {
    for (let i = 0; i < 4; i++)
      pendingCuts.push(circle(size * (0.3 + 0.4 * Math.random()), size * (0.3 + 0.4 * Math.random()), 4 + 5 * Math.random()));
  });
  $("pause").addEventListener("click", e => {
    paused = !paused;
    e.target.textContent = paused ? "▶ Play" : "⏸ Pause";
  });
  $("hidden").addEventListener("change", e => (showHidden = e.target.checked));
  bindRange("brush", "brushOut", v => v, v => (brush = v));
  bindRange("speed", "speedOut", v => `${v} step${v > 1 ? "s" : ""}/frame`, v => (speed = v));
  bindRange("angle", "angleOut", v => `${v}°`, v => nca && nca.setAngle((v * Math.PI) / 180));

  const requested = new URLSearchParams(location.search).get("m");
  loadModel(MODELS.some(m => m.name === requested) ? requested : "lizard");
  requestAnimationFrame(loop);
}

tf.ready().then(init).catch(err => { status.textContent = `Could not start: ${err.message}`; });
