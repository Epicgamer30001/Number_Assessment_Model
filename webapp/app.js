// ---- Model (loaded from weights.json) ----
let MODEL = null;

fetch("weights.json")
  .then((r) => r.json())
  .then((j) => { MODEL = j.layers; })
  .catch((e) => console.error("Failed to load weights.json", e));

function relu(v) {
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] > 0 ? v[i] : 0;
  return out;
}

// PyTorch Linear stores W as [out, in]; y[j] = sum_i W[j][i]*x[i] + b[j]
function dense(x, W, b) {
  const out = new Float32Array(W.length);
  for (let j = 0; j < W.length; j++) {
    const row = W[j];
    let s = b[j];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[j] = s;
  }
  return out;
}

function forward(input) {
  let h = relu(dense(input, MODEL[0].W, MODEL[0].b));
  h = relu(dense(h, MODEL[1].W, MODEL[1].b));
  return dense(h, MODEL[2].W, MODEL[2].b); // logits
}

function softmax(logits) {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// ---- Canvas drawing ----
const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

function clearPad() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
clearPad();

ctx.strokeStyle = "#fff";
ctx.lineWidth = 18;
ctx.lineCap = "round";
ctx.lineJoin = "round";

let drawing = false;
let last = null;

function posFromEvent(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) / rect.width) * canvas.width,
    y: ((e.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDraw(e) {
  drawing = true;
  last = posFromEvent(e);
  // dot for single taps
  ctx.beginPath();
  ctx.arc(last.x, last.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  e.preventDefault();
}

function moveDraw(e) {
  if (!drawing) return;
  const p = posFromEvent(e);
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  last = p;
  e.preventDefault();
}

function endDraw() { drawing = false; last = null; }

canvas.addEventListener("pointerdown", startDraw);
canvas.addEventListener("pointermove", moveDraw);
window.addEventListener("pointerup", endDraw);

// ---- Preprocess into MNIST-style 28x28 (0..1) ----
function preprocess() {
  const W = canvas.width, H = canvas.height;
  const img = ctx.getImageData(0, 0, W, H).data;

  // intensity from the red channel (white stroke on black bg)
  const intensity = new Float32Array(W * H);
  let minX = W, minY = H, maxX = -1, maxY = -1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = img[(y * W + x) * 4] / 255; // red channel
      intensity[y * W + x] = v;
      if (v > 0.05) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < 0) return null; // empty canvas

  // Crop bbox to an offscreen canvas, scale longest side to 20px.
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const sw = Math.max(1, Math.round(bw * scale));
  const sh = Math.max(1, Math.round(bh * scale));

  const off = document.createElement("canvas");
  off.width = sw;
  off.height = sh;
  const offCtx = off.getContext("2d", { willReadFrequently: true });
  offCtx.imageSmoothingEnabled = true;
  offCtx.drawImage(canvas, minX, minY, bw, bh, 0, 0, sw, sh);

  // Read scaled glyph intensities.
  const sData = offCtx.getImageData(0, 0, sw, sh).data;
  const glyph = new Float32Array(sw * sh);
  let mass = 0, cx = 0, cy = 0;
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const v = sData[(y * sw + x) * 4] / 255;
      glyph[y * sw + x] = v;
      mass += v;
      cx += v * x;
      cy += v * y;
    }
  }
  if (mass === 0) return null;
  cx /= mass;
  cy /= mass;

  // Paste into 28x28 so the glyph's center of mass lands at (14,14).
  const out = new Float32Array(28 * 28);
  const offX = Math.round(14 - cx);
  const offY = Math.round(14 - cy);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const tx = x + offX;
      const ty = y + offY;
      if (tx >= 0 && tx < 28 && ty >= 0 && ty < 28) {
        out[ty * 28 + tx] = glyph[y * sw + x];
      }
    }
  }
  return out;
}

// ---- Prediction + UI ----
const barsEl = document.getElementById("bars");
const verdictEl = document.getElementById("verdict-digit");

// build 10 bar rows once
const fills = [];
const pcts = [];
const rows = [];
for (let d = 0; d < 10; d++) {
  const li = document.createElement("li");
  const label = document.createElement("span");
  label.textContent = d;
  const track = document.createElement("div");
  track.className = "track";
  const fill = document.createElement("div");
  fill.className = "fill";
  track.appendChild(fill);
  const pct = document.createElement("span");
  pct.className = "pct";
  pct.textContent = "0%";
  li.append(label, track, pct);
  barsEl.appendChild(li);
  fills.push(fill);
  pcts.push(pct);
  rows.push(li);
}

function render(probs) {
  let best = 0;
  for (let d = 0; d < 10; d++) if (probs[d] > probs[best]) best = d;
  verdictEl.textContent = best;
  for (let d = 0; d < 10; d++) {
    fills[d].style.width = (probs[d] * 100).toFixed(1) + "%";
    pcts[d].textContent = (probs[d] * 100).toFixed(1) + "%";
    rows[d].classList.toggle("top", d === best);
  }
}

function predict() {
  if (!MODEL) { console.warn("Model not loaded yet"); return; }
  const input = preprocess();
  if (!input) return;
  const probs = softmax(Array.from(forward(input)));
  render(probs);
}

document.getElementById("predict").addEventListener("click", predict);
document.getElementById("clear").addEventListener("click", () => {
  clearPad();
  verdictEl.textContent = "–";
  render(new Array(10).fill(0));
});

// expose for the sanity-check harness
window.__mnist = { forward, softmax, preprocess };
