/* Crystal diffusion scrubber — renders precomputed denoising trajectories.
   Data: data/trajectories.json (written by viz/export_trajectory.py). */

const FRAME_MS = 120;            // playback speed (slow)
const ATOM_COLOR = "#111827";    // all atoms one dark color, on a white canvas
const CELL_COLOR = "#3a4250";    // dark unit-cell wireframe

let DATA = null;
let viewer = null;
let cur = 0;            // current crystal index
let frame = 0;          // current frame index
let playing = false;
let timer = null;

const $ = (id) => document.getElementById(id);

function fracToCart(fr, L) {
  // cart_j = sum_i frac_i * L[i][j]  (rows of L are lattice vectors)
  return [
    fr[0] * L[0][0] + fr[1] * L[1][0] + fr[2] * L[2][0],
    fr[0] * L[0][1] + fr[1] * L[1][1] + fr[2] * L[2][1],
    fr[0] * L[0][2] + fr[1] * L[1][2] + fr[2] * L[2][2],
  ];
}

function xyzString(f, L) {
  const lines = [String(f.elems.length), ""];
  for (let i = 0; i < f.elems.length; i++) {
    const p = fracToCart(f.frac[i], L);
    lines.push(`${f.elems[i]} ${p[0]} ${p[1]} ${p[2]}`);
  }
  return lines.join("\n");
}

function cellCorners(L) {
  const corners = [];
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++)
      for (let k = 0; k < 2; k++)
        corners.push({
          x: i * L[0][0] + j * L[1][0] + k * L[2][0],
          y: i * L[0][1] + j * L[1][1] + k * L[2][1],
          z: i * L[0][2] + j * L[1][2] + k * L[2][2],
          b: [i, j, k],
        });
  return corners;
}

function drawCell(L) {
  const c = cellCorners(L);
  for (let a = 0; a < c.length; a++) {
    for (let b = a + 1; b < c.length; b++) {
      const d = (c[a].b[0] !== c[b].b[0]) + (c[a].b[1] !== c[b].b[1]) + (c[a].b[2] !== c[b].b[2]);
      if (d === 1) {
        viewer.addCylinder({
          start: { x: c[a].x, y: c[a].y, z: c[a].z },
          end: { x: c[b].x, y: c[b].y, z: c[b].z },
          radius: 0.04, color: CELL_COLOR, fromCap: 1, toCap: 1,
        });
      }
    }
  }
}

function renderFrame(idx, recenter = false) {
  const crystal = DATA.crystals[cur];
  const L = crystal.lattice;            // fixed cell (final lattice) for every frame
  const f = crystal.frames[idx];
  viewer.removeAllModels();
  viewer.removeAllShapes();
  viewer.addModel(xyzString(f, L), "xyz");
  viewer.setStyle({}, { sphere: { scale: 0.34, color: ATOM_COLOR } });
  drawCell(L);
  if (recenter) { viewer.zoomTo(); viewer.zoom(0.78); }  // frame once; rotation persists after
  viewer.render();
  $("stepLabel").textContent = `t = ${f.t}`;
}

function loadCrystal(i) {
  cur = i;
  const crystal = DATA.crystals[i];
  $("slider").max = String(crystal.frames.length - 1);
  buildLegend(crystal);
  renderFrame(crystal.frames.length - 1, true);  // frame the finished crystal once
  setFrame(0);                                    // then start at pure noise
}

function buildLegend(crystal) {
  const last = crystal.frames[crystal.frames.length - 1];
  const counts = {};
  for (const e of last.elems) counts[e] = (counts[e] || 0) + 1;
  const box = $("legendItems");
  box.innerHTML = "";
  Object.keys(counts).sort().forEach((sym) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.textContent = sym;
    box.appendChild(item);
  });
}

function setFrame(idx) {
  frame = Math.max(0, Math.min(idx, DATA.crystals[cur].frames.length - 1));
  $("slider").value = String(frame);
  renderFrame(frame);
}

function stopPlay() {
  playing = false;
  $("play").textContent = "▶";
  clearInterval(timer);
}

function togglePlay() {
  if (playing) { stopPlay(); return; }
  const n = DATA.crystals[cur].frames.length;
  if (frame >= n - 1) setFrame(0);   // replay from noise if parked at the end
  playing = true;
  $("play").textContent = "⏸";
  timer = setInterval(() => {
    if (frame >= n - 1) { stopPlay(); return; }  // pause on the final crystal
    setFrame(frame + 1);
  }, FRAME_MS);
}

function init() {
  viewer = $3Dmol.createViewer($("viewer"), { backgroundColor: "white" });

  fetch("data/trajectories.json")
    .then((r) => r.json())
    .then((d) => {
      DATA = d;
      const m = d.meta;
      const tc = (m.tc_target_actual != null)
        ? `${m.tc_target_actual.toFixed(1)} K` : `scaled ${m.band_gap_scaled}`;
      $("meta").innerHTML =
        `target T<sub>c</sub> = ${tc} &nbsp;·&nbsp; guide&nbsp;w = ${m.guide_w}` +
        ` &nbsp;·&nbsp; ${m.n_steps_total} diffusion steps`;
      const sel = $("crystalSel");
      d.crystals.forEach((c, i) => {
        const o = document.createElement("option");
        o.value = String(i);
        o.textContent = `#${i + 1} — ${c.formula}`;
        sel.appendChild(o);
      });
      sel.onchange = () => { stopPlay(); loadCrystal(parseInt(sel.value, 10)); };
      $("slider").oninput = (e) => { stopPlay(); setFrame(parseInt(e.target.value, 10)); };
      $("play").onclick = togglePlay;
      loadCrystal(0);
    })
    .catch((err) => {
      $("viewer").innerHTML =
        `<div class="err">Could not load <code>data/trajectories.json</code>.<br>` +
        `Serve the folder locally: <code>python -m http.server -d docs</code><br>` +
        `<small>${err}</small></div>`;
    });
}

window.addEventListener("load", init);
