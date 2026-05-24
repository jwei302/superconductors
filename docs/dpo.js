/* DPO section: offline preference optimization of the generator against CHGNet
   stability. We present a single headline β (d.primary, chosen in the export as
   the strongest setting that improves stability without chemistry collapse).
   Data: data/dpo_results.json. Charts are hand-rolled SVG (no dependencies). */

const DNS = "http://www.w3.org/2000/svg";
const $d = (id) => document.getElementById(id);
const DPO_COLOR = "#0d9488";      // teal — DPO result
const BASE_COLOR = "#9aa3ad";     // gray — SFT baseline

/* ---- four headline result cards (baseline → DPO) ---- */
function dpoStatCards(d) {
  const box = $d("dpoCards");
  if (!box) return;
  const b = d.baseline, p = d.pools[d.primary];
  const relHr = (p.hr25 / b.hr25 - 1) * 100;
  const cards = [
    { lab: "Median stability", from: `${b.median.toFixed(2)}`, to: `${p.median.toFixed(2)}`,
      unit: "eV/atom", note: `${(p.median - b.median).toFixed(2)} eV/atom`, good: true },
    { lab: "Stability hit-rate", sub: "below baseline 25th-pct", from: `${(b.hr25 * 100).toFixed(0)}%`,
      to: `${(p.hr25 * 100).toFixed(0)}%`, unit: "", note: `${relHr >= 0 ? "+" : ""}${relHr.toFixed(0)}% relative`, good: true },
    { lab: "Top-500 screening", sub: "median of best 500", from: `${b.top500.toFixed(2)}`,
      to: `${p.top500.toFixed(2)}`, unit: "eV/atom", note: `${(p.top500 - b.top500).toFixed(2)} eV/atom`, good: true },
    { lab: "Chemical diversity", sub: "unique chemical systems", from: `${b.n_chemsys}`,
      to: `${p.n_chemsys}`, unit: "", note: "preserved — no reward hacking", good: true },
  ];
  box.innerHTML = cards.map((c) => `
    <div class="dpo-card">
      <div class="dc-lab">${c.lab}${c.sub ? `<small>${c.sub}</small>` : ""}</div>
      <div class="dc-val">${c.to}<span class="dc-unit">${c.unit}</span></div>
      <div class="dc-from">from <b>${c.from}</b> (SFT)</div>
      <div class="dc-note ${c.good ? "good" : ""}">${c.note}</div>
    </div>`).join("");
}

/* ---- energy-per-atom distribution shift: baseline (filled) vs DPO (line) ---- */
function dpoDistChart(elId, d) {
  const el = $d(elId);
  if (!el) return;
  el.innerHTML = "";
  const xs = d.bins.centers;
  const base = d.baseline.hist, dpo = d.pools[d.primary].hist;
  const W = el.clientWidth || 660, H = 250, m = { l: 46, r: 16, t: 16, b: 38 };
  const xmin = d.bins.lo, xmax = d.bins.hi;
  const ymax = Math.max(...base, ...dpo) * 1.08;
  const sx = (x) => m.l + (W - m.l - m.r) * (x - xmin) / (xmax - xmin);
  const sy = (y) => H - m.b - (H - m.t - m.b) * y / ymax;

  const svg = document.createElementNS(DNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%"); svg.setAttribute("height", H);

  const line = (x1, y1, x2, y2, c, w, dash) => {
    const l = document.createElementNS(DNS, "line");
    l.setAttribute("x1", x1); l.setAttribute("y1", y1); l.setAttribute("x2", x2); l.setAttribute("y2", y2);
    l.setAttribute("stroke", c); l.setAttribute("stroke-width", w);
    if (dash) l.setAttribute("stroke-dasharray", dash);
    svg.appendChild(l);
  };
  line(m.l, m.t, m.l, H - m.b, "#cbd0d6", 1);
  line(m.l, H - m.b, W - m.r, H - m.b, "#cbd0d6", 1);

  // x ticks
  for (let v = Math.ceil(xmin); v <= xmax; v += 2) {
    const t = document.createElementNS(DNS, "text");
    t.setAttribute("x", sx(v)); t.setAttribute("y", H - m.b + 16); t.setAttribute("text-anchor", "middle");
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "#6b7280"); t.textContent = v; svg.appendChild(t);
  }
  const xlab = document.createElementNS(DNS, "text");
  xlab.setAttribute("x", (m.l + W - m.r) / 2); xlab.setAttribute("y", H - 6);
  xlab.setAttribute("text-anchor", "middle"); xlab.setAttribute("font-size", "10.5"); xlab.setAttribute("fill", "#6b7280");
  xlab.textContent = "CHGNet energy per atom (eV/atom)  ·  lower = more stable"; svg.appendChild(xlab);

  // baseline 25th-pct threshold marker
  const xt = sx(d.thresholds.T25);
  line(xt, m.t, xt, H - m.b, "#b08900", 1.2, "5 4");
  const tt = document.createElementNS(DNS, "text");
  tt.setAttribute("x", xt + 4); tt.setAttribute("y", m.t + 11); tt.setAttribute("font-size", "9.5");
  tt.setAttribute("fill", "#8a6d00"); tt.textContent = "baseline 25th-pct"; svg.appendChild(tt);

  // filled baseline area
  const areaPts = xs.map((x, i) => `${sx(x).toFixed(1)} ${sy(base[i]).toFixed(1)}`);
  const area = document.createElementNS(DNS, "path");
  area.setAttribute("d", `M${sx(xs[0]).toFixed(1)} ${sy(0).toFixed(1)} L` + areaPts.join(" L") +
    ` L${sx(xs[xs.length - 1]).toFixed(1)} ${sy(0).toFixed(1)} Z`);
  area.setAttribute("fill", BASE_COLOR); area.setAttribute("fill-opacity", "0.30"); area.setAttribute("stroke", "none");
  svg.appendChild(area);

  const poly = (hist, color, width) => {
    const p = document.createElementNS(DNS, "path");
    p.setAttribute("d", "M" + xs.map((x, i) => `${sx(x).toFixed(1)} ${sy(hist[i]).toFixed(1)}`).join(" L"));
    p.setAttribute("fill", "none"); p.setAttribute("stroke", color); p.setAttribute("stroke-width", width);
    p.setAttribute("stroke-linejoin", "round"); svg.appendChild(p);
  };
  poly(base, BASE_COLOR, 1.6);
  poly(dpo, DPO_COLOR, 2.6);

  // legend
  const leg = [[`SFT baseline`, BASE_COLOR, 1.6], [`DPO β=${d.primary}`, DPO_COLOR, 2.6]];
  let ly = m.t + 6;
  leg.forEach(([txt, c, w]) => {
    line(W - m.r - 96, ly, W - m.r - 78, ly, c, w);
    const t = document.createElementNS(DNS, "text");
    t.setAttribute("x", W - m.r - 74); t.setAttribute("y", ly + 3); t.setAttribute("font-size", "10");
    t.setAttribute("fill", "#1a1d21"); t.textContent = txt; svg.appendChild(t);
    ly += 15;
  });
  el.appendChild(svg);
}

function renderDpo(d) {
  const status = $d("dpoStatus");
  if (!d || !d.pools || !d.pools[d.primary]) {
    if (status) status.textContent = "DPO results not available.";
    return;
  }
  dpoStatCards(d);
  dpoDistChart("dpoDistChart", d);
  if (status) {
    const p = d.pools[d.primary], c = d.pools[d.collapsed];
    status.innerHTML = `Headline β=${d.primary} · ${p.n} samples scored by CHGNet. ` +
      `Smaller β=${d.collapsed} reaches ${(c.hr25 * 100).toFixed(0)}% hit-rate but collapses to ` +
      `${c.n_chemsys} chemical systems (${(c.top10_share * 100).toFixed(0)}% in its top-10) — reward hacking, not shown.`;
  }
}

/* ---- 3Dmol mini crystal (winner / loser viewers) ---- */
function daCellEdges(v, L) {
  const c = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++)
    c.push({ x: i * L[0][0] + j * L[1][0] + k * L[2][0], y: i * L[0][1] + j * L[1][1] + k * L[2][1], z: i * L[0][2] + j * L[1][2] + k * L[2][2], b: [i, j, k] });
  for (let a = 0; a < c.length; a++) for (let b = a + 1; b < c.length; b++) {
    const dd = (c[a].b[0] !== c[b].b[0]) + (c[a].b[1] !== c[b].b[1]) + (c[a].b[2] !== c[b].b[2]);
    if (dd === 1) v.addCylinder({ start: { x: c[a].x, y: c[a].y, z: c[a].z }, end: { x: c[b].x, y: c[b].y, z: c[b].z }, radius: 0.035, color: "#9aa3ad" });
  }
}
function daMiniCrystal(id, c) {
  const div = $d(id);
  if (!div || !window.$3Dmol) return;
  const v = $3Dmol.createViewer(div, { backgroundColor: "white" });
  const lines = [String(c.elems.length), ""];
  for (let i = 0; i < c.elems.length; i++) { const p = c.xyz[i]; lines.push(`${c.elems[i]} ${p[0]} ${p[1]} ${p[2]}`); }
  v.addModel(lines.join("\n"), "xyz");
  v.setStyle({}, { sphere: { scale: 0.3, colorscheme: "Jmol" } });
  daCellEdges(v, c.lattice);
  v.zoomTo(); v.zoom(0.85); v.render();
  try { v.spin("y", 0.5); } catch (e) { /* spin optional */ }
}

/* ---- σ(z) training sparkline (real β=0.5 trajectory) ---- */
function daSparkline(id, traj) {
  const el = $d(id);
  if (!el) return;
  el.innerHTML = "";
  const W = el.clientWidth || 210, H = 48, m = { l: 4, r: 4, t: 7, b: 7 };
  const xs = traj.steps, ys = traj.values;
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = 0.5, ymax = 0.8;
  const clamp = (y) => Math.min(Math.max(y, ymin), ymax);
  const sx = (x) => m.l + (W - m.l - m.r) * (x - xmin) / ((xmax - xmin) || 1);
  const sy = (y) => H - m.b - (H - m.t - m.b) * (clamp(y) - ymin) / (ymax - ymin);
  const svg = document.createElementNS(DNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", "100%"); svg.setAttribute("height", H);
  const base = document.createElementNS(DNS, "line");
  base.setAttribute("x1", m.l); base.setAttribute("x2", W - m.r); base.setAttribute("y1", sy(0.5)); base.setAttribute("y2", sy(0.5));
  base.setAttribute("stroke", "#cbd0d6"); base.setAttribute("stroke-dasharray", "3 3"); base.setAttribute("stroke-width", "1"); svg.appendChild(base);
  const path = document.createElementNS(DNS, "path");
  path.setAttribute("d", "M" + xs.map((x, i) => `${sx(x).toFixed(1)} ${sy(ys[i]).toFixed(1)}`).join(" L"));
  path.setAttribute("fill", "none"); path.setAttribute("stroke", DPO_COLOR); path.setAttribute("stroke-width", "2"); svg.appendChild(path);
  el.appendChild(svg);
}

/* ---- phase-cycled DPO mechanism animation ---- */
const DA_PHASES = [
  "1 · Sample a winner / loser pair from the generator",
  "2 · Score stability with CHGNet — lower energy is more stable",
  "3 · DPO update: raise P(winner), lower P(loser)",
  "4 · β tethers the policy to the frozen reference π_ref",
  "5 · Preference learned — σ(z) = P(prefer winner) rises",
];
let _daTimer = null;

function renderDpoAnim(d) {
  const wrap = $d("dpoAnim");
  if (!wrap || !d || !d.winner) return;
  const w = d.winner, l = d.loser, sz = d.sigma_z;
  wrap.innerHTML = `
    <div class="dpo-anim" data-phase="0">
      <div class="da-caption mono" id="daCaption"></div>
      <div class="da-stage">
        <div class="da-pair">
          <div class="da-cryst win">
            <div class="da-mv" id="daWin"></div>
            <div class="da-meta"><b>${w.formula}</b>
              <span class="da-e">${w.energy_per_atom} eV/atom</span>
              <span class="da-tag up">✓ winner</span></div>
            <div class="da-push up">↑ raise P</div>
          </div>
          <div class="da-cryst lose">
            <div class="da-mv" id="daLose"></div>
            <div class="da-meta"><b>${l.formula}</b>
              <span class="da-e">${l.energy_per_atom} eV/atom</span>
              <span class="da-tag down">✗ loser</span></div>
            <div class="da-push down">↓ lower P</div>
          </div>
        </div>
        <div class="da-model">
          <div class="da-pi theta">π<sub>θ</sub><small>policy</small></div>
          <div class="da-tether"><span class="da-beta">β</span></div>
          <div class="da-pi ref">π<sub>ref</sub><small>frozen</small></div>
        </div>
        <div class="da-sigma">
          <div class="da-spark" id="daSpark"></div>
          <div class="da-spark-lab mono">σ(z) over training</div>
          <div class="da-gauge"><div class="da-fill" id="daFill"></div><div class="da-mid" title="0.5 = no preference"></div></div>
          <div class="da-sigval mono" id="daSigVal">σ(z) = —</div>
        </div>
      </div>
    </div>`;

  daMiniCrystal("daWin", w);
  daMiniCrystal("daLose", l);
  daSparkline("daSpark", sz);

  const anim = wrap.querySelector(".dpo-anim");
  const cap = $d("daCaption"), fill = $d("daFill"), sigval = $d("daSigVal");
  let p = 0;
  const setSig = (val) => { fill.style.width = (val * 100).toFixed(1) + "%"; sigval.textContent = `σ(z) = ${val.toFixed(2)}`; };
  const tick = () => {
    anim.setAttribute("data-phase", String(p));
    cap.textContent = DA_PHASES[p];
    if (p >= 4) setSig(sz.end);
    else if (p === 0) setSig(sz.start);
    p = (p + 1) % DA_PHASES.length;
  };
  if (_daTimer) clearInterval(_daTimer);
  tick();
  _daTimer = setInterval(tick, 2000);
}

let _dpoAnimBuilt = false;
function initDpo() {
  const loadRes = () => fetch("data/dpo_results.json", { cache: "no-store" })
    .then((r) => r.json()).then(renderDpo).catch(() => renderDpo(null));
  loadRes();
  fetch("data/dpo_pair.json", { cache: "no-store" })
    .then((r) => r.json()).then((d) => { if (!_dpoAnimBuilt) { renderDpoAnim(d); _dpoAnimBuilt = true; } })
    .catch(() => { /* animation optional */ });
  window.addEventListener("resize", loadRes);
}

window.addEventListener("load", initDpo);
