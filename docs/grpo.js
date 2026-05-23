/* GRPO section: an illustrative group + live reward curves from training.
   Data: data/grpo_progress.json (refreshed during runs by
   viz/export_grpo_progress.py). Charts are hand-rolled SVG (no dependencies).
   Supports multiple runs (e.g. stable vs high-LR) as overlaid series. */

const SVGNS = "http://www.w3.org/2000/svg";
const $g = (id) => document.getElementById(id);
const RUN_COLORS = { grpo_beeonly: "#2563eb", grpo_bee_hilr: "#dc2626" };
const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"];

function movingAverage(ys, w) {
  const out = [];
  for (let i = 0; i < ys.length; i++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, i - w + 1); j <= i; j++) {
      if (ys[j] != null && ys[j] === ys[j]) { s += ys[j]; c++; }
    }
    out.push(c ? s / c : null);
  }
  return out;
}

/* ---- illustrative GRPO group (shows the "group relative" idea) ---- */
function renderGroup() {
  const target = 10.0;
  const cands = [14.8, 11.2, 9.7, 6.1, 12.5, 3.4];
  const mean = cands.reduce((a, b) => a + b, 0) / cands.length;
  const box = $g("grpoGroup");
  if (!box) return;
  box.innerHTML = `<div class="group-label mono">one group · target T<sub>c</sub> = ${target} K · group mean = ${mean.toFixed(1)} K</div>`;
  const row = document.createElement("div");
  row.className = "chips";
  cands.forEach((tc) => {
    const adv = tc - mean;
    const chip = document.createElement("div");
    chip.className = "chip " + (adv >= 0 ? "up" : "down");
    chip.innerHTML = `<b>${tc.toFixed(1)} K</b><small>${adv >= 0 ? "+" : ""}${adv.toFixed(1)}</small>`;
    row.appendChild(chip);
  });
  box.appendChild(row);
  const cap = document.createElement("div");
  cap.className = "group-cap";
  cap.innerHTML = `<span class="sw up"></span>above mean → reinforced &nbsp;&nbsp; <span class="sw down"></span>below mean → suppressed`;
  box.appendChild(cap);
}

/* ---- multi-series SVG line chart (each series: {x, y, color, label, ma}) ---- */
function lineChart(elId, series, opts) {
  const el = $g(elId);
  if (!el) return;
  el.innerHTML = "";
  opts = opts || {};
  series = (series || []).filter((s) => s && s.x && s.y);
  const W = el.clientWidth || 660, H = 220, m = { l: 48, r: 16, t: 14, b: 30 };

  const all = [];
  series.forEach((s) => s.x.forEach((x, i) => {
    if (s.y[i] != null && s.y[i] === s.y[i]) all.push([x, s.y[i]]);
  }));
  if (all.length < 2) { el.innerHTML = `<div class="chart-empty">waiting for training data…</div>`; return; }

  const xmin = Math.min(...all.map((p) => p[0])), xmax = Math.max(...all.map((p) => p[0]));
  let ymin = Math.min(...all.map((p) => p[1])), ymax = Math.max(...all.map((p) => p[1]));
  if (opts.target != null) { ymin = Math.min(ymin, opts.target); ymax = Math.max(ymax, opts.target); }
  const pad = (ymax - ymin) * 0.1 || 1; ymin -= pad; ymax += pad;
  const sx = (x) => m.l + (W - m.l - m.r) * (x - xmin) / ((xmax - xmin) || 1);
  const sy = (y) => H - m.b - (H - m.t - m.b) * (y - ymin) / ((ymax - ymin) || 1);

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%"); svg.setAttribute("height", H);

  const axis = (x1, y1, x2, y2) => {
    const l = document.createElementNS(SVGNS, "line");
    l.setAttribute("x1", x1); l.setAttribute("y1", y1); l.setAttribute("x2", x2); l.setAttribute("y2", y2);
    l.setAttribute("stroke", "#cbd0d6"); l.setAttribute("stroke-width", "1"); svg.appendChild(l);
  };
  axis(m.l, m.t, m.l, H - m.b); axis(m.l, H - m.b, W - m.r, H - m.b);

  for (let i = 0; i <= 4; i++) {
    const yv = ymin + (ymax - ymin) * i / 4;
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", m.l - 8); t.setAttribute("y", sy(yv) + 3); t.setAttribute("text-anchor", "end");
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "#6b7280");
    t.textContent = yv.toFixed(Math.abs(ymax - ymin) < 5 ? 2 : 1); svg.appendChild(t);
  }
  [[xmin, m.l], [xmax, W - m.r]].forEach(([xv, xx]) => {
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", xx); t.setAttribute("y", H - m.b + 18); t.setAttribute("text-anchor", "middle");
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "#6b7280");
    t.textContent = Math.round(xv); svg.appendChild(t);
  });
  const xlab = document.createElementNS(SVGNS, "text");
  xlab.setAttribute("x", (m.l + W - m.r) / 2); xlab.setAttribute("y", H - 4);
  xlab.setAttribute("text-anchor", "middle"); xlab.setAttribute("font-size", "10"); xlab.setAttribute("fill", "#6b7280");
  xlab.textContent = "update"; svg.appendChild(xlab);

  if (opts.target != null) {
    const yt = sy(opts.target);
    const l = document.createElementNS(SVGNS, "line");
    l.setAttribute("x1", m.l); l.setAttribute("y1", yt); l.setAttribute("x2", W - m.r); l.setAttribute("y2", yt);
    l.setAttribute("stroke", "#e0a64a"); l.setAttribute("stroke-width", "1.3"); l.setAttribute("stroke-dasharray", "5 4");
    svg.appendChild(l);
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", W - m.r); t.setAttribute("y", yt - 4); t.setAttribute("text-anchor", "end");
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "#c9871f");
    t.textContent = `target ${(+opts.target).toFixed(1)} K`; svg.appendChild(t);
  }

  function path(xy, color, width, opacity) {
    if (xy.length < 2) return;
    const dd = xy.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ");
    const pa = document.createElementNS(SVGNS, "path");
    pa.setAttribute("d", dd); pa.setAttribute("fill", "none");
    pa.setAttribute("stroke", color); pa.setAttribute("stroke-width", width); pa.setAttribute("stroke-opacity", opacity);
    svg.appendChild(pa);
  }
  series.forEach((s) => {
    const raw = s.x.map((x, i) => [x, s.y[i]]).filter((p) => p[1] != null && p[1] === p[1]);
    path(raw, s.color, 1.2, s.ma ? 0.3 : 1);
    if (s.ma) {
      const ma = movingAverage(s.y, 6);
      path(s.x.map((x, i) => [x, ma[i]]).filter((p) => p[1] != null && p[1] === p[1]), s.color, 2.3, 1);
    }
  });

  // legend (top-left)
  if (series.length && series.some((s) => s.label)) {
    let ly = m.t + 4;
    series.forEach((s) => {
      const ln = document.createElementNS(SVGNS, "line");
      ln.setAttribute("x1", m.l + 6); ln.setAttribute("y1", ly); ln.setAttribute("x2", m.l + 24); ln.setAttribute("y2", ly);
      ln.setAttribute("stroke", s.color); ln.setAttribute("stroke-width", "2.3"); svg.appendChild(ln);
      const tx = document.createElementNS(SVGNS, "text");
      tx.setAttribute("x", m.l + 28); tx.setAttribute("y", ly + 3); tx.setAttribute("font-size", "10"); tx.setAttribute("fill", "#1a1d21");
      tx.textContent = s.label; svg.appendChild(tx);
      ly += 14;
    });
  }
  el.appendChild(svg);
}

function renderCharts(d) {
  const status = $g("grpoStatus");
  let runs = (d && d.runs) ? d.runs
    : (d && d.updates ? [{ name: "run", key: "run", updates: d.updates, reward: d.reward, tcad: d.tcad, target: d.target, n: d.n }] : []);
  runs = runs.filter((r) => r && (r.n || 0) > 0);
  if (!runs.length) {
    if (status) status.textContent = "GRPO run not started yet — charts will populate once training begins.";
    lineChart("rewardChart", []); lineChart("tcadChart", []);
    return;
  }
  const mk = (yk) => runs.map((r, i) => ({
    x: r.updates, y: r[yk], ma: true,
    color: RUN_COLORS[r.key] || PALETTE[i % PALETTE.length],
    label: r.name,
  }));
  lineChart("rewardChart", mk("reward"), {});
  const tgt = runs.map((r) => r.target).find((t) => t != null);
  lineChart("tcadChart", mk("tcad"), { target: tgt });
  if (status) {
    status.innerHTML = runs.map((r) => {
      const last = [...(r.tcad || [])].reverse().find((v) => v != null && v === v);
      return `<b>${r.name}</b>: ${r.n} updates` + (last != null ? `, latest mean T<sub>c</sub> ≈ ${last.toFixed(1)} K` : "");
    }).join(" &nbsp;·&nbsp; ");
  }
}

/* ---- gallery of real generated crystals (GRPO trains the generator) ---- */
function cellEdgesViewer(v, L) {
  const c = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++)
    c.push({ x: i * L[0][0] + j * L[1][0] + k * L[2][0], y: i * L[0][1] + j * L[1][1] + k * L[2][1], z: i * L[0][2] + j * L[1][2] + k * L[2][2], b: [i, j, k] });
  for (let a = 0; a < c.length; a++) for (let b = a + 1; b < c.length; b++) {
    const dd = (c[a].b[0] !== c[b].b[0]) + (c[a].b[1] !== c[b].b[1]) + (c[a].b[2] !== c[b].b[2]);
    if (dd === 1) v.addCylinder({ start: { x: c[a].x, y: c[a].y, z: c[a].z }, end: { x: c[b].x, y: c[b].y, z: c[b].z }, radius: 0.035, color: "#9aa3ad" });
  }
}

function renderMiniCrystal(id, c) {
  const div = $g(id);
  if (!div || !window.$3Dmol) return;
  const v = $3Dmol.createViewer(div, { backgroundColor: "white" });
  const lines = [String(c.elems.length), ""];
  for (let i = 0; i < c.elems.length; i++) { const p = c.xyz[i]; lines.push(`${c.elems[i]} ${p[0]} ${p[1]} ${p[2]}`); }
  v.addModel(lines.join("\n"), "xyz");
  v.setStyle({}, { sphere: { scale: 0.3, colorscheme: "Jmol" } });
  cellEdgesViewer(v, c.lattice);
  v.zoomTo(); v.zoom(0.85); v.render();
}

function renderSamples(d) {
  const wrap = $g("grpoSamples");
  if (!wrap) return;
  if (!d || (!d.baseline && !d.grpo)) {
    wrap.innerHTML = `<div class="chart-empty">generating sample crystals…</div>`;
    return;
  }
  const grp = d.grpo || d.baseline;
  const tgt = d.target_tc != null ? `${d.target_tc.toFixed(1)} K` : "";
  const label = d.grpo ? `GRPO-tuned generator (update ${d.grpo.update})` : "SFT baseline generator";
  let html = "";
  if (d.baseline && d.grpo) {
    html += `<p class="samples-stat mono">SFT baseline group mean T<sub>c</sub> = ${d.baseline.mean_tcad} K &nbsp;&rarr;&nbsp; GRPO group mean = ${d.grpo.mean_tcad} K &nbsp;·&nbsp; target ${tgt}</p>`;
  }
  html += `<div class="samples-label">${label} — one group at target T<sub>c</sub> ${tgt} (group mean ${grp.mean_tcad} K). Green beats the group mean (reinforced), red is below (suppressed).</div>`;
  html += `<div class="samples-grid">`;
  const show = grp.crystals.slice(0, 6);
  show.forEach((c, i) => {
    const cls = c.adv == null ? "" : (c.adv >= 0 ? "up" : "down");
    const tc = c.tcad != null ? `${c.tcad} K` : "invalid";
    const adv = c.adv != null ? ` (${c.adv >= 0 ? "+" : ""}${c.adv})` : "";
    html += `<div class="sample-card"><div class="mini-viewer" id="mv${i}"></div>` +
      `<div class="sample-cap"><b>${c.formula}</b><span class="tc ${cls}">${tc}${adv}</span></div></div>`;
  });
  html += `</div>`;
  wrap.innerHTML = html;
  show.forEach((c, i) => renderMiniCrystal(`mv${i}`, c));
}

function initGrpo() {
  renderGroup();
  const load = () => fetch("data/grpo_progress.json", { cache: "no-store" })
    .then((r) => r.json()).then(renderCharts).catch(() => renderCharts(null));
  load();
  fetch("data/grpo_samples.json", { cache: "no-store" })
    .then((r) => r.json()).then(renderSamples).catch(() => renderSamples(null));
  window.addEventListener("resize", load);
}

window.addEventListener("load", initGrpo);
