/* GRPO section: an illustrative group + live reward curves from training.
   Data: data/grpo_progress.json (refreshed during the run by
   viz/export_grpo_progress.py). Charts are hand-rolled SVG (no dependencies). */

const SVGNS = "http://www.w3.org/2000/svg";
const $g = (id) => document.getElementById(id);

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
  const cands = [14.8, 11.2, 9.7, 6.1, 12.5, 3.4];          // predicted Tc of a group
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

/* ---- minimal SVG line chart ---- */
function lineChart(elId, xs, ys, opts) {
  const el = $g(elId);
  if (!el) return;
  el.innerHTML = "";
  opts = opts || {};
  const W = el.clientWidth || 660, H = 220;
  const m = { l: 48, r: 16, t: 12, b: 30 };
  const pts = xs.map((x, i) => [x, ys[i]]).filter((p) => p[1] != null && p[1] === p[1]);
  if (pts.length < 2) {
    el.innerHTML = `<div class="chart-empty">waiting for training data…</div>`;
    return;
  }
  const xmin = Math.min(...pts.map((p) => p[0])), xmax = Math.max(...pts.map((p) => p[0]));
  let ymin = Math.min(...pts.map((p) => p[1])), ymax = Math.max(...pts.map((p) => p[1]));
  if (opts.target != null) { ymin = Math.min(ymin, opts.target); ymax = Math.max(ymax, opts.target); }
  const pad = (ymax - ymin) * 0.1 || 1; ymin -= pad; ymax += pad;
  const sx = (x) => m.l + (W - m.l - m.r) * (x - xmin) / (xmax - xmin || 1);
  const sy = (y) => H - m.b - (H - m.t - m.b) * (y - ymin) / (ymax - ymin || 1);

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%"); svg.setAttribute("height", H);

  // axes
  const axis = (x1, y1, x2, y2) => {
    const l = document.createElementNS(SVGNS, "line");
    l.setAttribute("x1", x1); l.setAttribute("y1", y1); l.setAttribute("x2", x2); l.setAttribute("y2", y2);
    l.setAttribute("stroke", "#cbd0d6"); l.setAttribute("stroke-width", "1"); svg.appendChild(l);
  };
  axis(m.l, m.t, m.l, H - m.b); axis(m.l, H - m.b, W - m.r, H - m.b);

  // y ticks
  for (let i = 0; i <= 4; i++) {
    const yv = ymin + (ymax - ymin) * i / 4;
    const yy = sy(yv);
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", m.l - 8); t.setAttribute("y", yy + 3); t.setAttribute("text-anchor", "end");
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "#6b7280");
    t.textContent = yv.toFixed(Math.abs(ymax - ymin) < 5 ? 2 : 1); svg.appendChild(t);
  }
  // x label endpoints
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

  // target dashed line
  if (opts.target != null) {
    const yt = sy(opts.target);
    const l = document.createElementNS(SVGNS, "line");
    l.setAttribute("x1", m.l); l.setAttribute("y1", yt); l.setAttribute("x2", W - m.r); l.setAttribute("y2", yt);
    l.setAttribute("stroke", "#e0a64a"); l.setAttribute("stroke-width", "1.3"); l.setAttribute("stroke-dasharray", "5 4");
    svg.appendChild(l);
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", W - m.r); t.setAttribute("y", yt - 4); t.setAttribute("text-anchor", "end");
    t.setAttribute("font-size", "10"); t.setAttribute("fill", "#c9871f");
    t.textContent = `target ${opts.target} K`; svg.appendChild(t);
  }

  // line(s): raw (light) + optional moving-average overlay (bold)
  const col = opts.color || "#2563eb";
  function drawSeries(ys2, width, opacity) {
    const p2 = xs.map((x, i) => [x, ys2[i]]).filter((p) => p[1] != null && p[1] === p[1]);
    if (p2.length < 2) return;
    const dd = p2.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join(" ");
    const pa = document.createElementNS(SVGNS, "path");
    pa.setAttribute("d", dd); pa.setAttribute("fill", "none");
    pa.setAttribute("stroke", col); pa.setAttribute("stroke-width", width);
    pa.setAttribute("stroke-opacity", opacity);
    svg.appendChild(pa);
  }
  drawSeries(ys, 1.2, opts.ma ? 0.32 : 1);
  if (opts.ma) drawSeries(opts.ma, 2.3, 1);
  el.appendChild(svg);
}

function renderCharts(d) {
  const status = $g("grpoStatus");
  if (!d || d.status === "waiting" || (d.n || 0) === 0) {
    if (status) status.textContent = "GRPO run not started yet — charts will populate once training begins.";
    lineChart("rewardChart", [], []); lineChart("tcadChart", [], []);
    return;
  }
  lineChart("rewardChart", d.updates, d.reward, { color: "#2563eb", ma: movingAverage(d.reward, 6) });
  lineChart("tcadChart", d.updates, d.tcad, { color: "#16a34a", target: d.target, ma: movingAverage(d.tcad, 6) });
  if (status) {
    const lastTc = [...(d.tcad || [])].reverse().find((v) => v != null && v === v);
    status.innerHTML = `${d.n} updates logged` +
      (d.target != null ? ` · target T<sub>c</sub> = ${d.target} K` : "") +
      (lastTc != null ? ` · latest mean predicted T<sub>c</sub> = ${lastTc.toFixed(1)} K` : "");
  }
}

function initGrpo() {
  renderGroup();
  fetch("data/grpo_progress.json", { cache: "no-store" })
    .then((r) => r.json())
    .then(renderCharts)
    .catch(() => renderCharts(null));
}

window.addEventListener("load", initGrpo);
window.addEventListener("resize", () => {
  fetch("data/grpo_progress.json", { cache: "no-store" }).then((r) => r.json()).then(renderCharts).catch(() => {});
});
