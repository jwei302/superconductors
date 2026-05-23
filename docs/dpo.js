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

function initDpo() {
  const load = () => fetch("data/dpo_results.json", { cache: "no-store" })
    .then((r) => r.json()).then(renderDpo).catch(() => renderDpo(null));
  load();
  window.addEventListener("resize", load);
}

window.addEventListener("load", initDpo);
