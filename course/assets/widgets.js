/* ============================================================
   widgets.js — the Evaluation course's own widgets.

   Loaded after the shared viz.js, and registered into its VIZ table, so they
   reuse its helpers (panel, ctlRow, slider, svgEl, svgText, C) and its theme
   handling: a theme change rebuilds every widget and replays its controls.

   Colour: this course's five hues mean ROLES in an evaluation
     --dim-1 the system under test   --dim-2 the baseline / reference
     --dim-3 human labels            --dim-4 the judge
     --dim-5 the uncertainty
   A widget uses a hue only where it draws that role; otherwise accent/stamp.

   Every formula here is the one in evalkit/ and is asserted by
   tests/test_evalkit.py; the widgets only draw it.
   ============================================================ */

const EV = (() => {
  const s = () => getComputedStyle(document.documentElement);
  const role = n => (s().getPropertyValue("--dim-" + n) || "").trim() || C.accent;
  return { sut: () => role(1), base: () => role(2), human: () => role(3), judge: () => role(4), unc: () => role(5) };
})();

/* standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9) */
function evNormInv(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  let q, r;
  if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  q = p - 0.5; r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/* A line chart with fixed geometry. series: [{name, ys, col, dash}], marker: {x, y, label}. */
function evChart(host, o) {
  host.innerHTML = "";
  const W = 620, H = 270, L = 58, R = 18, T = 14, B = 44;
  const xs = o.xs, all = o.series.flatMap(s => s.ys).filter(Number.isFinite);
  const xmin = o.xmin ?? Math.min(...xs), xmax = o.xmax ?? Math.max(...xs);
  const ymin = o.ymin ?? 0, ymax = o.ymax ?? Math.max(...all) * 1.08;
  const X = v => L + (v - xmin) / (xmax - xmin) * (W - L - R);
  const Y = v => H - B - (v - ymin) / (ymax - ymin) * (H - T - B);
  const sv = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": o.label || "chart" });
  const ticks = (lo, hi, n) => { const st = (hi - lo) / n; return Array.from({ length: n + 1 }, (_, i) => lo + i * st); };
  for (const v of ticks(ymin, ymax, 4)) {
    sv.appendChild(svgEl("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: C.grid, "stroke-width": 1 }));
    svgText(sv, L - 6, Y(v) + 4, (o.yfmt || (x => +x.toFixed(2)))(v), C.faint, { anchor: "end", size: 11, weight: 500 });
  }
  for (const v of (o.xticks || ticks(xmin, xmax, 5)))
    svgText(sv, X(v), H - B + 17, (o.xfmt || (x => String(Math.round(x))))(v), C.faint, { anchor: "middle", size: 11, weight: 500 });
  sv.appendChild(svgEl("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: C.gridHi, "stroke-width": 1.2 }));
  svgText(sv, (L + W - R) / 2, H - 6, o.xlabel || "", C.dim, { anchor: "middle", size: 12 });
  const yl = svgText(sv, 15, (T + H - B) / 2, o.ylabel || "", C.dim, { anchor: "middle", size: 12 });
  yl.setAttribute("transform", `rotate(-90 15 ${(T + H - B) / 2})`);
  o.series.forEach(s => {
    const pts = xs.map((x, i) => [X(x), Y(s.ys[i])]).filter(p => Number.isFinite(p[1]));
    sv.appendChild(svgEl("polyline", { points: pts.map(p => p.join(",")).join(" "), fill: "none",
      stroke: s.col, "stroke-width": 2.6, "stroke-dasharray": s.dash || "" }));
    if (s.name) { const p = pts[pts.length - 1]; svgText(sv, p[0] - 4, p[1] - 8, s.name, s.col, { anchor: "end", size: 12, weight: 700 }); }
  });
  (o.markers || []).forEach(m => {
    sv.appendChild(svgEl("line", { x1: X(m.x), x2: X(m.x), y1: Y(ymin), y2: Y(m.y), stroke: C.stamp, "stroke-width": 1.2, "stroke-dasharray": "3 3" }));
    sv.appendChild(svgEl("circle", { cx: X(m.x), cy: Y(m.y), r: 5, fill: C.stamp }));
    // label beside the point, below it when the point is high (so it never sits on a rising curve)
    const below = Y(m.y) < T + (H - T - B) * 0.45;
    if (m.label) svgText(sv, X(m.x) + (X(m.x) > W - 180 ? -10 : 10), Y(m.y) + (below ? 22 : -10), m.label, C.stamp,
      { anchor: X(m.x) > W - 180 ? "end" : "start", size: 12, weight: 700 });
  });
  host.appendChild(sv);
  return sv;
}

function evReadout(el, cls) {
  const r = document.createElement("div");
  r.className = "ev-read " + (cls || "");
  r.setAttribute("aria-live", "polite");
  r.style.cssText = "margin-top:10px;font-family:var(--mono);font-size:.82rem;color:var(--ink-2);line-height:1.8";
  el.appendChild(r);
  return r;
}

/* ---------------------------------------------------------------- Ch 1
   How many outputs must you read to see a failure mode at least once?
   P(seen) = 1 - (1-p)^n ;  n for 95% = ceil(ln 0.05 / ln(1-p))          */
function vizReads(el) {
  panel(el, "Reading outputs: what you will and will not see",
    "A failure mode that occurs in a fraction p of outputs. Read n of them: the curve is the chance you see it at least once.");
  const row = ctlRow(el);
  let pm = 20, n = 100;   // p in tenths of a percent
  slider(row, "failure rate (‰)", 2, 200, pm, v => { pm = v; draw(); });
  slider(row, "outputs read", 10, 400, n, v => { n = v; draw(); });
  const host = document.createElement("div"); el.appendChild(host);
  const read = evReadout(el, "reads-read");
  function draw() {
    const p = pm / 1000;
    const xs = Array.from({ length: 81 }, (_, i) => i * 5);
    const ys = xs.map(x => 1 - Math.pow(1 - p, x));
    const seen = 1 - Math.pow(1 - p, n);
    const need = Math.ceil(Math.log(0.05) / Math.log(1 - p));
    evChart(host, { xs, series: [{ ys, col: C.accent }], ymin: 0, ymax: 1, xmin: 0, xmax: 400,
      xlabel: "outputs read (n)", ylabel: "P(seen at least once)", yfmt: v => Math.round(v * 100) + "%",
      markers: [{ x: n, y: seen, label: `n = ${n}: ${(seen * 100).toFixed(1)}%` }],
      label: `Chance of seeing a ${(p * 100).toFixed(1)}% failure mode after reading ${n} outputs: ${(seen * 100).toFixed(1)} percent` });
    read.dataset.seen = seen.toFixed(4); read.dataset.need = need;
    read.textContent = `p = ${(p * 100).toFixed(1)}%  ·  after ${n} reads: ${(seen * 100).toFixed(1)}% chance you saw it, ` +
      `${((1 - seen) * 100).toFixed(1)}% you missed it entirely  ·  reads for 95%: ${need}`;
  }
  draw();
}

/* ---------------------------------------------------------------- Ch 2
   Half-width of a pass-rate interval: z * sqrt(p(1-p)/n)              */
function vizMargin(el) {
  panel(el, "How big must the golden set be?",
    "The 95% interval on a pass rate is ± z·√(p(1−p)/n). It shrinks with √n, so halving it costs four times the items.");
  const row = ctlRow(el);
  let n = 200, pp = 50;
  slider(row, "items (n)", 50, 3000, n, v => { n = v; draw(); });
  slider(row, "pass rate (%)", 5, 95, pp, v => { pp = v; draw(); });
  const host = document.createElement("div"); el.appendChild(host);
  const read = evReadout(el, "margin-read");
  const z = evNormInv(0.975);
  function draw() {
    const p = pp / 100;
    const xs = Array.from({ length: 60 }, (_, i) => 50 + i * 50);
    const hw = x => z * Math.sqrt(p * (1 - p) / x) * 100;
    evChart(host, { xs, series: [{ ys: xs.map(hw), col: EV.unc() }], ymin: 0, ymax: 15, xmin: 50, xmax: 3000,
      xticks: [50, 500, 1000, 1500, 2000, 2500, 3000],
      xlabel: "golden-set size (n)", ylabel: "± points (95%)", yfmt: v => "±" + v.toFixed(1),
      markers: [{ x: n, y: hw(n), label: `±${hw(n).toFixed(1)} pts` }],
      label: `With ${n} items at a ${pp}% pass rate the 95% interval is plus or minus ${hw(n).toFixed(1)} points` });
    const need2 = Math.ceil(z * z * p * (1 - p) / 0.0004);
    read.dataset.hw = hw(n).toFixed(2);
    read.textContent = `n = ${n}, p = ${pp}%  →  ±${hw(n).toFixed(1)} points  ·  for ±2 points you need ${need2.toLocaleString()} items`;
  }
  draw();
}

/* ---------------------------------------------------------------- Ch 3
   Cohen's kappa from a 2x2 table of two raters' pass/fail labels on 100 items.  */
function vizKappa(el) {
  panel(el, "Agreement is not kappa",
    "Two reviewers label the same 100 outputs pass or fail. Observed agreement counts matches; kappa subtracts the matches two reviewers with these pass rates would get by luck.");
  const row = ctlRow(el);
  let bp = 70, ao = 10, bo = 5;
  const sA = slider(row, "both pass", 0, 100, bp, v => { bp = v; fix(); draw(); });
  const sB = slider(row, "only A passes", 0, 50, ao, v => { ao = v; fix(); draw(); });
  const sC = slider(row, "only B passes", 0, 50, bo, v => { bo = v; fix(); draw(); });
  function fix() { if (bp + ao + bo > 100) { bp = 100 - ao - bo; sA.value = bp; sA.nextElementSibling.textContent = bp; } }
  const host = document.createElement("div"); el.appendChild(host);
  const read = evReadout(el, "kappa-read");
  function draw() {
    const bf = 100 - bp - ao - bo, n = 100;
    const po = (bp + bf) / n, pa = (bp + ao) / n, pb = (bp + bo) / n;
    const pe = pa * pb + (1 - pa) * (1 - pb), k = pe < 1 ? (po - pe) / (1 - pe) : NaN;
    host.innerHTML = "";
    const W = 620, H = 150, L = 170, R = 60;
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
      "aria-label": `Observed agreement ${(po * 100).toFixed(0)} percent, chance agreement ${(pe * 100).toFixed(1)} percent, kappa ${k.toFixed(2)}` });
    const X = v => L + v * (W - L - R);
    [["observed agreement", po, C.accent], ["agreement by chance", pe, C.faint], ["kappa", Math.max(0, k), EV.human()]].forEach(([lab, v, col], i) => {
      const y = 18 + i * 42;
      svgText(sv, L - 10, y + 17, lab, C.dim, { anchor: "end", size: 12 });
      sv.appendChild(svgEl("rect", { x: L, y, width: W - L - R, height: 24, fill: "none", stroke: C.grid }));
      sv.appendChild(svgEl("rect", { x: L, y, width: Math.max(0, X(v) - L), height: 24, fill: col }));
      svgText(sv, X(v) + 6, y + 17, i === 2 ? (Number.isFinite(k) ? k.toFixed(2) : "undefined") : (v * 100).toFixed(1) + "%", C.fg, { size: 12, weight: 700 });
    });
    host.appendChild(sv);
    read.dataset.kappa = Number.isFinite(k) ? k.toFixed(3) : "nan";
    read.textContent = `both fail ${bf}  ·  A passes ${(pa * 100).toFixed(0)}%, B passes ${(pb * 100).toFixed(0)}%  ·  ` +
      `p_o ${po.toFixed(2)}, p_e ${pe.toFixed(3)}, kappa ${Number.isFinite(k) ? k.toFixed(3) : "undefined"}`;
  }
  draw();
}

/* ---------------------------------------------------------------- Ch 4
   Reliability diagram: a system whose true chance of being right is q ~ U(0.5, 1)
   reports min(1, q + overconfidence). Bars = accuracy per confidence bin.          */
function vizCalib(el) {
  panel(el, "Calibration: does 90% confident mean 90% right?",
    "Each bar is the accuracy of the outputs whose stated confidence fell in that bin. A calibrated system's bars touch the diagonal; an overconfident one's fall below it.");
  const row = ctlRow(el);
  let oc = 10;
  slider(row, "overconfidence (points)", 0, 30, oc, v => { oc = v; draw(); });
  const host = document.createElement("div"); el.appendChild(host);
  const read = evReadout(el, "calib-read");
  function draw() {
    const r = rng(11), n = 4000, bins = Array.from({ length: 10 }, () => ({ n: 0, ok: 0, c: 0 }));
    let br = 0;
    for (let i = 0; i < n; i++) {
      const q = 0.5 + 0.5 * r(), c = Math.min(1, q + oc / 100), y = r() < q ? 1 : 0;
      const b = Math.min(9, Math.max(0, Math.ceil(c * 10) - 1));
      bins[b].n++; bins[b].ok += y; bins[b].c += c; br += (c - y) ** 2;
    }
    let e = 0; bins.forEach(b => { if (b.n) e += b.n / n * Math.abs(b.ok / b.n - b.c / b.n); });
    host.innerHTML = "";
    const W = 620, H = 300, L = 58, R = 18, T = 14, B = 44;
    const X = v => L + v * (W - L - R), Y = v => H - B - v * (H - T - B);
    const sv = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
      "aria-label": `Reliability diagram with ${oc} points of overconfidence: expected calibration error ${(e * 100).toFixed(1)} points` });
    for (let v = 0; v <= 1.0001; v += 0.25) {
      sv.appendChild(svgEl("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: C.grid }));
      svgText(sv, L - 6, Y(v) + 4, Math.round(v * 100) + "%", C.faint, { anchor: "end", size: 11, weight: 500 });
      svgText(sv, X(v), H - B + 17, Math.round(v * 100) + "%", C.faint, { anchor: "middle", size: 11, weight: 500 });
    }
    bins.forEach((b, i) => {
      if (!b.n) return;
      const acc = b.ok / b.n;
      sv.appendChild(svgEl("rect", { x: X(i / 10) + 3, y: Y(acc), width: (W - L - R) / 10 - 6, height: Y(0) - Y(acc), fill: EV.sut(), "fill-opacity": 0.85 }));
    });
    sv.appendChild(svgEl("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: C.stamp, "stroke-width": 2, "stroke-dasharray": "6 4" }));
    svgText(sv, X(0.04), Y(0.9), "perfect calibration", C.stamp, { size: 12, weight: 700 });
    svgText(sv, (L + W - R) / 2, H - 6, "stated confidence", C.dim, { anchor: "middle", size: 12 });
    const yl = svgText(sv, 15, (T + H - B) / 2, "actual accuracy", C.dim, { anchor: "middle", size: 12 });
    yl.setAttribute("transform", `rotate(-90 15 ${(T + H - B) / 2})`);
    host.appendChild(sv);
    read.dataset.ece = e.toFixed(4);
    read.textContent = `${n.toLocaleString()} simulated outputs  ·  ECE ${(e * 100).toFixed(1)} points  ·  Brier ${(br / n).toFixed(3)}`;
  }
  draw();
}

Object.assign(VIZ, { reads: vizReads, margin: vizMargin, kappa: vizKappa, calib: vizCalib });
