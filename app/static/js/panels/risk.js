/* Risk tab: regional what-if, model vs simple guess, risk curves, SHAP, clickable error chart. */
import { get, post } from "../api.js";
import { state, setState } from "../store.js";
import { C, baseLayout, config, $, featureLabel, shortRegion, esc } from "./common.js";

let curves = null;
let residualRows = [];
let latest = new Map();
let regionNames = [];
let wiTimer = null;
const DEFAULT_REGION = "National Capital Region (NCR)";

export async function renderRisk() {
  const [metrics, lat] = await Promise.all([get("/model/metrics"), get("/nowcast/latest")]);
  latest = new Map(lat.data.map((r) => [r.geolocation, r]));
  drawCompare(metrics.data);
  syncWhatIf();
  if (!curves) curves = (await get("/classifier/curves")).data;
  drawCurves(curves, metrics.data);
  await applyThreshold(Number($("#thrSlider").value));
  drawShapBar((await get("/nowcast/shap/global")).data);
  drawSwarm((await get("/nowcast/shap/beeswarm", { sample: 45 })).data);
  residualRows = (await get("/nowcast/residuals")).data;
  drawResiduals(residualRows);
  const pick = state.selectedPoint ||
    (residualRows.length ? (({ geolocation, date }) => ({ region: geolocation, date }))(
      residualRows.reduce((a, b) => (Math.abs(b.residual) > Math.abs(a.residual) ? b : a))) : null);
  if (pick) await showLocal(pick.region, pick.date); // default: the largest miss, so the panel is never empty
}

export function initRisk(meta) {
  regionNames = meta.regions.map((r) => r.geo_id).sort((a, b) => shortRegion(a).localeCompare(shortRegion(b)));
  const sel = $("#wiRegion");
  sel.innerHTML = regionNames.map((r) => `<option value="${esc(r)}">${esc(shortRegion(r))}</option>`).join("");
  sel.value = DEFAULT_REGION;
  sel.addEventListener("change", () => { setState({ region: sel.value }); });
  const rate = $("#wiRate");
  rate.addEventListener("input", () => { $("#wiRateVal").textContent = Number(rate.value).toFixed(1); queueWhatIf(); });
  $("#wiReset").addEventListener("click", () => {
    const l = latest.get(sel.value);
    if (l) { rate.value = l.infl_now; $("#wiRateVal").textContent = Number(l.infl_now).toFixed(1); runWhatIf(); }
  });
  const slider = $("#thrSlider");
  slider.addEventListener("input", () => { $("#thrVal").textContent = Number(slider.value).toFixed(2); });
  slider.addEventListener("change", () => applyThreshold(Number(slider.value)));
  $("#rocSqlBtn").addEventListener("click", () => {
    const drawer = $("#rocSql");
    drawer.classList.toggle("open");
    if (drawer.classList.contains("open")) {
      $("#rocSql pre").textContent =
        "Models are trained offline by models/train_models.py (python -m models.train_models)\n" +
        "and served from data/models/*.parquet.\n\n" +
        "POST /api/classifier/threshold  { \"threshold\": 0.50 }\n" +
        "POST /api/nowcast/whatif        { \"region\": \"...\", \"infl\": 5.0 }";
    }
  });
}

/* ---- regional what-if (changes model inputs, re-scores live) ---- */
function syncWhatIf() {
  const sel = $("#wiRegion");
  const region = state.region && regionNames.includes(state.region) ? state.region : (sel.value || DEFAULT_REGION);
  const changed = sel.value !== region || !$("#wiRate").dataset.ready;
  sel.value = region;
  if (changed) {
    const l = latest.get(region);
    if (l) { $("#wiRate").value = l.infl_now; $("#wiRateVal").textContent = Number(l.infl_now).toFixed(1); }
    $("#wiRate").dataset.ready = "1";
  }
  runWhatIf();
}

function queueWhatIf() { clearTimeout(wiTimer); wiTimer = setTimeout(runWhatIf, 150); }

async function runWhatIf() {
  const region = $("#wiRegion").value;
  const infl = Number($("#wiRate").value);
  const r = (await post("/nowcast/whatif", { region, infl })).data;
  const l = latest.get(region);
  const moved = Math.abs(infl - r.actual_now) > 0.05;
  const pc = (p) => `${Math.round(p * 100)}%`;
  const cards = [
    ["This month, actual", `${r.actual_now.toFixed(1)}%`, "last published rate"],
    ["Next month, simple guess", `${(moved ? infl : r.actual_now).toFixed(1)}%`,
      l ? `likely range ${(l.lo80 - l.infl_now + (moved ? infl : r.actual_now)).toFixed(1)}–${(l.hi80 - l.infl_now + (moved ? infl : r.actual_now)).toFixed(1)}%` : ""],
    ["Next month, model view", `${r.scenario_next.toFixed(1)}%`, moved ? `was ${r.baseline_next.toFixed(1)}% before your change` : "adds momentum and seasonal effects"],
    ["Chance above 4%", pc(r.scenario_proba), moved ? `was ${pc(r.baseline_proba)} before your change` : "model's estimate"],
  ];
  $("#wiKpis").innerHTML = cards.map(([k, v, d]) =>
    `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="d ${k === "Chance above 4%" && r.scenario_proba > r.baseline_proba ? "up" : ""}">${d}</div></div>`).join("");
  $("#wiNote").textContent = moved
    ? `If ${shortRegion(region)} were at ${infl.toFixed(1)}% this month instead of ${r.actual_now.toFixed(1)}%, the model re-computes every input that depends on it (recent averages and bumpiness) and re-scores the next month.`
    : "Drag the slider to ask: what if this month's inflation in this region were different?";
}

/* ---- is the model better than a simple guess? ---- */
function drawCompare(m) {
  const folds = m.nowcast?.folds || [];
  const k1 = "Persistence (this month carried forward)", k2 = "Boosted trees, change target + current rate (deployed)";
  const rows = folds.map((f) => {
    const a = f.regression?.[k1]?.MAE, b = f.regression?.[k2]?.MAE;
    return `<tr><td>${f.window}</td><td>${a?.toFixed(2)}</td><td>${b?.toFixed(2)}</td><td>${a <= b ? "Simple guess" : "Model"}</td></tr>`;
  }).join("");
  $("#compareTable").innerHTML =
    "<tr><th>Test period</th><th>Simple guess: avg miss (pts)</th><th>Model: avg miss (pts)</th><th>Closer</th></tr>" + rows;
  const cb = m.classifier_baseline, cm = m.classifier;
  $("#compareNote").textContent =
    `Honest reading: for the very next month, "same as this month" is hard to beat, and it won in every test period. ` +
    (cb && cm ? `For the yes/no question "above 4%?", the simple rule scored ${cb.AUC.toFixed(2)} (AUC) against the model's ${cm.AUC.toFixed(2)}. ` : "") +
    "The model's real value is the explanation (which factors push a region up) and the what-if tool, not a more accurate number.";
}

function drawCurves(c, m) {
  Plotly.react("rocChart", [
    { x: c.roc.map((r) => r.fpr), y: c.roc.map((r) => r.tpr), type: "scatter", mode: "lines",
      name: "Rises caught vs false alarms", line: { color: C.teal, width: 2 } },
    { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", name: "Coin flip", line: { color: C.muted, width: 1, dash: "dot" }, hoverinfo: "skip" },
  ], baseLayout({
    xaxis: { title: "False alarms (share of calm months flagged)", gridcolor: C.line, range: [0, 1] },
    yaxis: { title: "Rises caught", gridcolor: C.line, range: [0, 1.02] },
  }), config);
  const cm = m.classifier || {};
  $("#rocCaption").textContent =
    `Chance a region is above 4% next month. Closer to the top-left corner is better. Score (AUC) ${cm.AUC?.toFixed(3)}, ` +
    `${cm.n_test} test cases. The 2026 price surge makes high scores easier, so read them with care.`;
}

async function applyThreshold(threshold) {
  const m = (await post("/classifier/threshold", { threshold })).data;
  const cards = [
    ["Alarms that were right", m.precision.toFixed(2), `${m.tp} correct, ${m.fp} false alarms`],
    ["Rises caught", m.recall.toFixed(2), `${m.tp} caught, ${m.fn} missed`],
    ["Calm months correct", m.specificity.toFixed(2), `${m.tn} correct, ${m.fp} false alarms`],
    ["All months correct", m.accuracy.toFixed(2), `n = ${m.n}`],
  ];
  $("#thrKpis").innerHTML = cards.map(([k, v, d]) =>
    `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
  $("#confusionTable").innerHTML =
    "<tr><th>Flagged next month?</th><th>Actually above 4%</th><th>Actually below 4%</th></tr>" +
    `<tr><td class="name">Above 4%</td><td>${m.tp}</td><td>${m.fp}</td></tr>` +
    `<tr><td class="name">Not flagged</td><td>${m.fn}</td><td>${m.tn}</td></tr>`;
}

function drawShapBar(rows) {
  const sorted = [...rows].sort((a, b) => a.mean_abs_shap - b.mean_abs_shap);
  Plotly.react("shapBar", [{
    type: "bar", orientation: "h", x: sorted.map((r) => r.mean_abs_shap),
    y: sorted.map((r) => featureLabel(r.feature)), marker: { color: C.teal },
    hovertemplate: "%{y} · %{x:.3f} pts<extra></extra>",
  }], baseLayout({ margin: { l: 120, r: 18, t: 6, b: 36 },
    xaxis: { title: "Average push on next month (points)", gridcolor: C.line } }), config);
}

function drawSwarm(rows) {
  const features = [...new Set(rows.map((r) => r.feature))];
  const traces = features.map((f) => {
    const sub = rows.filter((r) => r.feature === f);
    return {
      type: "scatter", mode: "markers", name: featureLabel(f),
      x: sub.map((r) => r.shap_value), y: sub.map(() => featureLabel(f)),
      customdata: sub.map((r) => [String(r.date).slice(0, 10), r.geolocation]),
      marker: { size: 5, color: sub.map((r) => r.feature_value), colorscale: "RdYlBu", showscale: f === features[0],
        colorbar: f === features[0] ? { title: "value", thickness: 8, len: 0.6 } : undefined, opacity: 0.7 },
      hovertemplate: `${featureLabel(f)} · push %{x:.3f} pts<br>%{customdata[1]}, %{customdata[0]}<br><i>click to explain</i><extra></extra>`,
    };
  });
  Plotly.react("shapSwarm", traces, baseLayout({
    showlegend: false, margin: { l: 120, r: 14, t: 6, b: 36 },
    xaxis: { title: "Push on next month (points)", gridcolor: C.line, zeroline: true, zerolinecolor: C.line },
  }), config);
  const el = document.getElementById("shapSwarm");
  if (!el._bound) {
    el.on("plotly_click", (ev) => {
      const cd = ev.points?.[0]?.customdata;
      if (Array.isArray(cd) && cd.length > 1) selectPoint(cd[1], cd[0]);
    });
    el._bound = true;
  }
}

/** Cross-filter: choosing a region-month filters the map, KPIs, ranking and the explanation. */
function selectPoint(region, date) {
  setState({ region, focusDate: date, selectedPoint: { region, date } });
}

async function showLocal(region, date) {
  const body = await get("/nowcast/shap/local", { region, date });
  const row = residualRows.find((r) => r.geolocation === region && r.date === date);
  const bits = [`${shortRegion(region)}, ${date.slice(0, 7)}`];
  if (row) bits.push(`simple guess ${row.pred.toFixed(1)}%, model ${row.model_pred.toFixed(1)}%, actual next month ${row.actual.toFixed(1)}%`);
  $("#shapLocalLabel").textContent = bits.join(" · ");
  drawWaterfall(body);
}

function drawWaterfall(body) {
  const data = body?.data || [];
  if (!data.length) { $("#shapLocalLabel").textContent = "no stored explanation for that point"; return; }
  const rows = [...data].filter((r) => r.feature !== "infl" || true)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const start = body.start ?? 0, base = body.base ?? 0;
  const pred = body.prediction ?? start + base + rows.reduce((a, r) => a + r.contribution, 0);
  Plotly.react("shapLocal", [{
    type: "waterfall", orientation: "v",
    measure: ["absolute", "relative", ...rows.map(() => "relative"), "total"],
    x: ["This month", "Typical drift", ...rows.map((r) => `${featureLabel(r.feature)} = ${(+r.value).toFixed(1)}`), "Model's next month"],
    y: [start, base, ...rows.map((r) => r.contribution), pred],
    increasing: { marker: { color: C.vermilion } }, decreasing: { marker: { color: C.teal } },
    totals: { marker: { color: C.graphite } },
    hovertemplate: "%{x}<br>%{y:+.2f} pts<extra></extra>",
  }], baseLayout({
    margin: { l: 52, r: 12, t: 8, b: 96 }, xaxis: { tickangle: -35, tickfont: { size: 10 } },
    yaxis: { title: "Inflation (%)", gridcolor: C.line },
  }), config);
}

/* ---- where the one-month-ahead guess missed (click a point) ---- */
function drawResiduals(rows) {
  if (!rows.length) return;
  const rms = Math.sqrt(rows.reduce((a, r) => a + r.residual ** 2, 0) / rows.length);
  const sel = state.region;
  const big = (r) => Math.abs(r.residual) > 2 * rms;
  const color = (r) => (sel && r.geolocation !== sel ? "rgba(150,160,158,.35)" : big(r) ? C.vermilion : C.graphite);
  Plotly.react("residChart", [{
    x: rows.map((r) => r.date), y: rows.map((r) => r.residual), type: "scatter", mode: "markers",
    customdata: rows.map((r) => [r.geolocation, r.actual, r.pred]),
    marker: { size: rows.map((r) => (sel && r.geolocation === sel ? 9 : big(r) ? 9 : 6)), color: rows.map(color),
      line: { width: rows.map((r) => (state.selectedPoint && r.geolocation === state.selectedPoint.region && r.date === state.selectedPoint.date ? 2.5 : 0)), color: C.ink } },
    hovertemplate: "%{customdata[0]}<br>%{x|%b %Y}: guess %{customdata[2]:.1f}%, actual next month %{customdata[1]:.1f}%<br>miss %{y:+.2f} pts<br><i>click to explain this point</i><extra></extra>",
  }], baseLayout({
    margin: { l: 56, r: 18, t: 8, b: 40 }, showlegend: false,
    yaxis: { title: "Miss (points): actual − guess", gridcolor: C.line, zeroline: true, zerolinecolor: C.ink },
    shapes: [2 * rms, -2 * rms].map((y) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y, line: { color: C.amber, width: 1, dash: "dot" } })),
  }), config);
  $("#residCaption").textContent =
    `Each dot is one region in one month. Dots outside the dotted lines (red) missed by more than usual. ` +
    `Click one to jump to that region and month everywhere on the dashboard and see why the model said what it did.`;
  const el = document.getElementById("residChart");
  if (!el._bound) {
    el.on("plotly_click", (ev) => {
      const p = ev.points?.[0];
      if (p) selectPoint(p.customdata[0], String(p.x).slice(0, 10));
    });
    el._bound = true;
  }
}
