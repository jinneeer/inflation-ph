/* Forecast tab: live SARIMA/ETS fits with scenario controls and fan chart. */
import { post, get } from "../api.js";
import { C, baseLayout, config, $, $$, csvDownload, monthLabel } from "./common.js";

let last = null;

export async function renderBacktest() {
  const m = await get("/model/metrics"); // fallback if endpoint missing is handled below
  return m;
}

export async function initForecast(meta) {
  const nodeSel = $("#fcNode");
  if (nodeSel && nodeSel.options.length <= 1) {
    meta.shock_nodes.slice(0, 40).forEach((n) => {
      const o = document.createElement("option");
      o.value = n.code;
      o.textContent = `${n.name} (w ${n.weight}%)`;
      nodeSel.appendChild(o);
    });
  }
  const horizon = $("#fcHorizon"), shock = $("#fcShock");
  horizon.addEventListener("input", () => { $("#fcHorizonVal").textContent = horizon.value; });
  shock.addEventListener("input", () => { $("#fcShockVal").textContent = shock.value; });
  $("#fcRun").addEventListener("click", () => run());
  $$("#fcPresets [data-preset]").forEach((b) => b.addEventListener("click", () => {
    const [code, pct] = b.dataset.preset.split(":");
    if (![...nodeSel.options].some((o) => o.value === code)) return;
    nodeSel.value = code;
    shock.value = pct;
    $("#fcShockVal").textContent = pct;
    run();
  }));
  $("#fcAdvReset").addEventListener("click", () => {
    Object.entries({ fcP: 1, fcD: 1, fcQ: 1, fcSP: 1, fcSD: 1, fcSQ: 0 }).forEach(([id, v]) => { $(`#${id}`).value = v; });
    $("#fcTrend").value = "add"; $("#fcSeasonal").value = "add"; $("#fcDamped").checked = true;
    run();
  });
  $("#fcCsvBtn").addEventListener("click", () => {
    if (!last) return;
    const rows = last.data.flatMap((p) => p.rows.map((r) => ({ model: p.model, ...r })));
    csvDownload("forecast.csv", rows);
  });
  $("#fcSqlBtn").addEventListener("click", async () => {
    const drawer = $("#fcSql");
    drawer.classList.toggle("open");
    if (drawer.classList.contains("open")) {
      const sql = (await get("/sql/headline")).data.sql; // show pipeline provenance for the fit input
      $("#fcSql pre").textContent =
        "POST /api/forecast\n\n-- fit input series\n" + sql;
    }
  });
  bindNarrative();
}

function bindNarrative() {
  const holder = $("#backtestTable");
  get("/model/backtest").then((res) => {
    const rows = res.data || [];
    if (!rows.length) return;
    holder.innerHTML = "<tr><th>Fold</th><th>Model</th><th>Average error</th><th>RMS error</th></tr>" +
      rows.map((r) => `<tr><td>${r.window}</td><td class="name">${r.model}</td>` +
        `<td>${r.MAE.toFixed(2)}</td><td>${r.RMSE.toFixed(2)}</td></tr>`).join("");
  }).catch(() => {
    holder.innerHTML = `<tr><td colspan="4">Backtest artifact unavailable.</td></tr>`;
  });
}

export async function run(state) {
  const payload = {
    model: $("#fcModel").value,
    horizon: Number($("#fcHorizon").value),
    shock_node: $("#fcNode").value || null,
    shock_pct: Number($("#fcShock").value),
    shock_start: Number($("#fcShockStart").value),
    p: Number($("#fcP").value), d: Number($("#fcD").value), q: Number($("#fcQ").value),
    P: Number($("#fcSP").value), D: Number($("#fcSD").value), Q: Number($("#fcSQ").value),
    trend: $("#fcTrend").value, seasonal: $("#fcSeasonal").value, damped: $("#fcDamped").checked,
  };
  const btn = $("#fcRun");
  btn.disabled = true;
  btn.textContent = "Fitting…";
  try {
    last = await post("/forecast", payload);
    draw(last);
    if (last.meta?.error) {
      $("#fcMeta").textContent = `Fit failed: ${last.meta.error}`;
    }
  } catch (err) {
    $("#fcMeta").textContent = `Forecast request failed: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Fit forecast";
  }
}

function draw(body) {
  const traces = [];
  const hist = body.history || [];
  traces.push({
    x: hist.map((h) => h.date), y: hist.map((h) => h.value), type: "scatter",
    mode: "lines", name: "Actual", line: { color: C.ink, width: 1.6 },
  });
  const colors = { SARIMA: C.teal, ETS: C.vermilion };
  const fills = {
    SARIMA: ["rgba(13,110,99,.09)", "rgba(13,110,99,.18)"],
    ETS: ["rgba(196,73,45,.09)", "rgba(196,73,45,.18)"],
  };
  (body.data || []).forEach((path) => {
    const c = colors[path.model] || C.amber;
    const [fill95, fill80] = fills[path.model] || ["rgba(217,119,6,.09)", "rgba(217,119,6,.18)"];
    const rows = path.rows;
    traces.push({
      x: rows.map((r) => r.date), y: rows.map((r) => r.hi95), type: "scatter", mode: "lines",
      line: { width: 0 }, showlegend: false, hoverinfo: "skip",
    });
    traces.push({
      x: rows.map((r) => r.date), y: rows.map((r) => r.lo95), type: "scatter", mode: "lines",
      line: { width: 0 }, fill: "tonexty", fillcolor: fill95,
      name: `${path.model} 95%`, hoverinfo: "skip",
    });
    traces.push({
      x: rows.map((r) => r.date), y: rows.map((r) => r.hi80), type: "scatter", mode: "lines",
      line: { width: 0 }, showlegend: false, hoverinfo: "skip",
    });
    traces.push({
      x: rows.map((r) => r.date), y: rows.map((r) => r.lo80), type: "scatter", mode: "lines",
      line: { width: 0 }, fill: "tonexty", fillcolor: fill80,
      name: `${path.model} 80%`, hoverinfo: "skip",
    });
    traces.push({
      x: rows.map((r) => r.date), y: rows.map((r) => r.inflation), type: "scatter",
      mode: "lines", name: `${path.model} forecast`,
      line: { color: colors[path.model] || C.amber, width: 2, dash: "dash" },
      hovertemplate: "%{x|%b %Y} · %{y:.1f}%<extra></extra>",
    });
  });
  traces.push({
    x: [hist[0]?.date, (body.data?.[0]?.rows?.slice(-1)[0] || {}).date].filter(Boolean),
    y: [4, 4], mode: "lines", name: "4% ceiling",
    line: { color: C.vermilion, width: 1, dash: "dot" }, hoverinfo: "skip",
  });
  (body.baseline || []).forEach((path) => {
    traces.push({
      x: path.rows.map((r) => r.date), y: path.rows.map((r) => r.inflation),
      type: "scatter", mode: "lines", name: `${path.model} baseline`,
      line: { color: C.muted, width: 1.4, dash: "dot" },
      hovertemplate: "baseline %{x|%b %Y} · %{y:.1f}%<extra></extra>",
    });
  });
  Plotly.react("fanChart", traces, baseLayout({
    yaxis: { title: "Inflation (%)", gridcolor: C.line },
    xaxis: { gridcolor: C.line },
  }), config);
  const n = body.data?.[0]?.rows?.length || 0;
  const fit = body.meta?.fit_ms ? `, fit ${body.meta.fit_ms} ms` : "";
  const models = (body.data || []).map((p) => (p.aic != null ? `${p.model} (AIC ${p.aic})` : p.model)).join(" + ");
  $("#fcMeta").textContent = `${models}, ${n} months${fit}, ` +
    "shaded bands show likely ranges (80% and 95%); dotted = no price-change baseline; price changes use the basket-weight approximation.";
}
