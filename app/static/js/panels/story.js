/* Story tab: plain-language chapters written from the data, each with one annotated chart. */
import { get, post } from "../api.js";
import { setState } from "../store.js";
import { C, baseLayout, config, $, esc, shortRegion, monthLabel } from "./common.js";

const BAND = { type: "rect", xref: "paper", x0: 0, x1: 1, y0: 2, y1: 4, fillcolor: "rgba(47,122,79,.12)", line: { width: 0 }, layer: "below" };

export async function renderStory(state) {
  const root = $("#storyRoot");
  const body = (await get("/story", { end: state.end })).data;
  root.innerHTML = body.chapters.map((c, i) => `
    <article class="chapter" id="chapter-${c.id}">
      <div class="chapter-text">
        <p class="kicker">${i + 1} · ${esc(c.kicker)}</p>
        <h3>${esc(c.title)}</h3>
        ${c.big ? `<p class="big">${esc(c.big)}</p>` : ""}
        <ul>${c.bullets.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>
        ${c.id === "act" ? `<p class="chapter-links"><button class="btn" data-goto="forecast">Try a price scenario</button>
          <button class="btn" data-goto="risk">Test a region</button></p>` : ""}
      </div>
      <div class="chapter-viz">${c.chart ? `<div id="story-${c.chart}" class="plot"></div><p class="caption">${esc(c.caption)}</p>` : ""}</div>
    </article>`).join("") + footer(body);
  root.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () =>
    document.querySelector(`.tab[data-tab="${b.dataset.goto}"]`)?.click()));
  const draw = { headline: drawHeadline, map: drawRegions, treemap: drawDrivers, forecast: drawForecast, backtest: drawTrust };
  await Promise.all(body.chapters.filter((c) => draw[c.chart]).map((c) =>
    draw[c.chart](state).catch((e) => { $(`#story-${c.chart}`).innerHTML = `<div class="empty">Chart unavailable: ${esc(e.message)}</div>`; })));
}

function footer(body) {
  const news = (body.news || []).map((n) =>
    `<li><a href="${esc(n.url)}" target="_blank" rel="noopener">${esc(n.title)}</a> <span class="hint"> · ${esc(n.domain)}</span></li>`).join("");
  const words = body.glossary.map((g) => `<dt>${esc(g.term)}</dt><dd>${esc(g.plain)}</dd>`).join("");
  return `<div class="grid2 story-foot">
    ${news ? `<div class="card"><div class="card-head"><h3>In the news that month</h3></div><ul class="newslist">${news}</ul></div>` : ""}
    <div class="card"><div class="card-head"><h3>Plain-word guide</h3></div><dl class="glossary">${words}</dl></div>
  </div>`;
}

async function drawHeadline(state) {
  const rows = (await get("/series/headline", { start: "2018-01-01", end: state.end })).data;
  const last = rows[rows.length - 1];
  Plotly.react("story-headline", [{
    x: rows.map((r) => r.date), y: rows.map((r) => r.headline), type: "scatter", mode: "lines",
    line: { color: C.teal, width: 2.4 }, name: "Inflation", hovertemplate: "%{x|%b %Y}: %{y:.1f}%<extra></extra>",
  }], baseLayout({
    showlegend: false, shapes: [BAND], yaxis: { gridcolor: C.line, ticksuffix: "%", title: "Prices vs a year earlier" },
    annotations: [{ x: last.date, y: last.headline, text: `${last.headline.toFixed(1)}% now`, showarrow: true, ax: -50, ay: -28, font: { color: C.vermilion, size: 12 }, arrowcolor: C.vermilion },
      { xref: "paper", x: 0.01, y: 3, yref: "y", text: "2–4% target", showarrow: false, font: { size: 10.5, color: C.ok || "#2f7a4f" }, xanchor: "left" }],
  }), config);
}

async function drawRegions(state) {
  const rows = (await get("/map/choropleth", { date: state.end })).data
    .filter((r) => r.value != null).sort((a, b) => a.value - b.value);
  Plotly.react("story-map", [{
    type: "bar", orientation: "h", x: rows.map((r) => r.value), y: rows.map((r) => shortRegion(r.geo_id)),
    customdata: rows.map((r) => r.geo_id), marker: { color: rows.map((r) => (r.value > 4 ? C.vermilion : C.teal)) },
    text: rows.map((r) => `${r.value.toFixed(1)}%`), textposition: "outside", cliponaxis: false,
    hovertemplate: "%{y}: %{x:.1f}%<br><i>click to open this region</i><extra></extra>",
  }], baseLayout({
    margin: { l: 96, r: 44, t: 8, b: 40 }, xaxis: { ticksuffix: "%", gridcolor: C.line, title: "Inflation, year on year" },
    shapes: [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: 4, x1: 4, line: { color: C.ink, width: 1, dash: "dot" } }],
    height: 360,
  }), config);
  const el = document.getElementById("story-map");
  if (!el._bound) {
    el.on("plotly_click", (ev) => {
      const id = ev.points?.[0]?.customdata;
      if (id) { setState({ region: id }); window.scrollTo({ top: 0, behavior: "smooth" }); }
    });
    el._bound = true;
  }
}

async function drawDrivers(state) {
  const rows = (await get("/drivers/treemap", { date: state.end, region: "PHILIPPINES" })).data
    .filter((r) => r.value != null)
    .map((r) => ({ name: r.name.split(",")[0].toLowerCase().replace(/^./, (c) => c.toUpperCase()), pp: r.weight_pct * r.value / 100, v: r.value }))
    .sort((a, b) => a.pp - b.pp);
  Plotly.react("story-treemap", [{
    type: "bar", orientation: "h", x: rows.map((r) => r.pp), y: rows.map((r) => r.name),
    marker: { color: rows.map((r) => (r.v > 4 ? C.vermilion : r.v > 0 ? C.amber : C.teal)) },
    text: rows.map((r) => `${r.pp.toFixed(1)} pts`), textposition: "outside", cliponaxis: false,
    customdata: rows.map((r) => r.v), hovertemplate: "%{y}<br>adds %{x:.2f} points (prices up %{customdata:.1f}%)<extra></extra>",
  }], baseLayout({
    margin: { l: 190, r: 56, t: 8, b: 40 }, xaxis: { title: "Points added to inflation", gridcolor: C.line }, height: 360,
  }), config);
}

async function drawForecast() {
  const res = await post("/forecast", { model: "sarima", horizon: 12 });
  const hist = (res.history || []).slice(-36);
  const rows = res.data?.[0]?.rows || [];
  const band = (hi, lo, fill, name) => [
    { x: rows.map((r) => r.date), y: rows.map((r) => r[hi]), type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: rows.map((r) => r.date), y: rows.map((r) => r[lo]), type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: fill, name, hoverinfo: "skip" },
  ];
  Plotly.react("story-forecast", [
    ...band("hi95", "lo95", "rgba(13,110,99,.12)", "Very likely range (95%)"),
    ...band("hi80", "lo80", "rgba(13,110,99,.24)", "Likely range (80%)"),
    { x: hist.map((h) => h.date), y: hist.map((h) => h.value), type: "scatter", mode: "lines", name: "Actual", line: { color: C.ink, width: 2 } },
    { x: rows.map((r) => r.date), y: rows.map((r) => r.inflation), type: "scatter", mode: "lines", name: "Best guess", line: { color: C.teal, width: 2.4, dash: "dash" },
      hovertemplate: "%{x|%b %Y}: %{y:.1f}%<extra>best guess</extra>" },
  ], baseLayout({ shapes: [BAND], yaxis: { gridcolor: C.line, ticksuffix: "%" }, height: 340 }), config);
}

async function drawTrust() {
  const m = (await get("/model/metrics")).data;
  const folds = m.nowcast?.folds || [];
  const g = (f, k) => f.regression?.[k]?.MAE;
  Plotly.react("story-backtest", [
    { type: "bar", name: "Simple guess: same as this month", x: folds.map((f) => f.window), y: folds.map((f) => g(f, "Persistence (this month carried forward)")), marker: { color: C.teal } },
    { type: "bar", name: "Machine-learning model", x: folds.map((f) => f.window), y: folds.map((f) => g(f, "Boosted trees, change target + current rate (deployed)")), marker: { color: C.amber } },
  ], baseLayout({ barmode: "group", yaxis: { title: "Average miss (points) - lower is better", gridcolor: C.line }, height: 320, margin: { l: 56, r: 12, t: 28, b: 44 } }), config);
}
