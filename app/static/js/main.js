/* App bootstrap: state wiring, hero map + ranking, tabs, tools. */
import { initStore, state, setState, subscribe, clearRegion } from "./store.js";
import * as api from "./api.js";
import { lastMeta } from "./api.js";
import { PHMap } from "./map3d.js";
import { C, $, $$, monthLabel, csvDownload } from "./panels/common.js";
import { renderOverview } from "./panels/overview.js";
import { renderStory } from "./panels/story.js";
import { run as runForecast, initForecast } from "./panels/forecast.js";
import { renderDrivers } from "./panels/drivers.js";
import { renderRisk, initRisk } from "./panels/risk.js";
import { renderStructure } from "./panels/structure.js";
import { renderData, showSql } from "./panels/data.js";
import { highlightSector } from "./panels/drivers.js";
import * as motion from "./motion.js";

let meta = null;
let map = null;
let landingMap = null;
let activeTab = "story";
let storyEnd = null;
let playTimer = null;
let monthList = [];
let lastRegion = null;

const RENDERERS = {
  story: async (s) => {
    if (storyEnd === s.end && rendered.has("story")) return; // story reads the end month only
    storyEnd = s.end;
    await renderStory(s);
  },
  overview: renderOverview,
  forecast: (s) => runForecast(s),
  drivers: renderDrivers,
  risk: () => renderRisk(),
  structure: renderStructure,
  data: renderData,
};
const rendered = new Set();

function plotlyReady() {
  return new Promise((resolve) => {
    if (window.Plotly) return resolve();
    const t = setInterval(() => {
      if (window.Plotly) { clearInterval(t); resolve(); }
    }, 50);
  });
}

function buildDebugBadge() {
  const el = document.createElement("div");
  el.id = "debugBadge";
  document.body.appendChild(el);
  const paint = () => {
    const rows = [...lastMeta.values()].slice(-8).map((m) =>
      `<tr><td>${m.panel_id}</td><td>${m.row_count}</td>` +
      `<td>${m.elapsed_ms ?? "—"}</td><td>${m.rtt_ms ?? "—"}</td><td>${m.cache ?? "—"}</td></tr>`).join("");
    el.innerHTML = `<div>latency · ?debug=1</div><table><tr><th>panel</th><th>rows</th>` +
      `<th>server ms</th><th>rtt ms</th><th>cache</th></tr>${rows}</table>`;
  };
  window.addEventListener("api:meta", paint);
  paint();
}

function monthsBetween(start, end) {
  const out = [];
  const d = new Date(`${start}T00:00:00`);
  const stop = new Date(`${end}T00:00:00`);
  while (d <= stop) {
    out.push(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-01`);
    d.setMonth(d.getMonth() + 1);
  }
  return out;
}

async function bootstrap() {
  initStore();
  await plotlyReady();
  meta = await api.get("/meta");
  fillSelects();
  initForecast(meta);
  initRisk(meta);
  bindRail();
  bindTabs();
  bindTools();
  await initMap();
  subscribe(onState);
  lastState = { ...state };
  await renderHero();
  $("#landingGo").addEventListener("click", openDashboard, { once: true });
  await renderActive();
}

async function openDashboard() {
  const landing = $("#landing"), app = $(".app");
  scrollTo(0, 0);
  landing.classList.add("leaving"); // fixed on top so the app can sit under it during the slide
  document.body.classList.add("sliding");
  app.hidden = false;
  map.resize(); // it was measured at 0x0 while hidden
  window.dispatchEvent(new Event("resize")); // Plotly charts were drawn hidden too
  await motion.slideOver(landing, app);
  landing.hidden = true;
  landing.classList.remove("leaving");
  document.body.classList.remove("sliding");
  app.style.transform = ""; // any transform breaks the sticky rail
  landingMap = null; // stop feeding it once the landing is gone
  $("#main").focus({ preventScroll: true });
  motion.intro();
}

function fillSelects() {
  const region = $("#regionSelect");
  meta.regions
    .slice()
    .sort((a, b) => a.display_name.localeCompare(b.display_name))
    .forEach((r) => {
      const o = document.createElement("option");
      o.value = r.geo_id;
      o.textContent = `${r.short_code} – ${r.display_name}`;
      region.appendChild(o);
    });
  region.value = state.region || "";

  const sector = $("#sectorSelect");
  meta.sectors.forEach((s) => {
    const o = document.createElement("option");
    o.value = s.code;
    o.textContent = `${s.code} – ${s.name}`;
    sector.appendChild(o);
  });
  sector.value = state.sector || "";

  $("#startDate").value = state.start.slice(0, 7);
  $("#endDate").value = state.end.slice(0, 7);
  $("#metricSelect").value = state.metric;
  refreshMonths();
}

function refreshMonths() {
  monthList = monthsBetween(state.start, state.end);
  if (!monthList.includes(state.focusDate)) {
    state.focusDate = monthList[monthList.length - 1] || state.end;
  }
  const scrub = $("#timeScrubber");
  scrub.min = 0;
  scrub.max = Math.max(0, monthList.length - 1);
  const idx = Math.max(0, monthList.indexOf(state.focusDate));
  scrub.value = idx;
  $("#scrubLabel").textContent = monthLabel(state.focusDate);
}

let geoP; // geojson is static; fetch it once per page load
async function initMap() {
  geoP ??= fetch("/map-data").then((r) => r.json());
  const geojson = await geoP;
  map = new PHMap($("#map3d"), {
    onSelect: (id) => setState({ region: id }),
    onClear: () => clearRegion(),
    onHover: (row) => showTip($("#mapTip"), row),
  });
  await map.load(geojson, meta.regions);
  const luzon = [[-1.5, 10.5, 3.6], [-1.5, 0, -3]];
  landingMap = new PHMap($("#landingMap"), {
    onSelect: (id) => { landingMap?.zoomTo(id, 1.3); showLandingCard(id); }, // closer than the dashboard map, which frames neighbours too
    onClear: () => { landingMap?.flyTo(...luzon); cardSeq++; $("#landingCard").hidden = true; $("#landingTrend").hidden = true; },
    onHover: (row) => showTip($("#landingTip"), row),
  });
  await landingMap.load(geojson, meta.regions);
  landingMap.startAt(...luzon);
  $$("[data-view]").forEach((b) => b.addEventListener("click", () => map.setView(b.dataset.view)));
}

let cardSeq = 0, cardTree = null;
const yearEnd = (y) => `${y}-12-01`; // the treemap query snaps to the last month on or before this

async function showLandingCard(id) {
  const seq = ++cardSeq; // a later click or clear wins over a slow response
  const sel = $("#landingYear");
  if (!sel.options.length) {
    for (let y = +state.focusDate.slice(0, 4) - 1; y >= 2019; y--) sel.add(new Option(y, y));
    sel.addEventListener("change", async () => {
      const s = ++cardSeq;
      const past = await api.get("/drivers/treemap", { date: yearEnd(sel.value) }).catch(() => ({ data: [] }));
      if (s === cardSeq) drawPies(cardTree, past.data, sel.value);
    });
  }
  const [kpi, tree, past, series] = await Promise.all([
    api.get("/kpis", { end: state.focusDate, region: id }),
    api.get("/drivers/treemap", { date: state.focusDate }), // shortcut: regions only publish all-items CPI, so the breakdown is national; pass region once regional category data exists
    api.get("/drivers/treemap", { date: yearEnd(sel.value) }).catch(() => ({ data: [] })),
    api.get("/series/region", { region: id, end: state.focusDate }),
  ]);
  if (seq !== cardSeq) return;
  cardTree = tree.data;
  const k = kpi.data[0] || {};
  const card = $("#landingCard");
  $("#landingCardName").textContent = meta.regions.find((r) => r.geo_id === id)?.display_name ?? id;
  $("#landingCardValue").textContent = k.headline == null ? "No data" : `${k.headline.toFixed(1)}%`;
  const d = k.headline != null && k.prev != null ? k.headline - k.prev : null;
  $("#landingCardDelta").textContent = d == null ? "" :
    Math.abs(d) < 0.05 ? "same as last month" : `${d > 0 ? "up" : "down"} ${Math.abs(d).toFixed(1)} pts from last month`;
  card.hidden = false;
  drawPies(tree.data, past.data, sel.value);
  motion.revealPanel(card);
  drawTrend(series.data);
}

const small = () => matchMedia("(max-width: 720px)").matches;
const dark = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: "IBM Plex Sans, sans-serif", color: "#eef2ea" } };

// Both donuts use the latest month's top four sectors so colours mean the same thing side by side.
function drawPies(latest, past, year) {
  const sentence = (s) => s.charAt(0) + s.slice(1).toLowerCase();
  // Pie slices must be positive, so sectors pulling prices down are left out.
  const top = latest.filter((r) => r.contribution > 0).slice(0, 4).map((r) => r.code);
  const bucket = (rows) => {
    const by = new Map(rows.map((r) => [r.code, Math.max(r.contribution ?? 0, 0)]));
    const rest = rows.filter((r) => !top.includes(r.code)).reduce((s, r) => s + Math.max(r.contribution ?? 0, 0), 0);
    return top.map((c) => by.get(c) ?? 0).concat(rest);
  };
  const labels = top.map((c) => sentence(latest.find((r) => r.code === c).name)).concat("Everything else");
  const pie = (values, x) => ({
    type: "pie", hole: 0.62, labels, values, sort: false, textinfo: "none", domain: { x, y: [0, 1] },
    marker: { colors: [C.teal, C.amber, C.vermilion, C.tealSoft, "#6a7573"], line: { color: "#0b1f1d", width: 2 } },
    hovertemplate: "%{label}: %{percent}<extra></extra>",
  });
  const a = bucket(latest), b = bucket(past);
  const centre = (text, x) => ({ text, x, y: 0.5, xref: "paper", yref: "paper", showarrow: false, font: { size: 12 } });
  Plotly.react("landingPie", [pie(a, [0, 0.47]), pie(b, [0.53, 1])], {
    ...dark, margin: { t: 0, b: 0, l: 0, r: 0 }, height: small() ? 200 : 250,
    showlegend: true, legend: { orientation: "h", y: -0.05, font: { size: 12 } },
    annotations: [centre(monthLabel(state.focusDate), 0.235),
      centre(b.some((v) => v > 0) ? `Dec ${year}` : "No data", 0.765)],
  }, { displayModeBar: false, responsive: true });
}

function drawTrend(rows) {
  const box = $("#landingTrend");
  box.hidden = false;
  Plotly.react("landingLine", [{
    type: "scatter", mode: "lines", x: rows.map((r) => r.date), y: rows.map((r) => r.value),
    line: { color: C.teal, width: 2 }, hovertemplate: "%{x|%b %Y}: %{y:.1f}%<extra></extra>",
  }], {
    ...dark, height: small() ? 120 : 170, margin: { t: 6, b: 24, l: 32, r: 8 },
    xaxis: { showgrid: false, tickfont: { size: 11 } },
    yaxis: { ticksuffix: "%", gridcolor: "rgba(238,242,234,0.12)", zeroline: false, tickfont: { size: 11 } },
    // The 2–4% target band, so a reader sees at a glance when inflation ran hot.
    shapes: [{ type: "rect", xref: "paper", x0: 0, x1: 1, y0: 2, y1: 4, fillcolor: "rgba(238,242,234,0.08)", line: { width: 0 }, layer: "below" }],
  }, { displayModeBar: false, responsive: true });
  motion.revealPanel(box);
}

function showTip(tip, row) {
  if (!row) { tip.style.opacity = 0; return; }
  const status = row.value > 4 ? "above the 4% target" : "within target";
  tip.innerHTML = `<b>${row.id}</b><br>Inflation ${api.money(row.value)}%, rank ${row.rank}<br>` +
    `<span style="color:#9db4ad">${status}</span>`;
  tip.style.opacity = 1;
  tip.style.left = `${Math.min(row.x + 16, window.innerWidth - 280)}px`;
  tip.style.top = `${row.y + 16}px`;
}

function bindRail() {
  $("#regionSelect").addEventListener("change", (e) => {
    setState({ region: e.target.value || null });
  });
  $("#startDate").addEventListener("change", (e) => {
    const start = `${e.target.value}-01`;
    if (start >= state.end) return;
    setState({ start, focusDate: state.focusDate < start ? state.end : state.focusDate });
    refreshMonths();
  });
  $("#endDate").addEventListener("change", (e) => {
    const end = `${e.target.value}-01`;
    if (end <= state.start) return;
    setState({ end, focusDate: end });
    refreshMonths();
  });
  $("#metricSelect").addEventListener("change", (e) => setState({ metric: e.target.value }));
  $("#sectorSelect").addEventListener("change", (e) => setState({ sector: e.target.value || null }));
  $("#timeScrubber").addEventListener("input", (e) => {
    const date = monthList[Number(e.target.value)] || state.end;
    $("#scrubLabel").textContent = monthLabel(date);
    setState({ focusDate: date }, { silent: true });
    renderHero();
  });
  $("#timeScrubber").addEventListener("change", () => {
    setState({ focusDate: state.focusDate });
  });
  $("#playBtn").addEventListener("click", togglePlay);
  $("#timeScrubber").addEventListener("pointerdown", stopPlay);
  if (new URLSearchParams(location.search).get("debug") === "1") buildDebugBadge();
  $("#resetBtn").addEventListener("click", () => {
    stopPlay();
    clearRegion();
    setState({ start: "2018-01-01", end: "2026-10-01", focusDate: "2026-10-01", metric: "yoy", sector: null });
    $("#regionSelect").value = "";
    $("#sectorSelect").value = "";
    $("#metricSelect").value = "yoy";
    $("#startDate").value = "2018-01";
    $("#endDate").value = "2026-10";
    refreshMonths();
    api.clearCache();
  });
  $("#reloadBtn")?.addEventListener("click", () => {
    api.clearCache();
    rendered.clear();
    renderActive(true);
    renderHero();
  });
}

function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  const btn = $("#playBtn");
  btn.textContent = "Pause";
  let idx = monthList.indexOf(state.focusDate);
  if (idx >= monthList.length - 1) idx = 0; // restart from the beginning when at the end
  playTimer = setInterval(() => {
    idx += 1;
    if (idx >= monthList.length) { stopPlay(); return; }
    const date = monthList[idx];
    $("#timeScrubber").value = idx;
    $("#scrubLabel").textContent = monthLabel(date);
    setState({ focusDate: date }, { silent: true });
    renderHero();
  }, 650);
  updateNews();
}

function stopPlay() {
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  const btn = $("#playBtn");
  if (btn) btn.textContent = "Play";
  updateNews();
}

// Related news: national view, paused only.
let newsMonth = null;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function closeNews(refocus = false) {
  const stack = $("#newsStack");
  if (!stack || stack.hidden) return;
  stack.hidden = true;
  $("#newsBtn").setAttribute("aria-expanded", "false");
  if (refocus) $("#newsBtn").focus();
}

function updateNews() {
  const btn = $("#newsBtn");
  if (!btn) return;
  btn.hidden = Boolean(state.region || playTimer);
  if (btn.hidden || newsMonth !== state.focusDate) closeNews();
}

function driversLine(cur, prev) {
  if (!cur || !prev) return "";
  const before = new Map(prev.data.map((r) => [r.code, r.contribution]));
  const moves = cur.data
    .filter((r) => before.has(r.code))
    .map((r) => ({ name: r.name, d: r.contribution - before.get(r.code) }));
  const total = moves.reduce((s, m) => s + m.d, 0);
  if (!total) return "";
  const top = moves.filter((m) => Math.sign(m.d) === Math.sign(total))
    .sort((a, b) => Math.abs(b.d) - Math.abs(a.d)).slice(0, 2).map((m) => m.name.toLowerCase());
  const who = top.join(" and ");
  return `${who[0].toUpperCase()}${who.slice(1)} pushed inflation ${total > 0 ? "up" : "down"} from last month.`;
}


async function openNews() {
  const stack = $("#newsStack");
  const month = state.focusDate;
  newsMonth = month;
  stack.hidden = false;
  $("#newsBtn").setAttribute("aria-expanded", "true");
  $("#newsWhy").textContent = "Loading…";
  $("#newsCards").innerHTML = "";
  const prev = monthList[monthList.indexOf(month) - 1];
  const [news, cur, before] = await Promise.all([
    api.get("/news", { month }).catch(() => null),
    api.get("/drivers/treemap", { date: month }).catch(() => null),
    prev ? api.get("/drivers/treemap", { date: prev }).catch(() => null) : null,
  ]);
  if (stack.hidden || newsMonth !== month) return; // closed or month changed while loading
  const why = driversLine(cur, before);
  const articles = (news?.data || []).slice(0, 3);
  $("#newsWhy").textContent = why || (articles.length ? "" : `No related news found for ${monthLabel(month)}.`);
  $("#newsWhy").hidden = !$("#newsWhy").textContent;
  $("#newsCards").innerHTML = articles.map((a) => `
    <article class="news-card">
      ${a.image ? `<img src="${esc(a.image)}" alt="" loading="lazy">` : ""}
      <h3>${esc(a.title || a.domain)}</h3>
      ${a.summary ? `<p>${esc(a.summary.slice(0, 140).replace(/\s+\S*$/, "").replace(/[\s.,;:]+$/, ""))}....</p>` : ""}
      <a href="${esc(a.url)}" target="_blank" rel="noopener">Read more<span class="sr-only">: ${esc(a.title || a.domain)}</span></a>
      <button class="news-close" type="button" aria-label="Close related news">×</button>
    </article>`).join("");
  const first = stack.querySelector("a");
  if (first) first.focus();
  else { stack.tabIndex = -1; stack.focus(); }
}

$("#newsBtn")?.addEventListener("click", () => ($("#newsStack").hidden ? openNews() : closeNews()));
$("#newsCards")?.addEventListener("click", (e) => { if (e.target.closest(".news-close")) closeNews(true); });
$("#newsCards")?.addEventListener("error", (e) => { if (e.target.tagName === "IMG") e.target.remove(); }, true); // broken thumbnails vanish

function bindTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      activeTab = tab.dataset.tab;
      $$(".tab").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
      $$(".panel").forEach((p) => { p.hidden = p.id !== `panel-${activeTab}`; });
      motion.revealPanel($(`#panel-${activeTab}`));
      rendered.delete(activeTab);
      renderActive();
      requestAnimationFrame(() => {
        document.querySelectorAll(`#panel-${activeTab} .js-plotly-plot`)
          .forEach((p) => window.Plotly?.Plots.resize(p));
      });
    });
  });
}

function bindTools() {
  $$("[data-sql]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const panel = btn.dataset.sql;
      const args = (btn.dataset.sqlArgs || "")
        .replaceAll("MAPDATE", state.focusDate)
        .replaceAll("START", state.start)
        .replaceAll("END", state.end);
      const card = btn.closest(".card");
      let drawer = card.querySelector(".sql-drawer");
      if (!drawer) {
        drawer = document.createElement("div");
        drawer.className = "sql-drawer";
        drawer.innerHTML = '<div class="meta"></div><pre></pre>';
        card.appendChild(drawer);
      }
      drawer.classList.toggle("open");
      if (!drawer.classList.contains("open")) return;
      const pre = drawer.querySelector("pre");
      pre.textContent = "Loading…";
      try {
        const body = await api.get(`/sql/${panel}`);
        const meta = lastMeta.get(panel);
        const telemetry = meta
          ? `\n\n-- params: ${JSON.stringify(meta.params?.args ?? meta.params ?? {})}` +
            `\n-- rows: ${meta.row_count} · server: ${meta.elapsed_ms} ms · rtt: ${meta.rtt_ms} ms · cache: ${meta.cache}`
          : "";
        pre.textContent = `${panel}.sql\n\n${body.data.sql}${telemetry}`;
        drawer.querySelector(".meta").textContent = `/api/sql/${panel}`;
      } catch (err) {
        pre.textContent = `SQL preview unavailable: ${err.message}`;
      }
    });
  });

  $$("[data-export]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const panel = btn.dataset.export;
      const params = {
        map: { date: state.focusDate },
        headline: { start: state.start, end: state.end },
        calendar: { start: state.start, end: state.end },
        distribution: {},
        treemap: { date: state.focusDate },
        sector_heatmap: { end: state.end, start: state.start },
        region_table: {},
      }[panel] || {};
      api.exportPanel(panel, params);
    });
  });

  $$("[data-alt]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = document.getElementById(btn.dataset.alt);
      if (!t) return;
      t.hidden = !t.hidden;
      btn.textContent = t.hidden ? "Alt text" : "Hide alt text";
    });
  });
}

async function renderHero() {
  const kpiBody = await api.get("/kpis", {
    end: state.focusDate,
    region: state.region || undefined,
  });
  const k = kpiBody.data[0];
  renderKpis(k);
  renderChips(k);
  const mapRows = (await api.get("/map/choropleth", { date: state.focusDate })).data;
  const rows = state.metric === "mom"
    ? mapRows.map((r) => ({ geo_id: r.geo_id, value: r.mom_change, rank: r.rank }))
    : mapRows.map((r) => ({ geo_id: r.geo_id, value: r.value, rank: r.rank }));
  map.update(rows, state.metric === "mom" ? "mom" : "yoy");
  landingMap?.update(rows, state.metric === "mom" ? "mom" : "yoy");
  if (landingMap) {
    const mom = state.metric === "mom"; // ends mirror the fixed ranges in PHMap.update
    $("#landingLegendLabel").textContent = `${mom ? "Monthly change" : "Inflation"}, ${monthLabel(state.focusDate)}`;
    $("#landingLo").textContent = mom ? "-1.5% or less" : "0%";
    $("#landingHi").textContent = mom ? "1.5% or more" : "12% or more";
  }
  map.highlight(state.region);
  if (state.region !== lastRegion) {
    if (state.region) map.zoomTo(state.region);
    else map.resetView();
    lastRegion = state.region;
  }
  drawRank(rows);
  $("#playBtn").disabled = false;
  $("#mapBadge").textContent = `${state.metric === "mom" ? "Monthly change" : "Inflation"}, ${monthLabel(state.focusDate)}`;
  $("#rankHint").textContent = `${state.metric === "mom" ? "change vs prior month" : "latest inflation"}, ${monthLabel(state.focusDate)}`;
  updateNews();
}

function renderKpis(k) {
  const delta = k.prev == null ? null : k.headline - k.prev;
  const coreMeta = k.region && k.region !== "PHILIPPINES"
    ? "national figure (regional core not published)"
    : "steady underlying trend";
  const regional = k.region && k.region !== "PHILIPPINES";
  const range = regional && k.nowcast_lo80 != null ? `likely ${k.nowcast_lo80.toFixed(1)}–${k.nowcast_hi80.toFixed(1)}%` : null;
  const ab = k.regions_above_next;
  const cards = [
    ["Inflation now (vs a year ago)", api.pct(k.headline), delta == null ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} pts vs last month`, delta > 0 ? "up" : "down"],
    ["Price index (2018 = 100)", api.money(k.cpi), "the cost of a typical basket", ""],
    ["Inflation without food and fuel", api.pct(k.core), coreMeta, ""],
    ["Likely next month", api.pct(k.nowcast_next),
      regional ? (range || "this month carried forward") : "national forecast (SARIMA)", ""],
    regional
      ? ["Chance above 4% next month", k.breach_probability == null ? "—" : api.pct(k.breach_probability * 100), "model estimate", k.breach_probability > 0.5 ? "up" : ""]
      : ["Regions above 4% next month", ab?.count == null ? "—" : `${ab.count} of ${ab.total}`, "model estimate by region", ab?.count > 0 ? "up" : ""],
  ];
  const prevV = [...document.querySelectorAll("#kpis .v")].map((e) => e.textContent);
  $("#kpis").innerHTML = cards.map(([label, value, meta2, cls]) =>
    `<div class="kpi"><div class="k">${label}</div><div class="v">${value}</div>` +
    `<div class="d ${cls}">${meta2}</div></div>`).join("");
  $$("#kpis .v").forEach((el, i) => motion.countUp(el, prevV[i], !!playTimer));
}

function renderChips(k) {
  const above = k.headline > 4;
  $("#chips").innerHTML =
    `<span class="chip">${state.region ? "region" : "national"}, ${monthLabel(state.focusDate)}</span>` +
    `<span class="chip ${above ? "red" : "green"}">${above ? "above 2–4% target" : "within target"}</span>`;
  $("#pageTitle").textContent = state.region || "National overview";
  const live = $("#sr-live");
  if (live) live.textContent = state.region ? `${state.region} selected` : "National view";
}

function drawRank(rows) {
  const list = $("#rankList");
  if (!rows.length) { list.innerHTML = `<p class="rank-empty">No regional data for ${monthLabel(state.focusDate)}</p>`; return; }
  const prevW = new Map([...list.querySelectorAll(".rank-row")].map((r) => [r.dataset.id, r.querySelector(".bar > span")?.style.width]));
  const sorted = [...rows].sort((a, b) => (b.value ?? -99) - (a.value ?? -99));
  const max = Math.max(...rows.map((r) => Math.abs(r.value || 0)), 1);
  list.innerHTML = sorted.map((r, i) => {
    const hot = state.metric === "mom" ? (r.value || 0) > 0.5 : (r.value || 0) > 4;
    const active = state.region === r.geo_id;
    return `<div class="rank-row ${hot ? "hot" : ""} ${active ? "active" : ""}" role="option" ` +
      `aria-selected="${active}" tabindex="0" data-id="${r.geo_id}">` +
      `<span class="n">${String(i + 1).padStart(2, "0")}</span>` +
      `<span class="nm">${meta.regions.find((m) => m.geo_id === r.geo_id)?.short_code || r.geo_id}</span>` +
      `<span class="val">${r.value == null ? "—" : (r.value >= 0 && state.metric === "mom" ? "+" : "") + r.value.toFixed(1)}%</span>` +
      `<span class="bar"><span style="width:${prevW.get(r.geo_id) || "0%"}" data-w="${Math.min(100, (Math.abs(r.value || 0) / max) * 100)}"></span></span>` +
      `</div>`;
  }).join("");
  list.querySelectorAll(".rank-row").forEach((row) => {
    const go = () => setState({ region: row.dataset.id });
    row.addEventListener("click", go);
    row.addEventListener("pointerenter", () => map.setHover(row.dataset.id));
    row.addEventListener("pointerleave", () => map.setHover(null));
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
  motion.growBars(list, !!playTimer);
}

async function renderActive(force = false) {
  if (!force && rendered.has(activeTab)) return;
  const tab = activeTab;
  const panel = document.getElementById(`panel-${tab}`);
  panel?.querySelectorAll(":scope > .load-error").forEach((el) => el.remove()); // don't stack banners across retries
  try {
    await RENDERERS[tab](state);
    rendered.add(tab);
  } catch (err) {
    console.error(err);
    if (!panel) return;
    const banner = document.createElement("div");
    banner.className = "empty load-error";
    banner.textContent = `Panel failed to load: ${err.message} `;
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "btn";
    retry.textContent = "Retry";
    retry.addEventListener("click", () => { if (activeTab === tab) renderActive(true); });
    banner.append(retry);
    panel.prepend(banner);
  }
}

let lastState = null;

function onState() {
  if ($("#regionSelect").value !== (state.region || "")) {
    $("#regionSelect").value = state.region || "";
  }
  if ($("#sectorSelect").value !== (state.sector || "")) {
    $("#sectorSelect").value = state.sector || "";
  }
  if ($("#metricSelect").value !== state.metric) $("#metricSelect").value = state.metric;

  // Sector clicks only affect the drivers highlight, not the map, KPIs or charts.
  if (state.selectedPoint && state.selectedPoint.region !== state.region) state.selectedPoint = null;
  const keys = ["region", "start", "end", "focusDate", "metric", "sector"];
  const changed = !lastState || keys.some((k) => state[k] !== lastState[k]);
  const sectorOnly = lastState
    && state.sector !== lastState.sector
    && ["region", "start", "end", "focusDate", "metric"].every((k) => state[k] === lastState[k]);
  lastState = { ...state };
  if (!changed) return;
  if (sectorOnly) {
    if (activeTab === "drivers") highlightSector(state.sector);
    return;
  }

  refreshMonths();
  renderHero();
  renderActive(true);
}

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#newsStack")?.hidden) { closeNews(true); return; }
  if (landingMap && !$("#landing").hidden) { landingMap.highlight(null); landingMap.onClear(); } // landing has its own selection
  else clearRegion();
});

bootstrap().catch((err) => {
  console.error(err);
  document.querySelector(".main").insertAdjacentHTML("afterbegin",
    `<div class="empty">Failed to start the dashboard: ${err.message}</div>`);
});
