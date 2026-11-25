/* Data & SQL: region table, metrics, SQL drawers, exports. */
import { get } from "../api.js";
import { setState } from "../store.js";
import { $ } from "./common.js";

export async function renderData(state) {
  const regions = (await get("/table/geolocations")).data;
  drawRegionTable(regions, state);
  const metrics = (await get("/model/metrics")).data;
  drawMetrics(metrics);
}

function drawRegionTable(rows, state) {
  const table = $("#dataRegionTable");
  table.innerHTML =
    "<tr><th>Rank</th><th>Region</th><th>Inflation (year-on-year)</th><th>Previous month</th><th>Change vs prior</th><th>Status</th></tr>" +
    rows.map((r) => `<tr data-id="${r.region}" style="${state.region === r.region ? "background:var(--teal-soft)" : ""}">` +
      `<td>${r.rank}</td><td class="name">${r.region}</td>` +
      `<td>${r.infl.toFixed(1)}%</td><td>${r.prev == null ? "—" : r.prev.toFixed(1) + "%"}</td>` +
      `<td>${r.mom_change == null ? "—" : (r.mom_change >= 0 ? "+" : "") + r.mom_change.toFixed(1)}</td>` +
      `<td>${r.infl > 4 ? "Above 4%" : "Within band"}</td></tr>`).join("");
  table.querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.addEventListener("click", () => setState({ region: tr.dataset.id }));
  });
}

function drawMetrics(m) {
  const table = $("#metricsTable");
  const f = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));
  const pf = m.nowcast?.folds?.slice(-1)[0]?.regression || {};
  const carry = pf["Persistence (this month carried forward)"], dep = m.gbm;
  const nat = m.national_rolling?.mean || {};
  const iv = m.nowcast?.interval || {};
  const rows = [
    ["National SARIMA, avg / RMS miss, 4-window mean (pts)", `${f(nat.SARIMA?.MAE)} / ${f(nat.SARIMA?.RMSE)}`],
    ["National Holt-Winters, avg / RMS miss, 4-window mean", `${f(nat["Holt-Winters"]?.MAE)} / ${f(nat["Holt-Winters"]?.RMSE)}`],
    ["National SARIMA 95% band: share of months inside", f(nat.SARIMA?.coverage95)],
    ["Regional next month: simple guess (this month), avg / RMS miss", `${f(carry?.MAE)} / ${f(carry?.RMSE)}`],
    ["Regional next month: boosted-trees model, avg / RMS miss / R²", `${f(dep?.MAE)} / ${f(dep?.RMSE)} / ${f(dep?.R2)}`],
    ["Regional guess range: share of cases inside 80% / 95%", `${f(iv.coverage80_primary)} / ${f(iv.coverage95_primary)}`],
    ["Above-4% model: ranking score AUC / average precision", `${f(m.classifier?.AUC, 3)} / ${f(m.classifier?.average_precision, 3)}`],
    ["Above-4% simple rule (this month > 4%): AUC", f(m.classifier_baseline?.AUC, 3)],
    ["Main test window", m.nowcast?.primary_window ?? "—"],
    ["Series", `${m.series_start?.slice(0, 10)} to ${m.series_end?.slice(0, 10)}, n = ${m.n_obs}`],
    ["Trained at", m.trained_at],
  ];
  table.innerHTML = rows.map(([k, v]) => `<tr><td class="name">${k}</td><td>${v}</td></tr>`).join("");
}

export async function showSql(panel, argsText) {
  const box = $("#dataSql");
  const pre = $("#dataSqlText");
  box.classList.add("open");
  pre.textContent = "Loading…";
  try {
    const body = await get(`/sql/${panel}`);
    pre.textContent = `${panel}.sql\n\n${body.data.sql}` +
      (argsText ? `\n\n-- params: ${argsText}` : "");
  } catch (err) {
    pre.textContent = `SQL preview unavailable for ${panel}: ${err.message}`;
  }
}
