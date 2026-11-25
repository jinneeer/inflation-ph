/* Shared Plotly theme + small DOM helpers. */
export const C = {
  ink: "#14201f", muted: "#5b6b68", line: "#e2ddd0", paper: "#fffdf8",
  teal: "#0d6e63", tealSoft: "#8fb7a9", vermilion: "#c74b2b",
  amber: "#b8791f", graphite: "#3f4a48", navy: "#0f2f2c",
};

export const baseLayout = (overrides = {}) => ({
  paper_bgcolor: C.paper,
  plot_bgcolor: C.paper,
  font: { family: "IBM Plex Sans, sans-serif", size: 12, color: C.ink },
  margin: { l: 48, r: 18, t: 14, b: 40 },
  hoverlabel: { bgcolor: C.navy, font: { color: "#f6f4ee", family: "IBM Plex Mono, monospace", size: 11 } },
  xaxis: { gridcolor: C.line, zeroline: false, tickfont: { size: 10.5 } },
  yaxis: { gridcolor: C.line, zeroline: false, tickfont: { size: 10.5 } },
  legend: { orientation: "h", y: 1.12, x: 0, font: { size: 10.5 } },
  ...overrides,
});

export const config = { displayModeBar: false, responsive: true };

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function monthLabel(iso) {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleString("en", { month: "short", year: "numeric" });
}

export function statusChip(value) {
  return value > 4 ? "red" : "green";
}

export function clearNode(node) { while (node.firstChild) node.removeChild(node.firstChild); }

export function empty(container, text = "No data for this selection.") {
  container.innerHTML = `<div class="empty">${text}</div>`;
}

export function csvDownload(filename, rows) {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const lines = [keys.join(",")];
  rows.forEach((r) => lines.push(keys.map((k) => {
    const v = r[k] ?? "";
    return /[",\n]/.test(String(v)) ? `"${String(v).replaceAll('"', '""')}"` : v;
  }).join(",")));
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// Plain-language names for model inputs (the model itself uses the short codes).
export const FEATURE_LABELS = {
  infl: "This month's rate", lag1: "1 month ago", lag2: "2 months ago", lag3: "3 months ago",
  lag6: "6 months ago", lag12: "12 months ago", ma3: "3-month average", ma6: "6-month average",
  ma12: "12-month average", vol12: "How bumpy (12 mo)", month_sin: "Time of year (a)",
  month_cos: "Time of year (b)", time_idx: "Years since 2019",
};
export const featureLabel = (f) => FEATURE_LABELS[f] || f;

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

export function shortRegion(geo) {
  const [head, tail = ""] = String(geo).split(" (");
  const t = tail.replace(")", "");
  return t && t === t.toUpperCase() && t.length <= 6 ? t : head;
}
