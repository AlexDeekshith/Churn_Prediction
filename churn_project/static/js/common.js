/* Shared helpers: fetching, formatting and chart defaults. */
const COLORS = { stayed: '#6f8ea0', churned: '#b3261e', primary: '#0e6b6b', grid: '#e3e9ec',
                 low: '#2c7a57', Medium: '#b77a0b', High: '#b3261e', Low: '#2c7a57' };

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
const int = (x) => Number(x).toLocaleString('en-US');
const money = (x) => '$' + Number(x).toFixed(2);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, params = {}, options = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => { if (v !== '' && v != null) url.searchParams.set(k, v); });
  const res = await fetch(url, options);
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON error */ }
  if (!res.ok) { const e = new Error((body && body.error) || `Request failed (${res.status})`); e.details = body && body.details; throw e; }
  return body;
}

Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
Chart.defaults.color = '#44535f';
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.labels.boxWidth = 12;

const charts = {};
function drawChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
}

function titleOpts(text) { return { display: true, text, font: { size: 13, weight: '600' }, padding: { bottom: 10 } }; }
function axes(xTitle, yTitle, extra = {}) {
  return {
    x: { title: { display: !!xTitle, text: xTitle }, grid: { display: false }, ...(extra.x || {}) },
    y: { title: { display: !!yTitle, text: yTitle }, grid: { color: COLORS.grid }, beginAtZero: true, ...(extra.y || {}) },
  };
}

/* Churn-rate bar chart for [{label,total,churned,rate}] rows. Tooltip shows counts. */
function rateBar(id, title, rows, xTitle, horizontal = false) {
  const rateAxis = { ticks: { callback: (v) => v + '%' } };
  drawChart(id, {
    type: 'bar',
    data: { labels: rows.map((r) => r.label), datasets: [{ label: 'Churn rate', data: rows.map((r) => +(r.rate * 100).toFixed(1)), backgroundColor: COLORS.churned, borderRadius: 3 }] },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      plugins: { title: titleOpts(title), legend: { display: false },
        tooltip: { callbacks: { label: (c) => { const r = rows[c.dataIndex]; return `${r.rate * 100 > 0 ? (r.rate * 100).toFixed(1) : 0}% churn (${int(r.churned)} of ${int(r.total)})`; } } } },
      scales: horizontal ? { x: { ...axes('', 'Churn rate (%)').y, title: { display: true, text: 'Churn rate (%)' }, ...rateAxis }, y: { grid: { display: false } } }
                         : axes(xTitle, 'Churn rate (%)', { y: rateAxis }),
    },
  });
}

function showError(container, message) {
  container.innerHTML = `<div class="alert-err" role="alert">${esc(message)}</div>`;
}
