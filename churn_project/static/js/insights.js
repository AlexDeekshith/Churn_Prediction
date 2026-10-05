(async function () {
  let d;
  try { d = await api('/api/insights'); } catch (e) { showError($('#msg'), e.message); return; }
  $('#n').textContent = int(d.based_on_customers);
  $('#findings').innerHTML = d.findings.map((f) => `<div class="finding"><h3>${esc(f.title)}</h3><p>${esc(f.text)}</p></div>`).join('');
  const title = Object.fromEntries(d.findings.map((f) => [f.id, f.title]));
  $('#actions').innerHTML = d.actions.map((a) => `<div class="action"><h3>${esc(a.title)}</h3><p>${esc(a.rationale)}</p>
    <div class="meta">Based on: ${esc(title[a.based_on] || 'model risk scores')} · Customers to target: <strong>${int(a.target_customers)}</strong></div></div>`).join('');
})();
