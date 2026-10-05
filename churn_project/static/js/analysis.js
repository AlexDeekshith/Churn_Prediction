(function () {
  const hist = (id, title, h, xTitle) => drawChart(id, { type: 'bar',
    data: { labels: h.labels, datasets: [
      { label: 'Stayed', data: h.stayed, backgroundColor: COLORS.stayed },
      { label: 'Churned', data: h.churned, backgroundColor: COLORS.churned }] },
    options: { plugins: { title: titleOpts(title), legend: { position: 'bottom' } }, scales: axes(xTitle, 'Customers') } });

  async function load() {
    const by = $('#by').value;
    let d;
    try { d = await api('/api/analysis', { by }); } catch (e) { showError($('#msg'), e.message); return; }
    rateBar('c-seg', `Churn rate by ${d.by_label.toLowerCase()}`, d.segments, d.by_label);
    $('#seg-title').textContent = d.by_label;
    $('#seg-table').innerHTML = '<tr><th>Segment</th><th class="num">Customers</th><th class="num">Churned</th><th class="num">Churn rate</th></tr>' +
      d.segments.map((s) => `<tr><td>${esc(s.label)}</td><td class="num">${int(s.total)}</td><td class="num">${int(s.churned)}</td><td class="num">${pct(s.rate)}</td></tr>`).join('');
    hist('c-ten', 'Customer tenure: stayed vs churned', d.tenure_hist, 'Tenure (months)');
    hist('c-chg', 'Monthly charges: stayed vs churned', d.charges_hist, 'Monthly charges ($)');
    const names = { tenure: ['Tenure (months)', (x) => x.toFixed(1)], MonthlyCharges: ['Monthly charges', money], TotalCharges: ['Total charges', money] };
    $('#profile').innerHTML = '<tr><th>Measure</th><th class="num">Stayed (mean)</th><th class="num">Churned (mean)</th><th class="num">Stayed (median)</th><th class="num">Churned (median)</th></tr>' +
      Object.entries(d.profile).map(([k, v]) => { const [n, f] = names[k];
        return `<tr><td>${n}</td><td class="num">${f(v.stayed.mean)}</td><td class="num">${f(v.churned.mean)}</td><td class="num">${f(v.stayed.median)}</td><td class="num">${f(v.churned.median)}</td></tr>`; }).join('');
  }
  $('#by').addEventListener('change', load);
  load();
})();
