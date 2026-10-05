(async function () {
  const msg = $('#msg');
  const meta = await api('/api/meta');
  const opts = meta.filter_options;
  const fill = (id, values) => values.forEach((v) => $(id).insertAdjacentHTML('beforeend', `<option>${esc(v)}</option>`));
  fill('#f-contract', opts.contract); fill('#f-internet', opts.internet); fill('#f-gender', opts.gender);

  const params = () => ({ contract: $('#f-contract').value, internet: $('#f-internet').value, gender: $('#f-gender').value,
    senior: $('#f-senior').value, tenure_min: $('#f-tmin').value, tenure_max: $('#f-tmax').value });

  async function load() {
    msg.innerHTML = '';
    const p = params();
    if (p.tenure_min !== '' && p.tenure_max !== '' && +p.tenure_min > +p.tenure_max) {
      showError(msg, '"Tenure from" must not be larger than "Tenure to".'); return;
    }
    let d;
    try { d = await api('/api/dashboard', p); } catch (e) { showError(msg, e.message); return; }
    const k = d.kpis;
    if (k.total_customers === 0) msg.innerHTML = '<div class="alert-err">No customers match these filters. Widen the filters to see data.</div>';
    const cards = [['Total customers', int(k.total_customers)], ['Churned customers', int(k.churned_customers)],
      ['Churn rate', pct(k.churn_rate)], ['Average monthly charges', money(k.avg_monthly_charges)],
      ['Average tenure (months)', k.avg_tenure], ['High-risk customers', int(k.high_risk_customers), true]];
    $('#kpis').innerHTML = cards.map(([l, v, a]) => `<div class="card kpi ${a ? 'alert' : ''}"><div class="label">${l}</div><div class="value">${v}</div></div>`).join('');

    drawChart('c-dist', { type: 'doughnut',
      data: { labels: ['Stayed', 'Churned'], datasets: [{ data: [d.distribution.stayed, d.distribution.churned], backgroundColor: [COLORS.stayed, COLORS.churned], borderWidth: 2 }] },
      options: { cutout: '62%', plugins: { title: titleOpts('Overall churn distribution (customers)'), legend: { position: 'bottom' } } } });
    rateBar('c-contract', 'Churn rate by contract', d.by_contract, 'Contract type');
    rateBar('c-tenure', 'Churn rate by tenure', d.by_tenure, 'Customer tenure');
    rateBar('c-charges', 'Churn rate by monthly charges', d.by_charges, 'Monthly charges');
    rateBar('c-internet', 'Churn rate by internet service', d.by_internet, 'Internet service');
    rateBar('c-payment', 'Churn rate by payment method', d.by_payment, 'Payment method', true);
    rateBar('c-demo', 'Churn rate by customer demographics',
      d.by_demographics.map((r) => ({ ...r, label: `${r.group}: ${r.label}` })), 'Demographic group', true);
  }

  $$('#filters select, #filters input').forEach((el) => el.addEventListener('change', load));
  $('#reset').addEventListener('click', () => { $('#filters').reset(); load(); });
  load();
})();
