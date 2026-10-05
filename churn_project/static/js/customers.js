(async function () {
  const meta = await api('/api/meta');
  $('#low').value = meta.thresholds.low_max; $('#high').value = meta.thresholds.high_min;
  const s = { page: 1, sort: 'churn_probability', order: 'desc', risk: '', search: '' };
  const COLS = [['customer_id', 'Customer', ''], ['churn_probability', 'Churn probability', 'num'], [null, 'Risk', ''], ['Contract', 'Contract', ''],
    ['tenure', 'Tenure (months)', 'num'], ['MonthlyCharges', 'Monthly charges', 'num']];
  let timer;

  async function load() {
    $('#msg').innerHTML = '';
    let d;
    try { d = await api('/api/customers', { ...s, low_max: $('#low').value, high_min: $('#high').value }); }
    catch (e) { showError($('#msg'), e.message); return; }
    const c = d.risk_counts, all = c.High + c.Medium + c.Low;
    $('#chips').innerHTML = [['', 'All', all], ['High', 'High risk', c.High], ['Medium', 'Medium risk', c.Medium], ['Low', 'Low risk', c.Low]]
      .map(([v, l, n]) => `<button class="chip ${s.risk === v ? 'on' : ''}" data-risk="${v}">${l} (${int(n)})</button>`).join('');
    $('#tbl').innerHTML = '<tr>' + COLS.map(([k, l, cls]) => k
      ? `<th class="sortable ${cls}" data-sort="${k}" aria-sort="${s.sort === k ? (s.order === 'asc' ? 'ascending' : 'descending') : 'none'}">${l}${s.sort === k ? (s.order === 'asc' ? ' ▲' : ' ▼') : ''}</th>`
      : `<th>${l}</th>`).join('') + '</tr>' +
      (d.customers.length ? d.customers.map((r) => `<tr class="${r.risk === 'High' ? 'high' : ''}"><td>${esc(r.customer_id)}</td>
        <td class="num">${pct(r.churn_probability)}</td><td><span class="badge ${r.risk}">${r.risk}</span></td><td>${esc(r.Contract)}</td>
        <td class="num">${r.tenure}</td><td class="num">${money(r.MonthlyCharges)}</td></tr>`).join('')
        : '<tr><td colspan="6" class="muted">No customers match. Try a different ID or risk band.</td></tr>');
    $('#count').textContent = `${int(d.total)} customers`;
    $('#pageinfo').textContent = `Page ${d.page} of ${d.pages}`;
    $('#prev').disabled = d.page <= 1; $('#next').disabled = d.page >= d.pages;
  }

  $('#chips').addEventListener('click', (e) => { const b = e.target.closest('[data-risk]'); if (b) { s.risk = b.dataset.risk; s.page = 1; load(); } });
  $('#tbl').addEventListener('click', (e) => { const th = e.target.closest('[data-sort]'); if (!th) return;
    if (s.sort === th.dataset.sort) s.order = s.order === 'asc' ? 'desc' : 'asc'; else { s.sort = th.dataset.sort; s.order = 'desc'; }
    s.page = 1; load(); });
  $('#search').addEventListener('input', (e) => { clearTimeout(timer); timer = setTimeout(() => { s.search = e.target.value; s.page = 1; load(); }, 250); });
  $('#apply').addEventListener('click', () => { s.page = 1; load(); });
  $('#prev').onclick = () => { s.page--; load(); };
  $('#next').onclick = () => { s.page++; load(); };
  load();
})();
