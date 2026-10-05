(async function () {
  const meta = await api('/api/meta');
  const cats = meta.categories;
  const ADDONS = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'];
  const GROUPS = [
    ['Customer', [['gender', 'Gender'], ['SeniorCitizen', 'Senior citizen', 'flag'], ['Partner', 'Partner'], ['Dependents', 'Dependents']]],
    ['Account and billing', [['tenure', 'Tenure (months)', 'int'], ['Contract', 'Contract'], ['PaperlessBilling', 'Paperless billing'],
      ['PaymentMethod', 'Payment method'], ['MonthlyCharges', 'Monthly charges ($)', 'money'], ['TotalCharges', 'Total charges ($, optional)', 'money']]],
    ['Services', [['PhoneService', 'Phone service'], ['MultipleLines', 'Multiple lines'], ['InternetService', 'Internet service'],
      ['OnlineSecurity', 'Online security'], ['OnlineBackup', 'Online backup'], ['DeviceProtection', 'Device protection'],
      ['TechSupport', 'Tech support'], ['StreamingTV', 'Streaming TV'], ['StreamingMovies', 'Streaming movies']]],
  ];
  const DEFAULTS = { gender: 'Female', SeniorCitizen: '0', Partner: 'No', Dependents: 'No', tenure: 12, Contract: 'Month-to-month',
    PaperlessBilling: 'Yes', PaymentMethod: 'Electronic check', MonthlyCharges: 70, TotalCharges: '', PhoneService: 'Yes',
    MultipleLines: 'No', InternetService: 'Fiber optic', OnlineSecurity: 'No', OnlineBackup: 'No', DeviceProtection: 'No',
    TechSupport: 'No', StreamingTV: 'No', StreamingMovies: 'No' };
  // Example inputs for the form only; the probability is always computed by the model.
  const EXAMPLES = {
    high: { ...DEFAULTS, tenure: 2, MonthlyCharges: 95.5, StreamingTV: 'Yes', StreamingMovies: 'Yes' },
    low: { ...DEFAULTS, gender: 'Male', Partner: 'Yes', Dependents: 'Yes', tenure: 58, Contract: 'Two year', PaperlessBilling: 'No',
      PaymentMethod: 'Credit card (automatic)', MonthlyCharges: 64, TotalCharges: 3700, InternetService: 'DSL', OnlineSecurity: 'Yes', TechSupport: 'Yes' },
  };

  const control = (key, label, kind) => {
    if (kind === 'int') return `<label class="field" data-f="${key}">${label}<input type="number" name="${key}" min="0" max="120" step="1" required></label>`;
    if (kind === 'money') return `<label class="field" data-f="${key}">${label}<input type="number" name="${key}" min="0" step="0.01" ${key === 'TotalCharges' ? 'placeholder="Estimated if blank"' : 'required'}></label>`;
    if (kind === 'flag') return `<label class="field" data-f="${key}">${label}<select name="${key}"><option value="0">No</option><option value="1">Yes</option></select></label>`;
    return `<label class="field" data-f="${key}">${label}<select name="${key}">${cats[key].map((c) => `<option>${esc(c)}</option>`).join('')}</select></label>`;
  };
  $('#form-body').innerHTML = GROUPS.map(([t, fs]) => `<fieldset><legend>${t}</legend><div class="form-grid">${fs.map(([k, l, kind]) => control(k, l, kind)).join('')}</div></fieldset>`).join('');
  const form = $('#form');
  const el = (n) => form.elements[n];

  // Services that depend on internet / phone service.
  function syncDependents() {
    const net = el('InternetService').value !== 'No', phone = el('PhoneService').value === 'Yes';
    const rebuild = (name, hasService, noLabel) => {
      const sel = el(name), current = sel.value;
      const options = hasService ? cats[name].filter((c) => c !== noLabel) : [noLabel];
      sel.innerHTML = options.map((c) => `<option>${esc(c)}</option>`).join('');
      sel.value = options.includes(current) ? current : options[0];
      sel.disabled = !hasService;
    };
    ADDONS.forEach((n) => rebuild(n, net, 'No internet service'));
    rebuild('MultipleLines', phone, 'No phone service');
  }
  function setValues(v) {
    Object.entries(v).forEach(([k, val]) => { el(k).value = val; });
    syncDependents();
    Object.entries(v).forEach(([k, val]) => { if (!el(k).disabled) el(k).value = val; });
  }
  el('InternetService').addEventListener('change', syncDependents);
  el('PhoneService').addEventListener('change', syncDependents);
  setValues(DEFAULTS);
  $('#ex-high').onclick = () => setValues(EXAMPLES.high);
  $('#ex-low').onclick = () => setValues(EXAMPLES.low);

  // ----- gauge (the one expressive element on this page)
  function gauge(p, t) {
    const cx = 110, cy = 105, r = 84, pt = (f, rad = r) => { const a = Math.PI * (1 - f); return [cx + rad * Math.cos(a), cy - rad * Math.sin(a)]; };
    const arc = (f0, f1) => { const [x0, y0] = pt(f0), [x1, y1] = pt(f1); return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`; };
    const col = p >= t.high_min ? COLORS.High : p >= t.low_max ? COLORS.Medium : COLORS.Low;
    const [nx, ny] = pt(p, r - 20);
    return `<svg class="gauge" viewBox="0 0 220 125" role="img" aria-label="Churn probability ${pct(p, 0)}">
      <path d="${arc(0, t.low_max)}" stroke="${COLORS.Low}" stroke-opacity=".3" stroke-width="16" fill="none"/>
      <path d="${arc(t.low_max, t.high_min)}" stroke="${COLORS.Medium}" stroke-opacity=".3" stroke-width="16" fill="none"/>
      <path d="${arc(t.high_min, 1)}" stroke="${COLORS.High}" stroke-opacity=".3" stroke-width="16" fill="none"/>
      ${p > 0.004 ? `<path d="${arc(0, Math.min(p, 0.999))}" stroke="${col}" stroke-width="16" fill="none"/>` : ''}
      <line x1="${cx}" y1="${cy}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" stroke="#17212b" stroke-width="2.5" stroke-linecap="round"/>
      <circle cx="${cx}" cy="${cy}" r="5" fill="#17212b"/>
      <text x="14" y="122" font-size="9" fill="#5b6b77">0%</text><text x="188" y="122" font-size="9" fill="#5b6b77">100%</text></svg>`;
  }
  const factorRows = (list, cls) => {
    if (!list.length) return '<p class="muted small">No factors moved this customer\'s probability by 1 point or more.</p>';
    const max = Math.max(...list.map((f) => Math.abs(f.impact_points)));
    return list.map((f) => `<div class="factor ${cls}"><div>${esc(f.label)}: <strong>${esc(f.value)}</strong>
      <small>${f.impact_points > 0 ? '+' : ''}${f.impact_points} percentage points</small></div>
      <div class="bar"><i style="width:${(Math.abs(f.impact_points) / max * 100).toFixed(0)}%"></i></div></div>`).join('');
  };
  function render(r) {
    const t = r.thresholds;
    $('#result').innerHTML = `${gauge(r.probability, t)}
      <div class="gauge-value">${pct(r.probability, 0)}</div>
      <div class="muted small" style="text-align:center">Churn probability: ${pct(r.probability, 1)}</div>
      <div class="verdict ${r.churn ? 'churn' : 'stay'}">${esc(r.prediction)}</div>
      <p style="text-align:center"><span class="badge ${r.risk_level}">${r.risk_level} risk</span></p>
      ${r.notes.map((n) => `<p class="small muted">${esc(n)}</p>`).join('')}
      <h3 style="margin-top:1rem">Why this customer may be at risk</h3>
      <p class="small muted">Change in this customer's churn probability compared with replacing each detail by typical customer values, calculated from the trained model.</p>
      <h3 class="small">Pushing risk up</h3>${factorRows(r.factors.increase_risk, 'up')}
      <h3 class="small" style="margin-top:.6rem">Holding risk down</h3>${factorRows(r.factors.reduce_risk, 'down')}
      <p class="small muted" style="margin-top:.8rem">Risk bands are business rules: Low below ${pct(t.low_max, 0)}, High from ${pct(t.high_min, 0)}. They can be changed in the configuration (RISK_LOW_MAX, RISK_HIGH_MIN). The label uses a ${pct(t.decision, 0)} model cutoff.</p>`;
  }

  async function loadHistory() {
    const { predictions } = await api('/api/predictions');
    $('#history').innerHTML = predictions.length
      ? '<tr><th>Time (UTC)</th><th>Prediction</th><th class="num">Probability</th><th>Risk</th><th>Contract</th><th class="num">Tenure</th></tr>' +
        predictions.map((p) => `<tr><td>${esc(p.created_at.replace('T', ' ').replace('+00:00', ''))}</td><td>${esc(p.prediction)}</td><td class="num">${pct(p.churn_probability)}</td>
          <td><span class="badge ${p.risk_category}">${p.risk_category}</span></td><td>${esc(p.input.Contract)}</td><td class="num">${p.input.tenure}</td></tr>`).join('')
      : '<tr><td class="muted">No predictions yet.</td></tr>';
  }

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    $$('.field', form).forEach((f) => { f.classList.remove('invalid'); const e = $('.err', f); if (e) e.remove(); });
    const body = {};
    new FormData(form).forEach((v, k) => { body[k] = v; });
    ADDONS.concat('MultipleLines').forEach((n) => { body[n] = el(n).value; }); // disabled selects are skipped by FormData
    const btn = $('#submit'); btn.disabled = true;
    try {
      render(await api('/api/predict', {}, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }));
      loadHistory();
    } catch (e) {
      $('#result').innerHTML = `<div class="alert-err" role="alert">${esc(e.message)} Check the highlighted fields.</div>`;
      Object.entries(e.details || {}).forEach(([k, m]) => { const f = $(`[data-f="${k}"]`); if (f) { f.classList.add('invalid'); f.insertAdjacentHTML('beforeend', `<span class="err">${esc(m)}</span>`); } });
    } finally { btn.disabled = false; }
  });
  loadHistory();
})();
