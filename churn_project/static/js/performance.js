(async function () {
  let m;
  try { m = await api('/api/model-performance'); } catch (e) { showError($('#msg'), e.message); return; }
  const names = Object.keys(m.models), f3 = (x) => x.toFixed(3);
  $('#sub').textContent = `Metrics calculated on the held-out test set (${int(m.n_test)} customers). Each model uses its own churn cutoff, tuned on training data.`;
  $('#rule').textContent = `Selected model: ${m.selected_model}. Selection rule: ${m.selection_rule}. The test set was not used to choose it.`;
  $('#cmp').innerHTML = '<tr><th>Model</th><th class="num">Accuracy</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">ROC-AUC</th><th class="num">Cutoff</th><th class="num">CV F1 (train)</th></tr>' +
    names.map((n) => { const r = m.models[n];
      return `<tr class="${n === m.selected_model ? 'selected' : ''}"><td>${esc(n)} ${n === m.selected_model ? '<span class="badge Low">selected</span>' : ''}</td>
        <td class="num">${f3(r.accuracy)}</td><td class="num">${f3(r.precision)}</td><td class="num">${f3(r.recall)}</td><td class="num">${f3(r.f1)}</td><td class="num">${f3(r.roc_auc)}</td><td class="num">${pct(r.threshold, 0)}</td><td class="num">${f3(r.cv_f1)}</td></tr>`; }).join('');
  $('#which').innerHTML = names.map((n) => `<option ${n === m.selected_model ? 'selected' : ''}>${esc(n)}</option>`).join('');

  function details() {
    const n = $('#which').value, r = m.models[n], c = r.confusion_matrix;
    $('#cm').innerHTML = `<p class="small muted">${esc(n)}: rows are the actual outcome, columns the prediction.</p><div class="cm">
      <div class="h"></div><div class="h">Predicted stay</div><div class="h">Predicted churn</div>
      <div class="h">Actually stayed</div><div class="good"><b>${int(c.tn)}</b>true negatives</div><div class="bad"><b>${int(c.fp)}</b>false positives</div>
      <div class="h">Actually churned</div><div class="bad"><b>${int(c.fn)}</b>false negatives</div><div class="good"><b>${int(c.tp)}</b>true positives</div></div>`;
    const cr = r.classification_report;
    $('#report').innerHTML = '<tr><th>Class</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th></tr>' +
      ['Stayed', 'Churned', 'macro avg', 'weighted avg'].map((k) => `<tr><td>${k}</td><td class="num">${f3(cr[k].precision)}</td><td class="num">${f3(cr[k].recall)}</td><td class="num">${f3(cr[k]['f1-score'])}</td><td class="num">${int(cr[k].support)}</td></tr>`).join('') +
      `<tr><td>accuracy</td><td></td><td></td><td class="num">${f3(cr.accuracy)}</td><td class="num">${int(cr['macro avg'].support)}</td></tr>`;
    rocChart();
  }
  const palette = ['#0e6b6b', '#6f8ea0', '#b77a0b', '#7a4f9a', '#2c7a57'];
  function rocChart() {
    const sel = $('#which').value;
    const sets = names.map((n, i) => ({ label: `${n} (AUC ${f3(m.models[n].roc_auc)})`, showLine: true, pointRadius: 0, tension: 0,
      borderColor: palette[i % palette.length], borderWidth: n === sel ? 3.5 : 1.2, borderDash: n === sel ? [] : [], order: n === sel ? 0 : 1,
      data: m.models[n].roc_curve.fpr.map((x, j) => ({ x, y: m.models[n].roc_curve.tpr[j] })) }));
    sets.push({ label: 'Random guess', showLine: true, pointRadius: 0, borderColor: '#9aa8b1', borderDash: [5, 5], borderWidth: 1, data: [{ x: 0, y: 0 }, { x: 1, y: 1 }] });
    drawChart('c-roc', { type: 'scatter', data: { datasets: sets },
      options: { plugins: { title: titleOpts('ROC curves (test set)'), legend: { position: 'bottom' } },
        scales: { x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False positive rate' }, grid: { color: COLORS.grid } },
                  y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'True positive rate (recall)' }, grid: { color: COLORS.grid } } } } });
  }
  const imp = m.feature_importance.slice(0, 12), nice = (s) => s.replace(/([a-z])([A-Z])/g, '$1 $2');
  drawChart('c-imp', { type: 'bar', data: { labels: imp.map((i) => nice(i.feature)), datasets: [{ label: 'Drop in ROC-AUC when shuffled', data: imp.map((i) => +i.importance.toFixed(4)), backgroundColor: COLORS.primary, borderRadius: 3 }] },
    options: { indexAxis: 'y', plugins: { title: titleOpts(`Feature importance: ${m.selected_model} (permutation, test set)`), legend: { display: false } },
      scales: { x: { title: { display: true, text: 'Drop in ROC-AUC when the feature is shuffled' }, grid: { color: COLORS.grid } }, y: { grid: { display: false }, title: { display: true, text: 'Feature' } } } } });
  $('#cal').innerHTML = '<tr><th>Predicted probability</th><th class="num">Customers</th><th class="num">Average predicted</th><th class="num">Actual churn rate</th></tr>' +
    m.calibration.map((c) => `<tr><td>${c.range}</td><td class="num">${int(c.n)}</td><td class="num">${pct(c.predicted)}</td><td class="num">${pct(c.actual)}</td></tr>`).join('');
  $('#which').addEventListener('change', details);
  details();
})();
