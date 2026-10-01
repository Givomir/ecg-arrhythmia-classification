// ECG Analyzer - desktop UI. Talks to the Python side through window.pywebview.api.

const CLASS_COLORS = { N: '#2ca02c', S: '#ff7f0e', V: '#d62728', F: '#9467bd', Q: '#7f7f7f' };
const CLASS_NAMES = { N: 'Normal', S: 'Supraventricular', V: 'Ventricular', F: 'Fusion', Q: 'Unknown' };

const $ = (id) => document.getElementById(id);
let api = null;

function setStatus(el, text, kind = '') {
  el.textContent = text;
  el.className = 'status' + (kind ? ' ' + kind : '');
}

function decodeF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// --- settings panel --------------------------------------------------------
function applyState(state) {
  const dir = state.data_dir || '';
  // show the end of a long path - the record folder name is the useful part
  $('data-dir').textContent = !dir ? 'not selected' : (dir.length > 26 ? '…' + dir.slice(-25) : dir);
  $('data-dir').title = state.data_dir || '';

  const rec = $('record');
  const previous = rec.value || state.last_record;
  rec.innerHTML = '';
  for (const r of state.records) rec.add(new Option(r, r));
  if (state.records.includes(previous)) rec.value = previous;
  else if (state.records.includes('200')) rec.value = '200';

  const model = $('model');
  if (!model.options.length) {
    for (const m of state.models) model.add(new Option(m.label, m.id));
  }

  $('btn-analyze').disabled = !state.records.length;
  if (!state.data_dir) setStatus($('status'), 'Choose the MIT-BIH data folder.', 'warn');
  else if (!state.records.length) setStatus($('status'), 'No MIT-BIH records (.dat/.hea/.atr) in this folder.', 'warn');
  else setStatus($('status'), `${state.records.length} records found.`);

  if (!state.rag_available) {
    $('btn-recommend').disabled = true;
    setStatus($('rag-status'), 'Guideline index not bundled - recommendations unavailable.', 'warn');
  }
}

async function refreshOllama() {
  const models = await api.ollama_models();
  const sel = $('llama');
  sel.innerHTML = '';
  if (models === null) {
    sel.add(new Option('llama3.2', 'llama3.2'));
    return false;
  }
  const list = models.length ? models : ['llama3.2'];
  for (const m of list) sel.add(new Option(m, m));
  const preferred = list.find((m) => m.startsWith('llama3.2'));
  if (preferred) sel.value = preferred;
  return true;
}

// --- plot ------------------------------------------------------------------
function plotEcg(r, windowSec) {
  const signal = decodeF32(r.signal);
  const t = new Float32Array(signal.length);
  for (let i = 0; i < signal.length; i++) t[i] = i / r.fs;
  const duration = (signal.length - 1) / r.fs;

  // Same layout as app_web.plot_ecg: the full-resolution ECG and markers are
  // WebGL traces on y2 (not drawn in the range slider); the light overview
  // envelope is on the hidden primary y axis, so it shows only in the slider.
  const traces = [
    { x: decodeF32(r.env_t), y: decodeF32(r.env_y), type: 'scatter', mode: 'lines', yaxis: 'y',
      line: { color: '#555', width: 0.6 }, hoverinfo: 'skip', showlegend: false },
    { x: t, y: signal, type: 'scattergl', mode: 'lines', yaxis: 'y2',
      line: { color: '#333', width: 1 }, hoverinfo: 'skip', showlegend: false },
  ];

  for (const [cls, color] of Object.entries(CLASS_COLORS)) {
    const idx = [];
    r.pred.forEach((p, k) => { if (p === cls) idx.push(k); });
    if (!idx.length) continue;
    const label = cls + (cls !== 'N' ? '*' : '');
    traces.push({
      x: idx.map((k) => r.peaks[k] / r.fs), y: idx.map((k) => signal[r.peaks[k]]),
      type: 'scattergl', mode: 'markers+text', yaxis: 'y2',
      text: idx.map(() => label), textposition: 'top center',
      textfont: { color, size: 11 },
      marker: { color, size: 9, line: { color: 'white', width: 1 } },
      name: `${cls} (${CLASS_NAMES[cls]})`, customdata: idx.map((k) => r.true[k]),
      hovertemplate: `%{x:.2f} s<br>predicted: ${cls}<br>annotated: %{customdata}<extra></extra>`,
    });
  }

  const wrong = [];
  r.pred.forEach((p, k) => { if (p !== r.true[k]) wrong.push(k); });
  if (wrong.length) {
    traces.push({
      x: wrong.map((k) => r.peaks[k] / r.fs), y: wrong.map((k) => signal[r.peaks[k]]),
      type: 'scattergl', mode: 'markers', yaxis: 'y2',
      marker: { symbol: 'square-open', size: 22, color: 'red', line: { width: 1.5 } },
      name: 'Differs from annotation', hoverinfo: 'skip',
    });
  }

  const sorted = Float32Array.from(signal).sort();
  const lo = sorted[Math.floor(0.0005 * (sorted.length - 1))];
  const hi = sorted[Math.floor(0.9995 * (sorted.length - 1))];
  const pad = 0.1 * (hi - lo);

  const layout = {
    height: 480, margin: { l: 60, r: 20, t: 40, b: 20 },
    dragmode: 'pan', hovermode: 'closest',
    legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, x: 0 },
    xaxis: {
      title: { text: 'Time (seconds)' }, range: [0, Math.min(windowSec, duration)],
      rangeslider: { visible: true, thickness: 0.14, range: [0, duration], autorange: false,
                     yaxis: { rangemode: 'auto' } },
    },
    yaxis: { range: [1e6, 1e6 + 1], visible: false, fixedrange: true },
    yaxis2: { title: { text: 'Amplitude (mV)' }, overlaying: 'y', side: 'left',
              range: [lo - pad, hi + 2 * pad], fixedrange: true, zeroline: false,
              tickmode: 'auto', nticks: 8 },
  };
  Plotly.newPlot('plot', traces, layout, { responsive: true, displaylogo: false, scrollZoom: false });
}

function showCounts(r) {
  const box = $('counts');
  box.innerHTML = '';
  for (const [cls, cnt] of Object.entries(r.counts)) {
    const div = document.createElement('div');
    div.className = 'count';
    div.innerHTML = `<div class="label">${cls} — ${CLASS_NAMES[cls] || cls}</div>
      <div class="value" style="color:${CLASS_COLORS[cls] || '#000'}">${cnt}</div>
      <div class="pct">${(100 * cnt / r.total).toFixed(1)}%</div>`;
    box.appendChild(div);
  }
}

// --- actions ---------------------------------------------------------------
async function analyze() {
  const record = $('record').value;
  const model = $('model').value;
  $('btn-analyze').disabled = true;
  setStatus($('status'), `Classifying record ${record}…`);
  try {
    const r = await api.analyze(record, model);
    if (!r.ok) { setStatus($('status'), r.error, 'error'); return; }
    $('empty').hidden = true;
    $('results').hidden = false;
    const minutes = (r.n_samples / r.fs / 60).toFixed(1);
    $('plot-title').textContent = `ECG record ${r.record} (${r.model}, ${minutes} min)`;
    plotEcg(r, Number($('window').value));
    showCounts(r);
    document.querySelector('main').scrollTo(0, 0);
    $('passages').hidden = true;
    $('prompt-box').hidden = true;
    $('rag-source').textContent = '';
    $('answer').hidden = true;
    const dom = r.dominant ? `Dominant abnormality: ${r.dominant_name}.` : 'No dominant abnormality.';
    setStatus($('status'), `Done: ${r.total} beats. ${dom}`);
    setStatus($('rag-status'), '');
  } catch (e) {
    setStatus($('status'), String(e), 'error');
  } finally {
    $('btn-analyze').disabled = false;
  }
}

async function recommend() {
  $('btn-recommend').disabled = true;
  try {
    setStatus($('rag-status'), 'Searching the guidelines…');
    const p = await api.get_passages();
    if (!p.ok) { setStatus($('rag-status'), p.error, 'error'); return; }
    $('passages-list').innerHTML = p.passages.map((x, i) => `
      <div class="passage"><div class="src">[Source ${i + 1}] (${x.score.toFixed(3)}) ${escapeHtml(x.source)}</div>
      <p>${escapeHtml(x.text)}…</p></div>`).join('');
    $('passages').hidden = false;
    $('rag-source').textContent = `Passages from: ${p.source}. Query: "${p.query}"`;
    $('prompt').textContent = p.prompt;
    $('prompt-box').hidden = false;

    const running = await refreshOllama();
    if (!running) {
      setStatus($('rag-status'), 'Ollama is not running - showing the retrieved guideline passages only. '
        + 'Start Ollama (ollama serve) to generate the recommendation text.', 'warn');
      return;
    }
    const model = $('llama').value;
    setStatus($('rag-status'), `Generating with ${model} (may take a minute)…`);
    const g = await api.generate(model);
    if (!g.ok) { setStatus($('rag-status'), g.error, 'error'); return; }
    const html = escapeHtml(g.answer).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    $('answer').innerHTML = html + '\n\n<em>⚠ Educational prototype — not a diagnosis.</em>';
    $('answer').hidden = false;
    setStatus($('rag-status'), '');
  } catch (e) {
    setStatus($('rag-status'), String(e), 'error');
  } finally {
    $('btn-recommend').disabled = false;
  }
}

// --- guideline library (RAG microservice) ----------------------------------
let libraryBusy = false;

async function refreshLibrary() {
  if (libraryBusy) return;
  const st = await api.rag_status();
  $('rag-mode').value = st.mode;
  $('service-row').hidden = st.mode !== 'service';
  $('rag-dot').className = 'dot ' + (st.online ? 'online' : 'offline');
  $('btn-add-doc').disabled = !st.online;
  const body = $('docs').querySelector('tbody');
  if (!st.online) {
    $('rag-service-status').textContent = st.mode === 'service'
      ? `RAG service offline (${st.url}). Start it with "uvicorn rag_service.app:app --port 8001", or switch `
        + 'back to the built-in engine. Recommendations use the built-in engine meanwhile.'
      : `Built-in RAG engine failed: ${st.error}`;
    $('docs').hidden = true;
    return;
  }
  $('rag-service-status').textContent = st.mode === 'service'
    ? `RAG service online (${st.url}) — ${st.n_documents} documents, ${st.n_chunks} chunks, model ${st.model}.`
    : `Built-in engine — ${st.n_documents} documents, ${st.n_chunks} chunks, model ${st.model} (ONNX).`;
  const res = await api.list_documents();
  if (!res.ok) { setStatus($('docs-status'), res.error, 'error'); return; }
  body.innerHTML = '';
  for (const d of res.documents) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${escapeHtml(d.source)}</td><td class="num">${d.n_chunks}</td>
      <td><button class="secondary">Remove</button></td>`;
    tr.querySelector('button').addEventListener('click', () => removeDocument(d.source));
    body.appendChild(tr);
  }
  $('docs').hidden = !res.documents.length;
}

async function addDocument() {
  libraryBusy = true;
  $('btn-add-doc').disabled = true;
  setStatus($('docs-status'), 'Indexing… (a large guideline PDF takes about a minute)');
  try {
    const r = await api.add_document();
    for (const c of r.conflicts || []) {   // never replace a document silently
      if (!confirm(`"${c.source}" is already in the guideline index.\nReplace it with the new file?`)) continue;
      setStatus($('docs-status'), `Replacing ${c.source}…`);
      const rr = await api.upload_document_path(c.path, true);
      if (rr.ok) r.added.push(rr);
      else r.error = (r.error ? r.error + '; ' : '') + rr.error;
    }
    const names = r.added.map((a) => `${a.source} (${a.n_chunks} chunks`
      + (a.replaced_chunks ? `, replaced ${a.replaced_chunks}` : '') + ')').join(', ');
    if (r.error) setStatus($('docs-status'), (names ? `Added: ${names}. ` : '') + r.error, 'error');
    else setStatus($('docs-status'), names ? `Added: ${names}.` : '');
  } finally {
    libraryBusy = false;
    refreshLibrary();
  }
}

async function removeDocument(source) {
  if (!confirm(`Remove "${source}" from the guideline index?`)) return;
  libraryBusy = true;
  try {
    const r = await api.remove_document(source);
    setStatus($('docs-status'), r.ok ? `Removed ${source} (${r.removed_chunks} chunks).` : r.error,
      r.ok ? '' : 'error');
  } finally {
    libraryBusy = false;
    refreshLibrary();
  }
}

async function init() {
  api = window.pywebview.api;
  applyState(await api.get_state());
  refreshOllama();

  $('btn-folder').addEventListener('click', async () => applyState(await api.choose_folder()));
  $('btn-analyze').addEventListener('click', analyze);
  $('btn-recommend').addEventListener('click', recommend);
  $('rag-url').value = (await api.get_state()).rag_url;
  $('rag-mode').addEventListener('change', async () => {
    setStatus($('docs-status'), '');
    await api.set_rag_mode($('rag-mode').value);
    refreshLibrary();
  });
  $('btn-rag-url').addEventListener('click', async () => {
    await api.set_rag_mode('service', $('rag-url').value);
    refreshLibrary();
  });
  $('btn-add-doc').addEventListener('click', addDocument);
  refreshLibrary();
  setInterval(refreshLibrary, 15000);   // notice when the service starts/stops
  $('window').addEventListener('input', (e) => {
    $('window-val').textContent = e.target.value;
    const plot = $('plot');
    if (plot.layout) {   // move the right edge of the current view
      const start = plot.layout.xaxis.range[0];
      Plotly.relayout(plot, { 'xaxis.range': [start, start + Number(e.target.value)] });
    }
  });
}

if (window.pywebview && window.pywebview.api) init();
else window.addEventListener('pywebviewready', init);
