"""
ECG Analyzer - desktop application (single .exe, no installation)
==================================================================
A native window (pywebview, using the WebView2 engine built into Windows)
with an HTML/JS interface. The Python side reuses the project code as is:
feature extraction (utility.py), the trained models (artifacts_*), the
dominant-class logic and the guideline retrieval (rag_recommend.py).

RAG: the app has the full RAG engine built in. The embedding model runs
with ONNX (rag_embedder.py, no PyTorch), so the app embeds new guideline
documents and any query by itself. Its index lives in
%APPDATA%/ECGAnalyzer/rag_index (seeded with the bundled guidelines on the
first run); documents can be added and removed from the app. Optionally the
app can use the shared RAG microservice instead (rag_service/).
The LLM always receives the beat classification summary AND the retrieved
guideline passages (the app shows the exact prompt).

Not bundled in the .exe:
  * the MIT-BIH records - chosen with "Change..." (remembered between runs)
  * Ollama + Llama - optional; without them everything works except the
    generated recommendation text

EDUCATIONAL PROTOTYPE - not a medical device.

Run from source:
    python desktop_app.py
Build the .exe:
    python desktop/build_exe.py
Check a built .exe without the window (writes a JSON report):
    ECGAnalyzer.exe --selftest <data_dir> <record> <report.json>
"""
import os
import sys
import json
import base64
import shutil
import logging
import threading
import traceback
from collections import Counter

import numpy as np
import requests

# The models are pickled objects - import their modules explicitly so that
# PyInstaller bundles them (cascade.CascadeClassifier, sklearn estimators)
import cascade  # noqa: F401
import sklearn.ensemble  # noqa: F401
import sklearn.neural_network  # noqa: F401
import sklearn.preprocessing  # noqa: F401
import sklearn.decomposition  # noqa: F401
import sklearn.pipeline  # noqa: F401
import joblib

from train_arrhythmia import load_beats
from utility import record_features, uses_rhythm_features
from rag_recommend import (CLASS_NAMES_EN, OLLAMA_URL, find_dominant, summarize_counts,
                           build_query_for_class, retrieve, build_prompt, ask_ollama)
from rag_store import IndexStore, DocumentExists
import rag_embedder

APP_NAME = 'ECG Analyzer'
# Bundled files (models, index, UI) - inside the .exe they are unpacked to _MEIPASS
RES_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
EXE_DIR = (os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
           else os.path.dirname(os.path.abspath(__file__)))
# User data: settings, log, the built-in RAG index (ECG_ANALYZER_HOME overrides it)
SETTINGS_DIR = os.environ.get('ECG_ANALYZER_HOME') or os.path.join(
    os.environ.get('APPDATA', os.path.expanduser('~')), 'ECGAnalyzer')
SETTINGS_FILE = os.path.join(SETTINGS_DIR, 'settings.json')
INDEX_DIR = os.path.join(RES_DIR, 'rag_index')   # bundled guideline index (read-only)
# The built-in RAG engine: its own index (writable) and the ONNX embedding model
LOCAL_INDEX_DIR = os.path.join(SETTINGS_DIR, 'rag_index')
LOCAL_DOCS_DIR = os.path.join(SETTINGS_DIR, 'guidelines')
EMBEDDER_DIR = next((d for d in (os.path.join(RES_DIR, 'embedder'),
                                 os.path.join(RES_DIR, 'desktop', 'onnx_model'))
                     if os.path.exists(os.path.join(d, 'model.onnx'))), None)
DEFAULT_RAG_URL = 'http://localhost:8001'        # optional RAG microservice
# The UI is bundled as ui/ in the .exe and lives in desktop/ui/ in the source tree
UI_DIR = next(d for d in (os.path.join(RES_DIR, 'ui'), os.path.join(RES_DIR, 'desktop', 'ui'))
              if os.path.exists(os.path.join(d, 'index.html')))

# Models offered in the app (the CNN needs TensorFlow and is not bundled)
MODELS = [
    ('artifacts_cascade', 'Cascade RF (recommended)'),
    ('artifacts', 'Random Forest'),
    ('artifacts_mlp', 'MLP'),
]

os.makedirs(SETTINGS_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(SETTINGS_DIR, 'app.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(APP_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_records(data_dir):
    """Record names in a folder that have the .dat, .hea and .atr files."""
    if not data_dir or not os.path.isdir(data_dir):
        return []
    names = {os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.dat')}
    ok = [n for n in names
          if all(os.path.exists(os.path.join(data_dir, n + ext)) for ext in ('.hea', '.atr'))]
    return sorted(ok, key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else 0, n))


def find_default_data_dir():
    """A folder with MIT-BIH records next to the .exe (searched 2 levels deep)."""
    for root, dirs, _ in os.walk(EXE_DIR):
        if root[len(EXE_DIR):].count(os.sep) > 2:
            dirs[:] = []
            continue
        if list_records(root):
            return root
    return ''


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)


def overview_envelope(signal, fs, bucket_sec=0.25):
    """Min/max envelope of the whole signal - for the scroll bar under the plot."""
    b = max(1, int(bucket_sec * fs))
    n = len(signal) // b * b
    blocks = signal[:n].reshape(-1, b)
    t = (np.arange(len(blocks)) * b + b / 2) / fs
    return np.repeat(t, 2), np.column_stack([blocks.min(1), blocks.max(1)]).ravel()


def _b64(arr):
    """float32 array -> base64 (compact transfer of ~650k samples to the UI)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype='<f4').tobytes()).decode('ascii')


# ---------------------------------------------------------------------------
# API exposed to the JavaScript UI (window.pywebview.api.*)
# ---------------------------------------------------------------------------
class Api:
    # Only public methods are exposed to JavaScript. State lives in "_" attributes:
    # pywebview walks public attributes, and the window object would recurse.
    def __init__(self):
        self._window = None
        self._settings = load_settings()
        if not list_records(self._settings.get('data_dir')):
            self._settings['data_dir'] = find_default_data_dir()
        self._models = {}
        self._last = None   # the last analysis (for the recommendation)
        self._store = None  # the built-in RAG index (created on first use)
        self._store_lock = threading.Lock()

    # -- state --------------------------------------------------------------
    def get_state(self):
        data_dir = self._settings.get('data_dir', '')
        models = [{'id': m, 'label': label} for m, label in MODELS
                  if os.path.exists(os.path.join(RES_DIR, m, 'model.pkl'))]
        return {
            'data_dir': data_dir,
            'records': list_records(data_dir),
            'last_record': self._settings.get('last_record', ''),
            'models': models,
            'rag_available': os.path.exists(os.path.join(INDEX_DIR, 'chunks.json')),
            'rag_url': self._rag_url(),
        }

    def choose_folder(self):
        import webview
        start = self._settings.get('data_dir') or EXE_DIR
        result = self._window.create_file_dialog(webview.FileDialog.FOLDER, directory=start)
        if result:
            path = result[0] if isinstance(result, (list, tuple)) else result
            self._settings['data_dir'] = path
            save_settings(self._settings)
        return self.get_state()

    # -- analysis -----------------------------------------------------------
    def _load_model(self, model_id):
        if model_id not in self._models:
            d = os.path.join(RES_DIR, model_id)
            self._models[model_id] = (joblib.load(os.path.join(d, 'model.pkl')),
                                      joblib.load(os.path.join(d, 'scaler.pkl')),
                                      joblib.load(os.path.join(d, 'label_encoder.pkl')),
                                      uses_rhythm_features(d))
        return self._models[model_id]

    def analyze(self, record, model_id):
        try:
            data_dir = self._settings.get('data_dir', '')
            model, scaler, le, rhythm = self._load_model(model_id)
            signals, fs, beats = load_beats(os.path.join(data_dir, record))
            # features from the original float64 signal (exactly as in training);
            # float32 only for the transfer to the UI
            signal = signals[:, 0]
            X, kept = record_features(signal, [b[0] for b in beats], fs, rhythm=rhythm)
            pred = le.inverse_transform(model.predict(scaler.transform(np.nan_to_num(X))))
            true = [beats[i][1] for i in kept]
            peaks = [int(beats[i][0]) for i in kept]

            counts = Counter(pred.tolist())
            dominant = find_dominant(counts)
            self._last = {'record': record, 'fs': fs, 'counts': counts, 'dominant': dominant}
            self._settings['last_record'] = record
            save_settings(self._settings)

            env_t, env_y = overview_envelope(signal, fs)
            label = dict(MODELS).get(model_id, model_id)
            log.info("analyzed %s with %s: %s", record, model_id, dict(counts))
            return {
                'ok': True, 'record': record, 'model': label, 'fs': float(fs),
                'n_samples': int(len(signal)), 'signal': _b64(signal),
                'env_t': _b64(env_t), 'env_y': _b64(env_y),
                'peaks': peaks, 'pred': pred.tolist(), 'true': true,
                'counts': dict(counts.most_common()), 'total': len(pred),
                'dominant': dominant,
                'dominant_name': CLASS_NAMES_EN.get(dominant) if dominant else None,
            }
        except Exception as e:
            log.error("analyze failed: %s", traceback.format_exc())
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    # -- RAG: built-in engine (default) or the optional microservice -----------
    def _rag_url(self):
        return (self._settings.get('rag_url') or DEFAULT_RAG_URL).rstrip('/')

    def _rag_mode(self):
        return self._settings.get('rag_mode', 'local')

    def _local_store(self):
        """The built-in index (created on first use from the bundled guidelines)."""
        with self._store_lock:
            if self._store is None:
                if not os.path.exists(os.path.join(LOCAL_INDEX_DIR, 'config.json')):
                    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
                    for name in ('embeddings.npy', 'chunks.json', 'config.json'):
                        shutil.copyfile(os.path.join(INDEX_DIR, name),
                                        os.path.join(LOCAL_INDEX_DIR, name))
                    log.info("built-in index created in %s", LOCAL_INDEX_DIR)

                def onnx_encoder(model_name):
                    embedder = rag_embedder.OnnxEmbedder(EMBEDDER_DIR)
                    if embedder.model_name != model_name:
                        raise ValueError(f"the index uses {model_name}, the built-in "
                                         f"embedder is {embedder.model_name}")
                    return embedder
                self._store = IndexStore(LOCAL_INDEX_DIR, LOCAL_DOCS_DIR, onnx_encoder,
                                         logger=log)
            return self._store

    def set_rag_mode(self, mode, url=None):
        self._settings['rag_mode'] = 'service' if mode == 'service' else 'local'
        if url is not None:
            url = (url or '').strip() or DEFAULT_RAG_URL
            if '://' not in url:
                url = 'http://' + url
            self._settings['rag_url'] = url
        save_settings(self._settings)
        return self.rag_status()

    def _service_status(self):
        url = self._rag_url()
        try:
            r = requests.get(url + '/health', timeout=1.5)
            r.raise_for_status()
            return {'online': True, 'url': url, **r.json()}
        except Exception:
            return {'online': False, 'url': url}

    def rag_status(self):
        """Which RAG engine is used, and its index stats."""
        if self._rag_mode() == 'service':
            return {'mode': 'service', **self._service_status()}
        try:
            return {'mode': 'local', 'online': True, 'location': LOCAL_INDEX_DIR,
                    'url': self._rag_url(), **self._local_store().stats()}
        except Exception as e:
            log.error("built-in RAG failed: %s", traceback.format_exc())
            return {'mode': 'local', 'online': False, 'url': self._rag_url(),
                    'error': f"{type(e).__name__}: {e}"}

    def list_documents(self):
        try:
            if self._rag_mode() == 'service':
                r = requests.get(self._rag_url() + '/documents', timeout=5)
                r.raise_for_status()
                return {'ok': True, 'documents': r.json()}
            docs = sorted(self._local_store().documents().items())
            return {'ok': True, 'documents': [{'source': s, 'n_chunks': n} for s, n in docs]}
        except Exception as e:
            return {'ok': False, 'error': f"Guideline library unavailable: {e}"}

    def add_document(self):
        """Opens a file dialog and indexes the chosen guideline(s)."""
        import webview
        paths = self._window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True,
            file_types=('Guideline documents (*.pdf;*.txt)', 'All files (*.*)'))
        if not paths:
            return {'ok': True, 'added': [], 'conflicts': []}
        added, conflicts, errors = [], [], []
        for path in paths:
            r = self.upload_document_path(path)
            if r['ok']:
                added.append(r)
            elif r.get('exists'):
                conflicts.append({'path': path, 'source': r['source']})   # UI asks
            else:
                errors.append(r)
        return {'ok': not errors, 'added': added, 'conflicts': conflicts,
                'error': '; '.join(e['error'] for e in errors) if errors else None}

    def upload_document_path(self, path, replace=False):
        """Indexes one file (extract, chunk, embed). If a document with the same
        name is indexed, nothing changes unless replace is True (the UI asks
        the user first)."""
        name = os.path.basename(path)
        exists = {'ok': False, 'exists': True, 'source': name,
                  'error': f"{name} is already in the index"}
        try:
            if self._rag_mode() == 'service':
                with open(path, 'rb') as f:
                    r = requests.post(self._rag_url() + '/documents',
                                      params={'replace': 'true' if replace else 'false'},
                                      files={'file': (name, f)}, timeout=1800)
                if r.status_code == 409:
                    return exists
                if r.status_code != 200:
                    return {'ok': False, 'error': f"{name}: {r.json().get('detail', r.text)}"}
                result = r.json()
            else:
                result = self._local_store().add_file(path, replace=replace)
            log.info("indexed document %s", result)
            return {'ok': True, **result}
        except DocumentExists:
            return exists
        except Exception as e:
            log.error("indexing failed: %s", traceback.format_exc())
            return {'ok': False, 'error': f"{name}: {e}"}

    def remove_document(self, source):
        try:
            if self._rag_mode() == 'service':
                r = requests.delete(
                    f"{self._rag_url()}/documents/{requests.utils.quote(source, safe='')}",
                    timeout=60)
                r.raise_for_status()
                result = r.json()
            else:
                removed = self._local_store().remove(source)
                if not removed:
                    return {'ok': False, 'error': f"{source} is not in the index"}
                result = {'source': source, 'removed_chunks': removed}
            log.info("removed document %s", source)
            return {'ok': True, **result}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def _search(self, query, top_k=4):
        """(passages, description of where they came from)."""
        if self._rag_mode() == 'service':
            status = self._service_status()
            if status['online']:
                return (retrieve(query, INDEX_DIR, top_k=top_k, service_url=status['url']),
                        f"RAG service ({status['n_documents']} documents, live search)")
        try:
            store = self._local_store()
            hits = store.search(query, top_k)
            note = ('' if self._rag_mode() == 'local'
                    else ' - RAG service offline, using the built-in engine')
            return ([({'text': h['text'], 'source': h['source']}, h['score']) for h in hits],
                    f"built-in RAG ({store.stats()['n_documents']} documents, live search){note}")
        except Exception:
            log.error("built-in RAG failed: %s", traceback.format_exc())
            return (retrieve(query, INDEX_DIR, top_k=top_k, service_url=''),
                    "bundled guidelines (fixed per-class queries)")

    # -- recommendation: classification + RAG passages -> LLM ------------------
    def ollama_models(self):
        """Installed Ollama models, or None if Ollama is not running."""
        try:
            base = OLLAMA_URL.rsplit('/api/', 1)[0]
            r = requests.get(base + '/api/tags', timeout=2)
            r.raise_for_status()
            return [m['name'] for m in r.json().get('models', [])]
        except Exception:
            return None

    def get_passages(self):
        """
        Retrieves the guideline passages for the dominant class and builds the
        LLM prompt from the classification summary + those passages.
        Uses the RAG microservice when it runs, else the bundled index.
        """
        if not self._last:
            return {'ok': False, 'error': 'Analyze a record first.'}
        last = self._last
        try:
            query = build_query_for_class(last['dominant'])
            passages, source = self._search(query, top_k=4)
            summary = summarize_counts(last['record'], last['counts'], last['fs'])
            dominant_name = CLASS_NAMES_EN.get(last['dominant']) if last['dominant'] else None
            last['passages'] = passages
            last['prompt'] = build_prompt(summary, dominant_name, passages)
            return {'ok': True, 'query': query, 'source': source, 'prompt': last['prompt'],
                    'passages': [{'source': p['source'], 'score': s, 'text': p['text'][:600]}
                                 for p, s in passages]}
        except Exception as e:
            log.error("retrieval failed: %s", traceback.format_exc())
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}

    def generate(self, llama_model):
        """Sends the prompt (classification + passages) to the local LLM."""
        if not self._last or 'prompt' not in self._last:
            return {'ok': False, 'error': 'Analyze a record first.'}
        try:
            return {'ok': True, 'answer': ask_ollama(self._last['prompt'], model=llama_model)}
        except requests.exceptions.ConnectionError:
            return {'ok': False, 'error': 'Cannot connect to Ollama. Start it (ollama serve) '
                                          'and make sure the model is pulled.'}
        except Exception as e:
            log.error("generation failed: %s", traceback.format_exc())
            return {'ok': False, 'error': f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def selftest(data_dir, record, report_path):
    """Runs the backend without a window - for checking a built .exe."""
    report = {'res_dir': RES_DIR, 'frozen': bool(getattr(sys, 'frozen', False))}
    try:
        api = Api()
        api._settings['data_dir'] = data_dir
        report['state'] = api.get_state()
        for model_id, _ in MODELS:
            r = api.analyze(record, model_id)
            report[model_id] = {k: r[k] for k in ('ok', 'counts', 'dominant', 'error') if k in r}
        p = api.get_passages()
        report['passages'] = [(x['source'][:40], round(x['score'], 3)) for x in p.get('passages', [])]
        report['rag_source'] = p.get('source')
        # the built-in engine: index a new document, find it, remove it
        doc = os.path.join(SETTINGS_DIR, 'selftest_document.txt')
        with open(doc, 'w', encoding='utf-8') as f:
            f.write("Selftest guideline about Wolff-Parkinson-White syndrome. Catheter "
                    "ablation of the accessory pathway is recommended for symptomatic "
                    "patients with pre-excitation and recurrent tachycardia. " * 3)
        added = api.upload_document_path(doc)
        again = api.upload_document_path(doc)
        hits = api._local_store().search('Wolff-Parkinson-White accessory pathway ablation', 3)
        report['builtin_rag'] = {
            'added': added, 'second_upload_refused': bool(again.get('exists')),
            'found_new_document': any(h['source'] == 'selftest_document.txt' for h in hits),
            'removed': api.remove_document('selftest_document.txt').get('removed_chunks')}
        report['rag_status'] = api.rag_status()
        prompt = p.get('prompt', '')
        report['prompt_has_findings'] = 'ECG FINDINGS' in prompt and 'GUIDELINE EXCERPTS' in prompt
        report['ollama_models'] = api.ollama_models()
        report['torch_loaded'] = 'torch' in sys.modules
    except Exception:
        report['error'] = traceback.format_exc()
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def main():
    if len(sys.argv) >= 5 and sys.argv[1] == '--selftest':
        selftest(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    import webview
    api = Api()
    window = webview.create_window(
        APP_NAME, url=os.path.join(UI_DIR, 'index.html'), js_api=api,
        width=1400, height=900, min_size=(1000, 700))
    api._window = window
    log.info("starting (res_dir=%s)", RES_DIR)
    webview.start(http_server=True)


if __name__ == '__main__':
    main()
