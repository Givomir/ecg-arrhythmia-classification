"""
Shared fixtures for the unit and integration tests.

The real inputs (MIT-BIH records, trained models, guideline PDFs, the RAG
index, the ONNX model) are not in git, so the tests build their own:
  * a synthetic ECG record in the real MIT-BIH format (written with wfdb)
    with normal (N), premature supraventricular (S) and wide ventricular (V)
    beats
  * small models trained on that record (same code paths as training)
  * a deterministic fake embedding model (hashed bag of words) instead of
    sentence-transformers / ONNX
"""
import os
import sys
import hashlib
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# desktop_app writes its settings/log/index here - never into the real profile
os.environ.setdefault('ECG_ANALYZER_HOME', tempfile.mkdtemp(prefix='ecg-test-home-'))

FS = 360
MODEL_NAME = 'fake-bow-64'


# ---------------------------------------------------------------------------
# Fake embedding model
# ---------------------------------------------------------------------------
class FakeEncoder:
    """Deterministic bag-of-words embeddings: texts sharing words are similar."""
    dim = 64

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip('.,;:()')
                if word:
                    h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                    out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)


@pytest.fixture
def fake_encoder_factory():
    return lambda model_name: FakeEncoder(model_name)


# ---------------------------------------------------------------------------
# Synthetic ECG
# ---------------------------------------------------------------------------
def synth_beats(n_beats=300, rr=0.8, seed=0):
    """Beat plan: list of (r_peak_sample, symbol). Every 6th beat is a premature
    S beat (narrow), every 15th a premature V beat (wide); each ectopic beat
    is followed by a compensatory pause."""
    rng = np.random.default_rng(seed)
    t, beats = 1.0, []
    for k in range(n_beats):
        sym = 'V' if k % 15 == 7 else ('A' if k % 6 == 3 else 'N')
        if sym == 'N':
            t += rr * rng.uniform(0.97, 1.03)
        else:
            t += rr * 0.62
        beats.append((int(t * FS), sym))
        if sym != 'N':
            t += rr * 0.3          # pause after the ectopic beat
    return beats


def synth_signal(beats, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n_samples)
    sig = 0.03 * rng.standard_normal(n_samples)
    for r, sym in beats:
        width = 0.012 * FS if sym != 'V' else 0.045 * FS        # wide QRS for V
        amp = 1.0 if sym != 'V' else 1.6
        sig += amp * np.exp(-0.5 * ((x - r) / width) ** 2)
        if sym == 'N':                                           # P wave before N
            sig += 0.15 * np.exp(-0.5 * ((x - (r - 0.16 * FS)) / (0.025 * FS)) ** 2)
        sig += 0.25 * np.exp(-0.5 * ((x - (r + 0.25 * FS)) / (0.05 * FS)) ** 2)   # T wave
    return sig


@pytest.fixture(scope='session')
def synthetic_record(tmp_path_factory):
    """A synthetic record '900' in MIT-BIH format: (data_dir, record, beats)."""
    import wfdb
    d = tmp_path_factory.mktemp('mitbih')
    beats = synth_beats()
    n = beats[-1][0] + 2 * FS
    sig = synth_signal(beats, n)
    wfdb.wrsamp('900', fs=FS, units=['mV'], sig_name=['MLII'],
                p_signal=sig.reshape(-1, 1), fmt=['212'], write_dir=str(d))
    wfdb.wrann('900', 'atr', np.array([b[0] for b in beats]),
               symbol=[b[1] for b in beats], write_dir=str(d))
    return str(d), '900', beats


# ---------------------------------------------------------------------------
# Tiny trained models (the same code paths as the training scripts)
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def trained_artifacts(synthetic_record, tmp_path_factory):
    """Trains a cascade (rhythm features) and a plain RF on the synthetic record
    and saves them like train_cascade.py / train_arrhythmia.py do.
    Returns the folder that holds artifacts_cascade/ and artifacts/."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from cascade import CascadeClassifier
    from train_arrhythmia import load_beats
    from utility import record_features

    data_dir, rec, _ = synthetic_record
    signals, fs, beats = load_beats(os.path.join(data_dir, rec))
    root = tmp_path_factory.mktemp('artifacts')
    for name, rhythm in [('artifacts_cascade', True), ('artifacts', False)]:
        X, kept = record_features(signals[:, 0], [b[0] for b in beats], fs, rhythm=rhythm)
        X = np.nan_to_num(X)
        y_raw = np.array([beats[i][1] for i in kept])
        le = LabelEncoder().fit(np.array(['F', 'N', 'Q', 'S', 'V']))
        y = le.transform(y_raw)
        scaler = StandardScaler().fit(X)
        if rhythm:
            model = CascadeClassifier(s_class=3, n_class=1, n_classes=5, n_estimators=20,
                                      n_estimators_stage2=20, n_pca=5, random_state=0)
        else:
            model = RandomForestClassifier(n_estimators=20, random_state=0)
        model.fit(scaler.transform(X), y)
        out = root / name
        out.mkdir()
        joblib.dump(model, out / 'model.pkl')
        joblib.dump(scaler, out / 'scaler.pkl')
        joblib.dump(le, out / 'label_encoder.pkl')
        if rhythm:
            joblib.dump({'rhythm_features': True, 'threshold': 0.5,
                         'n_features': X.shape[1]}, out / 'meta.pkl')
    return str(root)


# ---------------------------------------------------------------------------
# Guideline text and a small index
# ---------------------------------------------------------------------------
AF_TEXT = ("Atrial fibrillation guideline. Anticoagulation for stroke prevention is "
           "recommended based on the CHA2DS2-VASc score. Rate control and rhythm control "
           "strategies for atrial fibrillation are discussed for supraventricular "
           "arrhythmia patients with palpitations. ") * 6
VA_TEXT = ("Ventricular arrhythmia guideline. Premature ventricular contractions and "
           "sudden cardiac death risk are evaluated. An implantable cardioverter "
           "defibrillator ICD and catheter ablation are options for ventricular "
           "tachycardia in structural heart disease. ") * 6


@pytest.fixture
def guideline_files(tmp_path):
    af, va = tmp_path / 'af-guideline.txt', tmp_path / 'va-guideline.txt'
    af.write_text(AF_TEXT, encoding='utf-8')
    va.write_text(VA_TEXT, encoding='utf-8')
    return str(af), str(va)


@pytest.fixture
def small_index(tmp_path, guideline_files):
    """A two-document index built with the fake encoder (index_dir)."""
    from build_rag_index import chunk_text, clean_text, save_index, source_name
    chunks = []
    for path in guideline_files:
        with open(path, encoding='utf-8') as f:
            chunks += chunk_text(clean_text(f.read()), source_name(path), 60, 10)
    emb = FakeEncoder().encode([c['text'] for c in chunks])
    index_dir = tmp_path / 'rag_index'
    save_index(str(index_dir), chunks, emb, MODEL_NAME)
    return str(index_dir)
