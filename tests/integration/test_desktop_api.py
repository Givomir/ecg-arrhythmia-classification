"""The desktop app backend (desktop_app.Api) end to end, without a window:
analysis of a record, the built-in RAG engine and the LLM prompt."""
import base64
import os

import numpy as np
import pytest

import desktop_app as D
import rag_embedder
from tests.conftest import MODEL_NAME, FakeEncoder

pytestmark = pytest.mark.integration


@pytest.fixture
def api(tmp_path, monkeypatch, synthetic_record, trained_artifacts, small_index):
    """An Api wired to the synthetic record, the tiny models, a small bundled
    index and the fake embedder (instead of the ONNX model)."""
    data_dir, _, _ = synthetic_record
    monkeypatch.setattr(D, 'RES_DIR', trained_artifacts)
    monkeypatch.setattr(D, 'INDEX_DIR', small_index)
    monkeypatch.setattr(D, 'LOCAL_INDEX_DIR', str(tmp_path / 'local_index'))
    monkeypatch.setattr(D, 'LOCAL_DOCS_DIR', str(tmp_path / 'local_docs'))
    monkeypatch.setattr(D, 'SETTINGS_FILE', str(tmp_path / 'settings.json'))
    monkeypatch.setattr(rag_embedder, 'OnnxEmbedder', lambda _dir: FakeEncoder(MODEL_NAME))
    a = D.Api()
    a._settings = {'data_dir': data_dir, 'rag_url': 'http://127.0.0.1:9'}   # no service there
    return a


def test_state_lists_records_and_available_models(api):
    st = api.get_state()
    assert st['records'] == ['900']
    assert [m['id'] for m in st['models']] == ['artifacts_cascade', 'artifacts']   # no MLP here


def test_analyze(api):
    r = api.analyze('900', 'artifacts_cascade')
    assert r['ok'], r.get('error')
    signal = np.frombuffer(base64.b64decode(r['signal']), dtype='<f4')
    assert len(signal) == r['n_samples']
    assert len(r['peaks']) == len(r['pred']) == len(r['true']) == r['total']
    assert r['dominant'] == 'S' and r['dominant_name'] == 'supraventricular ectopic'
    assert sum(r['counts'].values()) == r['total']


def test_analyze_reports_errors_instead_of_raising(api):
    r = api.analyze('does-not-exist', 'artifacts_cascade')
    assert r['ok'] is False and r['error']


def test_passages_and_prompt_need_an_analysis_first(api):
    assert api.get_passages()['ok'] is False
    assert api.generate('llama3.2')['ok'] is False


def test_prompt_contains_classification_and_guideline_passages(api):
    api.analyze('900', 'artifacts_cascade')
    p = api.get_passages()
    assert p['ok'] and p['source'].startswith('built-in RAG')
    assert len(p['passages']) == 4
    assert p['passages'][0]['source'] == 'af-guideline.txt'          # S -> AF guideline
    prompt = p['prompt']
    assert '=== ECG FINDINGS ===' in prompt and 'ECG record 900' in prompt
    assert 'Predominant abnormality: supraventricular ectopic' in prompt
    assert '=== GUIDELINE EXCERPTS ===' in prompt and '[Source 1: af-guideline.txt]' in prompt


def test_generate_sends_the_prompt_to_the_llm(api, monkeypatch):
    sent = {}
    monkeypatch.setattr(D, 'ask_ollama', lambda prompt, model: sent.update(
        prompt=prompt, model=model) or 'educational note')
    api.analyze('900', 'artifacts_cascade')
    prompt = api.get_passages()['prompt']
    r = api.generate('llama3.2')
    assert r == {'ok': True, 'answer': 'educational note'}
    assert sent == {'prompt': prompt, 'model': 'llama3.2'}


def test_builtin_library_add_conflict_replace_remove(api, tmp_path):
    doc = tmp_path / 'wpw.txt'
    doc.write_text(("Wolff-Parkinson-White accessory pathway ablation guideline text. " * 30),
                   encoding='utf-8')
    status = api.rag_status()
    assert status['mode'] == 'local' and status['online'] and status['n_documents'] == 2
    added = api.upload_document_path(str(doc))
    assert added['ok'] and added['source'] == 'wpw.txt'
    again = api.upload_document_path(str(doc))
    assert again['ok'] is False and again['exists'] is True
    replaced = api.upload_document_path(str(doc), replace=True)
    assert replaced['ok'] and replaced['replaced_chunks'] == added['n_chunks']
    docs = {d['source'] for d in api.list_documents()['documents']}
    assert docs == {'af-guideline.txt', 'va-guideline.txt', 'wpw.txt'}
    assert api.remove_document('wpw.txt')['ok']
    assert api.remove_document('wpw.txt')['ok'] is False
    assert api.rag_status()['n_documents'] == 2


def test_builtin_index_is_created_from_the_bundled_one(api):
    assert not os.path.exists(D.LOCAL_INDEX_DIR)
    api.rag_status()
    assert sorted(os.listdir(D.LOCAL_INDEX_DIR)) == ['chunks.json', 'config.json', 'embeddings.npy']


def test_new_document_is_used_in_recommendations(api, tmp_path):
    api.analyze('900', 'artifacts_cascade')
    doc = tmp_path / 'svt-update.txt'
    # written in the words of the S query, so it must rank among the top passages
    doc.write_text(("Update on atrial fibrillation and supraventricular arrhythmia: "
                    "anticoagulation for stroke prevention, rate control and rhythm "
                    "control, CHA2DS2-VASc. ") * 10, encoding='utf-8')
    api.upload_document_path(str(doc))
    sources = {x['source'] for x in api.get_passages()['passages']}
    assert 'svt-update.txt' in sources


def test_service_mode_falls_back_to_the_builtin_engine(api):
    st = api.set_rag_mode('service', '127.0.0.1:9')
    assert st['mode'] == 'service' and st['online'] is False
    assert api._settings['rag_url'] == 'http://127.0.0.1:9'           # scheme added
    api.analyze('900', 'artifacts_cascade')
    p = api.get_passages()
    assert p['ok'] and 'RAG service offline, using the built-in engine' in p['source']
    assert api.set_rag_mode('local')['mode'] == 'local'
