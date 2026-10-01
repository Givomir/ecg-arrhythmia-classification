"""The RAG microservice REST API (rag_service/app.py) with a fake embedding model."""
import importlib
import os

import pytest

from tests.conftest import AF_TEXT, MODEL_NAME, VA_TEXT, FakeEncoder

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv('RAG_INDEX_DIR', str(tmp_path / 'index'))
    monkeypatch.setenv('RAG_DOCS_DIR', str(tmp_path / 'docs'))
    monkeypatch.setenv('RAG_MODEL', MODEL_NAME)
    monkeypatch.setenv('RAG_CHUNK_SIZE', '60')
    monkeypatch.setenv('RAG_OVERLAP', '10')
    import rag_service.app as svc
    svc = importlib.reload(svc)                          # read the env vars
    monkeypatch.setattr(svc, 'sentence_transformer', lambda name: FakeEncoder(name))
    with TestClient(svc.app) as c:                       # runs the startup event
        c.docs_dir = tmp_path / 'docs'
        yield c


def upload(client, name, text, replace=False):
    return client.post('/documents', params={'replace': str(replace).lower()},
                       files={'file': (name, text.encode('utf-8'))})


def test_health_of_an_empty_index(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'model': MODEL_NAME, 'n_chunks': 0, 'n_documents': 0}


def test_upload_search_and_list(client):
    assert upload(client, 'af.txt', AF_TEXT).status_code == 200
    assert upload(client, 'va.txt', VA_TEXT).status_code == 200
    docs = {d['source']: d['n_chunks'] for d in client.get('/documents').json()}
    assert set(docs) == {'af.txt', 'va.txt'} and all(n > 0 for n in docs.values())
    hits = client.post('/search', json={'query': 'implantable cardioverter defibrillator',
                                        'top_k': 2}).json()
    assert hits[0]['source'] == 'va.txt' and hits[0]['score'] >= hits[1]['score']
    assert (client.docs_dir / 'af.txt').exists()
    assert not [f for f in os.listdir(client.docs_dir) if f.startswith('.upload-')]


def test_same_name_conflict_and_replace(client):
    first = upload(client, 'doc.txt', AF_TEXT).json()
    r = upload(client, 'doc.txt', VA_TEXT)
    assert r.status_code == 409
    assert (client.docs_dir / 'doc.txt').read_text(encoding='utf-8') == AF_TEXT   # untouched
    r = upload(client, 'doc.txt', VA_TEXT, replace=True)
    assert r.status_code == 200 and r.json()['replaced_chunks'] == first['n_chunks']
    assert (client.docs_dir / 'doc.txt').read_text(encoding='utf-8') == VA_TEXT


def test_delete(client):
    upload(client, 'af.txt', AF_TEXT)
    r = client.delete('/documents/af.txt')
    assert r.status_code == 200 and r.json()['removed_chunks'] > 0
    assert client.delete('/documents/af.txt').status_code == 404


def test_bad_uploads(client):
    assert upload(client, 'notes.docx', AF_TEXT).status_code == 400
    assert upload(client, 'empty.txt', '   ').status_code == 422
    assert client.get('/health').json()['n_documents'] == 0


def test_top_k_is_limited(client):
    upload(client, 'af.txt', AF_TEXT)
    upload(client, 'va.txt', VA_TEXT)
    assert len(client.post('/search', json={'query': 'guideline', 'top_k': 0}).json()) == 1
    assert len(client.post('/search', json={'query': 'guideline', 'top_k': 999}).json()) <= 20


def test_embed_endpoint(client):
    r = client.post('/embed', json={'texts': ['atrial fibrillation', 'ICD']})
    emb = r.json()['embeddings']
    assert r.json()['model'] == MODEL_NAME and len(emb) == 2 and len(emb[0]) == FakeEncoder.dim
