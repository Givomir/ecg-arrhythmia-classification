"""End-to-end on a synthetic MIT-BIH record: features -> trained models -> RAG
analysis, and the classification REST API (app.py + predict.py)."""
import importlib
import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_record_analysis_with_the_cascade(synthetic_record, trained_artifacts):
    from rag_recommend import analyze_record
    data_dir, rec, beats = synthetic_record
    res = analyze_record(data_dir, rec, os.path.join(trained_artifacts, 'artifacts_cascade'))
    truth = {'N': 0, 'S': 0, 'V': 0}
    for _, sym in beats:
        truth[{'N': 'N', 'A': 'S', 'V': 'V'}[sym]] += 1
    assert res['total'] == sum(res['counts'].values())
    assert set(res['counts']) <= {'N', 'S', 'V', 'F', 'Q'}
    assert sum(res['true_counts'].values()) == res['total']
    assert res['true_counts']['S'] <= truth['S']                 # edge beats may be skipped
    assert res['dominant'] == 'S'                                 # ~16% S, ~7% V
    assert res['summary'].startswith(f"ECG record {rec}: {res['total']} beats analyzed")


def test_record_analysis_with_the_plain_rf(synthetic_record, trained_artifacts):
    from rag_recommend import analyze_record
    data_dir, rec, _ = synthetic_record
    res = analyze_record(data_dir, rec, os.path.join(trained_artifacts, 'artifacts'))
    assert res['total'] > 250 and 'N' in res['counts']


@pytest.fixture
def api_client(trained_artifacts, monkeypatch):
    """The FastAPI app with the cascade model (ARTIFACT_DIR is read at import)."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv('ARTIFACT_DIR', os.path.join(trained_artifacts, 'artifacts_cascade'))
    import predict
    import app
    importlib.reload(predict)
    importlib.reload(app)
    return TestClient(app.app)


def beat_vector(synthetic_record, k, rhythm=True):
    from train_arrhythmia import load_beats
    from utility import record_features
    data_dir, rec, _ = synthetic_record
    signals, fs, beats = load_beats(os.path.join(data_dir, rec))
    X, kept = record_features(signals[:, 0], [b[0] for b in beats], fs, rhythm=rhythm)
    return np.nan_to_num(X[k]).tolist(), beats[kept[k]][1]


def test_rest_api_health(api_client):
    r = api_client.get('/health')
    assert r.status_code == 200
    assert r.json()['n_features'] == 247 and r.json()['model_type'] == 'sklearn'


def test_rest_api_predicts_a_beat(api_client, synthetic_record):
    features, true = beat_vector(synthetic_record, 50)
    r = api_client.post('/predict', json={'features': features})
    assert r.status_code == 200
    body = r.json()
    assert body['prediction'] in {'N', 'S', 'V', 'F', 'Q'}
    assert body['prediction'] == true
    assert abs(sum(body['probabilities'].values()) - 1) < 1e-3


def test_rest_api_rejects_a_wrong_vector_length(api_client):
    r = api_client.post('/predict', json={'features': [0.0] * 239})
    assert r.status_code == 422
    assert 'Expected 247 features' in r.json()['detail']
