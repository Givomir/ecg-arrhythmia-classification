"""The two-stage cascade classifier (cascade.py)."""
import pickle

import numpy as np
import pytest

from cascade import (CascadeClassifier, MORPH_COLS, N_FEATURES, RHYTHM_COLS, STAGE1_COLS)
from utility import N_CONTEXT, N_MORPH

S, N, V = 3, 1, 4


def make_data(n=600, seed=0):
    """N/S/V beats: S differs from N only in rhythm, V in morphology."""
    rng = np.random.default_rng(seed)
    y = rng.choice([N, S, V], size=n, p=[0.7, 0.2, 0.1])
    X = rng.normal(size=(n, N_FEATURES)).astype(np.float64)
    X[y == V, :20] += 4.0                       # V: different morphology
    X[y == S, RHYTHM_COLS[3]] -= 4.0            # S: early beat (normalized RR)
    return X, y


@pytest.fixture(scope='module')
def model():
    X, y = make_data()
    return CascadeClassifier(s_class=S, n_class=N, n_classes=5, n_estimators=30,
                             n_estimators_stage2=30, n_pca=5, random_state=0).fit(X, y)


def test_column_layout():
    assert N_FEATURES == 247
    assert len(MORPH_COLS) == N_MORPH
    assert not set(range(N_MORPH + 5, N_MORPH + N_CONTEXT)) & set(STAGE1_COLS)   # no P wave
    assert len(set(STAGE1_COLS)) == len(STAGE1_COLS)


def test_predict_and_proba_shapes(model):
    X, _ = make_data(100, seed=1)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert pred.shape == (100,)
    assert proba.shape == (100, 5)
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-6)
    assert set(pred) <= {N, S, V}


def test_learns_the_synthetic_classes(model):
    X, y = make_data(400, seed=2)
    pred = model.predict(X)
    assert np.mean(pred == y) > 0.9
    assert np.mean(pred[y == S] == S) > 0.8


def test_threshold_controls_s_calls(model):
    X, _ = make_data(200, seed=3)
    P1, gate, p2 = model.stage_scores(X)
    all_s = model.predict_from_scores(P1, gate, p2, threshold=0.0)
    no_s = model.predict_from_scores(P1, gate, p2, threshold=1.01)
    assert np.all(all_s[gate] == S)              # every gated beat becomes S
    assert not np.any(no_s == S)                 # nothing passes the threshold
    assert np.all(all_s[~gate] == no_s[~gate])   # non-gated beats are unaffected


def test_stage2_only_sees_beats_stage1_calls_n_or_s(model):
    X, _ = make_data(200, seed=4)
    P1, gate, p2 = model.stage_scores(X)
    assert np.array_equal(gate, np.isin(P1.argmax(1), [N, S]))
    assert np.all(p2[~gate] == 0)


def test_pickle_roundtrip_keeps_predictions(model):
    X, _ = make_data(50, seed=5)
    clone = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(clone.predict(X), model.predict(X))
