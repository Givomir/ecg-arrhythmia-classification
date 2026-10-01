"""Feature extraction (utility.py) - the single source for training and deployment."""
import joblib
import numpy as np
import pytest

from utility import (N_CONTEXT, N_MORPH, N_RHYTHM, extract_beat_features, feature_groups,
                     preprocess_beat, record_features, rhythm_features, uses_rhythm_features)
from tests.conftest import FS, synth_beats, synth_signal

BASE = N_MORPH + N_CONTEXT          # 239


def regular_peaks(n=50, rr=0.8):
    step = round(rr * FS)          # exact integer step -> a perfectly regular rhythm
    return [FS + k * step for k in range(n)]


def test_beat_vector_layout():
    sig = np.zeros(10 * FS)
    v = extract_beat_features(sig, 4 * FS, int(3.2 * FS), int(4.8 * FS), FS, local_mean_rr=0.8)
    assert v.shape == (BASE,)
    prev_rr, next_rr, ratio, prev_norm, next_norm = v[N_MORPH:N_MORPH + 5]
    assert prev_rr == pytest.approx(0.8)
    assert next_rr == pytest.approx(0.8)
    assert ratio == pytest.approx(1.0)
    assert prev_norm == pytest.approx(1.0)
    assert next_norm == pytest.approx(1.0)


def test_premature_beat_has_normalized_rr_below_one():
    sig = np.zeros(10 * FS)
    v = extract_beat_features(sig, 4 * FS, int(3.5 * FS), int(4.9 * FS), FS, local_mean_rr=0.8)
    assert v[N_MORPH + 3] == pytest.approx(0.5 / 0.8)     # came early
    assert v[N_MORPH + 4] > 1                            # followed by a pause


def test_beats_too_close_to_the_edge_are_skipped():
    sig = np.zeros(10 * FS)
    assert extract_beat_features(sig, 50, None, 300, FS) is None              # start
    assert extract_beat_features(sig, len(sig) - 20, len(sig) - 300, None, FS) is None


def test_missing_neighbours_and_local_mean_fall_back_to_neutral_values():
    sig = np.zeros(10 * FS)
    v = extract_beat_features(sig, 4 * FS, None, None, FS, local_mean_rr=None)
    assert v[N_MORPH] == 0 and v[N_MORPH + 1] == 0 and v[N_MORPH + 2] == 0
    assert v[N_MORPH + 3] == 1.0 and v[N_MORPH + 4] == 1.0


def test_p_wave_energy_is_higher_when_a_p_wave_exists():
    x = np.arange(10 * FS)
    r = 4 * FS
    flat = np.exp(-0.5 * ((x - r) / 4) ** 2)
    with_p = flat + 0.2 * np.exp(-0.5 * ((x - (r - 0.16 * FS)) / 9) ** 2)
    e_flat = extract_beat_features(flat, r, r - FS, r + FS, FS)[N_MORPH + 5]
    e_p = extract_beat_features(with_p, r, r - FS, r + FS, FS)[N_MORPH + 5]
    assert e_p > e_flat


def test_record_features_keeps_order_and_skips_edges():
    peaks = [30] + regular_peaks(20)                      # first peak too close to the start
    sig = np.zeros(peaks[-1] + 2 * FS)
    X, kept = record_features(sig, peaks, FS)
    assert kept == list(range(1, len(peaks)))
    assert X.shape == (len(peaks) - 1, BASE)


def test_record_features_rhythm_adds_the_extra_columns():
    peaks = regular_peaks(30)
    sig = np.zeros(peaks[-1] + 2 * FS)
    X, _ = record_features(sig, peaks, FS)
    Xr, _ = record_features(sig, peaks, FS, rhythm=True)
    assert Xr.shape[1] == BASE + N_RHYTHM
    np.testing.assert_allclose(Xr[:, :BASE], X)


def test_rhythm_features_regular_rhythm():
    peaks = regular_peaks(60)
    sig = np.zeros(peaks[-1] + 2 * FS)
    X, _ = record_features(sig, peaks, FS)
    R = rhythm_features(X)
    late = R[20:-1]
    np.testing.assert_allclose(late[:, 0], 1.0, atol=1e-6)       # prev / long-window mean
    np.testing.assert_allclose(late[:, 2], 0.0, atol=1e-6)       # cv of a regular rhythm
    assert np.all(late[:, 4] == 0)                               # pnn50


def test_rhythm_features_premature_beat_in_regular_rhythm():
    beats = synth_beats(80)
    sig = np.zeros(beats[-1][0] + 2 * FS)
    X, kept = record_features(sig, [b[0] for b in beats], FS, rhythm=True)
    R = X[:, BASE:]
    sym = [beats[i][1] for i in kept]
    s_rows = [k for k, s in enumerate(sym) if s == 'A' and k > 15]
    n_rows = [k for k, s in enumerate(sym) if s == 'N' and k > 15 and sym[k - 1] == 'N']
    assert np.mean(R[s_rows, 0]) < 0.8 < np.mean(R[n_rows, 0])   # early vs. long-window mean
    assert np.all(np.abs(R[:, 5]) <= 10) and np.all(np.abs(R[:, 6]) <= 10)   # z is clipped


def test_feature_groups_cover_the_vector():
    g239 = feature_groups(BASE)
    g247 = feature_groups(BASE + N_RHYTHM)
    assert sum(len(c) for c in g239.values()) == BASE
    assert sum(len(c) for c in g247.values()) == BASE + N_RHYTHM
    assert set(g247) - set(g239) == {'Long-window RR (rhythm)', 'Rhythm irregularity'}


def test_uses_rhythm_features_reads_meta(tmp_path):
    assert uses_rhythm_features(str(tmp_path)) is False
    joblib.dump({'n_context': 9}, tmp_path / 'meta.pkl')              # CNN-style meta
    assert uses_rhythm_features(str(tmp_path)) is False
    joblib.dump({'rhythm_features': True}, tmp_path / 'meta.pkl')
    assert uses_rhythm_features(str(tmp_path)) is True


def test_preprocess_beat_uses_the_saved_scaler_and_handles_nan():
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.array([[0.0, 0.0], [2.0, 4.0]]))
    out = preprocess_beat([1.0, np.nan], sc)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[0.0, -1.0]])


def test_synthetic_signal_features_are_finite():
    beats = synth_beats(40)
    sig = synth_signal(beats, beats[-1][0] + 2 * FS)
    X, _ = record_features(sig, [b[0] for b in beats], FS, rhythm=True)
    assert np.isfinite(X).all()
