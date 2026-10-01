"""Smoke tests on REAL MIT-BIH records (marker: data).

Uses MITBIH_DIR if set, otherwise downloads records 100 and 232 from
PhysioNet (about 6 MB) into tests/.data/mitdb. Skipped when neither works.
"""
import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.data]

RECORDS = ['100', '232']
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def mitbih_dir():
    d = os.environ.get('MITBIH_DIR') or os.path.join(ROOT, '.data', 'mitdb')
    missing = [r for r in RECORDS
               if not all(os.path.exists(os.path.join(d, r + ext)) for ext in ('.dat', '.hea', '.atr'))]
    if missing:
        try:
            import wfdb
            os.makedirs(d, exist_ok=True)
            wfdb.dl_database('mitdb', dl_dir=d, records=missing)
        except Exception as e:                       # no network
            pytest.skip(f"MIT-BIH records not available: {e}")
    return d


def test_annotations_and_features_of_record_232(mitbih_dir):
    from train_arrhythmia import load_beats
    from utility import record_features
    signals, fs, beats = load_beats(os.path.join(mitbih_dir, '232'))
    assert fs == 360 and signals.shape[0] == 650000
    X, kept = record_features(signals[:, 0], [b[0] for b in beats], fs, rhythm=True)
    assert X.shape == (1780, 247)
    true = [beats[i][1] for i in kept]
    assert true.count('S') == 1382 and true.count('N') == 398      # the record studied in README
    assert np.isfinite(np.nan_to_num(X)).all()


def test_cascade_trains_and_analyzes_real_records(mitbih_dir, tmp_path):
    import joblib
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from cascade import CascadeClassifier
    from rag_recommend import analyze_record
    from train_cascade import load_records

    X, y_raw, _ = load_records(['100'], mitbih_dir, 0)
    le = LabelEncoder().fit(np.array(['F', 'N', 'Q', 'S', 'V']))
    scaler = StandardScaler().fit(X)
    model = CascadeClassifier(s_class=3, n_class=1, n_classes=5, n_estimators=10,
                              n_estimators_stage2=10, n_pca=5, random_state=0)
    model.fit(scaler.transform(X), le.transform(y_raw))
    for name, obj in [('model', model), ('scaler', scaler), ('label_encoder', le)]:
        joblib.dump(obj, tmp_path / f'{name}.pkl')
    joblib.dump({'rhythm_features': True}, tmp_path / 'meta.pkl')

    res = analyze_record(mitbih_dir, '232', str(tmp_path))
    assert res['total'] == 1780
    assert set(res['counts']) <= {'N', 'S', 'V', 'F', 'Q'}
