"""Smaller units: AAMI mapping and data split, RAG evaluation helpers, desktop helpers."""
import base64
import os

import numpy as np
import pytest

import desktop_app as D
import evaluate_rag as E
from train_arrhythmia import AAMI_GROUPS, DS1, DS2, VAL_RECORDS, get_split


# -- data split -------------------------------------------------------------------
def test_aami_mapping_covers_the_five_classes():
    assert set(AAMI_GROUPS.values()) == {'N', 'S', 'V', 'F', 'Q'}
    assert AAMI_GROUPS['A'] == 'S' and AAMI_GROUPS['V'] == 'V' and AAMI_GROUPS['L'] == 'N'


def test_inter_patient_split_has_no_overlap():
    assert not set(DS1) & set(DS2)
    assert set(VAL_RECORDS) <= set(DS1)
    assert not {'102', '104', '107', '217'} & (set(DS1) | set(DS2))    # paced records


def test_get_split_uses_only_present_records(tmp_path):
    for rec in ['101', '106', '118', '100', '103']:
        (tmp_path / f'{rec}.dat').write_bytes(b'')
    train, test = get_split(str(tmp_path))
    assert train == ['101', '106', '118'] and test == ['100', '103']
    train, val, test = get_split(str(tmp_path), with_val=True)
    assert val == ['118'] and '118' not in train


def test_get_split_without_records_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_split(str(tmp_path))


# -- RAG evaluation helpers ----------------------------------------------------------
@pytest.mark.parametrize('source, expected', [
    ('al-khatib-et-al-2018-ventricular.pdf', 'ventricular'),
    ('joglar-et-al-2023-atrial-fibrillation.pdf', 'atrial'),
    ('something-else.pdf', 'unknown'),
])
def test_classify_source(source, expected):
    assert E.classify_source(source) == expected


def test_term_regex_matches_word_starts():
    # callers lowercase the text; the term is lowercased by term_regex itself
    rx = E.term_regex('Ablation')
    assert rx.search('catheter ablation is')
    assert rx.search('ablations')                # prefix of a word
    assert not rx.search('preablation')          # not inside a word


# -- desktop helpers -------------------------------------------------------------------
def test_list_records_needs_dat_hea_and_atr(tmp_path):
    for ext in ('.dat', '.hea', '.atr'):
        (tmp_path / f'100{ext}').write_bytes(b'')
    (tmp_path / '101.dat').write_bytes(b'')            # incomplete record
    for ext in ('.dat', '.hea', '.atr'):
        (tmp_path / f'20{ext}').write_bytes(b'')
    assert D.list_records(str(tmp_path)) == ['20', '100']    # numeric order
    assert D.list_records(str(tmp_path / 'missing')) == []


def test_overview_envelope_keeps_min_and_max():
    sig = np.zeros(3600)
    sig[1000], sig[2000] = 5.0, -3.0
    t, y = D.overview_envelope(sig, 360, bucket_sec=0.5)
    assert y.max() == 5.0 and y.min() == -3.0
    assert len(t) == len(y) == 2 * (3600 // 180)
    assert np.all(np.diff(t[::2]) > 0)


def test_float32_transfer_roundtrip():
    x = np.array([0.5, -1.25, 3.0])
    back = np.frombuffer(base64.b64decode(D._b64(x)), dtype='<f4')
    np.testing.assert_allclose(back, x)


def test_settings_live_in_the_test_home():
    assert os.environ['ECG_ANALYZER_HOME'] == D.SETTINGS_DIR
