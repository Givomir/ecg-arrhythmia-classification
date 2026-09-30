"""
Multi-class ECG Arrhythmia Classification (MIT-BIH)
====================================================
Clean, correct version of the pipeline:
  * Segmentation of every beat around the R peak (morphology)
  * Combined features: beat morphology + local RR context
  * Multi-class classification by AAMI groups (not binary)
  * Standard inter-patient DS1/DS2 split (de Chazal et al., 2004) -
    no leakage between patients and comparable with the literature
  * The StandardScaler is fitted on train only and is SAVED
  * SMOTE is applied to train only
  * Saves: model, scaler, label encoder -> for deployment

Usage:
    python train_arrhythmia.py --data_dir /path/to/mit-bih --out_dir artifacts
"""

import os
import argparse
import numpy as np
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Feature extraction lives in utility.py (a single source for training
# and deployment). We re-export extract_beat_features for older code.
from utility import extract_beat_features, record_features  # noqa: F401


# ---------------------------------------------------------------------------
# 1. AAMI grouping of the annotations
# ---------------------------------------------------------------------------
# MIT-BIH uses ~19 symbols. The AAMI EC57 standard reduces them to 5 clinically
# meaningful super-groups. This keeps ALL pathologies as separate classes
# instead of merging them into "abnormal".
AAMI_GROUPS = {
    # N - Normal / bundle branch blocks (normal conduction)
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S - Supraventricular ectopic
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V - Ventricular ectopic
    'V': 'V', 'E': 'V',
    # F - Fusion (fusion beats)
    'F': 'F',
    # Q - Unknown / paced
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

# Symbols that are NOT beats (rhythm annotations, noise, etc.) - skipped
NON_BEAT = set(['+', '~', '|', '"', '[', ']', '!', 'x', '(', ')', 'p', 't', 'u'])


# ---------------------------------------------------------------------------
# 2. Inter-patient split (de Chazal et al., 2004)
# ---------------------------------------------------------------------------
# The standard in the literature: DS1 for training, DS2 for testing. Both sets are
# balanced by arrhythmia type. The paced records 102/104/107/217 are
# excluded (AAMI recommendation).
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
       '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
       '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
       '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
       '233', '234']

# Validation records (a subset of DS1) - for early stopping of MLP/CNN.
# Validation is by PATIENT (whole records), separated BEFORE oversampling,
# so it contains no copies of training beats. Records with S and V beats
# were chosen so there is a meaningful signal for the rare classes.
VAL_RECORDS = ['118', '207', '215', '223']


def available(records, data_dir):
    """Keeps only the records actually present in the folder."""
    return [r for r in records if os.path.exists(os.path.join(data_dir, r + '.dat'))]


def get_split(data_dir, with_val=False):
    """
    Returns (train, test) or (train, val, test) lists of records.
    train/val come from DS1, test is DS2.
    """
    ds1 = available(DS1, data_dir)
    ds2 = available(DS2, data_dir)
    if not ds1 or not ds2:
        raise FileNotFoundError(f"No DS1/DS2 records found in {data_dir}")
    if not with_val:
        return ds1, ds2
    val = [r for r in ds1 if r in VAL_RECORDS]
    train = [r for r in ds1 if r not in VAL_RECORDS]
    return train, val, ds2


# ---------------------------------------------------------------------------
# 3. Processing a list of records -> X, y
# ---------------------------------------------------------------------------
def load_beats(rec_path):
    """Reads a record and returns (signals, fs, [(r_peak, aami_class), ...])."""
    import wfdb
    record = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, 'atr')
    beats = [(s, AAMI_GROUPS[sym]) for s, sym in zip(annotation.sample, annotation.symbol)
             if sym in AAMI_GROUPS]
    return record.p_signal, record.fs, beats


def process_records(record_list, data_dir, lead=0):
    X, y = [], []
    for rec in record_list:
        try:
            signals, fs, beats = load_beats(os.path.join(data_dir, rec))
        except Exception as e:
            print(f"  [!] Skipped record {rec}: {e}")
            continue
        feats, kept = record_features(signals[:, lead], [b[0] for b in beats], fs)
        if len(kept):
            X.append(feats)
            y.extend(beats[i][1] for i in kept)
    return np.vstack(X), np.array(y)


def load_split(data_dir, lead=0, with_val=False):
    """Extracts the features for train(/val)/test and prints the class distributions."""
    split = get_split(data_dir, with_val=with_val)
    names = ['train', 'val', 'test'] if with_val else ['train', 'test']
    out = []
    for name, recs in zip(names, split):
        print(f"Extracting features ({name}, {len(recs)} records)...")
        X, y = process_records(recs, data_dir, lead)
        print(f"  {name}: X={X.shape} | {dict(Counter(y))}")
        out.extend([X, y])
    return out


# ---------------------------------------------------------------------------
# 4. Main training
# ---------------------------------------------------------------------------
def main():
    from imblearn.over_sampling import SMOTE

    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Folder with the MIT-BIH .dat/.hea/.atr files')
    ap.add_argument('--out_dir', default='artifacts', help='Where to save the model/scaler/encoder')
    ap.add_argument('--lead', type=int, default=0, help='Which lead to use (0 = usually MLII)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # DS1 -> training, DS2 -> test (split by PATIENT)
    X_train, y_train, X_test, y_test = load_split(args.data_dir, args.lead)

    # -- Label encoding --
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    print("Classes:", list(le.classes_))

    # -- Scaling: fit ONLY on train, then save --
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)   # transform only!

    # -- SMOTE on train only (to balance the rare classes) --
    # k_neighbors is lowered if some class is very small
    min_count = min(Counter(y_train_enc).values())
    k = min(5, max(1, min_count - 1))
    try:
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train_s, y_train_enc)
        print("After SMOTE:", Counter(y_res))
    except ValueError as e:
        print(f"  [!] SMOTE skipped ({e}); training on the original data")
        X_res, y_res = X_train_s, y_train_enc

    # -- Model: Random Forest (robust, naturally multi-class) --
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    print("Training Random Forest...")
    clf.fit(X_res, y_res)

    # -- Evaluation --
    y_pred = clf.predict(X_test_s)
    labels = np.arange(len(le.classes_))
    print("\n===== Classification Report =====")
    print(classification_report(y_test_enc, y_pred, labels=labels,
                                target_names=le.classes_, zero_division=0))
    print("===== Confusion Matrix =====")
    print("Rows=true, columns=predicted; row/column order =", list(le.classes_))
    print(confusion_matrix(y_test_enc, y_pred, labels=labels))

    # -- Save EVERYTHING needed for deployment --
    joblib.dump(clf, os.path.join(args.out_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(args.out_dir, 'label_encoder.pkl'))
    print(f"\nSaved: model.pkl, scaler.pkl, label_encoder.pkl in {args.out_dir}/")


if __name__ == '__main__':
    main()
