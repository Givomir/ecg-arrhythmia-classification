"""
Cascade model: multi-class RF + dedicated S-vs-N detector
==========================================================
The model chosen by the S ablation study (ablation_s.py, "cascade_rf"):
  * Stage 1: Random Forest on morphology + rhythm features (no P wave)
  * Stage 2: S-vs-N Random Forest on PCA morphology + rhythm features,
    applied to the beats stage 1 calls N or S
  * Rhythm features include rhythm IRREGULARITY, so that "early" beats in
    atrial fibrillation are not called S (see cascade.py, utility.py)

Threshold: the stage-2 S threshold is tuned on out-of-fold scores from a
GroupKFold by RECORD over DS1 (a model never sees the patient it scores).
DS2 is used only for the final evaluation.

Evaluation on DS2 is also reported WITHOUT record 232: it holds 1382 of the
1837 DS2 S beats, and its S beats form the dominant rhythm (runs of atrial
ectopy between long pauses) rather than isolated premature beats, so it
would otherwise dominate the S metrics.

Saves (for deployment): model.pkl, scaler.pkl, label_encoder.pkl and
meta.pkl ({'rhythm_features': True, ...}) - the apps read meta.pkl and
extract the features with record_features(rhythm=True).

Usage:
    python train_cascade.py --data_dir /path/to/mit-bih --out_dir artifacts_cascade
"""

import os
import argparse
import numpy as np
import joblib
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

from cascade import CascadeClassifier, N_FEATURES
from train_arrhythmia import get_split, load_beats
from utility import record_features

DOMINANT_RECORD = '232'


def load_records(records, data_dir, lead):
    """Features (with rhythm features), labels and record id of every beat."""
    X, y, g = [], [], []
    for rec in records:
        signals, fs, beats = load_beats(os.path.join(data_dir, rec))
        feats, kept = record_features(signals[:, lead], [b[0] for b in beats], fs,
                                      rhythm=True)
        if not len(kept):
            continue
        X.append(feats)
        y.extend(beats[i][1] for i in kept)
        g.extend([rec] * len(kept))
    return np.nan_to_num(np.vstack(X)), np.array(y), np.array(g)


def s_metrics(y, pred, s):
    tp = np.sum((y == s) & (pred == s))
    rec = tp / max(1, np.sum(y == s))
    prec = tp / max(1, np.sum(pred == s))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return rec, prec, f1


def tune_threshold(model, P1, gate, p2, y):
    """The stage-2 threshold with the highest S F1."""
    best = (-1.0, 0.5)
    for t in np.linspace(0.02, 0.98, 97):
        pred = model.predict_from_scores(P1, gate, p2, threshold=t)
        f1 = s_metrics(y, pred, model.s_class)[2]
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='Folder with the MIT-BIH .dat/.hea/.atr files')
    ap.add_argument('--out_dir', default='artifacts_cascade')
    ap.add_argument('--lead', type=int, default=0)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds1, ds2 = get_split(args.data_dir)
    print(f"Extracting features (DS1, {len(ds1)} records)...")
    X1, y1_raw, g1 = load_records(ds1, args.data_dir, args.lead)
    print(f"Extracting features (DS2, {len(ds2)} records)...")
    X2, y2_raw, g2 = load_records(ds2, args.data_dir, args.lead)
    assert X1.shape[1] == N_FEATURES
    print(f"  DS1: {X1.shape} | {dict(Counter(y1_raw))}")
    print(f"  DS2: {X2.shape} | {dict(Counter(y2_raw))}")

    le = LabelEncoder().fit(np.concatenate([y1_raw, y2_raw]))
    y1, y2 = le.transform(y1_raw), le.transform(y2_raw)
    s, n = int(le.transform(['S'])[0]), int(le.transform(['N'])[0])
    print("Classes:", list(le.classes_))

    def new_model():
        return CascadeClassifier(s_class=s, n_class=n, n_classes=len(le.classes_),
                                 random_state=args.seed)

    # -- 1. Out-of-fold scores by patient over DS1 -> threshold --
    print(f"\nOut-of-fold scores ({args.folds}-fold GroupKFold by record over DS1)...")
    P1 = np.zeros((len(y1), len(le.classes_)))
    gate = np.zeros(len(y1), dtype=bool)
    p2 = np.zeros(len(y1))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=args.folds).split(X1, y1, g1), 1):
        sc = StandardScaler().fit(X1[tr])
        m = new_model().fit(sc.transform(X1[tr]), y1[tr])
        P1[va], gate[va], p2[va] = m.stage_scores(sc.transform(X1[va]))
        print(f"  fold {k}/{args.folds} done")
    threshold, oof_f1 = tune_threshold(new_model(), P1, gate, p2, y1)
    print(f"Stage-2 S threshold: {threshold:.2f} (DS1 out-of-fold S F1 = {oof_f1:.3f})")

    # -- 2. Final model on all of DS1 --
    print("\nTraining the final cascade on all of DS1...")
    scaler = StandardScaler().fit(X1)
    model = new_model().fit(scaler.transform(X1), y1)
    model.threshold = threshold

    # -- 3. Evaluation on DS2 --
    pred = model.predict(scaler.transform(X2))
    labels = np.arange(len(le.classes_))
    print("\n===== Classification Report (cascade, DS2) =====")
    print(classification_report(y2, pred, labels=labels,
                                target_names=le.classes_, zero_division=0))
    print("===== Confusion Matrix =====")
    print("Rows=true, columns=predicted; row/column order =", list(le.classes_))
    print(confusion_matrix(y2, pred, labels=labels))

    rest = g2 != DOMINANT_RECORD
    print("\n===== S detection =====")
    for name, mask in [('DS2', slice(None)), (f'DS2 without {DOMINANT_RECORD}', rest)]:
        r, p, f = s_metrics(y2[mask], pred[mask], s)
        print(f"  {name:18s} S recall {r:.3f} | S precision {p:.3f} | S F1 {f:.3f}")
    print(f"\n  {'record':>6} {'S':>5} {'caught':>7} {'N->S':>6}")
    for rec in ds2:
        m = g2 == rec
        ns = int(np.sum(y2[m] == s))
        tp = int(np.sum((y2[m] == s) & (pred[m] == s)))
        fp = int(np.sum((y2[m] == n) & (pred[m] == s)))
        if ns or fp:
            print(f"  {rec:>6} {ns:>5} {tp:>7} {fp:>6}")

    # -- 4. Save everything needed for deployment --
    joblib.dump(model, os.path.join(args.out_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(args.out_dir, 'label_encoder.pkl'))
    joblib.dump({'rhythm_features': True, 'threshold': threshold,
                 'n_features': N_FEATURES}, os.path.join(args.out_dir, 'meta.pkl'))
    print(f"\nSaved: model.pkl, scaler.pkl, label_encoder.pkl, meta.pkl in {args.out_dir}/")


if __name__ == '__main__':
    main()
