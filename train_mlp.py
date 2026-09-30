"""
MLP (neural network) for ECG classification
==========================================
Uses the SAME 239 features as the Random Forest pipeline -> a clean comparison.
Differences from train_arrhythmia.py:
  * Classifier: MLPClassifier (multi-layer perceptron) instead of Random Forest
  * Balancing: oversampling by repeating real examples instead of SMOTE
  * Early stopping by macro F1 on SEPARATE validation records (patients).
    sklearn's built-in early_stopping takes its validation set from the already
    oversampled data -> copies of the same beats end up in both train
    and validation, and stopping becomes meaningless. So we train manually
    epoch by epoch with partial_fit and keep the best epoch.

Split: DS1 (without VAL_RECORDS) -> train, VAL_RECORDS -> validation,
DS2 -> test (see train_arrhythmia.get_split).

Usage:
    python train_mlp.py --data_dir /path/to/mit-bih --out_dir artifacts_mlp
"""

import os
import copy
import argparse
import numpy as np
import joblib
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Reuse the existing feature extraction and split logic,
# so that it is GUARANTEED to be the same as for Random Forest.
from train_arrhythmia import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='artifacts_mlp')
    ap.add_argument('--lead', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_split(
        args.data_dir, args.lead, with_val=True)

    # -- Label encoding --
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    classes = np.arange(len(le.classes_))
    print("Classes:", list(le.classes_))

    # -- Scaling (mandatory for neural networks!) - fit on train only --
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # -- MLP architecture --
    # Two hidden layers (128, 64). No built-in early_stopping (see above).
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,                 # L2 regularization
        batch_size=256,
        learning_rate_init=1e-3,
        random_state=42,
    )

    # -- Epoch-by-epoch training with early stopping on validation macro F1 --
    # Every epoch: a fresh balanced oversample of train (repeating real
    # examples, no synthetic data like SMOTE). Validation is NOT touched.
    print("Training MLP...")
    best_f1, best_epoch, best_model, bad = -1.0, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        idx = _balanced_indices(y_train_enc, random_state=epoch)
        clf.partial_fit(X_train_s[idx], y_train_enc[idx], classes=classes)
        val_f1 = f1_score(y_val_enc, clf.predict(X_val_s), labels=classes,
                          average='macro', zero_division=0)
        print(f"  epoch {epoch:3d} | loss {clf.loss_:.4f} | val macro F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch, best_model, bad = val_f1, epoch, copy.deepcopy(clf), 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  Early stopping: {args.patience} epochs without improvement")
                break
    clf = best_model
    print(f"Best epoch: {best_epoch} (val macro F1 = {best_f1:.4f})")

    # -- Evaluation on DS2 --
    y_pred = clf.predict(X_test_s)
    print("\n===== Classification Report (MLP) =====")
    print(classification_report(y_test_enc, y_pred, labels=classes,
                                target_names=le.classes_, zero_division=0))
    print("===== Confusion Matrix =====")
    print("Rows=true, columns=predicted; row/column order =", list(le.classes_))
    print(confusion_matrix(y_test_enc, y_pred, labels=classes))

    # -- Saving --
    joblib.dump(clf, os.path.join(args.out_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(args.out_dir, 'label_encoder.pkl'))
    print(f"\nSaved: model.pkl, scaler.pkl, label_encoder.pkl in {args.out_dir}/")


def _balanced_indices(y, random_state=42):
    """
    Returns indices in which every class is oversampled to the size
    of the largest class (by repeating real examples, not synthetic ones).
    Lighter than SMOTE and works well with neural networks.
    """
    rng = np.random.default_rng(random_state)
    counts = Counter(y)
    max_n = max(counts.values())
    all_idx = []
    for cls in counts:
        cls_idx = np.where(y == cls)[0]
        # sample with replacement up to max_n
        chosen = rng.choice(cls_idx, size=max_n, replace=True)
        all_idx.append(chosen)
    out = np.concatenate(all_idx)
    rng.shuffle(out)
    return out


if __name__ == '__main__':
    main()
