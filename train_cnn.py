"""
1D CNN (Convolutional Neural Network) for ECG classification
==========================================================
The third model. Unlike RF and MLP, the convolutional network learns the
beat morphology BY ITSELF from the raw signal instead of relying only on
hand-crafted features.

Hybrid architecture (two input heads):
  1. Morphology: the raw beat segment -> Conv1D blocks (learn the shape)
  2. RR/P context: the last 9 engineered features -> straight into the dense part
     (to keep the timing information, which helped class S a lot)

The two branches are merged, followed by dense + softmax for the 5 AAMI classes.

Uses the SAME extraction logic and the SAME split (DS1/DS2 + separate
validation records) as the other two models, so the comparison is clean.

Requires: tensorflow
Usage:
    python train_cnn.py --data_dir /path/to/mit-bih --out_dir artifacts_cnn
"""

import os
import argparse
import numpy as np
import joblib
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from train_arrhythmia import load_split
from utility import N_CONTEXT  # 9 engineered features at the end of the vector


def build_model(morph_len, n_context, n_classes):
    """Hybrid 1D CNN: a Conv branch for morphology + a dense branch for RR/P context."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    # --- Branch 1: morphology through convolutions ---
    morph_in = layers.Input(shape=(morph_len, 1), name='morphology')
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(morph_in)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    # --- Branch 2: RR/P context directly ---
    ctx_in = layers.Input(shape=(n_context,), name='context')
    c = layers.Dense(16, activation='relu')(ctx_in)

    # --- Merge ---
    merged = layers.concatenate([x, c])
    z = layers.Dense(64, activation='relu')(merged)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(n_classes, activation='softmax')(z)

    model = Model(inputs=[morph_in, ctx_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def split_features(X):
    """Splits every vector into (morphology, context)."""
    morph = X[:, :-N_CONTEXT]
    ctx = X[:, -N_CONTEXT:]
    return morph, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', default='artifacts_cnn')
    ap.add_argument('--lead', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # DS1 (without VAL_RECORDS) -> train, VAL_RECORDS -> validation, DS2 -> test
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(
        args.data_dir, args.lead, with_val=True)

    # -- Label encoding --
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    labels = np.arange(len(le.classes_))
    print("Classes:", list(le.classes_))

    # -- Scaling (fit on train only, then saved) --
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # -- Split into morphology and context --
    Xtr_morph, Xtr_ctx = split_features(X_train_s)
    Xva_morph, Xva_ctx = split_features(X_val_s)
    Xte_morph, Xte_ctx = split_features(X_test_s)
    morph_len = Xtr_morph.shape[1]

    # Reshape the morphology for Conv1D: (samples, length, 1 channel)
    Xtr_morph = Xtr_morph[..., np.newaxis]
    Xva_morph = Xva_morph[..., np.newaxis]
    Xte_morph = Xte_morph[..., np.newaxis]

    # -- Balancing via class weights (natural for neural networks) --
    classes = np.unique(y_train_enc)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_enc)
    # We CAP the weights: classes with a handful of examples (Q=15) would otherwise get
    # a weight of ~1000, which pulls training in the wrong direction and destabilizes
    # the other classes. A cap of 50 is a reasonable compromise.
    weights = np.clip(weights, None, 50.0)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights (capped):", class_weight)

    # -- Model --
    import tensorflow as tf
    model = build_model(morph_len, N_CONTEXT, len(le.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True,
                                         monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5,
                                             monitor='val_loss', min_lr=1e-5),
    ]

    print("Training CNN...")
    model.fit(
        {'morphology': Xtr_morph, 'context': Xtr_ctx},
        y_train_enc,
        # Validation by PATIENT (separate records), not the last 10% of beats
        validation_data=({'morphology': Xva_morph, 'context': Xva_ctx}, y_val_enc),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # -- Evaluation --
    probs = model.predict({'morphology': Xte_morph, 'context': Xte_ctx})
    y_pred = np.argmax(probs, axis=1)

    print("\n===== Classification Report (CNN) =====")
    print(classification_report(y_test_enc, y_pred, labels=labels,
                                target_names=le.classes_, zero_division=0))
    print("===== Confusion Matrix =====")
    print("Rows=true, columns=predicted; row/column order =", list(le.classes_))
    print(confusion_matrix(y_test_enc, y_pred, labels=labels))

    # -- Saving --
    model.save(os.path.join(args.out_dir, 'model.keras'))
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(args.out_dir, 'label_encoder.pkl'))
    # Also save how many features are context (needed for deployment)
    joblib.dump({'n_context': N_CONTEXT, 'morph_len': morph_len},
                os.path.join(args.out_dir, 'meta.pkl'))
    print(f"\nSaved: model.keras, scaler.pkl, label_encoder.pkl, meta.pkl in {args.out_dir}/")


if __name__ == '__main__':
    main()