"""
Visualization and explanation of ECG classification
==============================================
Reads a standard MIT-BIH record (.hea/.dat/.atr), classifies all beats
with the trained model, and shows:

  1. A PLOT of the whole signal with beats marked and colored by class
     (green=N, orange=S, red=V, purple=F, gray=Q). Misclassified
     beats are outlined so they stand out clearly.

  2. An EXPLANATION of which features weigh in the decision - both on the plot
     (stars over the deciding region) and in the terminal (text with the
     actual RR/P values + importance of the feature groups).

Usage:
    python visualize_ecg.py --data_dir "C:\\...\\mit-bih" --record 200
    python visualize_ecg.py --data_dir "..." --record 208 --start 0 --seconds 20

Arguments:
    --data_dir  folder with the MIT-BIH records
    --record    record number (e.g. 200)
    --artifacts model folder (artifacts_cascade / artifacts / artifacts_mlp / artifacts_cnn)
    --start     second at which the plot starts (default 0)
    --seconds   how many seconds to show (default 10)
"""
import os
import argparse
import numpy as np
import wfdb
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from train_arrhythmia import AAMI_GROUPS
from utility import record_features, uses_rhythm_features, feature_groups

# Colors per class
CLASS_COLORS = {
    'N': '#2ca02c',   # green
    'S': '#ff7f0e',   # orange
    'V': '#d62728',   # red
    'F': '#9467bd',   # purple
    'Q': '#7f7f7f',   # gray
}
CLASS_NAMES = {
    'N': 'Normal', 'S': 'Supraventricular', 'V': 'Ventricular',
    'F': 'Fusion', 'Q': 'Unclassifiable',
}

def load_model(artifacts_dir):
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
    le = joblib.load(os.path.join(artifacts_dir, 'label_encoder.pkl'))
    keras_path = os.path.join(artifacts_dir, 'model.keras')
    if os.path.exists(keras_path):
        import tensorflow as tf
        model = tf.keras.models.load_model(keras_path)
        meta = joblib.load(os.path.join(artifacts_dir, 'meta.pkl'))
        return ('cnn', model, scaler, le, meta['n_context'])
    else:
        model = joblib.load(os.path.join(artifacts_dir, 'model.pkl'))
        return ('sklearn', model, scaler, le, 9)


def predict_all(model_type, model, scaler, le, n_context, X):
    Xs = scaler.transform(X)
    if model_type == 'cnn':
        morph = Xs[:, :-n_context][..., np.newaxis]
        ctx = Xs[:, -n_context:]
        probs = model.predict({'morphology': morph, 'context': ctx}, verbose=0)
        idx = np.argmax(probs, axis=1)
    else:
        idx = model.predict(Xs)
        probs = model.predict_proba(Xs) if hasattr(model, 'predict_proba') else None
    labels = le.inverse_transform(idx)
    return labels, probs


def group_importance(model_type, model, scaler, le, n_context, X, y_idx, groups):
    """
    Permutation importance by feature GROUP: we shuffle each group and
    measure how much the accuracy drops. Works for any model (RF, MLP, CNN).
    Returns a dict {group_name: accuracy_drop}.
    """
    Xs = scaler.transform(X)

    def acc(Xin):
        if model_type == 'cnn':
            morph = Xin[:, :-n_context][..., np.newaxis]
            ctx = Xin[:, -n_context:]
            p = model.predict({'morphology': morph, 'context': ctx}, verbose=0)
            pred = np.argmax(p, axis=1)
        else:
            pred = model.predict(Xin)
        return np.mean(pred == y_idx)

    base = acc(Xs)
    rng = np.random.default_rng(0)
    importance = {}
    for name, cols in groups.items():
        Xperm = Xs.copy()
        perm = rng.permutation(Xperm.shape[0])
        for c in cols:
            Xperm[:, c] = Xperm[perm, c]
        importance[name] = float(base - acc(Xperm))
    return importance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--record', default='200')
    ap.add_argument('--artifacts', default='artifacts_cascade')
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--seconds', type=float, default=10)
    ap.add_argument('--lead', type=int, default=0)
    args = ap.parse_args()

    rec_path = os.path.join(args.data_dir, args.record)
    record = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, 'atr')
    signal = record.p_signal[:, args.lead]
    fs = record.fs

    beats = [(s, sym) for s, sym in zip(annotation.sample, annotation.symbol)
             if sym in AAMI_GROUPS]
    # Extract the features for all beats (same logic as in training)
    X, kept = record_features(signal, [b[0] for b in beats], fs,
                              rhythm=uses_rhythm_features(args.artifacts))
    X = np.nan_to_num(X)
    y_true = [AAMI_GROUPS[beats[i][1]] for i in kept]
    kept_beats = [(i, beats[i][0], beats[i][1]) for i in kept]

    print(f"Record {args.record}: {len(X)} beats, {fs} Hz")

    # Model + predictions
    model_type, model, scaler, le, n_context = load_model(args.artifacts)
    y_pred, probs = predict_all(model_type, model, scaler, le, n_context, X)
    y_true_idx = le.transform(y_true)

    acc = np.mean(y_pred == np.array(y_true))
    print(f"Model: {model_type} ({args.artifacts}) | record accuracy: {acc:.1%}")

    # --- Explainability: importance of the feature groups ---
    groups = feature_groups(X.shape[1])
    print("\nWhat the model bases its decision on (accuracy drop when shuffled):")
    imp = group_importance(model_type, model, scaler, le, n_context,
                           X, y_true_idx, groups)
    imp_sorted = sorted(imp.items(), key=lambda kv: -kv[1])
    max_imp = max(abs(v) for v in imp.values()) or 1.0
    for name, val in imp_sorted:
        stars = '*' * int(round(10 * abs(val) / max_imp))
        print(f"  {name:32s} {val:+.3f}  {stars}")

    # --- Class distribution ---
    print("\nDistribution of the predicted beats:")
    from collections import Counter
    for cls, cnt in Counter(y_pred).most_common():
        print(f"  {cls} ({CLASS_NAMES.get(cls,'?')}): {cnt}")

    # --- Plot ---
    start_sample = int(args.start * fs)
    end_sample = int((args.start + args.seconds) * fs)
    end_sample = min(end_sample, len(signal))
    t = np.arange(start_sample, end_sample) / fs

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(t, signal[start_sample:end_sample], color='#333333',
            linewidth=0.8, zorder=1)

    shown_classes = set()
    for k, (i, r_peak, sym) in enumerate(kept_beats):
        if not (start_sample <= r_peak < end_sample):
            continue
        pred = y_pred[k]
        true = y_true[k]
        color = CLASS_COLORS.get(pred, '#000000')
        tsec = r_peak / fs

        # R-peak marker, colored by the predicted class
        ax.scatter([tsec], [signal[r_peak]], color=color, s=80, zorder=3,
                   edgecolors='white', linewidths=0.8,
                   label=f"{pred} ({CLASS_NAMES.get(pred)})" if pred not in shown_classes else None)
        shown_classes.add(pred)

        # Text with the predicted class above the beat
        ax.annotate(pred, (tsec, signal[r_peak]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=9, fontweight='bold', color=color)

        # Misclassified -> outline with a red dashed box
        if pred != true:
            ax.annotate(f"!={true}", (tsec, signal[r_peak]),
                        textcoords="offset points", xytext=(0, -18),
                        ha='center', fontsize=7, color='red')
            rect = Rectangle((tsec - 0.12, signal[r_peak] - 0.4), 0.24, 0.8,
                             fill=False, edgecolor='red', linestyle='--',
                             linewidth=1.2, zorder=2)
            ax.add_patch(rect)

        # Star for non-normal beats (anomalies) - marks the deciding region
        if pred != 'N':
            ax.annotate('*', (tsec, signal[r_peak]),
                        textcoords="offset points", xytext=(8, 4),
                        ha='center', fontsize=16, color=color)

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude (mV)', fontsize=11)
    top_group = imp_sorted[0][0]
    ax.set_title(
        f"ECG record {args.record} — beat classification ({model_type})\n"
        f"Accuracy: {acc:.1%} | Most decisive feature group: {top_group} | "
        f"* = anomaly, --- = misclassified",
        fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    out_png = f"ecg_{args.record}_classified.png"
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"\nPlot saved to: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()