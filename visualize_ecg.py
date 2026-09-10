"""
Визуализация и обяснение на ЕКГ класификация
==============================================
Чете стандартен MIT-BIH запис (.hea/.dat/.atr), класифицира всички удари
с обучения модел, и показва:

  1. ГРАФИКА на целия сигнал с маркирани и оцветени удари по клас
     (зелено=N, оранжево=S, червено=V, лилаво=F, сиво=Q). Грешно
     класифицираните се ограждат, за да се виждат ясно.

  2. ОБЯСНЕНИЕ кои признаци тежат при решението - и на графиката
     (звездички върху решаващата област), и в терминала (текст с
     конкретните RR/P стойности + важност на групите признаци).

Използване:
    python visualize_ecg.py --data_dir "C:\\...\\mit-bih" --record 200
    python visualize_ecg.py --data_dir "..." --record 208 --start 0 --seconds 20

Аргументи:
    --data_dir  папка с MIT-BIH записите
    --record    номер на записа (напр. 200)
    --artifacts папка с модела (artifacts / artifacts_mlp / artifacts_cnn)
    --start     от коя секунда да започне графиката (по подразбиране 0)
    --seconds   колко секунди да покаже (по подразбиране 10)
"""
import os
import argparse
import numpy as np
import wfdb
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from train_arrhythmia import AAMI_GROUPS, extract_beat_features

# Цветове по клас
CLASS_COLORS = {
    'N': '#2ca02c',   # зелено
    'S': '#ff7f0e',   # оранжево
    'V': '#d62728',   # червено
    'F': '#9467bd',   # лилаво
    'Q': '#7f7f7f',   # сиво
}
CLASS_NAMES = {
    'N': 'Нормален', 'S': 'Надкамерен', 'V': 'Камерен',
    'F': 'Сливен', 'Q': 'Неопределим',
}

# Групи признаци (за обяснимост). Векторът е:
# [230 морфология][prev_rr, next_rr, rr_ratio, prev_rr_norm, next_rr_norm][4 P]
def feature_groups(n_total):
    n_morph = n_total - 9
    return {
        'Морфология (форма на QRS)': list(range(0, n_morph)),
        'RR интервали (тайминг)': [n_morph, n_morph + 1, n_morph + 2],
        'Нормализиран RR (ритъм)': [n_morph + 3, n_morph + 4],
        'P-вълна': [n_morph + 5, n_morph + 6, n_morph + 7, n_morph + 8],
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
    Permutation importance по ГРУПИ признаци: разбъркваме всяка група и
    гледаме колко пада точността. Работи за всеки модел (RF, MLP, CNN).
    Връща речник {име_на_група: спад_в_точността}.
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
    ap.add_argument('--artifacts', default='artifacts_mlp')
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
    peaks = [b[0] for b in beats]
    rr_all = [(peaks[k] - peaks[k - 1]) / fs for k in range(1, len(peaks))]

    # Извличаме признаци за всички удари
    X, y_true, kept_beats = [], [], []
    for i, (r_peak, sym) in enumerate(beats):
        prev_r = beats[i - 1][0] if i > 0 else None
        next_r = beats[i + 1][0] if i < len(beats) - 1 else None
        lo = max(0, i - 10)
        window_rr = rr_all[lo:i] if i > 0 else []
        lmr = float(np.mean(window_rr)) if window_rr else None
        feats = extract_beat_features(signal, r_peak, prev_r, next_r, fs,
                                      local_mean_rr=lmr)
        if feats is None:
            continue
        X.append(feats)
        y_true.append(AAMI_GROUPS[sym])
        kept_beats.append((i, r_peak, sym))
    X = np.array(X)

    print(f"Запис {args.record}: {len(X)} удара, {fs} Hz")

    # Модел + предсказания
    model_type, model, scaler, le, n_context = load_model(args.artifacts)
    y_pred, probs = predict_all(model_type, model, scaler, le, n_context, X)
    y_true_idx = le.transform(y_true)

    acc = np.mean(y_pred == np.array(y_true))
    print(f"Модел: {model_type} ({args.artifacts}) | точност на записа: {acc:.1%}")

    # --- Обяснимост: важност на групите признаци ---
    groups = feature_groups(X.shape[1])
    print("\nНа база кои показатели решава моделът (спад в точността при разбъркване):")
    imp = group_importance(model_type, model, scaler, le, n_context,
                           X, y_true_idx, groups)
    imp_sorted = sorted(imp.items(), key=lambda kv: -kv[1])
    max_imp = max(abs(v) for v in imp.values()) or 1.0
    for name, val in imp_sorted:
        stars = '*' * int(round(10 * abs(val) / max_imp))
        print(f"  {name:32s} {val:+.3f}  {stars}")

    # --- Разпределение по класове ---
    print("\nРазпределение на предсказаните удари:")
    from collections import Counter
    for cls, cnt in Counter(y_pred).most_common():
        print(f"  {cls} ({CLASS_NAMES.get(cls,'?')}): {cnt}")

    # --- Графика ---
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

        # Маркер на R-пика, оцветен по предсказан клас
        ax.scatter([tsec], [signal[r_peak]], color=color, s=80, zorder=3,
                   edgecolors='white', linewidths=0.8,
                   label=f"{pred} ({CLASS_NAMES.get(pred)})" if pred not in shown_classes else None)
        shown_classes.add(pred)

        # Текст с предсказания клас над удара
        ax.annotate(pred, (tsec, signal[r_peak]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=9, fontweight='bold', color=color)

        # Грешно класифициран -> ограждаме с червен пунктир
        if pred != true:
            ax.annotate(f"!={true}", (tsec, signal[r_peak]),
                        textcoords="offset points", xytext=(0, -18),
                        ha='center', fontsize=7, color='red')
            rect = Rectangle((tsec - 0.12, signal[r_peak] - 0.4), 0.24, 0.8,
                             fill=False, edgecolor='red', linestyle='--',
                             linewidth=1.2, zorder=2)
            ax.add_patch(rect)

        # Звездичка за не-нормалните удари (аномалии) - маркира решаващата област
        if pred != 'N':
            ax.annotate('*', (tsec, signal[r_peak]),
                        textcoords="offset points", xytext=(8, 4),
                        ha='center', fontsize=16, color=color)

    ax.set_xlabel('Време (секунди)', fontsize=11)
    ax.set_ylabel('Амплитуда (mV)', fontsize=11)
    top_group = imp_sorted[0][0]
    ax.set_title(
        f"ЕКГ запис {args.record} — класификация на удари ({model_type})\n"
        f"Точност: {acc:.1%} | Най-решаващ показател: {top_group} | "
        f"* = аномалия, --- = грешно",
        fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    out_png = f"ecg_{args.record}_classified.png"
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"\nГрафиката е записана: {out_png}")
    plt.show()


if __name__ == '__main__':
    main()