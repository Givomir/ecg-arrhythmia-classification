"""
Ablation study for S (supraventricular ectopic) beats
======================================================
Goal: find out WHAT limits S recall - the features, class balancing, the
decision threshold, or the model structure.

The core difficulty with S is S <-> N confusion (narrow QRS, nearly the same
shape), not S <-> V. The discriminators are rhythm (the beat comes early) and
the P wave. We test:

  * Feature groups:  morphology / RR / normalized RR / P wave /
                     + RR relative to the record mean (offline, de Chazal 2004)
                     + RR relative to a long PRECEDING window (works real-time)
                     + rhythm irregularity (so that an "early" beat inside an
                       already irregular rhythm, e.g. atrial fibrillation, is
                       not mistaken for S)
  * Balancing:       none / class_weight / undersampling of N
  * S threshold:     argmax vs. a threshold chosen by cross-validation
  * Cascade:         stage 1 = multi-class RF, stage 2 = dedicated S-vs-N
                     detector applied to the beats stage 1 calls N or S

Protocol (DS2 takes part in NO selection):
  1. GroupKFold by RECORD over DS1 (22 records) -> out-of-fold probabilities
     for every DS1 beat from a model that has never seen that patient.
  2. The S threshold is chosen on all OOF probabilities.
  3. Final model on all of DS1 -> evaluation on DS2 with argmax and threshold.
  4. Everything is repeated for several seeds -> mean ± standard deviation.
  5. DS2 metrics are also reported WITHOUT record 232, which alone holds ~75%
     of the DS2 S beats and would otherwise dominate the result.

Usage:
    python ablation_s.py --data_dir /path/to/mit-bih
    python ablation_s.py --data_dir ... --only full_cw,cascade_rrlong_irr --seeds 0,1,2
"""

import os
import json
import time
import argparse
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, f1_score

from train_arrhythmia import get_split, load_beats
from utility import record_features, rhythm_features

CLASSES = ['F', 'N', 'Q', 'S', 'V']
S = CLASSES.index('S')
N = CLASSES.index('N')
V = CLASSES.index('V')
F = CLASSES.index('F')

# Record that dominates the S class in DS2 (1382 of 1837 S beats)
DOMINANT_RECORD = 232

# Columns of the vector from utility.extract_beat_features (230 + 5 + 4 = 239)
# plus the extra rhythm features from _extra_rr (239..250)
COLS = {
    'morph':  np.arange(0, 230),
    'rr':     np.arange(230, 233),   # prev_rr, next_rr, rr_ratio
    'rrn':    np.arange(233, 235),   # prev/next relative to the local MEAN (10 beats)
    'p':      np.arange(235, 239),   # P wave
    'rrmed':  np.arange(239, 241),   # prev/next relative to the local MEDIAN (10 beats)
    'rrrec':  np.arange(241, 243),   # prev/next relative to the mean RR of the WHOLE record
    'rrlong': np.arange(243, 245),   # prev/next relative to the mean of the preceding 300
    'irr':    np.arange(245, 251),   # rhythm irregularity (see utility.rhythm_features)
}


# ---------------------------------------------------------------------------
# Data (cached - feature extraction takes minutes)
# ---------------------------------------------------------------------------
def _extra_rr(X):
    """
    Extra rhythm features for ONE record (beats are in record order).
    rrmed:  the median is robust to the ectopic beats themselves, which
            otherwise distort the local mean.
    rrrec:  RR relative to the mean of the whole record (de Chazal 2004).
            Needs the whole record -> offline analysis only.
    rrlong: like rrrec, but only from the PRECEDING 300 beats -> usable real-time.
    irr:    rhythm irregularity over the preceding 10 beats (cv, rmssd,
            pnn50, prev_z, next_z) plus cv over the preceding 40 beats.
    rrlong and irr come from utility.rhythm_features.
    """
    prev_rr, next_rr = X[:, 230], X[:, 231]
    valid = prev_rr[prev_rr > 0]
    rec_mean = float(np.mean(valid)) if len(valid) else 1.0
    out = np.zeros((len(X), 4))
    for i in range(len(X)):
        win = prev_rr[max(0, i - 10):i]
        win = win[win > 0]
        med = float(np.median(win)) if len(win) else rec_mean
        out[i] = [prev_rr[i] / med, next_rr[i] / med,
                  prev_rr[i] / rec_mean, next_rr[i] / rec_mean]
    # rrlong + irr: the same code the deployed cascade model uses
    return np.hstack([out, rhythm_features(X)])


def load_records(records, data_dir, lead):
    X, y, g = [], [], []
    for rec in records:
        signals, fs, beats = load_beats(os.path.join(data_dir, rec))
        feats, kept = record_features(signals[:, lead], [b[0] for b in beats], fs)
        if not len(kept):
            continue
        X.append(np.hstack([feats, _extra_rr(feats)]))
        y.extend(CLASSES.index(beats[i][1]) for i in kept)
        g.extend([int(rec)] * len(kept))
    return np.nan_to_num(np.vstack(X)), np.array(y), np.array(g)


def load_data(data_dir, lead, cache):
    if os.path.exists(cache):
        print(f"Loading from cache: {cache}")
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    ds1, ds2 = get_split(data_dir)
    data = {}
    for name, recs in [('ds1', ds1), ('ds2', ds2)]:
        print(f"Extracting features ({name}, {len(recs)} records)...")
        X, y, g = load_records(recs, data_dir, lead)
        print(f"  {name}: {X.shape} | {dict(Counter(CLASSES[i] for i in y))}")
        data.update({f'X_{name}': X, f'y_{name}': y, f'g_{name}': g})
    np.savez_compressed(cache, **data)
    return data


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
# name    - experiment name
# groups  - feature groups next to the morphology
# morph   - 'raw' | 'pca' | None (no morphology)
# balance - 'none' | 'cw' (class_weight) | 'under' (undersampling of N)
# stage2  - None, or a dedicated S-vs-N detector:
#           {'groups': [...], 'morph': 'pca' | None, 'model': 'rf' | 'hgb'}
# "base" = morphology + rr + rrn (no P wave)
RHYTHM = ['rr', 'rrn', 'rrlong', 'irr']
EXPERIMENTS = [
    # --- references from the previous round ---
    dict(name='full_cw',            groups=['rr', 'rrn', 'p'],            morph='raw', balance='cw'),
    dict(name='base_rrlong_cw',     groups=['rr', 'rrn', 'rrlong'],       morph='raw', balance='cw'),
    dict(name='base_rrrec_cw',      groups=['rr', 'rrn', 'rrrec'],        morph='raw', balance='cw'),
    # --- + rhythm irregularity ---
    dict(name='base_rrlong_irr_cw', groups=RHYTHM,                        morph='raw', balance='cw'),
    dict(name='base_rrrec_irr_cw',  groups=['rr', 'rrn', 'rrrec', 'irr'], morph='raw', balance='cw'),
    dict(name='base_rrlong_irr_none', groups=RHYTHM,                      morph='raw', balance='none'),
    # --- cascade: stage 1 multi-class + stage 2 S-vs-N detector ---
    dict(name='cascade_rf',         groups=RHYTHM, morph='raw', balance='cw',
         stage2={'groups': RHYTHM, 'morph': 'pca', 'model': 'rf'}),
    dict(name='cascade_hgb',        groups=RHYTHM, morph='raw', balance='cw',
         stage2={'groups': RHYTHM, 'morph': 'pca', 'model': 'hgb'}),
    dict(name='cascade_rf_rhythm_only', groups=RHYTHM, morph='raw', balance='cw',
         stage2={'groups': RHYTHM, 'morph': None, 'model': 'rf'}),
]


def build_features(X_fit, X_list, groups, morph, n_pca, seed):
    """The scaler/PCA are fitted ONLY on X_fit (the training part)."""
    def fit_block(idx, use_pca):
        sc = StandardScaler().fit(X_fit[:, idx])
        pca = (PCA(n_components=n_pca, random_state=seed).fit(sc.transform(X_fit[:, idx]))
               if use_pca else None)

        def tr(X):
            Z = sc.transform(X[:, idx])
            return pca.transform(Z) if pca is not None else Z
        return tr

    blocks = []
    if morph is not None:
        blocks.append(fit_block(COLS['morph'], morph == 'pca'))
    if groups:
        blocks.append(fit_block(np.concatenate([COLS[g] for g in groups]), False))
    return [np.hstack([b(X) for b in blocks]) for X in X_list]


def rebalance(X, y, how, seed):
    if how == 'under':
        # N is reduced to 5x the number of S; the rare classes stay untouched
        from imblearn.under_sampling import RandomUnderSampler
        c = Counter(y)
        target = {N: min(c[N], 5 * c[S])}
        return RandomUnderSampler(sampling_strategy=target,
                                  random_state=seed).fit_resample(X, y)
    return X, y


def fit_stage1(Xtr, ytr, X_eval, exp, args, seed):
    """Trains the multi-class RF and returns probabilities (ordered by CLASSES)."""
    feats = build_features(Xtr, [Xtr] + X_eval, exp['groups'], exp['morph'],
                           args.n_pca, seed)
    Ftr, ytr_b = rebalance(feats[0], ytr, exp['balance'], seed)
    clf = RandomForestClassifier(
        n_estimators=args.trees, n_jobs=-1, random_state=seed,
        class_weight='balanced' if exp['balance'] == 'cw' else None,
        min_samples_leaf=2,
    )
    clf.fit(Ftr, ytr_b)
    out = []
    for Fe in feats[1:]:
        P = np.zeros((len(Fe), len(CLASSES)))
        P[:, clf.classes_] = clf.predict_proba(Fe)   # Q may be missing from train
        out.append(P)
    return out


def fit_stage2(Xtr, ytr, X_eval, cfg, args, seed):
    """
    Dedicated S-vs-N detector trained ONLY on N and S beats, mostly on
    rhythm features plus (optionally) compressed morphology.
    model='rf':  Random Forest with larger leaves (more robust across patients)
    model='hgb': gradient boosting (tends to overfit the few S patients)
    Returns P(S) for every beat in X_eval.
    """
    m = (ytr == N) | (ytr == S)
    feats = build_features(Xtr[m], [Xtr[m]] + X_eval, cfg['groups'], cfg['morph'],
                           args.n_pca_stage2, seed)
    if cfg.get('model', 'rf') == 'hgb':
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            class_weight='balanced', random_state=seed,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample',
            n_jobs=-1, random_state=seed,
        )
    clf.fit(feats[0], (ytr[m] == S).astype(int))
    return [clf.predict_proba(Fe)[:, 1] for Fe in feats[1:]]


def fit_scores(Xtr, ytr, X_eval, exp, args, seed):
    """
    Returns, for every X in X_eval, a triple (rest_pred, s_score, default_pred):
      rest_pred    - the best non-S class (used when the beat is not called S)
      s_score      - the score compared against the S threshold
      default_pred - prediction without threshold tuning (argmax for a single
                     stage, stage-2 threshold 0.5 for a cascade)
    Single stage: s_score = P1(S).
    Cascade:      s_score = P2(S) for beats that stage 1 calls N or S, else 0.
    """
    P1s = fit_stage1(Xtr, ytr, X_eval, exp, args, seed)
    P2s = (fit_stage2(Xtr, ytr, X_eval, exp['stage2'], args, seed)
           if exp.get('stage2') else [None] * len(X_eval))
    out = []
    for P1, P2 in zip(P1s, P2s):
        rest = P1.copy()
        rest[:, S] = -1
        rest_pred = rest.argmax(1)
        if P2 is None:
            s_score = P1[:, S]
            default_pred = P1.argmax(1)
        else:
            gate = np.isin(P1.argmax(1), [N, S])
            s_score = np.where(gate, P2, 0.0)
            # a non-S beat that stage 1 called S falls back to N
            rest_pred = np.where(gate, N, rest_pred)
            default_pred = predict_with_threshold(rest_pred, s_score, 0.5)
        out.append((rest_pred, s_score, default_pred))
    return out


def predict_with_threshold(rest_pred, s_score, t):
    """S if s_score >= t, otherwise the best non-S class."""
    pred = rest_pred.copy()
    pred[s_score >= t] = S
    return pred


def tune_threshold(rest_pred, s_score, y, criterion, min_precision):
    """
    criterion='f1':     the threshold with the highest S F1.
    criterion='recall': highest S recall with S precision >= min_precision
                        (if no such threshold -> highest F1).
    """
    best_f1, best_rec = (-1, 0.5), None
    for t in np.linspace(0.02, 0.98, 97):
        pred = predict_with_threshold(rest_pred, s_score, t)
        tp = np.sum((pred == S) & (y == S))
        prec = tp / max(1, np.sum(pred == S))
        rec = tp / max(1, np.sum(y == S))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best_f1[0]:
            best_f1 = (f1, t)
        if prec >= min_precision and (best_rec is None or rec > best_rec[0]):
            best_rec = (rec, t)
    if criterion == 'recall' and best_rec is not None:
        return float(best_rec[1])
    return float(best_f1[1])


def metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=np.arange(len(CLASSES)))
    tp = cm[S, S]
    s_rec = tp / max(1, cm[S].sum())
    s_prec = tp / max(1, cm[:, S].sum())
    return {
        'acc': float(np.trace(cm) / cm.sum()),
        'macro_f1': float(f1_score(y, pred, labels=[N, S, V, F],
                                   average='macro', zero_division=0)),
        'S_rec': float(s_rec),
        'S_prec': float(s_prec),
        'S_f1': float(2 * s_prec * s_rec / max(1e-9, s_prec + s_rec)),
        'V_rec': float(cm[V, V] / max(1, cm[V].sum())),
        'S_as_N': int(cm[S, N]),
        'S_as_V': int(cm[S, V]),
        'N_as_S': int(cm[N, S]),
        'cm': cm.tolist(),
    }


def run_seed(exp, data, args, seed):
    X1, y1, g1 = data['X_ds1'], data['y_ds1'], data['g_ds1']
    X2, y2, g2 = data['X_ds2'], data['y_ds2'], data['g_ds2']

    # 1. Out-of-fold scores by patient over DS1
    oof_rest = np.zeros(len(y1), dtype=int)
    oof_score = np.zeros(len(y1))
    for tr, va in GroupKFold(n_splits=args.folds).split(X1, y1, g1):
        [(rest, score, _)] = fit_scores(X1[tr], y1[tr], [X1[va]], exp, args, seed)
        oof_rest[va], oof_score[va] = rest, score

    # 2. Threshold from the OOF scores
    t = tune_threshold(oof_rest, oof_score, y1, args.criterion, args.min_precision)

    # 3. Final model on all of DS1 -> DS2
    [(rest2, score2, pred_argmax)] = fit_scores(X1, y1, [X2], exp, args, seed)
    pred_tuned = predict_with_threshold(rest2, score2, t)
    m = g2 != DOMINANT_RECORD
    return {
        'seed': seed, 'threshold': t,
        'oof_tuned': metrics(y1, predict_with_threshold(oof_rest, oof_score, t)),
        'argmax': metrics(y2, pred_argmax),
        'tuned': metrics(y2, pred_tuned),
        'tuned_no232': metrics(y2[m], pred_tuned[m]),
    }


SUMMARY_KEYS = ['S_rec', 'S_prec', 'S_f1', 'S_as_N', 'S_as_V', 'N_as_S',
                'V_rec', 'macro_f1', 'acc']
PARTS = ['oof_tuned', 'argmax', 'tuned', 'tuned_no232']


def summarize(runs):
    """Mean and standard deviation across seeds."""
    out = {'threshold': [float(np.mean([r['threshold'] for r in runs])),
                         float(np.std([r['threshold'] for r in runs]))]}
    for part in PARTS:
        out[part] = {k: [float(np.mean([r[part][k] for r in runs])),
                         float(np.std([r[part][k] for r in runs]))]
                     for k in SUMMARY_KEYS}
    return out


def run(exp, data, args):
    t0 = time.time()
    runs = [run_seed(exp, data, args, s) for s in args.seed_list]
    summ = summarize(runs)
    u, x = summ['tuned'], summ['tuned_no232']
    print(f"  {exp['name']:22s} | thr {summ['threshold'][0]:.2f}±{summ['threshold'][1]:.2f} "
          f"| S rec {u['S_rec'][0]:.2f} prec {u['S_prec'][0]:.2f} F1 {u['S_f1'][0]:.2f} "
          f"| w/o 232: rec {x['S_rec'][0]:.2f} prec {x['S_prec'][0]:.2f} F1 {x['S_f1'][0]:.2f} "
          f"| {time.time() - t0:.0f}s", flush=True)
    return {**exp, 'summary': summ, 'runs': runs}


def _describe(exp):
    feats = {'raw': 'morph+', 'pca': 'pca+', None: ''}[exp['morph']] + '+'.join(exp['groups'])
    if exp.get('stage2'):
        s2 = exp['stage2']
        feats += (f" → S/N {s2.get('model', 'rf')}: "
                  + {'pca': 'pca+', None: ''}[s2['morph']] + '+'.join(s2['groups']))
    return feats


def write_markdown(results, path, args):
    def ms(v, fmt='.3f'):
        return f"{v[0]:{fmt}} ± {v[1]:{fmt}}"

    def mi(v):
        return f"{v[0]:.0f} ± {v[1]:.0f}"

    lines = [
        "# S ablation (DS2 test set)",
        "",
        f"Protocol: S threshold chosen on out-of-fold scores ({args.folds}-fold GroupKFold "
        f"by record over DS1, criterion `{args.criterion}`), final model on all of DS1, "
        f"tested on DS2. Seeds: {args.seeds}. Values are mean ± std. "
        "Macro F1 is over N, S, V and F (Q excluded). "
        f"\"w/o {DOMINANT_RECORD}\" excludes record {DOMINANT_RECORD}, which holds ~75% "
        "of the DS2 S beats. Stage 1 is a Random Forest; the cascade's S/N stage "
        "model is named in the Features column. For cascades, \"argmax\" means stage-2 threshold 0.5.",
        "",
        "## DS2 with the tuned threshold",
        "",
        "| Experiment | Features | Balance | threshold | S rec | S prec | S F1 "
        f"| N→S | S→N | V rec | macro F1 | S rec w/o {DOMINANT_RECORD} "
        f"| S prec w/o {DOMINANT_RECORD} | S F1 w/o {DOMINANT_RECORD} |",
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        s = r['summary']
        u, x = s['tuned'], s['tuned_no232']
        lines.append(
            f"| {r['name']} | {_describe(r)} | {r['balance']} | {ms(s['threshold'], '.2f')} "
            f"| {ms(u['S_rec'])} | {ms(u['S_prec'])} | {ms(u['S_f1'])} "
            f"| {mi(u['N_as_S'])} | {mi(u['S_as_N'])} | {ms(u['V_rec'])} | {ms(u['macro_f1'])} "
            f"| {ms(x['S_rec'])} | {ms(x['S_prec'])} | {ms(x['S_f1'])} |")
    lines += [
        "",
        "## DS2 with argmax (no threshold tuning)",
        "",
        "| Experiment | S rec | S prec | S F1 | V rec | macro F1 |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        a = r['summary']['argmax']
        lines.append(f"| {r['name']} | {ms(a['S_rec'])} | {ms(a['S_prec'])} "
                     f"| {ms(a['S_f1'])} | {ms(a['V_rec'])} | {ms(a['macro_f1'])} |")
    lines += [
        "",
        "## DS1 out-of-fold (with threshold) - for comparison with DS2",
        "",
        "| Experiment | S rec | S prec | S F1 |",
        "|---|--:|--:|--:|",
    ]
    for r in results:
        o = r['summary']['oof_tuned']
        lines.append(f"| {r['name']} | {ms(o['S_rec'])} | {ms(o['S_prec'])} | {ms(o['S_f1'])} |")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--lead', type=int, default=0)
    ap.add_argument('--out_dir', default='ablation_results')
    ap.add_argument('--trees', type=int, default=150)
    ap.add_argument('--n_pca', type=int, default=20,
                    help='PCA components of the morphology in stage 1 (morph="pca")')
    ap.add_argument('--n_pca_stage2', type=int, default=10,
                    help='PCA components of the morphology in the S/N stage')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--criterion', choices=['f1', 'recall'], default='f1',
                    help='How the S threshold is chosen on the OOF scores')
    ap.add_argument('--min_precision', type=float, default=0.4,
                    help='Minimum S precision with --criterion recall')
    ap.add_argument('--only', default='', help='Comma-separated experiment names')
    args = ap.parse_args()
    args.seed_list = [int(s) for s in args.seeds.split(',')]

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_data(args.data_dir, args.lead,
                     os.path.join(args.out_dir, f'features_v3_lead{args.lead}.npz'))

    exps = EXPERIMENTS
    if args.only:
        wanted = set(args.only.split(','))
        exps = [e for e in EXPERIMENTS if e['name'] in wanted]

    print(f"\n{len(exps)} experiments (RF {args.trees} trees, {args.folds} folds, "
          f"seeds {args.seeds}, criterion {args.criterion})")
    results = [run(e, data, args) for e in exps]

    with open(os.path.join(args.out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    write_markdown(results, os.path.join(args.out_dir, 'results.md'), args)
    print(f"\nSaved: {args.out_dir}/results.json and results.md")


if __name__ == '__main__':
    main()
