"""
Feature extraction - the SINGLE source of truth.
=====================================================
All scripts (training, API, visualization, RAG, web app) use the
functions from here, so training and deployment never diverge.
The module has no heavy dependencies (numpy only) - handy for the API server.

IMPORTANT: scaling uses the SAVED scaler (it does not fit again).
"""
import numpy as np

# Number of engineered features at the end of every vector:
# prev_rr, next_rr, rr_ratio, prev_rr_norm, next_rr_norm + 4 P wave = 9
N_CONTEXT = 9

# Number of preceding beats for the local mean RR
RR_WINDOW = 10

# Length of the beat morphology segment (win_before + win_after)
N_MORPH = 230

# Optional rhythm features, appended per record by record_features(rhythm=True).
# Used by the cascade model (train_cascade.py); the RF/MLP/CNN models use the
# 239 base features only. All of them look only at PRECEDING beats (plus the
# next RR, as the base features do), so they also work on a live signal:
#   prev/next RR relative to the mean of the preceding LONG_WINDOW beats (2)
#   irregularity of the preceding IRR_SHORT RRs: cv, rmssd, pnn50,
#   prev_z, next_z (5)
#   cv of the preceding IRR_LONG RRs (1)
N_RHYTHM = 8
LONG_WINDOW = 300   # ~4-5 minutes
IRR_SHORT = 10
IRR_LONG = 40


def _p_wave_features(segment, fs, r_index):
    """
    P-wave features - the segment BEFORE the QRS complex.
    The P wave reflects atrial depolarization. In supraventricular
    ectopic beats (S) it is often missing, abnormally shaped or of
    reversed polarity - which makes it a strong S-vs-N discriminator.
    The P region is roughly 50-220 ms before the R peak.
    """
    p_start = max(0, r_index - int(0.22 * fs))
    p_end = max(0, r_index - int(0.05 * fs))
    if p_end - p_start < 3:
        return [0.0, 0.0, 0.0, 0.0]
    p = segment[p_start:p_end]
    energy = float(np.sum(p ** 2))                       # is there a P wave at all
    p_ampl = float(np.max(p) - np.min(p))                # amplitude
    p_max = float(np.max(p))                             # polarity/height
    p_peak_pos = float(np.argmax(np.abs(p))) / len(p)    # position relative to R
    return [energy, p_ampl, p_max, p_peak_pos]


def extract_beat_features(signal, r_peak, prev_r, next_r, fs,
                          local_mean_rr=None, win_before=120, win_after=110):
    """
    Returns the feature vector for one beat:
      * morphology: signal segment around the R peak (win_before+win_after samples)
      * RR context: previous RR, next RR, their ratio
      * NORMALIZED RR: deviation of the beat from the local rhythm.
        A key signal for supraventricular ectopic beats (S) - they come "early".
      * P-WAVE features: energy/amplitude/polarity of the region before the QRS.
    local_mean_rr: the mean RR of the last ~10 beats (in seconds)
    Returns None if the beat is too close to the edges of the record.
    """
    start = r_peak - win_before
    end = r_peak + win_after
    if start < 0 or end > len(signal):
        return None

    morphology = signal[start:end]

    # RR features (in seconds)
    prev_rr = (r_peak - prev_r) / fs if prev_r is not None else 0.0
    next_rr = (next_r - r_peak) / fs if next_r is not None else 0.0
    rr_ratio = (prev_rr / next_rr) if next_rr > 0 else 0.0

    # RR features normalized to the local rhythm
    if local_mean_rr and local_mean_rr > 0:
        prev_rr_norm = prev_rr / local_mean_rr   # <1 => came EARLY (typical for S)
        next_rr_norm = next_rr / local_mean_rr
    else:
        prev_rr_norm = 1.0
        next_rr_norm = 1.0

    # P-wave features (the R peak is at position win_before in the segment)
    p_feats = _p_wave_features(morphology, fs, win_before)

    return np.concatenate([
        morphology,
        [prev_rr, next_rr, rr_ratio, prev_rr_norm, next_rr_norm],
        p_feats
    ])


# Old name - kept for compatibility (test_api.py etc.)
extract_beat_features_raw = extract_beat_features


def _irregularity(win, cur_prev, cur_next):
    """
    Irregularity of the rhythm in the PRECEDING RR window `win`:
      cv       - coefficient of variation (std / mean)
      rmssd    - RMSSD of successive differences, normalized by the mean
      pnn50    - fraction of successive differences > 50 ms
      prev_z / next_z - how unusual the current RRs are relative to the local
                 variability. In a regular rhythm a premature S beat gives a
                 large |z|; in atrial fibrillation std is large, so |z| stays
                 small even for short RRs.
    """
    if len(win) < 3:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    mean, std = float(np.mean(win)), float(np.std(win))
    diffs = np.diff(win)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    pnn50 = float(np.mean(np.abs(diffs) > 0.05))
    denom = std + 0.02            # 20 ms floor - avoids huge z in perfectly regular rhythm
    prev_z = np.clip((cur_prev - mean) / denom, -10, 10)
    next_z = np.clip((cur_next - mean) / denom, -10, 10)
    return [std / mean, rmssd / mean, pnn50, float(prev_z), float(next_z)]


def rhythm_features(X):
    """
    The N_RHYTHM extra rhythm features for ONE record.
    X: the base feature matrix of the record (beats in record order), as
    returned by record_features(rhythm=False).
    """
    prev_rr, next_rr = X[:, N_MORPH], X[:, N_MORPH + 1]
    out = np.zeros((len(X), N_RHYTHM))
    for i in range(len(X)):
        long_win = prev_rr[max(0, i - LONG_WINDOW):i]
        long_win = long_win[long_win > 0]
        # the first beats have no history -> use the current RR (ratio 1)
        long_mean = float(np.mean(long_win)) if len(long_win) else (prev_rr[i] or 1.0)
        win = prev_rr[max(0, i - IRR_SHORT):i]
        win = win[win > 0]
        win40 = prev_rr[max(0, i - IRR_LONG):i]
        win40 = win40[win40 > 0]
        cv40 = float(np.std(win40) / np.mean(win40)) if len(win40) >= 3 else 0.0
        out[i] = ([prev_rr[i] / long_mean, next_rr[i] / long_mean]
                  + _irregularity(win, prev_rr[i], next_rr[i]) + [cv40])
    return out


def feature_groups(n_features):
    """Column groups of the feature vector (for explainability)."""
    m = N_MORPH
    groups = {
        'Morphology (QRS shape)': list(range(0, m)),
        'RR intervals (timing)': [m, m + 1, m + 2],
        'Normalized RR (rhythm)': [m + 3, m + 4],
        'P wave': [m + 5, m + 6, m + 7, m + 8],
    }
    if n_features == m + N_CONTEXT + N_RHYTHM:
        b = m + N_CONTEXT
        groups['Long-window RR (rhythm)'] = [b, b + 1]
        groups['Rhythm irregularity'] = list(range(b + 2, b + N_RHYTHM))
    return groups


def uses_rhythm_features(artifacts_dir):
    """True if the model in artifacts_dir was trained with rhythm=True features."""
    import os
    import joblib
    path = os.path.join(artifacts_dir, 'meta.pkl')
    return bool(os.path.exists(path) and joblib.load(path).get('rhythm_features', False))


def record_features(signal, peaks, fs, window=RR_WINDOW, rhythm=False):
    """
    Extracts the features for ALL beats in a record - with the same RR context
    as in training (local mean of the last `window` RR intervals).
    peaks: list of R-peak indices (in record order)
    rhythm: also append the N_RHYTHM rhythm features (for the cascade model)
    Returns (X, kept) - the feature matrix and the indices of the beats that
    were kept (edge beats, too close to the border, are skipped).
    """
    rr_all = [(peaks[k] - peaks[k - 1]) / fs for k in range(1, len(peaks))]
    X, kept = [], []
    for i, r_peak in enumerate(peaks):
        prev_r = peaks[i - 1] if i > 0 else None
        next_r = peaks[i + 1] if i < len(peaks) - 1 else None
        window_rr = rr_all[max(0, i - window):i] if i > 0 else []
        lmr = float(np.mean(window_rr)) if window_rr else None
        feats = extract_beat_features(signal, r_peak, prev_r, next_r, fs,
                                      local_mean_rr=lmr)
        if feats is not None:
            X.append(feats)
            kept.append(i)
    X = np.array(X)
    if rhythm and len(X):
        X = np.hstack([X, rhythm_features(X)])
    return X, kept


def preprocess_beat(beat_features, scaler):
    """
    Scales one feature vector with the SAVED scaler.
    beat_features: 1D array (morphology + RR), as in training
    scaler: a loaded StandardScaler (only .transform, no fit!)
    """
    data = np.nan_to_num(np.asarray(beat_features, dtype=float))
    data = data.reshape(1, -1)
    return scaler.transform(data)   # transform only
