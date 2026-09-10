"""
Предобработка за deployment.
ВАЖНО: използва ЗАПАЗЕНИЯ scaler (не прави fit наново - това беше бъг).
"""
import numpy as np
from scipy.signal import resample


def _p_wave_features(segment, fs, r_index):
    """Признаци за P-вълната - трябва да съвпада ТОЧНО с трениращия скрипт."""
    p_start = max(0, r_index - int(0.22 * fs))
    p_end = max(0, r_index - int(0.05 * fs))
    if p_end - p_start < 3:
        return [0.0, 0.0, 0.0, 0.0]
    p = segment[p_start:p_end]
    energy = float(np.sum(p ** 2))
    p_ampl = float(np.max(p) - np.min(p))
    p_max = float(np.max(p))
    p_peak_pos = float(np.argmax(np.abs(p))) / len(p)
    return [energy, p_ampl, p_max, p_peak_pos]


def extract_beat_features_raw(signal, r_peak, prev_r, next_r, fs,
                              local_mean_rr=None, win_before=120, win_after=110):
    """Същата логика като при трениране - трябва да съвпада ТОЧНО."""
    start = r_peak - win_before
    end = r_peak + win_after
    if start < 0 or end > len(signal):
        return None
    morphology = signal[start:end]
    prev_rr = (r_peak - prev_r) / fs if prev_r is not None else 0.0
    next_rr = (next_r - r_peak) / fs if next_r is not None else 0.0
    rr_ratio = (prev_rr / next_rr) if next_rr > 0 else 0.0

    if local_mean_rr and local_mean_rr > 0:
        prev_rr_norm = prev_rr / local_mean_rr
        next_rr_norm = next_rr / local_mean_rr
    else:
        prev_rr_norm = 1.0
        next_rr_norm = 1.0

    p_feats = _p_wave_features(morphology, fs, win_before)

    return np.concatenate([
        morphology,
        [prev_rr, next_rr, rr_ratio, prev_rr_norm, next_rr_norm],
        p_feats
    ])


def preprocess_beat(beat_features, scaler):
    """
    Скалира един вектор характеристики със ЗАПАЗЕНИЯ scaler.
    beat_features: 1D масив (морфология + RR), както при трениране
    scaler: зареден StandardScaler (само .transform, без fit!)
    """
    data = np.nan_to_num(np.asarray(beat_features, dtype=float))
    data = data.reshape(1, -1)
    return scaler.transform(data)   # само transform