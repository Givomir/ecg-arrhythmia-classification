"""
Тестов клиент за ЕКГ API-то.
Взима реален удар от избран запис, извлича същите 239 признака като при
трениране, и го праща на живия FastAPI сървър.

Използване (сървърът трябва да работи в друг терминал):
    python test_api.py --data_dir "C:\\...\\mit-bih..." --record 200 --beat 10

Аргументи:
    --data_dir  папка с MIT-BIH записите (.dat/.hea/.atr)
    --record    номер на записа (по подразбиране 200)
    --beat      кой пореден удар да се тества (по подразбиране 10)
    --url       адрес на сървъра (по подразбиране http://127.0.0.1:8000)
"""
import os
import argparse
import numpy as np
import wfdb
import requests

from utility import extract_beat_features_raw
from train_arrhythmia import AAMI_GROUPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--record', default='200')
    ap.add_argument('--beat', type=int, default=10)
    ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--lead', type=int, default=0)
    args = ap.parse_args()

    rec_path = os.path.join(args.data_dir, args.record)
    record = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, 'atr')
    signal = record.p_signal[:, args.lead]
    fs = record.fs

    # Само истински удари с валиден AAMI клас (както при трениране)
    beats = [(s, sym) for s, sym in zip(annotation.sample, annotation.symbol)
             if sym in AAMI_GROUPS]

    i = args.beat
    if i < 1 or i >= len(beats) - 1:
        raise SystemExit(f"Изберете --beat между 1 и {len(beats) - 2}")

    r_peak = beats[i][0]
    true_symbol = beats[i][1]
    true_class = AAMI_GROUPS[true_symbol]

    # Локална средна на RR (както при трениране)
    peaks = [b[0] for b in beats]
    rr_all = [(peaks[k] - peaks[k - 1]) / fs for k in range(1, len(peaks))]
    lo = max(0, i - 10)
    window_rr = rr_all[lo:i]
    local_mean_rr = float(np.mean(window_rr)) if window_rr else None

    feats = extract_beat_features_raw(
        signal, r_peak, beats[i - 1][0], beats[i + 1][0], fs,
        local_mean_rr=local_mean_rr)
    if feats is None:
        raise SystemExit("Ударът е твърде близо до края на записа, изберете друг --beat")

    print(f"Запис {args.record}, удар #{i}")
    print(f"Истински символ: '{true_symbol}' -> AAMI клас: {true_class}")
    print(f"Дължина на вектора: {len(feats)} признака")
    print(f"Изпращам към {args.url}/predict ...\n")

    resp = requests.post(f"{args.url}/predict",
                         json={"features": feats.tolist()}, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    print("Отговор от сървъра:")
    print(f"  Модел:      {result.get('model_type')}")
    print(f"  Предсказан: {result.get('prediction')}  ({result.get('diagnosis')})")
    print(f"  Истински:   {true_class}")
    match = "верно" if result.get('prediction') == true_class else "грешно"
    print(f"  Резултат:   {match}")
    if result.get('probabilities'):
        print("  Вероятности:")
        for cls, p in sorted(result['probabilities'].items(),
                             key=lambda kv: -kv[1]):
            print(f"      {cls}: {p:.3f}")


if __name__ == '__main__':
    main()