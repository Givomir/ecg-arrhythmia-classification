"""
Test client for the ECG API.
Takes a real beat from a chosen record, extracts the same features as in
training (239, or 247 for a model trained with rhythm features - the server's
/health tells which), and sends it to the running FastAPI server.

Usage (the server must be running in another terminal):
    python test_api.py --data_dir "C:\\...\\mit-bih..." --record 200 --beat 10

Arguments:
    --data_dir  folder with the MIT-BIH records (.dat/.hea/.atr)
    --record    record number (default 200)
    --beat      which beat (by index) to test (default 10)
    --url       server address (default http://127.0.0.1:8000)
"""
import os
import argparse
import numpy as np
import wfdb
import requests

from utility import record_features, N_MORPH, N_CONTEXT
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

    # Only real beats with a valid AAMI class (as in training)
    beats = [(s, sym) for s, sym in zip(annotation.sample, annotation.symbol)
             if sym in AAMI_GROUPS]

    i = args.beat
    if i < 1 or i >= len(beats) - 1:
        raise SystemExit(f"Choose --beat between 1 and {len(beats) - 2}")

    true_symbol = beats[i][1]
    true_class = AAMI_GROUPS[true_symbol]

    # The rhythm features need the preceding beats, so the features are
    # extracted for the whole record (as in training) and the beat is picked
    n_expected = requests.get(f"{args.url}/health", timeout=30).json()['n_features']
    X, kept = record_features(signal, [b[0] for b in beats], fs,
                              rhythm=n_expected > N_MORPH + N_CONTEXT)
    if i not in kept:
        raise SystemExit("The beat is too close to the end of the record, choose another --beat")
    feats = np.nan_to_num(X[kept.index(i)])

    print(f"Record {args.record}, beat #{i}")
    print(f"True symbol: '{true_symbol}' -> AAMI class: {true_class}")
    print(f"Vector length: {len(feats)} features")
    print(f"Sending to {args.url}/predict ...\n")

    resp = requests.post(f"{args.url}/predict",
                         json={"features": feats.tolist()}, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    print("Server response:")
    print(f"  Model:      {result.get('model_type')}")
    print(f"  Predicted:  {result.get('prediction')}  ({result.get('diagnosis')})")
    print(f"  True:       {true_class}")
    match = "correct" if result.get('prediction') == true_class else "wrong"
    print(f"  Result:     {match}")
    if result.get('probabilities'):
        print("  Probabilities:")
        for cls, p in sorted(result['probabilities'].items(),
                             key=lambda kv: -kv[1]):
            print(f"      {cls}: {p:.3f}")


if __name__ == '__main__':
    main()