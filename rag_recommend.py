"""
RAG клинична препоръка на база ЕКГ класификация
==================================================
Свързва трите части:
  1. Изход от ЕКГ модела (класификация на удари в запис)
  2. Retrieval - намира релевантни пасажи от guidelines (векторно търсене)
  3. Ollama (Llama 3.2) - генерира оценка и препоръка САМО на база пасажите

ВАЖНО - ОБРАЗОВАТЕЛЕН ПРОТОТИП:
Тази система е учебен проект, НЕ медицинско изделие. Не замества лекар.
Не се използва за реална диагностика или лечение.

Изисква:
  * изграден индекс (build_rag_index.py)
  * работеща Ollama с модел llama3.2 (ollama serve)
  * обучен ЕКГ модел (artifacts_mlp/)

Стартиране:
    python rag_recommend.py --data_dir "C:\\...\\mit-bih" --record 208
"""
import os
import json
import argparse
import numpy as np
import requests

DISCLAIMER = (
    "⚠  ОБРАЗОВАТЕЛЕН ПРОТОТИП — не е медицинско изделие. Не замества "
    "лекарска оценка. Не използвайте за реална диагностика или лечение."
)

OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------------------------
# 1. Анализ на ЕКГ запис -> обобщение на находките
# ---------------------------------------------------------------------------
def analyze_record(data_dir, record, artifacts, lead=0):
    """Класифицира всички удари и връща текстово обобщение на находките."""
    import wfdb
    import joblib
    from collections import Counter
    from train_arrhythmia import AAMI_GROUPS, extract_beat_features

    rec_path = os.path.join(data_dir, record)
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    signal = rec.p_signal[:, lead]
    fs = rec.fs

    beats = [(s, sym) for s, sym in zip(ann.sample, ann.symbol)
             if sym in AAMI_GROUPS]
    peaks = [b[0] for b in beats]
    rr_all = [(peaks[k] - peaks[k - 1]) / fs for k in range(1, len(peaks))]

    X = []
    for i, (r_peak, sym) in enumerate(beats):
        prev_r = beats[i - 1][0] if i > 0 else None
        next_r = beats[i + 1][0] if i < len(beats) - 1 else None
        lo = max(0, i - 10)
        window_rr = rr_all[lo:i] if i > 0 else []
        lmr = float(np.mean(window_rr)) if window_rr else None
        feats = extract_beat_features(signal, r_peak, prev_r, next_r, fs,
                                      local_mean_rr=lmr)
        if feats is not None:
            X.append(feats)
    X = np.array(X)

    # Зареждаме модела (sklearn или keras)
    scaler = joblib.load(os.path.join(artifacts, 'scaler.pkl'))
    le = joblib.load(os.path.join(artifacts, 'label_encoder.pkl'))
    Xs = scaler.transform(X)
    keras_path = os.path.join(artifacts, 'model.keras')
    if os.path.exists(keras_path):
        import tensorflow as tf
        model = tf.keras.models.load_model(keras_path)
        meta = joblib.load(os.path.join(artifacts, 'meta.pkl'))
        nc = meta['n_context']
        p = model.predict({'morphology': Xs[:, :-nc][..., np.newaxis],
                           'context': Xs[:, -nc:]}, verbose=0)
        pred = le.inverse_transform(np.argmax(p, axis=1))
    else:
        model = joblib.load(os.path.join(artifacts, 'model.pkl'))
        pred = le.inverse_transform(model.predict(Xs))

    counts = Counter(pred)
    total = len(pred)
    names = {'N': 'normal', 'S': 'supraventricular ectopic',
             'V': 'ventricular ectopic', 'F': 'fusion', 'Q': 'unknown/paced'}

    # Текстово обобщение (на английски - за Llama и за guidelines)
    lines = [f"ECG record {record}: {total} beats analyzed, sampling {fs} Hz."]
    for cls, cnt in counts.most_common():
        pct = 100.0 * cnt / total
        lines.append(f"  - {names.get(cls, cls)} ({cls}): {cnt} beats ({pct:.1f}%)")
    summary = "\n".join(lines)

    # Определяме доминиращата аномалия (без N)
    abnormal = {c: n for c, n in counts.items() if c != 'N'}
    dominant = max(abnormal, key=abnormal.get) if abnormal else None
    return summary, counts, total, dominant, names


# ---------------------------------------------------------------------------
# 2. Retrieval - намира релевантни пасажи от guidelines
# ---------------------------------------------------------------------------
def retrieve(query, index_dir, top_k=5):
    """Векторно търсене - връща top_k най-релевантни части."""
    from sentence_transformers import SentenceTransformer

    with open(os.path.join(index_dir, 'config.json')) as f:
        cfg = json.load(f)
    with open(os.path.join(index_dir, 'chunks.json'), encoding='utf-8') as f:
        chunks = json.load(f)
    embeddings = np.load(os.path.join(index_dir, 'embeddings.npy'))

    model = SentenceTransformer(cfg['model'])
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    # Косинусова близост (embeddings са нормализирани -> скаларно произведение)
    scores = embeddings @ q_emb
    top_idx = np.argsort(-scores)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# 3. Ollama - генерира препоръка на база пасажите
# ---------------------------------------------------------------------------
def ask_ollama(prompt, model='llama3.2', temperature=0.2):
    """Праща заявка към локалната Ollama."""
    resp = requests.post(OLLAMA_URL, json={
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': temperature},
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()['response']


def build_prompt(ecg_summary, dominant_name, passages):
    """Съставя промпта за Llama - строго на база подадените пасажи."""
    context = "\n\n".join(
        f"[Source {i+1}: {p['source']}]\n{p['text']}"
        for i, (p, _) in enumerate(passages)
    )
    return f"""You are a clinical decision-support assistant for EDUCATIONAL purposes.
You are given (1) automated ECG classification findings and (2) excerpts from
official ACC/AHA clinical practice guidelines.

Base your answer STRICTLY on the provided guideline excerpts. If the excerpts
do not cover something, say so explicitly. Do NOT invent facts, drug doses, or
recommendations that are not in the excerpts. Cite sources as [Source N].

=== ECG FINDINGS ===
{ecg_summary}

Predominant abnormality: {dominant_name or "none (mostly normal)"}

=== GUIDELINE EXCERPTS ===
{context}

=== TASK ===
Write a short, structured educational note with:
1. ASSESSMENT: what the ECG findings suggest, in plain language.
2. RELEVANT GUIDELINE POINTS: what the excerpts say that applies here, with [Source N] citations.
3. GENERAL CONSIDERATIONS: what such findings generally warrant (monitoring, further evaluation), strictly per the excerpts.
4. LIMITATIONS: note that this is an automated educational estimate, not a diagnosis.

Keep it concise and factual."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--record', default='208')
    ap.add_argument('--artifacts', default='artifacts_mlp')
    ap.add_argument('--index', default='rag_index')
    ap.add_argument('--model', default='llama3.2')
    ap.add_argument('--top_k', type=int, default=5)
    args = ap.parse_args()

    print("=" * 70)
    print(DISCLAIMER)
    print("=" * 70)

    # 1. Анализ на ЕКГ
    print(f"\n[1/3] Анализ на ЕКГ запис {args.record}...")
    summary, counts, total, dominant, names = analyze_record(
        args.data_dir, args.record, args.artifacts)
    print(summary)

    # 2. Retrieval
    dominant_name = names.get(dominant) if dominant else None
    query = (f"management and evaluation of {dominant_name} arrhythmia "
             f"clinical recommendations") if dominant else \
            "normal sinus rhythm evaluation"
    print(f"\n[2/3] Търсене в guidelines: '{query}'...")
    passages = retrieve(query, args.index, args.top_k)
    for i, (p, score) in enumerate(passages):
        print(f"  [{i+1}] ({score:.3f}) {p['source'][:45]}... {p['text'][:80]}...")

    # 3. Llama препоръка
    print(f"\n[3/3] Генериране на препоръка с {args.model}...")
    prompt = build_prompt(summary, dominant_name, passages)
    try:
        answer = ask_ollama(prompt, model=args.model)
    except requests.exceptions.ConnectionError:
        print("\n[!] Не мога да се свържа с Ollama на localhost:11434.")
        print("    Увери се, че Ollama работи (ollama serve) и моделът е свален:")
        print(f"    ollama pull {args.model}")
        return

    print("\n" + "=" * 70)
    print("КЛИНИЧНА ПРЕПОРЪКА (образователна)")
    print("=" * 70)
    print(answer)
    print("\n" + "=" * 70)
    print(DISCLAIMER)
    print("=" * 70)


if __name__ == '__main__':
    main()
