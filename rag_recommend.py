"""
RAG clinical recommendation based on ECG classification
==================================================
Connects the three parts:
  1. Output of the ECG model (classification of the beats in a record)
  2. Retrieval - finds relevant passages from the guidelines (vector search)
  3. Ollama (Llama 3.2) - generates an assessment and recommendation ONLY from the passages

IMPORTANT - EDUCATIONAL PROTOTYPE:
This system is a student project, NOT a medical device. It does not replace a physician.
Not to be used for real diagnosis or treatment.

Requires:
  * a built index (build_rag_index.py)
  * a running Ollama with the llama3.2 model (ollama serve)
  * a trained ECG model (artifacts_cascade/ - cascade RF, best routing on DS2)

Usage:
    python rag_recommend.py --data_dir "C:\\...\\mit-bih" --record 208
"""
import os
import json
import argparse
import numpy as np
import requests
from functools import lru_cache

from build_rag_index import index_signature, load_index_files

DISCLAIMER = (
    "⚠  EDUCATIONAL PROTOTYPE — not a medical device. Does not replace "
    "clinical judgment. Do not use for real diagnosis or treatment."
)

OLLAMA_URL = "http://localhost:11434/api/generate"


CLASS_NAMES_EN = {'N': 'normal', 'S': 'supraventricular ectopic',
                  'V': 'ventricular ectopic', 'F': 'fusion', 'Q': 'unknown/paced'}

# Threshold for a "dominant abnormality": a non-N class counts as clinically
# significant only if it makes up at least MIN_BURDEN of the beats. Otherwise a single
# ectopic beat in an otherwise normal record (e.g. 0.1% V) would trigger the whole "ventricular" path.
# 5% is a conservative cut-off (a PVC burden below ~1-5% is usually benign;
# >10% is associated with a risk of cardiomyopathy).
MIN_BURDEN = 0.05


# ---------------------------------------------------------------------------
# 1. Analysis of an ECG record -> summary of the findings
# ---------------------------------------------------------------------------
def find_dominant(counts, min_burden=MIN_BURDEN):
    """The most frequent non-N class if it is at least min_burden of the beats; otherwise None."""
    total = sum(counts.values())
    abnormal = {c: n for c, n in counts.items() if c != 'N'}
    if not total or not abnormal:
        return None
    dominant = max(abnormal, key=abnormal.get)
    return dominant if abnormal[dominant] / total >= min_burden else None


def summarize_counts(record, counts, fs=None):
    """Text summary of the findings (in English - for Llama and the guidelines)."""
    total = sum(counts.values())
    hz = f", sampling {fs} Hz" if fs else ""
    lines = [f"ECG record {record}: {total} beats analyzed{hz}."]
    for cls, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * cnt / total
        lines.append(f"  - {CLASS_NAMES_EN.get(cls, cls)} ({cls}): {cnt} beats ({pct:.1f}%)")
    return "\n".join(lines)


def classify_beats(X, artifacts):
    """Classifies a feature matrix with the model in artifacts (sklearn or keras)."""
    import joblib
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
        return le.inverse_transform(np.argmax(p, axis=1))
    model = joblib.load(os.path.join(artifacts, 'model.pkl'))
    return le.inverse_transform(model.predict(Xs))


def analyze_record(data_dir, record, artifacts, lead=0, min_burden=MIN_BURDEN):
    """
    Classifies all beats and returns a text summary of the findings.
    Also returns the true (annotated) classes - for evaluating the pipeline.
    """
    from collections import Counter
    from train_arrhythmia import load_beats
    from utility import record_features, uses_rhythm_features

    signals, fs, beats = load_beats(os.path.join(data_dir, record))
    X, kept = record_features(signals[:, lead], [b[0] for b in beats], fs,
                              rhythm=uses_rhythm_features(artifacts))
    X = np.nan_to_num(X)
    pred = classify_beats(X, artifacts)
    true = [beats[i][1] for i in kept]

    counts = Counter(pred)
    summary = summarize_counts(record, counts, fs)
    dominant = find_dominant(counts, min_burden)
    return {'summary': summary, 'counts': counts, 'total': len(pred),
            'dominant': dominant, 'true_counts': Counter(true), 'fs': fs}


# ---------------------------------------------------------------------------
# 2. Retrieval - finds relevant passages from the guidelines
# ---------------------------------------------------------------------------
# Embeddings of the fixed per-class queries (build_query_for_class), computed
# once when the index is built. With them, retrieval for the app's queries
# needs no embedding model at run time (no PyTorch in the desktop app), and
# the result is identical to encoding the query live.
QUERY_CACHE_FILE = 'query_embeddings.json'


def _file_stamp(path):
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except FileNotFoundError:
        return None


@lru_cache(maxsize=4)
def _load_index_cached(index_dir, stamp):
    """Reads the index files; cached per version of the files (stamp)."""
    model_name, chunks, embeddings = load_index_files(index_dir)
    query_cache = {}
    cache_path = os.path.join(index_dir, QUERY_CACHE_FILE)
    if os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            query_cache = json.load(f)
    return {'model': model_name}, chunks, embeddings, query_cache


def _load_index(index_dir):
    """The index: config, chunks, embeddings, query cache. Reloaded automatically
    when its files change on disk (e.g. the RAG service added a document or
    build_rag_index.py rebuilt it) - no restart needed."""
    stamp = (index_signature(index_dir), _file_stamp(os.path.join(index_dir, QUERY_CACHE_FILE)))
    return _load_index_cached(index_dir, stamp)


@lru_cache(maxsize=2)
def _load_encoder(model_name):
    """The embedding model - loaded only for queries that are not cached."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_query(query, index_dir):
    """Normalized query embedding: from the cache if possible, else encoded live."""
    cfg, _, _, query_cache = _load_index(os.path.abspath(index_dir))
    if query in query_cache:
        return np.asarray(query_cache[query], dtype=np.float32)
    return _load_encoder(cfg['model']).encode([query], normalize_embeddings=True)[0]


def save_class_query_embeddings(index_dir):
    """Computes and saves the embeddings of all per-class queries."""
    cfg, _, _, _ = _load_index(os.path.abspath(index_dir))
    model = _load_encoder(cfg['model'])
    queries = sorted({build_query_for_class(c) for c in [None, 'N', 'S', 'V', 'F', 'Q']})
    vectors = model.encode(queries, normalize_embeddings=True)
    with open(os.path.join(index_dir, QUERY_CACHE_FILE), 'w', encoding='utf-8') as f:
        json.dump({q: v.tolist() for q, v in zip(queries, vectors)}, f)
    return len(queries)


# RAG microservice (rag_service/, optional). When a service URL is set,
# retrieval goes through it: live embeddings, any query, uploaded documents.
# Otherwise the local index in index_dir is used.
RAG_SERVICE_URL = os.environ.get('RAG_SERVICE_URL', '')


def retrieve(query, index_dir, top_k=5, service_url=None):
    """Vector search - returns the top_k most relevant chunks as (chunk, score)."""
    url = (RAG_SERVICE_URL if service_url is None else service_url).rstrip('/')
    if url:
        resp = requests.post(url + '/search', json={'query': query, 'top_k': top_k}, timeout=60)
        resp.raise_for_status()
        return [({'text': h['text'], 'source': h['source']}, h['score']) for h in resp.json()]

    _, chunks, embeddings, _ = _load_index(os.path.abspath(index_dir))
    q_emb = embed_query(query, index_dir)

    # Cosine similarity (the embeddings are normalized -> dot product)
    scores = embeddings @ q_emb
    top_idx = np.argsort(-scores)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# 2b. Building a targeted query per class
# ---------------------------------------------------------------------------
def build_query_for_class(dominant):
    """
    Builds a query that matches the LANGUAGE of the target guideline.
    Important: for supraventricular (S) we avoid the word 'ventricular' (a substring of
    'supraventricular'), which pulls the search towards the wrong guideline.
    Instead we use the terms of the atrial fibrillation guideline.
    """
    if dominant == 'V':
        return ("ventricular arrhythmia premature ventricular contractions "
                "ICD sudden cardiac death risk management catheter ablation")
    if dominant == 'S':
        return ("atrial fibrillation supraventricular anticoagulation stroke "
                "prevention rate rhythm control CHA2DS2-VASc")
    if dominant in ('F', 'Q'):
        return ("ventricular arrhythmia evaluation structural heart disease "
                "monitoring")
    # normal / no dominant abnormality
    return "normal sinus rhythm evaluation monitoring low risk"


# ---------------------------------------------------------------------------
# 3. Ollama - generates a recommendation from the passages
# ---------------------------------------------------------------------------
def ask_ollama(prompt, model='llama3.2', temperature=0.2, seed=None):
    """Sends a request to the local Ollama. seed -> reproducible answer."""
    options = {'temperature': temperature}
    if seed is not None:
        options['seed'] = seed
    resp = requests.post(OLLAMA_URL, json={
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': options,
    }, timeout=600)
    resp.raise_for_status()
    return resp.json()['response']


def build_prompt(ecg_summary, dominant_name, passages, max_chars=700):
    """Builds the prompt for Llama - strictly based on the given passages.
    The passages are truncated so they do not overload the small local model."""
    context = "\n\n".join(
        f"[Source {i+1}: {p['source'][:40]}]\n{p['text'][:max_chars]}"
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
    ap.add_argument('--artifacts', default='artifacts_cascade')
    ap.add_argument('--index', default='rag_index')
    ap.add_argument('--model', default='llama3.2')
    ap.add_argument('--top_k', type=int, default=4)
    ap.add_argument('--min_burden', type=float, default=MIN_BURDEN,
                    help='minimum share of a non-N class for it to be "dominant"')
    args = ap.parse_args()

    print("=" * 70)
    print(DISCLAIMER)
    print("=" * 70)

    # 1. ECG analysis
    print(f"\n[1/3] Analyzing ECG record {args.record}...")
    res = analyze_record(args.data_dir, args.record, args.artifacts,
                         min_burden=args.min_burden)
    summary, dominant = res['summary'], res['dominant']
    print(summary)

    # 2. Retrieval
    dominant_name = CLASS_NAMES_EN.get(dominant) if dominant else None
    query = build_query_for_class(dominant)
    print(f"\n[2/3] Searching the guidelines: '{query}'...")
    passages = retrieve(query, args.index, args.top_k)
    for i, (p, score) in enumerate(passages):
        print(f"  [{i+1}] ({score:.3f}) {p['source'][:45]}... {p['text'][:80]}...")

    # 3. Llama recommendation
    print(f"\n[3/3] Generating the recommendation with {args.model}...")
    prompt = build_prompt(summary, dominant_name, passages)
    try:
        answer = ask_ollama(prompt, model=args.model)
    except requests.exceptions.ConnectionError:
        print("\n[!] Cannot connect to Ollama on localhost:11434.")
        print("    Make sure Ollama is running (ollama serve) and the model is pulled:")
        print(f"    ollama pull {args.model}")
        return

    print("\n" + "=" * 70)
    print("CLINICAL RECOMMENDATION (educational)")
    print("=" * 70)
    print(answer)
    print("\n" + "=" * 70)
    print(DISCLAIMER)
    print("=" * 70)


if __name__ == '__main__':
    main()
